"""Vector + keyword store backed by Neon PostgreSQL with pgvector.

Retrieval pipeline:
  1. Vector search   — Gemini embeddings + cosine similarity (pgvector)
  2. Keyword search  — ILIKE token matching
  3. RRF fusion      — Reciprocal Rank Fusion combines both ranked lists
  4. Gemini rerank   — single LLM call re-orders the top candidates by relevance

Embeddings: Google Gemini gemini-embedding-001 (768-dim), L2-normalized.
"""
import asyncio
import json
import logging
import re
from typing import Any, Dict, List

import numpy as np
import google.generativeai as genai
from sqlalchemy import text

from backend.config import settings
from backend.database import neon_session

logger = logging.getLogger(__name__)


_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "has",
    "are", "was", "were", "but", "not", "you", "your", "they", "them",
    "their", "there", "here", "when", "what", "which", "who", "whom",
    "how", "why", "into", "out", "about", "over", "under", "between",
    "any", "all", "some", "more", "most", "other", "such", "than",
    "then", "also", "just", "only", "very", "much", "many",
}


def _run_async(coro):
    """Run an async coroutine from a sync method, even if a loop is already running."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


class NeonVectorStore:
    EMBED_MODEL = "models/gemini-embedding-001"
    EMBED_DIM = 768

    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        logger.info(
            "NeonVectorStore initialized with Gemini gemini-embedding-001 "
            f"({self.EMBED_DIM}-dim)."
        )

    def _embed(
        self,
        texts: List[str],
        task_type: str = "retrieval_document",
    ) -> List[List[float]]:
        """Embed texts via Gemini and L2-normalize each vector.

        gemini-embedding-001 only accepts a single content per call, so we
        iterate one-by-one rather than batching.
        """
        if not texts:
            return []

        genai.configure(api_key=settings.GEMINI_API_KEY)

        out: List[List[float]] = []
        for content in texts:
            result = genai.embed_content(
                model=self.EMBED_MODEL,
                content=content,
                task_type=task_type,
                output_dimensionality=self.EMBED_DIM,
            )
            vec = result["embedding"]
            arr = np.asarray(vec, dtype=np.float32)
            norm = float(np.linalg.norm(arr))
            if norm > 0:
                arr = arr / norm
            out.append(arr.tolist())
        return out

    def _vec_literal(self, vec: List[float]) -> str:
        return "[" + ",".join(f"{v:.6f}" for v in vec) + "]"

    def _gemini_rerank(
        self, query: str, candidates: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        """Use a single Gemini call to rerank RRF candidates by true relevance.

        Returns the top-k candidates in the new order. Falls back to the
        original RRF order if Gemini is unavailable or returns bad output.
        """
        if len(candidates) <= top_k:
            return candidates

        snippet_list = "\n\n".join(
            f"[{i + 1}] {c['chunk'][:350]}" for i, c in enumerate(candidates)
        )
        prompt = (
            f"You are a relevance ranking expert.\n\n"
            f"Question: {query}\n\n"
            f"Document chunks:\n{snippet_list}\n\n"
            "Rank the chunks from most to least relevant to answering the question.\n"
            "Return ONLY a JSON array of chunk numbers (1-indexed), e.g. [3, 1, 5, 2, 4].\n"
            "No explanation, no markdown — just the JSON array."
        )
        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.0, "max_output_tokens": 120},
            )
            raw = (response.text or "").strip()
            m = re.search(r"\[[\d,\s]+\]", raw)
            if m:
                indices = json.loads(m.group())
                seen: set = set()
                reranked: List[Dict[str, Any]] = []
                for idx in indices:
                    if isinstance(idx, int) and 0 < idx <= len(candidates):
                        item = candidates[idx - 1]
                        key = id(item)
                        if key not in seen:
                            reranked.append(item)
                            seen.add(key)
                # Append any candidates not mentioned by Gemini
                for item in candidates:
                    if id(item) not in seen:
                        reranked.append(item)
                return reranked[:top_k]
        except Exception as e:
            logger.warning(f"Gemini rerank failed (falling back to RRF order): {e}")

        return candidates[:top_k]

    def add_documents(self, chunks: List[str], doc_id: str) -> int:
        return _run_async(self._add_documents_async(chunks, doc_id))

    async def _add_documents_async(self, chunks: List[str], doc_id: str) -> int:
        if not chunks:
            return 0

        embeddings = self._embed(chunks, task_type="retrieval_document")
        inserted = 0

        async with neon_session() as session:
            for idx, (chunk_text, vec) in enumerate(zip(chunks, embeddings)):
                chunk_text = chunk_text.replace("\x00", "")
                vec_lit = self._vec_literal(vec)
                try:
                    await session.execute(
                        text(
                            """
                            INSERT INTO document_chunks
                                (doc_name, chunk_index, chunk_text, embedding,
                                 search_vector, metadata)
                            VALUES
                                (:doc_name, :chunk_index, :chunk_text,
                                 CAST(:embedding AS vector),
                                 to_tsvector('english', :chunk_text),
                                 CAST(:metadata AS jsonb))
                            """
                        ),
                        {
                            "doc_name": doc_id,
                            "chunk_index": idx,
                            "chunk_text": chunk_text,
                            "embedding": vec_lit,
                            "metadata": '{"doc_id": "%s", "chunk_index": %d}'
                                        % (doc_id.replace('"', '\\"'), idx),
                        },
                    )
                    inserted += 1
                    if inserted % 20 == 0:
                        await session.commit()
                except Exception as e:
                    logger.warning(f"Failed to insert chunk {idx} for {doc_id}: {e}")
                    try:
                        await session.rollback()
                    except Exception:
                        pass

        return inserted

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return _run_async(self._search_async(query, top_k))

    async def _search_async(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        async with neon_session() as session:
            count_result = await session.execute(
                text("SELECT COUNT(*) FROM document_chunks")
            )
            total = count_result.scalar() or 0
            if total == 0:
                return []

            query_vec = self._embed([query], task_type="retrieval_query")[0]
            vec_lit = self._vec_literal(query_vec)

            vector_results: List[Dict[str, Any]] = []
            try:
                vec_res = await session.execute(
                    text(
                        """
                        SELECT id, doc_name, chunk_text,
                               1 - (embedding <=> CAST(:qvec AS vector)) AS similarity
                        FROM document_chunks
                        WHERE 1 - (embedding <=> CAST(:qvec AS vector)) > 0.12
                        ORDER BY embedding <=> CAST(:qvec AS vector) ASC
                        LIMIT :lim
                        """
                    ),
                    {"qvec": vec_lit, "lim": top_k + 5},
                )
                for rank, row in enumerate(vec_res.fetchall()):
                    vector_results.append(
                        {
                            "id": row[0],
                            "doc_name": row[1],
                            "chunk_text": row[2],
                            "similarity": float(row[3]),
                            "rank": rank,
                        }
                    )
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")

            # Build a PostgreSQL tsquery from the query for BM25-like full-text search.
            # We use the stored search_vector (tsvector) column with ts_rank which
            # considers term frequency + IDF + document length — much better than ILIKE.
            words = [
                w.lower()
                for w in re.findall(r"[A-Za-z][A-Za-z0-9]+", query)
                if len(w) >= 3 and w.lower() not in _STOPWORDS
            ]

            keyword_results: List[Dict[str, Any]] = []
            if words:
                try:
                    # Use plainto_tsquery which handles multi-word queries safely
                    tsquery_str = " & ".join(words)
                    kw_res = await session.execute(
                        text(
                            """
                            SELECT id, doc_name, chunk_text,
                                   ts_rank(search_vector,
                                           to_tsquery('english', :tsq)) AS bm25_score
                            FROM document_chunks
                            WHERE search_vector @@ to_tsquery('english', :tsq)
                            ORDER BY bm25_score DESC
                            LIMIT :lim
                            """
                        ),
                        {"tsq": tsquery_str, "lim": top_k + 5},
                    )
                    rows = kw_res.fetchall()

                    # Fallback to ILIKE if tsquery returns nothing
                    # (happens when words don't match PostgreSQL stemming)
                    if not rows and words:
                        conditions = " OR ".join(
                            f"chunk_text ILIKE :w{i}" for i in range(len(words))
                        )
                        params = {f"w{i}": f"%{w}%" for i, w in enumerate(words)}
                        params["lim"] = top_k + 5
                        kw_res2 = await session.execute(
                            text(
                                f"SELECT id, doc_name, chunk_text, 0.1 AS bm25_score "
                                f"FROM document_chunks WHERE {conditions} LIMIT :lim"
                            ),
                            params,
                        )
                        rows = kw_res2.fetchall()

                    for rank, row in enumerate(rows):
                        keyword_results.append(
                            {
                                "id": row[0],
                                "doc_name": row[1],
                                "chunk_text": row[2],
                                "similarity": float(row[3]),
                                "rank": rank,
                            }
                        )
                except Exception as e:
                    logger.warning(f"BM25 keyword search failed: {e}")

            vector_weight = 0.65
            keyword_weight = 0.35

            merged: Dict[int, Dict[str, Any]] = {}
            vector_ids = set()
            keyword_ids = set()

            for r in vector_results:
                vector_ids.add(r["id"])
                merged[r["id"]] = {
                    "id": r["id"],
                    "chunk_text": r["chunk_text"],
                    "score": vector_weight / (r["rank"] + 60),
                }

            for r in keyword_results:
                keyword_ids.add(r["id"])
                add = keyword_weight / (r["rank"] + 60)
                if r["id"] in merged:
                    merged[r["id"]]["score"] += add
                else:
                    merged[r["id"]] = {
                        "id": r["id"],
                        "chunk_text": r["chunk_text"],
                        "score": add,
                    }

            for rid in vector_ids & keyword_ids:
                merged[rid]["score"] += 0.15

            ordered = sorted(merged.values(), key=lambda x: x["score"], reverse=True)

            # Build candidate list for reranking (grab a few extra for the reranker)
            rerank_pool = [
                {"chunk": item["chunk_text"], "score": float(item["score"])}
                for item in ordered[: top_k + 5]
            ]

            # Gemini rerank: re-orders by true relevance, falls back to RRF order
            reranked = self._gemini_rerank(query, rerank_pool, top_k)

            return [
                {"chunk": c["chunk"], "score": c["score"], "rank": i}
                for i, c in enumerate(reranked)
            ]

    def get_doc_list(self) -> List[Dict[str, Any]]:
        return _run_async(self._get_doc_list_async())

    async def _get_doc_list_async(self) -> List[Dict[str, Any]]:
        async with neon_session() as session:
            result = await session.execute(
                text(
                    "SELECT doc_name, COUNT(*) AS c FROM document_chunks "
                    "GROUP BY doc_name ORDER BY doc_name"
                )
            )
            return [
                {"filename": row[0], "chunk_count": int(row[1])}
                for row in result.fetchall()
            ]

    def total_chunks(self) -> int:
        return _run_async(self._total_chunks_async())

    async def _total_chunks_async(self) -> int:
        async with neon_session() as session:
            result = await session.execute(text("SELECT COUNT(*) FROM document_chunks"))
            return int(result.scalar() or 0)

    def clear(self) -> None:
        _run_async(self._clear_async())

    async def _clear_async(self) -> None:
        async with neon_session() as session:
            await session.execute(text("TRUNCATE document_chunks RESTART IDENTITY"))


neon_store = NeonVectorStore()
