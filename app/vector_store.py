import json
from sqlalchemy import bindparam, text
from app.db import SessionLocal


def _to_vector_literal(values):
    return "[" + ",".join(str(value) for value in values) + "]"


def upsert_chunks(source, chunks, embeddings, document_metadata, chunk_metadatas):
    with SessionLocal() as db:
        document_row = db.execute(text("""
        INSERT INTO document (source, metadata, updated_at)
        VALUES (:source, CAST(:metadata AS jsonb), NOW())
        ON CONFLICT (source)
        DO UPDATE SET metadata = EXCLUDED.metadata,
                      updated_at = NOW()
        RETURNING id
        """), {
            "source": source,
            "metadata": json.dumps(document_metadata)
        }).mappings().one()

        document_id = document_row["id"]
        db.execute(text("""
        DELETE FROM document_chunk
        WHERE document_id = :document_id
        """), {"document_id": document_id})

        for i, (content, emb, md) in enumerate(zip(chunks, embeddings, chunk_metadatas)):
            db.execute(text("""
            INSERT INTO document_chunk (
                document_id,
                chunk_index,
                content,
                metadata,
                embedding,
                updated_at
            )
            VALUES (
                :document_id,
                :chunk_index,
                :content,
                CAST(:metadata AS jsonb),
                CAST(:embedding AS vector),
                NOW()
            )
            """), {
                "document_id": document_id,
                "chunk_index": i,
                "content": content,
                "metadata": json.dumps(md),
                "embedding": _to_vector_literal(emb)
            })
        db.commit()

def similarity_search(query_embedding, top_k, source_ids=None):
    sql = """
    SELECT d.source,
           d.metadata AS document_metadata,
           dc.chunk_index::text AS chunk_id,
           dc.content,
           dc.metadata,
           1 - (dc.embedding <=> CAST(:qvec AS vector)) AS score
    FROM document_chunk dc
    JOIN document d ON d.id = dc.document_id
    {where}
    ORDER BY dc.embedding <=> CAST(:qvec AS vector)
    LIMIT :k
    """
    where = ""
    params = {"qvec": _to_vector_literal(query_embedding), "k": top_k}
    stmt = None
    if source_ids:
        where = "WHERE d.source IN :source_ids"
        params["source_ids"] = source_ids
    with SessionLocal() as db:
        stmt = text(sql.format(where=where))
        if source_ids:
            stmt = stmt.bindparams(bindparam("source_ids", expanding=True))
        rows = db.execute(stmt, params).mappings().all()
        return [dict(r) for r in rows]


def get_documents(source_ids=None):
    sql = """
    SELECT d.source,
           d.metadata
    FROM document d
    {where}
    ORDER BY d.updated_at DESC, d.source
    """
    where = ""
    params = {}
    stmt = None
    if source_ids:
        where = "WHERE d.source IN :source_ids"
        params["source_ids"] = source_ids
    with SessionLocal() as db:
        stmt = text(sql.format(where=where))
        if source_ids:
            stmt = stmt.bindparams(bindparam("source_ids", expanding=True))
        rows = db.execute(stmt, params).mappings().all()
        return [dict(r) for r in rows]


def get_document_chunks(source_ids=None):
    sql = """
    SELECT d.source,
           dc.chunk_index,
           dc.content,
           d.metadata AS document_metadata,
           dc.metadata AS chunk_metadata
    FROM document_chunk dc
    JOIN document d ON d.id = dc.document_id
    {where}
    ORDER BY d.source, dc.chunk_index
    """
    where = ""
    params = {}
    stmt = None
    if source_ids:
        where = "WHERE d.source IN :source_ids"
        params["source_ids"] = source_ids
    with SessionLocal() as db:
        stmt = text(sql.format(where=where))
        if source_ids:
            stmt = stmt.bindparams(bindparam("source_ids", expanding=True))
        rows = db.execute(stmt, params).mappings().all()
        return [dict(r) for r in rows]


def get_analysis_cache(cache_key, analysis_type="default"):
    with SessionLocal() as db:
        row = db.execute(text("""
        SELECT cache_key, source_ids, analysis_type, result, created_at, updated_at
        FROM document_analysis
        WHERE cache_key = :cache_key AND analysis_type = :analysis_type
        """), {
            "cache_key": cache_key,
            "analysis_type": analysis_type,
        }).mappings().first()
        return dict(row) if row else None


def upsert_analysis_cache(cache_key, source_ids, result, analysis_type="default"):
    with SessionLocal() as db:
        db.execute(text("""
        INSERT INTO document_analysis (cache_key, source_ids, analysis_type, result, updated_at)
        VALUES (
            :cache_key,
            CAST(:source_ids AS jsonb),
            :analysis_type,
            CAST(:result AS jsonb),
            NOW()
        )
        ON CONFLICT (cache_key)
        DO UPDATE SET source_ids = EXCLUDED.source_ids,
                      analysis_type = EXCLUDED.analysis_type,
                      result = EXCLUDED.result,
                      updated_at = NOW()
        """), {
            "cache_key": cache_key,
            "source_ids": json.dumps(source_ids),
            "analysis_type": analysis_type,
            "result": json.dumps(result),
        })
        db.commit()


def append_conversation_message(conversation_id, role, content, metadata=None):
    with SessionLocal() as db:
        db.execute(text("""
        INSERT INTO conversation_message (conversation_id, role, content, metadata)
        VALUES (:conversation_id, :role, :content, CAST(:metadata AS jsonb))
        """), {
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "metadata": json.dumps(metadata or {}),
        })
        db.commit()


def get_conversation_messages(conversation_id, limit=6):
    with SessionLocal() as db:
        rows = db.execute(text("""
        SELECT role, content, metadata, created_at
        FROM conversation_message
        WHERE conversation_id = :conversation_id
        ORDER BY created_at DESC
        LIMIT :limit
        """), {
            "conversation_id": conversation_id,
            "limit": limit,
        }).mappings().all()
        messages = [dict(row) for row in rows]
        messages.reverse()
        return messages
