from io import BytesIO
import json
import re
import secrets
from urllib.parse import unquote, urlparse
from uuid import uuid4

from fastapi import Depends, FastAPI, UploadFile, File, Form, HTTPException, Query, status
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pypdf import PdfReader
from app.db import init_db
from app.embeddings import embed_texts
from app.vector_store import (
    append_conversation_message,
    get_analysis_cache,
    get_conversation_messages,
    get_document_chunks,
    get_documents,
    upsert_analysis_cache,
    upsert_chunks,
    similarity_search,
)
from app.config import settings
import httpx

security = HTTPBasic()


def require_basic_auth(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    provided_username = credentials.username.strip()
    provided_password = credentials.password.strip()
    expected_username = settings.basic_auth_username.strip()
    expected_password = settings.basic_auth_password.strip()

    correct_username = secrets.compare_digest(provided_username, expected_username)
    correct_password = secrets.compare_digest(provided_password, expected_password)

    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials.",
            headers={"WWW-Authenticate": "Basic"},
        )

    return provided_username


app = FastAPI(
    title=settings.swagger_title,
    description=settings.swagger_description,
    version=settings.swagger_version,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    dependencies=[Depends(require_basic_auth)],
)


def _clean_text(value: str) -> str:
    return value.replace("\x00", "").strip()


def _extract_text_segments(file_bytes: bytes, filename: str | None) -> list[dict]:
    lowered_name = (filename or "").lower()

    if lowered_name.endswith(".pdf") or file_bytes.startswith(b"%PDF"):
        reader = PdfReader(BytesIO(file_bytes))
        segments = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = _clean_text(page.extract_text() or "")
            if not page_text:
                continue
            segments.append({
                "text": page_text,
                "page_start": page_number,
                "page_end": page_number,
            })
        return segments

    decoded = file_bytes.decode("utf-8", errors="ignore")
    cleaned = _clean_text(decoded)
    if not cleaned:
        return []
    return [{
        "text": cleaned,
        "page_start": 1,
        "page_end": 1,
    }]


def _filename_from_url(file_url: str) -> str:
    parsed = urlparse(file_url)
    name = unquote(parsed.path.rsplit("/", 1)[-1]).strip()
    return name or "remote-file"


async def _fetch_remote_file(file_url: str) -> tuple[bytes, str, str]:
    normalized_url = (file_url or "").strip()
    parsed = urlparse(normalized_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail="file_url must be a valid http or https URL.")

    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            response = await client.get(normalized_url)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch file from URL (HTTP {exc.response.status_code}).",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Failed to fetch file from URL.") from exc

    content_type = (response.headers.get("content-type") or "application/octet-stream").split(";", 1)[0].strip()
    return response.content, _filename_from_url(normalized_url), content_type or "application/octet-stream"


def _split_text_units(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    if len(paragraphs) > 1:
        return paragraphs

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if len(sentences) > 1:
        return sentences

    return [part.strip() for part in text.splitlines() if part.strip()] or [text.strip()]


def _build_chunks(segments: list[dict], chunk_size: int = 900, overlap_size: int = 150) -> tuple[list[str], list[dict]]:
    chunks = []
    chunk_metadatas = []

    for segment in segments:
        units = _split_text_units(segment["text"])
        current_parts = []

        def flush_chunk():
            nonlocal current_parts
            chunk_text = "\n\n".join(part for part in current_parts if part).strip()
            if not chunk_text:
                current_parts = []
                return

            chunks.append(chunk_text)
            chunk_metadatas.append({
                "page_start": segment["page_start"],
                "page_end": segment["page_end"],
            })

            overlap_text = chunk_text[-overlap_size:].strip()
            current_parts = [overlap_text] if overlap_text else []

        for unit in units:
            unit = unit.strip()
            if not unit:
                continue

            if len(unit) > chunk_size:
                if current_parts:
                    flush_chunk()
                for i in range(0, len(unit), chunk_size - overlap_size):
                    piece = unit[i:i + chunk_size].strip()
                    if not piece:
                        continue
                    chunks.append(piece)
                    chunk_metadatas.append({
                        "page_start": segment["page_start"],
                        "page_end": segment["page_end"],
                    })
                current_parts = []
                continue

            candidate_parts = current_parts + [unit]
            candidate_text = "\n\n".join(candidate_parts).strip()
            if len(candidate_text) <= chunk_size:
                current_parts = candidate_parts
                continue

            flush_chunk()
            current_parts.append(unit)

        if current_parts:
            flush_chunk()

    return chunks, chunk_metadatas


def _format_page_label(metadata: dict | None) -> str:
    if not isinstance(metadata, dict):
        return ""

    page_start = metadata.get("page_start")
    page_end = metadata.get("page_end")
    if not page_start:
        return ""
    if not page_end or page_end == page_start:
        return f"page {page_start}"
    return f"pages {page_start}-{page_end}"


def _format_chunk_header(source: str, chunk_id: str | int, metadata: dict | None) -> str:
    page_label = _format_page_label(metadata)
    if page_label:
        return f"{source}#{chunk_id} ({page_label})"
    return f"{source}#{chunk_id}"


def _normalize_source_ids(source_id: str | None, source_ids: list[str] | None) -> list[str] | None:
    values = []

    if source_id:
        values.extend(part.strip() for part in source_id.split(","))

    if source_ids:
        for item in source_ids:
            values.extend(part.strip() for part in item.split(","))

    cleaned = [value for value in values if value]
    return cleaned or None


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    normalized = normalized.strip("-")
    return normalized or "untitled-document"


def _tokenize_for_ranking(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]{2,}", (text or "").lower())
        if token
    }


def _score_document_relevance(question: str, document: dict) -> float:
    metadata = document.get("metadata") if isinstance(document, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}

    tags = metadata.get("tags", [])
    if not isinstance(tags, list):
        tags = []

    searchable_text = " ".join([
        document.get("source", "") or "",
        metadata.get("source_title", "") or "",
        metadata.get("summary", "") or "",
        " ".join(tag for tag in tags if isinstance(tag, str)),
    ])
    question_tokens = _tokenize_for_ranking(question)
    document_tokens = _tokenize_for_ranking(searchable_text)

    overlap = len(question_tokens & document_tokens)
    title_bonus = 2 if metadata.get("source_title") else 0
    summary_bonus = 1 if metadata.get("summary") else 0
    return overlap * 3 + title_bonus + summary_bonus


def _prefilter_source_ids(question: str, requested_source_ids: list[str] | None, max_sources: int = 3) -> list[str] | None:
    documents = get_documents(requested_source_ids)
    if not documents:
        return requested_source_ids

    ranked = sorted(
        documents,
        key=lambda document: (_score_document_relevance(question, document), document.get("source", "")),
        reverse=True,
    )
    selected = [document["source"] for document in ranked[:max_sources] if document.get("source")]
    return selected or requested_source_ids


def _score_chunk_relevance(question: str, hit: dict) -> float:
    question_tokens = _tokenize_for_ranking(question)
    content_tokens = _tokenize_for_ranking(hit.get("content", ""))
    metadata = hit.get("document_metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    metadata_tokens = _tokenize_for_ranking(" ".join([
        metadata.get("source_title", "") or "",
        metadata.get("summary", "") or "",
    ]))
    overlap = len(question_tokens & content_tokens)
    metadata_overlap = len(question_tokens & metadata_tokens)
    vector_score = float(hit.get("vector_score", hit.get("score", 0)))
    keyword_score = float(hit.get("keyword_score", 0))
    return vector_score * 10 + keyword_score * 5 + overlap * 2 + metadata_overlap


def _rerank_hits(question: str, hits: list[dict], top_k: int) -> list[dict]:
    ranked = sorted(hits, key=lambda hit: _score_chunk_relevance(question, hit), reverse=True)
    return ranked[:top_k]


def _build_ask_context(hits: list[dict]) -> str:
    context_parts = []

    for hit in hits:
        document_metadata = hit.get("document_metadata")
        if not isinstance(document_metadata, dict):
            document_metadata = {}

        tags = document_metadata.get("tags", [])
        if not isinstance(tags, list):
            tags = []

        header = _format_chunk_header(hit["source"], hit["chunk_id"], hit["metadata"])
        context_parts.append(
            "\n".join([
                f"SOURCE_ID: {hit['source']}",
                f"TITLE: {document_metadata.get('source_title', '')}",
                f"SUMMARY: {document_metadata.get('summary', '')}",
                f"TAGS: {', '.join(tag for tag in tags if isinstance(tag, str))}",
                f"CHUNK: {header}",
                hit["content"],
            ]).strip()
        )

    return "\n\n".join(context_parts)


def _serialize_document(document: dict) -> dict:
    metadata = document.get("metadata") if isinstance(document, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}

    tags = metadata.get("tags", [])
    if not isinstance(tags, list):
        tags = []

    return {
        "source_id": document.get("source"),
        "source_title": metadata.get("source_title", ""),
        "summary": metadata.get("summary", ""),
        "tags": [tag for tag in tags if isinstance(tag, str)],
        "filename": metadata.get("filename", ""),
        "file_type": metadata.get("file_type", ""),
        "file_size_bytes": metadata.get("file_size_bytes"),
        "page_count": metadata.get("page_count"),
    }


def _score_document_search(query_text: str, document: dict) -> float:
    base_score = _score_document_relevance(query_text, document)
    metadata = document.get("metadata") if isinstance(document, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}

    normalized_query = query_text.strip().lower()
    searchable_text = " ".join([
        document.get("source", "") or "",
        metadata.get("source_title", "") or "",
        metadata.get("summary", "") or "",
        " ".join(tag for tag in metadata.get("tags", []) if isinstance(tag, str)),
    ]).lower()
    exact_bonus = 4 if normalized_query and normalized_query in searchable_text else 0
    return base_score + exact_bonus


def _search_documents(query_text: str | None, source_ids: list[str] | None, limit: int) -> list[dict]:
    documents = get_documents(source_ids)
    if not query_text:
        return [_serialize_document(document) for document in documents[:limit]]

    ranked = sorted(
        documents,
        key=lambda document: (_score_document_search(query_text, document), document.get("source", "")),
        reverse=True,
    )
    filtered = [document for document in ranked if _score_document_search(query_text, document) > 0]
    return [_serialize_document(document) for document in filtered[:limit]]


def _build_analysis_cache_key(source_ids: list[str], analysis_type: str = "default") -> str:
    ordered = sorted(str(source_id) for source_id in source_ids)
    return f"{analysis_type}:{'|'.join(ordered)}"


def _build_conversation_context(messages: list[dict]) -> str:
    if not messages:
        return ""

    parts = []
    for message in messages:
        role = str(message.get("role", "")).upper()
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        parts.append(f"{role}:\n{content}")
    return "\n\n".join(parts)


def _get_memory_source_ids(messages: list[dict]) -> list[str] | None:
    for message in reversed(messages):
        metadata = message.get("metadata")
        if not isinstance(metadata, dict):
            continue

        source_ids = metadata.get("source_ids")
        if not isinstance(source_ids, list):
            continue

        cleaned = [str(source_id).strip() for source_id in source_ids if str(source_id).strip()]
        if cleaned:
            return cleaned

    return None


def _merge_hits(vector_hits: list[dict], keyword_hits: list[dict], top_k: int) -> list[dict]:
    merged = {}

    for hit in vector_hits:
        key = (hit["source"], str(hit["chunk_id"]))
        merged[key] = {
            **hit,
            "chunk_id": str(hit["chunk_id"]),
            "vector_score": float(hit.get("score", 0)),
            "keyword_score": 0.0,
        }

    for hit in keyword_hits:
        key = (hit["source"], str(hit["chunk_id"]))
        if key in merged:
            merged[key]["keyword_score"] = max(merged[key].get("keyword_score", 0.0), float(hit.get("keyword_score", 0)))
            continue

        merged[key] = {
            **hit,
            "chunk_id": str(hit["chunk_id"]),
            "score": 0.0,
            "vector_score": 0.0,
            "keyword_score": float(hit.get("keyword_score", 0)),
        }

    ranked = sorted(
        merged.values(),
        key=lambda hit: (
            float(hit.get("vector_score", 0)) + float(hit.get("keyword_score", 0)),
            hit["source"],
            hit["chunk_id"],
        ),
        reverse=True,
    )
    return ranked[:top_k]


def _keyword_search(query_text: str, source_ids: list[str] | None, top_k: int) -> list[dict]:
    chunks = get_document_chunks(source_ids)
    query_tokens = _tokenize_for_ranking(query_text)
    if not query_tokens:
        return []

    matches = []
    normalized_query = query_text.strip().lower()

    for chunk in chunks:
        document_metadata = chunk.get("document_metadata")
        if not isinstance(document_metadata, dict):
            document_metadata = {}

        chunk_metadata = chunk.get("chunk_metadata")
        if not isinstance(chunk_metadata, dict):
            chunk_metadata = {}

        tags = document_metadata.get("tags", [])
        if not isinstance(tags, list):
            tags = []

        searchable_text = " ".join([
            chunk.get("source", "") or "",
            document_metadata.get("source_title", "") or "",
            document_metadata.get("summary", "") or "",
            " ".join(tag for tag in tags if isinstance(tag, str)),
            chunk.get("content", "") or "",
        ])
        searchable_lower = searchable_text.lower()
        overlap = len(query_tokens & _tokenize_for_ranking(searchable_text))
        exact_bonus = 2 if normalized_query and normalized_query in searchable_lower else 0
        if overlap <= 0 and exact_bonus <= 0:
            continue

        matches.append({
            "source": chunk["source"],
            "document_metadata": document_metadata,
            "chunk_id": str(chunk["chunk_index"]),
            "content": chunk["content"],
            "metadata": chunk_metadata,
            "score": 0.0,
            "keyword_score": overlap * 0.3 + exact_bonus * 0.2,
        })

    ranked = sorted(matches, key=lambda hit: (hit["keyword_score"], hit["source"], hit["chunk_id"]), reverse=True)
    return ranked[:top_k]


def _build_hit_lookup(hits: list[dict]) -> dict:
    lookup = {}
    for hit in hits:
        key = (str(hit["source"]), str(hit["chunk_id"]))
        lookup[key] = hit
    return lookup


def _normalize_citations(citations, hits: list[dict]) -> list[dict]:
    if not isinstance(citations, list):
        return []

    hit_lookup = _build_hit_lookup(hits)
    normalized = []

    for citation in citations:
        if not isinstance(citation, dict):
            continue

        source = citation.get("source")
        chunk_id = citation.get("chunk_id")
        ref = citation.get("ref")

        if (not source or not chunk_id) and isinstance(ref, str) and "#" in ref:
            source, chunk_id = ref.split("#", 1)

        if source is None or chunk_id is None:
            continue

        key = (str(source), str(chunk_id))
        hit = hit_lookup.get(key)
        if not hit:
            continue

        page_label = _format_page_label(hit.get("metadata"))
        normalized.append({
            "source": key[0],
            "chunk_id": key[1],
            "ref": f"{key[0]}#{key[1]}",
            "quote": citation.get("quote", "") if isinstance(citation.get("quote"), str) else "",
            "page": page_label,
        })

    return normalized


def _normalize_suggested_questions(values) -> list[str]:
    if not isinstance(values, list):
        return []

    normalized = []
    seen = set()
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(cleaned)
    return normalized[:3]


def _normalize_ask_response(payload, hits: list[dict], fallback_answer: str = "") -> dict:
    if not isinstance(payload, dict):
        return {
            "intent": "",
            "relevant_information": [],
            "answer": fallback_answer,
            "suggested_questions": [],
            "citations": [],
        }

    relevant_information = payload.get("relevant_information", [])
    if not isinstance(relevant_information, list):
        relevant_information = []
    relevant_information = [item.strip() for item in relevant_information if isinstance(item, str) and item.strip()]

    answer = payload.get("answer", "")
    if not isinstance(answer, str):
        answer = str(answer) if answer is not None else ""

    intent = payload.get("intent", "")
    if not isinstance(intent, str):
        intent = str(intent) if intent is not None else ""

    return {
        "intent": intent.strip(),
        "relevant_information": relevant_information,
        "answer": answer.strip() or fallback_answer,
        "suggested_questions": _normalize_suggested_questions(payload.get("suggested_questions", [])),
        "citations": _normalize_citations(payload.get("citations", []), hits),
    }


async def _rewrite_query(question: str) -> str:
    system = """Return JSON only:
{
  "search_query": "string"
}
Rewrite the user's question into a concise retrieval query. Preserve the original meaning, keep key entities, and optimize for document search.
"""

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                json={
                    "model": settings.openai_chat_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": question},
                    ],
                    "temperature": 0.0,
                },
            )
            r.raise_for_status()
            result = r.json()["choices"][0]["message"]["content"]
    except Exception:
        return question

    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return question

    rewritten = payload.get("search_query")
    if not isinstance(rewritten, str):
        return question

    rewritten = rewritten.strip()
    return rewritten or question


async def _generate_document_title(text: str, filename: str | None) -> str:
    system = """Return JSON only:
{
  "title": "string"
}
Generate a concise document title in Indonesian based on the file content. Keep it under 12 words.
"""
    excerpt = text[:4000]
    fallback_name = (filename or "Untitled").rsplit(".", 1)[0]

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                json={
                    "model": settings.openai_chat_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": f"FILENAME: {filename or '-'}\n\nDOCUMENT:\n{excerpt}"},
                    ],
                    "temperature": 0.1,
                },
            )
            r.raise_for_status()
            result = r.json()["choices"][0]["message"]["content"]
    except Exception:
        return fallback_name

    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return fallback_name

    title = payload.get("title")
    if not isinstance(title, str):
        return fallback_name

    cleaned = title.strip()
    return cleaned or fallback_name


async def _generate_document_summary(text: str) -> str:
    system = """Return JSON only:
{
  "summary": "string"
}
Generate a concise summary in Indonesian of the document in 2 to 4 sentences. Focus on the main purpose, scope, and important context.
"""
    excerpt = text[:8000]

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                json={
                    "model": settings.openai_chat_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": f"DOCUMENT:\n{excerpt}"},
                    ],
                    "temperature": 0.1,
                },
            )
            r.raise_for_status()
            result = r.json()["choices"][0]["message"]["content"]
    except Exception:
        return ""

    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return ""

    summary = payload.get("summary")
    if not isinstance(summary, str):
        return ""
    return summary.strip()


async def _generate_document_tags(text: str) -> list[str]:
    system = """Return JSON only:
{
  "tags": ["string"]
}
Generate 3 to 8 short topical tags for the document. Tags must be concise, lowercase, and non-duplicated.
"""
    excerpt = text[:6000]

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                json={
                    "model": settings.openai_chat_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": f"DOCUMENT:\n{excerpt}"},
                    ],
                    "temperature": 0.1,
                },
            )
            r.raise_for_status()
            result = r.json()["choices"][0]["message"]["content"]
    except Exception:
        return []

    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return []

    tags = payload.get("tags", [])
    if not isinstance(tags, list):
        return []

    cleaned_tags = []
    seen = set()
    for tag in tags:
        if not isinstance(tag, str):
            continue
        normalized = tag.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        cleaned_tags.append(normalized)

    return cleaned_tags[:8]


async def _generate_document_analysis(source_ids: list[str] | None, analysis_type: str = "default"):
    if not source_ids:
        raise HTTPException(status_code=400, detail="Provide at least one source_id.")

    cache_key = _build_analysis_cache_key(source_ids, analysis_type)
    cached = get_analysis_cache(cache_key, analysis_type)
    if cached:
        result = cached.get("result")
        if isinstance(result, dict):
            return {**result, "cached": True}

    chunks = get_document_chunks(source_ids)
    if not chunks:
        raise HTTPException(status_code=404, detail="No document content found for the selected source_id values.")

    sources = []
    seen_sources = set()
    context_parts = []

    for chunk in chunks:
        source = chunk["source"]
        if source not in seen_sources:
            seen_sources.add(source)
            sources.append(source)
        context_parts.append(
            f"{_format_chunk_header(source, chunk['chunk_index'], chunk['chunk_metadata'])}\n{chunk['content']}"
        )

    context = "\n\n".join(context_parts)
    system = """Return JSON only:
{
  "overview": "string",
  "key_points": ["string"],
  "risks": ["string"],
  "recommended_actions": ["string"],
  "tags": ["string"]
}
Analyze the provided document set. Be concise and factual. If multiple sources are provided, synthesize them together.
"""

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            json={
                "model": settings.openai_chat_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"SOURCES: {', '.join(sources)}\n\nDOCUMENTS:\n{context[:18000]}"},
                ],
                "temperature": 0.2,
            },
        )
        r.raise_for_status()
        result = r.json()["choices"][0]["message"]["content"]

    try:
        parsed = json.loads(result)
    except json.JSONDecodeError:
        parsed = {
            "overview": result,
            "key_points": [],
            "risks": [],
            "recommended_actions": [],
            "tags": [],
        }

    parsed["sources"] = sources
    parsed["chunks_analyzed"] = len(chunks)
    parsed["cached"] = False
    upsert_analysis_cache(cache_key, sources, parsed, analysis_type)
    return parsed

@app.on_event("startup")
def startup():
    init_db()


@app.get("/openapi.json", include_in_schema=False)
def openapi_json(_username: str = Depends(require_basic_auth)):
    return JSONResponse(
        get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
    )


@app.get("/docs", include_in_schema=False)
def swagger_ui(_username: str = Depends(require_basic_auth)):
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=f"{app.title} - Swagger UI",
    )


@app.post("/ingest-file")
async def ingest_file(
    file: UploadFile | None = File(None),
    file_url: str | None = Form(None),
    source_id: str | None = Form(None),
):
    normalized_file_url = file_url.strip() if file_url and file_url.strip() else None
    if file and normalized_file_url:
        raise HTTPException(status_code=400, detail="Provide either file or file_url, not both.")
    if not file and not normalized_file_url:
        raise HTTPException(status_code=400, detail="Provide either file or file_url.")

    source_url = None
    if file:
        file_bytes = await file.read()
        filename = file.filename or "uploaded-file"
        file_type = file.content_type or "application/octet-stream"
    else:
        source_url = normalized_file_url
        file_bytes, filename, file_type = await _fetch_remote_file(source_url)

    if not file_bytes:
        raise HTTPException(status_code=400, detail="File is empty.")

    segments = _extract_text_segments(file_bytes, filename)
    if not segments:
        raise HTTPException(status_code=400, detail="File does not contain extractable text.")

    text = "\n".join(segment["text"] for segment in segments)
    chunks, chunk_page_metadatas = _build_chunks(segments)
    embs = await embed_texts(chunks)
    source_title = await _generate_document_title(text, filename)
    resolved_source_id = source_id.strip() if source_id and source_id.strip() else _slugify(source_title)
    summary = await _generate_document_summary(text)
    tags = await _generate_document_tags(text)
    document_metadata = {
        "filename": filename,
        "file_type": file_type,
        "file_size_bytes": len(file_bytes),
        "page_count": max(segment["page_end"] for segment in segments),
        "source_title": source_title,
        "summary": summary,
        "tags": tags,
    }
    if source_url:
        document_metadata["source_url"] = source_url
    chunk_metadatas = []
    for page_metadata in chunk_page_metadatas:
        chunk_metadatas.append({
            "filename": filename,
            "page_start": page_metadata["page_start"],
            "page_end": page_metadata["page_end"],
        })
    upsert_chunks(resolved_source_id, chunks, embs, document_metadata, chunk_metadatas)
    return {
        "ok": True,
        "source_id": resolved_source_id,
        "source_title": source_title,
        "summary": summary,
        "page_count": document_metadata["page_count"],
        "chunks": len(chunks),
        "tags": tags,
    }

@app.post("/ask")
async def ask(
    question: str = Form(...),
    source_id: str | None = Form(None),
    source_ids: list[str] | None = Form(None),
    conversation_id: str | None = Form(None),
):
    resolved_conversation_id = conversation_id.strip() if conversation_id and conversation_id.strip() else str(uuid4())
    history = get_conversation_messages(resolved_conversation_id, limit=6)
    rewritten_query = await _rewrite_query(question)
    q_emb = (await embed_texts([rewritten_query]))[0]
    requested_sources = _normalize_source_ids(source_id, source_ids) or _get_memory_source_ids(history)
    selected_sources = _prefilter_source_ids(rewritten_query, requested_sources)
    vector_hits = similarity_search(q_emb, max(settings.top_k * 3, settings.top_k), selected_sources)
    keyword_hits = _keyword_search(rewritten_query, selected_sources, max(settings.top_k * 3, settings.top_k))
    initial_hits = _merge_hits(vector_hits, keyword_hits, max(settings.top_k * 4, settings.top_k))
    hits = _rerank_hits(question, initial_hits, settings.top_k)
    if not hits:
        return {
            "intent": "",
            "relevant_information": [],
            "answer": "Tidak ditemukan informasi yang relevan pada dokumen yang tersedia.",
            "suggested_questions": [],
            "citations": [],
            "conversation_id": resolved_conversation_id,
        }
    conversation_context = _build_conversation_context(history)
    context = _build_ask_context(hits)

    system = """Return JSON:
{
  "intent": "string",
  "relevant_information": [
    "string"
  ],
  "answer": "string",
  "suggested_questions": [
    "string"
  ],
  "citations": [
    {"source":"string","chunk_id":"string","ref":"string","quote":"string","page":"string"}
  ]
}
Process the answer in this order:
1. Identify the user's intent.
2. Select only the most relevant facts from the context.
3. Compose the final answer from those facts.
Answer ONLY using context. Keep `relevant_information` concise and factual.
Use the document metadata (`TITLE`, `SUMMARY`, `TAGS`) to disambiguate which document is most relevant before answering.
Return 2 or 3 short `suggested_questions` that would logically continue the user's exploration.
"""

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            json={
                "model": settings.openai_chat_model,
                "messages": [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": (
                            f"RECENT_CONVERSATION:\n{conversation_context or '(empty)'}\n\n"
                            f"QUESTION:\n{question}\n\n"
                            f"REWRITTEN_SEARCH_QUERY:\n{rewritten_query}\n\n"
                            f"CONTEXT:\n{context}"
                        ),
                    }
                ],
                "temperature": 0.2
            }
        )
        r.raise_for_status()
        result = r.json()["choices"][0]["message"]["content"]

    append_conversation_message(
        resolved_conversation_id,
        "user",
        question,
        {"rewritten_query": rewritten_query, "source_ids": selected_sources or []},
    )
    try:
        normalized = _normalize_ask_response(json.loads(result), hits, fallback_answer=result)
    except json.JSONDecodeError:
        normalized = _normalize_ask_response({}, hits, fallback_answer=result)

    append_conversation_message(
        resolved_conversation_id,
        "assistant",
        normalized.get("answer", ""),
        {
            "intent": normalized.get("intent", ""),
            "citations": normalized.get("citations", []),
        },
    )
    normalized["conversation_id"] = resolved_conversation_id
    return normalized


@app.get("/documents")
def list_documents(
    q: str | None = Query(None),
    source_id: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
):
    selected_sources = _normalize_source_ids(source_id, None)
    documents = _search_documents(q, selected_sources, limit)
    return {
        "items": documents,
        "count": len(documents),
    }


@app.post("/analyze-documents")
async def analyze_documents(
    source_id: str | None = Form(None),
    source_ids: list[str] | None = Form(None),
):
    selected_sources = _normalize_source_ids(source_id, source_ids)
    return await _generate_document_analysis(selected_sources)

@app.get("/health")
def health():
    return {"ok": True}
