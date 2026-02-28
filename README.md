# Agentic RAG

API FastAPI sederhana untuk:
- ingest dokumen ke PostgreSQL + pgvector
- pecah dokumen menjadi beberapa chunk berbasis paragraf/section dengan overlap kecil
- simpan embedding per chunk
- tanya jawab berbasis retrieval dari dokumen yang sudah di-upload

## Prasyarat

- Python 3.11+
- Docker Desktop
- PostgreSQL dengan extension `vector` aktif (sudah disiapkan lewat `docker-compose.yml`)
- OpenAI API key

## Setup

1. Jalankan database:

```bash
docker compose up -d
```

2. Buat file environment:

```bash
cp .env.example .env
```

Untuk Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

3. Isi `.env` minimal dengan:

```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/postgres
OPENAI_API_KEY=your_api_key
OPENAI_EMBED_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini
TOP_K=6
BASIC_AUTH_USERNAME=
BASIC_AUTH_PASSWORD=
SWAGGER_TITLE=Agentic RAG API
SWAGGER_DESCRIPTION=Protected API documentation for the Agentic RAG service.
SWAGGER_VERSION=1.0.0
```

4. Install dependency:

```bash
pip install -r requirements.txt
```

5. Jalankan API:

```bash
uvicorn app.main:app --reload
```

API akan aktif di `http://localhost:8248`.

Semua endpoint, termasuk Swagger, dilindungi HTTP Basic Auth.

Swagger UI:

```text
http://localhost:8248/docs
```

Gunakan username dan password dari `.env`:
- `BASIC_AUTH_USERNAME`
- `BASIC_AUTH_PASSWORD`

## Endpoint

### `GET /health`

Cek service aktif.

Contoh:

```bash
curl -u basic:auth http://localhost:8248/health
```

### `GET /documents`

Lihat daftar dokumen yang sudah di-ingest, sekaligus pencarian berbasis `source_title`, `summary`, `tags`, dan `source_id`.

Query parameter:
- `q`: opsional, keyword pencarian metadata dokumen
- `source_id`: opsional, filter satu atau beberapa `source_id` dipisah koma
- `limit`: opsional, default `20`

Contoh list dokumen:

```bash
curl -u basic:auth "http://localhost:8248/documents"
```

Contoh search dokumen:

```bash
curl -u basic:auth "http://localhost:8248/documents?q=feedback%20system&limit=10"
```

Contoh response:

```json
{
  "items": [
    {
      "source_id": "my-doc",
      "source_title": "Ringkasan Kebutuhan Sistem Feedback",
      "summary": "Dokumen ini menjelaskan kebutuhan pengadaan sistem feedback berbasis AI.",
      "tags": ["procurement", "feedback system"],
      "filename": "file.pdf",
      "file_type": "application/pdf",
      "file_size_bytes": 123456,
      "page_count": 24
    }
  ],
  "count": 1
}
```

### `POST /ingest-file`

Upload dokumen lalu simpan hasil chunk + embedding ke database.

Form field:
- `file`: file yang akan di-ingest
- `file_url`: opsional, URL `http`/`https` ke file yang akan di-ingest
- `source_id`: opsional, id unik dokumen. Jika tidak dikirim, akan dibuat dari judul AI

Catatan:
- Kirim salah satu: `file` atau `file_url`
- PDF akan diekstrak teksnya dengan `pypdf`
- File non-PDF akan dibaca sebagai teks UTF-8
- Metadata dokumen menyimpan `filename`, `file_type`, `file_size_bytes`, `page_count`, `source_title`, dan `summary`
- Jika ingest via URL, metadata dokumen juga menyimpan `source_url`
- Tag dokumen akan digenerate otomatis oleh AI dan disimpan di metadata `document`
- Summary dokumen akan digenerate otomatis oleh AI agar lebih mudah mengidentifikasi dokumen yang relevan
- Metadata chunk menyimpan informasi halaman (`page_start`, `page_end`) agar hasil retrieval bisa ditelusuri ke halaman sumber
- Jika `source_id` tidak dikirim, nilai `source` internal akan diisi dari judul file yang digenerate AI
- Jika `source_id` yang sama di-upload lagi, chunk lama akan diganti

Contoh:

```bash
curl -u basic:auth -X POST http://localhost:8248/ingest-file \
  -F "file=@./file.pdf" \
  -F "source_id=my-doc"
```

Atau dari URL file:

```bash
curl -u basic:auth -X POST http://localhost:8248/ingest-file \
  -F "file_url=https://example.com/file.pdf" \
  -F "source_id=my-doc"
```

Contoh response:

```json
{
  "ok": true,
  "source_id": "ringkasan-kebutuhan-sistem-feedback",
  "source_title": "Ringkasan Kebutuhan Sistem Feedback",
  "summary": "Dokumen ini menjelaskan kebutuhan pengadaan sistem feedback berbasis AI, ruang lingkup pekerjaan, dan sasaran implementasi.",
  "page_count": 24,
  "chunks": 12,
  "tags": ["procurement", "feedback system", "artificial intelligence"]
}
```

### `POST /ask`

Kirim pertanyaan, sistem akan:
- melakukan query rewriting agar query pencarian lebih eksplisit
- membuat embedding untuk pertanyaan
- melakukan prefilter dokumen berdasarkan metadata (`source_title`, `summary`, `tags`)
- mengambil chunk dengan hybrid search (vector search + keyword search)
- melakukan reranking chunk sebelum dikirim ke model
- mengirim context ke model chat
- melakukan normalisasi citation agar `source`, `chunk_id`, dan `page` konsisten
- mengembalikan jawaban model

Form field:
- `question`: pertanyaan user
- `source_id`: opsional, satu `source_id` atau string dipisah koma
- `source_ids`: opsional, kirim beberapa field `source_ids` untuk multi dokumen
- `conversation_id`: opsional, untuk melanjutkan percakapan multi-turn pada sesi yang sama

Catatan memory:
- Jika `conversation_id` yang sama dipakai lagi, server akan memuat beberapa pesan terakhir sebagai memory percakapan.
- Jika pada follow-up Anda tidak mengirim `source_id` lagi, server akan mencoba memakai `source_id` terakhir dari percakapan tersebut.

Contoh:

```bash
curl -u basic:auth -X POST http://localhost:8248/ask \
  -F "question=Apa isi utama dokumen ini?" \
  -F "source_id=my-doc"
```

Contoh lanjutan percakapan:

```bash
curl -u basic:auth -X POST http://localhost:8248/ask \
  -F "question=Jelaskan lebih detail risikonya" \
  -F "conversation_id=existing-conversation-id"
```

Contoh multi source dengan koma:

```bash
curl -u basic:auth -X POST http://localhost:8248/ask \
  -F "question=Bandingkan isi dua dokumen ini" \
  -F "source_id=doc-a,doc-b"
```

Contoh multi source dengan field berulang:

```bash
curl -u basic:auth -X POST http://localhost:8248/ask \
  -F "question=Bandingkan isi dua dokumen ini" \
  -F "source_ids=doc-a" \
  -F "source_ids=doc-b"
```

Contoh response:

```json
{
  "intent": "Menjelaskan isi utama dokumen",
  "relevant_information": [
    "Dokumen membahas kebutuhan pengadaan sistem feedback berbasis AI.",
    "Ruang lingkup mencakup riset, implementasi, dan optimalisasi."
  ],
  "answer": "Dokumen membahas ...",
  "suggested_questions": [
    "Apa risiko utama dari implementasi sistem ini?",
    "Apa ruang lingkup pekerjaan yang paling penting?"
  ],
  "conversation_id": "7f8d3145-8a1f-40b2-9f3b-b2d994f30346",
  "citations": [
    {
      "source": "my-doc",
      "chunk_id": "0",
      "ref": "my-doc#0",
      "quote": "...",
      "page": "page 3"
    }
  ]
}
```

Jika model tidak mengembalikan JSON, API akan fallback ke format:

```json
{
  "intent": "",
  "relevant_information": [],
  "answer": "raw model output",
  "suggested_questions": [],
  "conversation_id": "7f8d3145-8a1f-40b2-9f3b-b2d994f30346",
  "citations": []
}
```

Catatan response:
- `citations` akan dinormalisasi ulang oleh server. Jika model memberi citation yang valid tapi field `ref` atau `page` tidak rapi, server akan mengisi ulang dari chunk yang benar.

### `POST /analyze-documents`

Generate analisis dokumen terstruktur dari satu atau banyak dokumen yang sudah di-ingest.
Hasil analisis disimpan di cache database dan akan dipakai ulang untuk kombinasi `source_id` yang sama.

Form field:
- `source_id`: opsional, satu `source_id` atau string dipisah koma
- `source_ids`: opsional, kirim beberapa field `source_ids` untuk multi dokumen

Contoh:

```bash
curl -u basic:auth -X POST http://localhost:8248/analyze-documents \
  -F "source_id=my-doc"
```

Contoh multi source:

```bash
curl -u basic:auth -X POST http://localhost:8248/analyze-documents \
  -F "source_ids=doc-a" \
  -F "source_ids=doc-b"
```

Contoh response:

```json
{
  "overview": "Dokumen membahas kebutuhan sistem dan ruang lingkup implementasi.",
  "key_points": ["...", "..."],
  "risks": ["...", "..."],
  "recommended_actions": ["...", "..."],
  "tags": ["procurement", "feedback system"],
  "sources": ["my-doc"],
  "chunks_analyzed": 12,
  "cached": false
}
```

## Alur Penggunaan

1. Jalankan database dan API.
2. Upload dokumen lewat `/ingest-file`.
3. Simpan nilai `source_id` yang dipakai saat upload.
4. Gunakan `/ask` untuk Q&A berbasis retrieval atau `/analyze-documents` untuk analisis dokumen terstruktur.

## Struktur Data

Data disimpan di dua tabel:
- `document`: metadata dokumen per `source`, dengan primary key UUID
  Metadata ini mencakup `filename`, `file_type`, `file_size_bytes`, `page_count`, `source_title`, `summary`, dan `tags` hasil generate AI.
- `document_chunk`: isi chunk, metadata chunk, dan embedding per potongan teks, dengan primary key UUID
  Metadata chunk mencakup `filename`, `page_start`, dan `page_end`.
- `document_analysis`: cache hasil analisis dokumen untuk kombinasi `source_id` tertentu.
- `conversation_message`: riwayat percakapan per `conversation_id` untuk multi-turn Q&A.

## Catatan

- PDF berbasis scan/gambar mungkin tidak menghasilkan teks yang bagus karena belum ada OCR.
- Kualitas jawaban sangat tergantung pada kualitas ekstraksi teks dan hasil retrieval.
- Pastikan model embedding dan dimensi kolom `vector(1536)` tetap cocok.
- Perubahan skema `id` dari integer ke UUID hanya berlaku untuk tabel baru. Jika tabel lama sudah terbuat, reset DB atau buat migrasi manual agar tipenya ikut berubah.
