CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    file_name TEXT NOT NULL UNIQUE,
    title TEXT,
    report_year INT,
    source_path TEXT NOT NULL,
    raw_html TEXT,
    parsed_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_key TEXT NOT NULL UNIQUE,
    chunk_index_global INT NOT NULL,
    chunk_index_in_section INT NOT NULL,
    company TEXT NOT NULL,
    report_year INT,
    major_section TEXT,
    sub_section TEXT,
    section_type TEXT NOT NULL,
    note_no INT,
    note_title TEXT,
    topic TEXT,
    content TEXT NOT NULL,
    char_count INT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_report_year ON documents(report_year);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_report_year ON chunks(report_year);
CREATE INDEX IF NOT EXISTS idx_chunks_sub_section ON chunks(sub_section);
CREATE INDEX IF NOT EXISTS idx_chunks_note_no ON chunks(note_no);
CREATE INDEX IF NOT EXISTS idx_chunks_topic ON chunks(topic);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
