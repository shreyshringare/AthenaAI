-- AthenaAI database schema
-- Run automatically by docker-compose on first postgres start.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- RAG chunks table
CREATE TABLE IF NOT EXISTS chunks (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id TEXT NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector(1536),
    source      TEXT DEFAULT '',
    metadata    JSONB DEFAULT '{}',
    chunk_index INTEGER NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Conversation memory table
CREATE TABLE IF NOT EXISTS conversation_messages (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS conv_session_idx ON conversation_messages (session_id, created_at);

-- Semantic memory table
CREATE TABLE IF NOT EXISTS semantic_memories (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id  TEXT NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector(1536),
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS sem_session_idx ON semantic_memories (session_id);
CREATE INDEX IF NOT EXISTS sem_embedding_idx ON semantic_memories USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
