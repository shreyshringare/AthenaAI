-- AthenaAI database schema
-- Run automatically by docker-compose on first postgres start.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ivfflat `lists` tuning note (both ivfflat indexes below):
-- pgvector's own guidance is lists ~= rows / 1000 for large tables. At
-- demo/test scale (single-digit to low-hundreds of rows), lists=100 trains
-- mostly-empty clusters; with ivfflat.probes defaulting to 1, a query only
-- scans one cluster, so it can miss the true nearest rows entirely. This
-- was caught as a non-deterministic 0-or-partial-result failure in the RAG
-- and semantic-memory integration tests. lists=1 degrades to an exact scan
-- at this scale -- re-tune upward only once real row counts justify it.

-- RAG chunks table
-- id TEXT + created_at DOUBLE PRECISION to match athenai.rag.retriever
-- .PgVectorRetriever's own _CREATE_TABLE_SQL -- see conversation_messages
-- comment below for why this has to match exactly.
CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector(1536),
    source      TEXT DEFAULT '',
    metadata    JSONB DEFAULT '{}',
    chunk_index INTEGER NOT NULL DEFAULT 0,
    created_at  DOUBLE PRECISION NOT NULL DEFAULT EXTRACT(EPOCH FROM NOW())
);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 1);

-- Conversation memory table
-- Column set/types must match athenai.memory.conversation.ConversationMemory's
-- own _ensure_schema() DDL exactly (id TEXT not UUID, created_at DOUBLE
-- PRECISION not TIMESTAMPTZ, metadata JSONB) since that INSERT runs against
-- whichever version of this table exists first -- CREATE TABLE IF NOT EXISTS
-- there is a no-op once this script has created it.
CREATE TABLE IF NOT EXISTS conversation_messages (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  DOUBLE PRECISION NOT NULL DEFAULT EXTRACT(EPOCH FROM NOW()),
    metadata    JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS conv_session_idx ON conversation_messages (session_id, created_at);

-- Semantic memory table
-- id TEXT + created_at DOUBLE PRECISION (not UUID / TIMESTAMPTZ) to match
-- athenai.memory.semantic.SemanticMemory's own _ensure_schema() DDL -- see
-- conversation_messages comment above for why.
CREATE TABLE IF NOT EXISTS semantic_memories (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector(1536),
    metadata    JSONB DEFAULT '{}',
    created_at  DOUBLE PRECISION NOT NULL DEFAULT EXTRACT(EPOCH FROM NOW())
);

CREATE INDEX IF NOT EXISTS sem_session_idx ON semantic_memories (session_id);
CREATE INDEX IF NOT EXISTS sem_embedding_idx ON semantic_memories USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 1);
