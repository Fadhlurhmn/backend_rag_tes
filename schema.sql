-- ============================================================
-- Supabase SQL Schema — BrandnPurpose HR Chat
-- Jalankan di: Supabase Dashboard → SQL Editor → New Query
-- ============================================================

-- Enable UUID extension (biasanya sudah aktif di Supabase)
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ─────────────────────────────────────
-- Tabel: conversations
-- ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS conversations (
  id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMPTZ DEFAULT now(),
  title      TEXT        -- opsional
);

-- ─────────────────────────────────────
-- Tabel: messages
-- ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS messages (
  id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID        REFERENCES conversations(id) ON DELETE CASCADE,
  created_at      TIMESTAMPTZ DEFAULT now(),
  role            TEXT        NOT NULL CHECK (role IN ('user', 'assistant')),
  content         TEXT        NOT NULL,
  agent_role      TEXT        CHECK (agent_role IN ('manager', 'specialist')),
  input_tokens    INT,
  output_tokens   INT,
  total_tokens    INT
);

-- Index untuk query history cepat
CREATE INDEX IF NOT EXISTS idx_messages_conv_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);

-- ─────────────────────────────────────
-- Row Level Security (RLS) — opsional
-- Aktifkan jika ingin isolasi per user
-- ─────────────────────────────────────
-- ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY "public access" ON conversations FOR ALL USING (true);
-- CREATE POLICY "public access" ON messages FOR ALL USING (true);
