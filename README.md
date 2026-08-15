<div align="center">

<br/>

# 📑 Samsung Audit RAG

**Audit-report Q&A over 10 years of Samsung Electronics filings · PostgreSQL + pgvector**

**Team 5** — Jinju Kim · Jungyeon Na · Jisoo Bang · Jungwon Lee · Yeonjun Cho · Hyeonju Cho

<br/>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Poetry](https://img.shields.io/badge/Poetry-60A5FA?style=flat-square&logo=poetry&logoColor=white)](https://python-poetry.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?style=flat-square&logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/🤗_Transformers-sentence--transformers-FFD21E?style=flat-square)](https://huggingface.co)

<br/>

[![🔄 Pipeline](https://img.shields.io/badge/🔄_Pipeline-Parse_to_RAG-2E7D32?style=for-the-badge)](#pipeline)
[![🗄️ Schema](https://img.shields.io/badge/🗄️_Database-Schema-1565C0?style=for-the-badge)](#database-schema)

<br/>

</div>

---

## Overview

**Samsung Audit RAG** is a Retrieval-Augmented Generation system built over **10 years of Samsung Electronics audit reports** (2014–2024, `.htm/.html`). It decodes and parses legacy EUC-KR/CP949 HTML into structured JSON, chunks it section by section, embeds it, and stores everything in a single **PostgreSQL + pgvector** instance — then answers natural-language questions either by extractive search or by a locally-hosted LLM, through a CLI or a Streamlit chat UI with a **ReAct tool-calling agent**.

| | `app.ingest` | `app.search` | `app.qa` | `streamlit_app.py` |
|---|---|---|---|---|
| **Role** | Parse → chunk → embed → load | Vector similarity search | CLI question answering | Chat UI |
| **Output** | Rows in `documents` / `chunks` | Top-K chunks | Extractive or LLM answer | Streaming chat |
| **Options** | — | — | `--llm` `--stream` `--debug` `--test` | RAG streaming / Agent (ReAct) |

---

## Quick Start

**1 · Python & Poetry** (macOS)

```bash
brew install pyenv poetry
pyenv install 3.12
pyenv local 3.12.13
poetry env use $(pyenv which python)
poetry install
cp .env.example .env
```

**2 · Start PostgreSQL (pgvector)**

```bash
docker compose up -d
```

**3 · Add audit reports**

```bash
data/html/감사보고서_2014.htm
data/html/감사보고서_2015.htm
...
```

**4 · Ingest — parse, chunk, embed, load**

```bash
poetry run python -m app.ingest
```

**5 · Search & QA**

```bash
poetry run python -m app.search "현금및현금성자산"
poetry run python -m app.qa "2020년도 감사의견근거는 무엇이야?"
poetry run python -m app.qa --llm --stream "2023년도 기타사항에는 뭐가 적혀 있어?"
poetry run python -m app.qa --test              # run 10 sample questions
```

**6 · Run the chat UI**

```bash
poetry run streamlit run app/streamlit_app.py
```

---

## Project Structure

```
samsung-audit-rag/
├── app/
│   ├── config.py                 # Env vars, DB connection
│   ├── db.py                     # Database queries
│   ├── parser.py                 # HTML → structured JSON
│   ├── chunker.py                # Section-based chunking
│   ├── embedder.py                # Text embedding
│   ├── ingest.py                 # Ingest pipeline entry point
│   ├── search.py                 # Vector search
│   ├── qa.py                     # RAG question answering
│   ├── generator.py              # Local LLM generation (streaming)
│   └── streamlit_app.py          # Chat UI
├── data/
│   ├── html/                     # Raw audit report .htm files
│   └── parsed/                   # Parsed JSON output
├── sql/init.sql                  # DB schema
├── tests/test_data_schema.py
├── docker-compose.yml            # PostgreSQL + pgvector
└── pyproject.toml                # Poetry dependencies
```

---

## Pipeline

1. **Ingest** — read raw `.htm/.html` files
2. **Parse** — EUC-KR/CP949 decoding + defensive HTML parsing (`lxml → html5lib → html.parser` fallback)
3. **Chunk** — financial statements chunked by row group, notes chunked by section
4. **Embed** — `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dim)
5. **Store** — document + chunk metadata and embeddings in PostgreSQL
6. **Search** — vectorize the query, retrieve top-K similar chunks
7. **Generate** — extractive answer by default, or a locally-hosted LLM (`transformers`, streaming) with `--llm`

The same pipeline powers the Streamlit UI, in two modes: **RAG streaming** (retrieve → stream an answer) and **Agent (ReAct) streaming** (tool-calling loop → stream the final answer), with tools `search_audit_report`, `get_specific_section`, and `compare_years`.

---

## Chunking Strategy

**Financial statements** — never chunk a whole table at once; group table rows (e.g. current/non-current assets, operating/investing/financing cash flows) to keep chunks retrievable and coherent.

**Notes** — split by note number first, merge `p`/`table` elements under the same note into one logical unit, and further split if still too long.

---

## Database Schema

**`documents`** — `file_name` · `title` · `report_year` · `source_path` · `raw_html` · `parsed_json`

**`chunks`** — `chunk_key` · `chunk_index_global` · `chunk_index_in_section` · `major_section` · `sub_section` · `section_type` · `note_no` · `note_title` · `topic` · `content` · `embedding`

---

## Design Decisions

| Decision | Reasoning |
|---|---|
| Single PostgreSQL + pgvector, no separate vector DB | 10 years of reports is a small enough corpus — one database is simpler to operate than adding Pinecone or similar |
| Parser fallback chain (`lxml → html5lib → html.parser`) | Legacy audit-report HTML breaks strict parsers on some files; the fallback maximizes ingest robustness |
| Row-grouped chunking for tables | Whole-table chunks hurt retrieval precision; grouping rows by logical section fixed it |
| Extractive answer by default | Grounds answers in retrieved context and avoids hallucination; LLM generation is opt-in via `--llm` |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Parsing | BeautifulSoup4 · lxml / html5lib / html.parser |
| Chunking | langchain-text-splitters |
| Embedding | sentence-transformers (multilingual MiniLM) |
| Generation | 🤗 Transformers (local LLM, streaming) · PyTorch |
| Storage | PostgreSQL + pgvector · Docker Compose |
| UI | Streamlit |
| Dependency mgmt | Poetry |

---

<details>
<summary>🇰🇷 &nbsp;한국어 설명 보기</summary>
<br/>

## 개요

**Samsung Audit RAG**는 삼성전자 감사보고서 10년치(2014–2024, `.htm/.html`)를 대상으로 한 RAG 시스템입니다. EUC-KR/CP949 HTML을 방어적으로 파싱해 구조화된 JSON으로 변환하고, 섹션 단위로 청킹한 뒤 임베딩을 생성해 **PostgreSQL + pgvector** 하나에 통합 저장합니다. 이후 추출식 검색 또는 로컬 LLM으로 질문에 답하며, CLI와 **ReAct 툴콜링 에이전트**를 갖춘 Streamlit 챗 UI를 모두 지원합니다.

## 빠른 시작

```bash
# Python/Poetry 환경 구성
pyenv install 3.12
pyenv local 3.12.13
poetry env use $(pyenv which python)
poetry install
cp .env.example .env

# PostgreSQL 실행
docker compose up -d

# 감사보고서 파일을 data/html/ 에 넣은 뒤 적재
poetry run python -m app.ingest

# 검색 / 질의응답
poetry run python -m app.search "현금및현금성자산"
poetry run python -m app.qa --llm --stream "2023년도 기타사항에는 뭐가 적혀 있어?"

# 챗 UI
poetry run streamlit run app/streamlit_app.py
```

## 파이프라인

수집 → 파싱(EUC-KR 디코딩 + 다중 파서 fallback) → 청킹(재무제표는 행 단위, 주석은 섹션 단위) → 임베딩 → PostgreSQL 저장 → 검색 → 생성(기본은 추출식, `--llm`로 로컬 LLM 생성) 순으로 동작합니다.

</details>
