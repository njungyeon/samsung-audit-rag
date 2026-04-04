from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from app.chunker import build_chunks
from app.config import settings
from app.db import get_conn
from app.embedder import Embedder
from app.parser import parse_html_file, save_parsed_json


def upsert_document(cur, report) -> int:
    cur.execute(
        """
        INSERT INTO documents (file_name, title, report_year, source_path, raw_html, parsed_json)
        VALUES (%s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (file_name) DO UPDATE SET
            title = EXCLUDED.title,
            report_year = EXCLUDED.report_year,
            source_path = EXCLUDED.source_path,
            raw_html = EXCLUDED.raw_html,
            parsed_json = EXCLUDED.parsed_json
        RETURNING id
        """,
        (
            report.file_name,
            report.title,
            report.report_year,
            str(report.file_path),
            report.raw_html,
            __import__('json').dumps(report.structured, ensure_ascii=False),
        ),
    )
    return cur.fetchone()["id"]


def replace_chunks(cur, document_id: int, report, chunks, embeddings, embedding_model: str) -> None:
    cur.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))

    for chunk, embedding in zip(chunks, embeddings, strict=True):
        cur.execute(
            """
            INSERT INTO chunks (
                id, document_id, chunk_key, chunk_index_global, chunk_index_in_section,
                company, report_year, major_section, sub_section, section_type,
                note_no, note_title, topic, content, char_count,
                embedding_model, embedding
            )
            VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s::vector
            )
            """,
            (
                chunk.id,
                document_id,
                chunk.chunk_key,
                chunk.chunk_index_global,
                chunk.chunk_index_in_section,
                report.structured["document_meta"]["company"],
                report.report_year,
                chunk.major_section,
                chunk.sub_section,
                chunk.section_type,
                chunk.note_no,
                chunk.note_title,
                chunk.topic,
                chunk.content,
                chunk.char_count,
                embedding_model,
                embedding,
            ),
        )


def ingest_one(html_path: Path, embedder: Embedder) -> None:
    report = parse_html_file(html_path)
    save_parsed_json(report, settings.parsed_dir)
    chunks = build_chunks(report)
    if not chunks:
        print(f"[SKIP] {html_path.name}: usable chunk가 없습니다.")
        return

    embeddings = embedder.encode_texts([chunk.content for chunk in chunks])

    with get_conn() as conn:
        with conn.cursor() as cur:
            document_id = upsert_document(cur, report)
            replace_chunks(cur, document_id, report, chunks, embeddings, embedder.model_name)

    print(
        f"[DONE] {html_path.name} | parser={report.parser_used} | year={report.report_year} | chunks={len(chunks)}"
    )


def main() -> None:
    html_dir = Path(settings.html_dir)
    html_files = sorted(list(html_dir.glob("*.htm")) + list(html_dir.glob("*.html")))
    if not html_files:
        raise FileNotFoundError(f"HTML 파일이 없습니다: {html_dir}")

    embedder = Embedder()
    for html_path in tqdm(html_files, desc="ingest"):
        ingest_one(html_path, embedder)


if __name__ == "__main__":
    main()
