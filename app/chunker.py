from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.parser import ParsedReport, normalize_compact, normalize_space


@dataclass(slots=True)
class ChunkRecord:
    id: str
    chunk_key: str
    chunk_index_global: int
    chunk_index_in_section: int
    major_section: str | None
    sub_section: str | None
    section_type: str
    note_no: int | None
    note_title: str | None
    topic: str | None
    content: str
    char_count: int

def normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", "", str(text).strip())


def build_account_aliases(account_name: str) -> list[str]:
    raw = str(account_name).strip()
    normalized = normalize_for_match(raw)

    aliases = {raw, normalized}

    # 자주 나오는 한국어 계정명 띄어쓰기 보정
    alias_map = {
        "판매비와관리비": ["판매비와 관리비"],
        "기타수익": ["기타 수익", "기 타 수 익"],
        "기타비용": ["기타 비용", "기 타 비 용"],
        "금융수익": ["금융 수익", "금 융 수 익"],
        "금융비용": ["금융 비용", "금 융 비 용"],
        "매출원가": ["매출 원가", "매 출 원 가"],
        "매출총이익": ["매출 총이익", "매 출 총 이 익"],
        "영업이익": ["영업 이익", "영 업 이 익"],
        "법인세비용": ["법인세 비용", "법 인 세 비 용"],
        "당기순이익": ["당기 순이익", "당 기 순 이 익"],
    }

    for alias in alias_map.get(normalized, []):
        aliases.add(alias)
        aliases.add(normalize_for_match(alias))

    return sorted(aliases)


def _slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", text)


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = normalize_space(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            window = text[start:end]
            split_at = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(". "), window.rfind(" "))
            if split_at > max(50, chunk_size // 3):
                end = start + split_at + (2 if window[split_at:split_at + 2] == ". " else 1)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _table_rows(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _group_rows(rows: list[str], group_size: int) -> list[list[str]]:
    return [rows[i:i + group_size] for i in range(0, len(rows), group_size)]


def _append_chunk(
    chunks: list[ChunkRecord],
    stem: str,
    major_section: str | None,
    sub_section: str | None,
    section_type: str,
    note_no: int | None,
    note_title: str | None,
    topic: str | None,
    content: str,
    chunk_index_in_section: int,
) -> None:
    content = normalize_space(content)
    if len(content) < 30:
        return
    chunks.append(
        ChunkRecord(
            id=str(uuid.uuid4()),
            chunk_key=f"{stem}__{len(chunks):04d}",
            chunk_index_global=len(chunks),
            chunk_index_in_section=chunk_index_in_section,
            major_section=major_section,
            sub_section=sub_section,
            section_type=section_type,
            note_no=note_no,
            note_title=note_title,
            topic=topic,
            content=content,
            char_count=len(content),
        )
    )


def _chunk_general_text(prefix: str, text: str) -> list[str]:
    return [f"{prefix}\n\n{chunk}".strip() for chunk in _split_text(text, settings.text_chunk_size, settings.text_chunk_overlap) if chunk.strip()]


def _chunk_note_text(prefix: str, text: str) -> list[str]:
    return [f"{prefix}\n\n{chunk}".strip() for chunk in _split_text(text, settings.note_chunk_size, settings.note_chunk_overlap) if chunk.strip()]


def _guess_topic(text: str) -> str | None:
    compact = normalize_compact(text)
    keywords = [
        "현금및현금성자산",
        "단기금융상품",
        "매출채권",
        "재고자산",
        "유형자산",
        "무형자산",
        "차입금",
        "충당부채",
        "법인세비용",
        "현금흐름표",
        "영업활동현금흐름",
        "투자활동현금흐름",
        "재무활동현금흐름",
        "재무위험관리",
    ]
    for keyword in keywords:
        if keyword in compact:
            return keyword
    return None


def _financial_statement_chunks(stem: str, major_title: str, section: dict[str, Any], chunks: list[ChunkRecord]) -> None:
    section_title = section["title"]
    table_meta = section.get("table_meta", {})
    unit = table_meta.get("unit")
    current_period_label = table_meta.get("current_period_label") or "당기"
    prior_period_label = table_meta.get("prior_period_label") or "전기"

    block_idx = 0

    # ⭐ 핵심: table_rows 사용
    for row in section.get("table_rows", []):
        content = f"""
            [{section_title} - 구조화 데이터]
            계정명: {row['account_name']}
            정규화계정명: {row['account_name_normalized']}
            계층레벨: {row.get('hierarchy_level')}
            노드유형: {row.get('node_type')}
            부모행ID: {row.get('parent_row_id')}
            루트행ID: {row.get('root_row_id')}
            계층경로: {' > '.join(row.get('lineage_labels', []))}
            현재기간라벨: {current_period_label}
            이전기간라벨: {prior_period_label}
            당기금액: {row['current_amount']}
            전기금액: {row['prior_amount']}
            주석참조: {", ".join(row['note_refs']) if row['note_refs'] else "-"}
            단위: {unit}
            """.strip()

        _append_chunk(
            chunks,
            stem,
            major_title,
            section_title,
            "financial_statement_row",   # ⭐ 타입도 구분
            None,
            None,
            row["account_name_normalized"],
            content,
            block_idx,
        )

        block_idx += 1


def _collect_section_text(sec: dict[str, Any]) -> str:
    parts: list[str] = []
    title = sec.get("title", "")
    if title:
        parts.append(title)
    for block in sec.get("content_blocks", []):
        parts.append(block["text"])
    for child in sec.get("children", []):
        child_text = _collect_section_text(child)
        if child_text:
            parts.append(child_text)
    return "\n\n".join(p for p in parts if p and p.strip())


def _note_subsection_chunks(
    stem: str,
    major_title: str,
    sub_section: str,
    note: dict[str, Any],
    node: dict[str, Any],
    chunks: list[ChunkRecord],
    start_idx: int,
    path: list[str] | None = None,
) -> int:
    note_no = note["note_no"]
    note_title = note.get("note_title") or f"주석 {note_no}"
    code = str(node.get("code") or "").strip()
    title = str(node.get("title") or "").strip()
    current_path = list(path or [])
    if code:
        current_path.append(code)
    if title:
        current_path.append(title)
    idx = start_idx

    sec_text = _collect_section_text(node)
    if sec_text:
        path_label = " > ".join(current_path)
        prefix = f"[주석 {note_no}. {note_title} > {path_label}]"
        parts = _chunk_note_text(prefix, sec_text)
        topic = _guess_topic(sec_text) or normalize_for_match(f"{note_title}{path_label}")
        for part in parts:
            _append_chunk(
                chunks,
                stem,
                major_title,
                sub_section,
                "note_subsection",
                note_no,
                note_title,
                topic,
                part,
                idx,
            )
            idx += 1

    for child in node.get("children", []):
        idx = _note_subsection_chunks(
            stem,
            major_title,
            sub_section,
            note,
            child,
            chunks,
            idx,
            current_path,
        )

    return idx


def _note_table_cell_chunks(
    stem: str,
    major_title: str,
    sub_section: str,
    note: dict[str, Any],
    chunks: list[ChunkRecord],
    start_idx: int,
) -> int:
    note_no = note["note_no"]
    note_title = note.get("note_title") or f"주석 {note_no}"
    idx = start_idx

    def _iter_table_cells(
        raw_map: dict[str, Any],
        num_map: dict[str, Any],
        parent_label: str | None = None,
    ) -> list[tuple[str, Any, Any]]:
        items: list[tuple[str, Any, Any]] = []
        for key, raw_value in raw_map.items():
            key_text = str(key)
            col_label = f"{parent_label}_{key_text}" if parent_label else key_text
            numeric_value = num_map.get(key) if isinstance(num_map, dict) else None

            if isinstance(raw_value, dict):
                nested_num_map = numeric_value if isinstance(numeric_value, dict) else {}
                items.extend(_iter_table_cells(raw_value, nested_num_map, col_label))
            else:
                items.append((col_label, raw_value, numeric_value))
        return items

    for table_idx, table in enumerate(note.get("tables", []), start=1):
        unit = table.get("unit")
        for row in table.get("rows", []):
            row_label = str(row.get("row_label") or "")
            raw_values = row.get("raw_values", {}) or {}
            values = row.get("values", {}) or {}

            for col_label, raw_value, numeric_value in _iter_table_cells(raw_values, values):
                content = (
                    f"[주석 {note_no}. {note_title} - 테이블 셀]\n"
                    f"표번호: {table_idx}\n"
                    f"행: {row_label}\n"
                    f"열: {col_label}\n"
                    f"값(raw): {raw_value}\n"
                    f"값(numeric): {numeric_value}\n"
                    f"단위: {unit}"
                )

                _append_chunk(
                    chunks,
                    stem,
                    major_title,
                    sub_section,
                    "note_table_cell",
                    note_no,
                    note_title,
                    normalize_for_match(f"{note_title}{row_label}{col_label}"),
                    content,
                    idx,
                )
                idx += 1

    return idx


def _notes_chunks(stem: str, major_title: str, section: dict[str, Any], chunks: list[ChunkRecord]) -> None:
    for note in section.get("notes", []):
        note_no = note["note_no"]
        note_title = note.get("note_title") or f"주석 {note_no}"

        block_idx = 0
        block_idx = _note_table_cell_chunks(
            stem,
            major_title,
            section["title"],
            note,
            chunks,
            block_idx,
        )

        merged_parts: list[str] = []
        for block in note.get("intro_blocks", []):
            merged_parts.append(block["text"])
        for sec in note.get("subsections", []):
            block_idx = _note_subsection_chunks(
                stem,
                major_title,
                section["title"],
                note,
                sec,
                chunks,
                block_idx,
            )
            sec_text = _collect_section_text(sec)
            if sec_text:
                merged_parts.append(sec_text)

        merged_text = "\n\n".join(part for part in merged_parts if part.strip())
        if not merged_text:
            continue

        prefix = f"[주석 {note_no}. {note_title}]"
        parts = _chunk_note_text(prefix, merged_text)
        for idx, part in enumerate(parts, start=block_idx):
            _append_chunk(
                chunks,
                stem,
                major_title,
                section["title"],
                "note",
                note_no,
                note_title,
                _guess_topic(merged_text) or note_title,
                part,
                idx,
            )


def _major_text_chunks(stem: str, major: dict[str, Any], chunks: list[ChunkRecord]) -> None:
    title = major["title"]
    merged = "\n\n".join(block["text"] for block in major.get("content_blocks", []))
    if not merged:
        return
    for idx, part in enumerate(_chunk_general_text(f"[{title}]", merged)):
        _append_chunk(
            chunks,
            stem,
            title,
            None,
            major.get("major_type", "other"),
            None,
            None,
            _guess_topic(merged),
            part,
            idx,
        )


def build_chunks(report: ParsedReport) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    stem = _slugify(report.file_path.stem)

    for major in report.structured.get("major_sections", []):
        major_title = major["title"]
        _major_text_chunks(stem, major, chunks)

        for section in major.get("sections", []):
            section_type = section.get("section_type")
            if section_type == "financial_statement":
                _financial_statement_chunks(stem, major_title, section, chunks)
            elif section_type == "notes_section":
                _notes_chunks(stem, major_title, section, chunks)
            else:
                merged = "\n\n".join(block["text"] for block in section.get("content_blocks", []))
                if not merged:
                    continue
                for idx, part in enumerate(_chunk_general_text(f"[{section['title']}]", merged)):
                    _append_chunk(
                        chunks,
                        stem,
                        major_title,
                        section["title"],
                        section_type or "section",
                        None,
                        None,
                        _guess_topic(merged),
                        part,
                        idx,
                    )

    return chunks