from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from bs4 import BeautifulSoup, FeatureNotFound, NavigableString, Tag


STATEMENT_TITLES = {
    "재무상태표",
    "손익계산서",
    "포괄손익계산서",
    "자본변동표",
    "현금흐름표",
}

SKIP_COMPACT_TEXTS = {
    "별첨주석은본재무제표의일부입니다.",
    "계속;",
    "계속",
}

YEAR_RE = re.compile(r"(20\d{2})년")
FILENAME_YEAR_RE = re.compile(r"(20\d{2})")

# note parser regex
NOTE_START_RE = re.compile(r"^\s*(\d{1,3})\s*[.\)]\s*(.+?)\s*$")
DECIMAL_SECTION_RE = re.compile(r"^\s*(\d{1,3}\.\d{1,3})\s+(.+?)\s*$")
KOREAN_SECTION_RE = re.compile(r"^\s*([가-하])\.\s*(.+?)\s*$")
PAREN_SECTION_RE = re.compile(r"^\s*\((\d{1,3})\)\s*(.+?)\s*$")
CONTINUATION_RE = re.compile(r"^\s*(.+?)\s*,?\s*계\s*속\s*:?\s*$")
ONLY_CONTINUATION_RE = re.compile(r"^\s*계\s*속\s*:?\s*$")


@dataclass(slots=True)
class ParsedReport:
    file_path: Path
    file_name: str
    title: str
    report_year: int | None
    raw_html: str
    parser_used: str
    structured: dict[str, Any]


def normalize_space(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n+ *", "\n", text)
    return text.strip()


def normalize_compact(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", "", text)
    return text.strip()


def read_html_text(file_path: str | Path) -> str:
    path = Path(file_path)
    raw = path.read_bytes()
    for enc in ("euc-kr", "cp949", "utf-8"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("euc-kr", errors="ignore")


def build_soup_with_fallback(html_text: str) -> tuple[BeautifulSoup, str]:
    errors: list[str] = []
    for parser in ("lxml", "html5lib", "html.parser"):
        try:
            return BeautifulSoup(html_text, parser), parser
        except FeatureNotFound as exc:
            errors.append(f"{parser}: {exc}")
        except Exception as exc:
            errors.append(f"{parser}: {exc}")
    raise RuntimeError("No HTML parser available: " + " | ".join(errors))


def tag_text(tag: Tag) -> str:
    return normalize_space(tag.get_text(" ", strip=True))


def tag_text_br(tag: Tag) -> str:
    """tag_text variant that preserves <br> as newlines, used for note paragraphs."""
    for br in tag.find_all("br"):
        br.replace_with("\n")
    text = tag.get_text(" ", strip=True)
    # collapse spaces but keep newlines
    text = re.sub(r"[ \t\r\f\v]+", " ", text.replace("\xa0", " "))
    text = re.sub(r" *\n+ *", "\n", text)
    return text.strip()


def get_section_level(tag: Tag) -> Optional[int]:
    for cls in (tag.get("class") or []):
        m = re.match(r"(?i)^section-(\d+)$", str(cls).strip())
        if m:
            return int(m.group(1))
    return None


def is_section_header(tag: Tag) -> bool:
    return isinstance(tag, Tag) and tag.name in ("h2", "h3") and get_section_level(tag) in (1, 2)


def get_section_name(header_tag: Tag) -> str:
    text = normalize_space(header_tag.get_text(" ", strip=True))
    if text:
        return text

    level = get_section_level(header_tag)
    for sib in header_tag.next_siblings:
        if isinstance(sib, NavigableString):
            continue
        if not isinstance(sib, Tag):
            break
        if is_section_header(sib):
            break
        sib_level = get_section_level(sib)
        if sib_level == level and sib.name in ("p", "div", "span"):
            text = normalize_space(sib.get_text(" ", strip=True))
            if text:
                return text
        if sib.name not in ("p", "div", "span"):
            break
    return "(섹션명 없음)"


def iter_section_nodes(header_tag: Tag) -> Iterable[Tag]:
    level = get_section_level(header_tag) or 1
    for sib in header_tag.next_siblings:
        if isinstance(sib, NavigableString):
            continue
        if not isinstance(sib, Tag):
            continue
        if is_section_header(sib):
            sib_level = get_section_level(sib) or 99
            if sib_level <= level:
                break
        yield sib


def _table_to_grid(table_tag: Tag) -> list[list[str]]:
    """
    HTML table -> 2차원 grid 로 펼친다.
    rowspan / colspan 을 반영해서 병합 셀도 실제 좌표에 맞게 채운다.
    """
    grid: list[list[str]] = []
    pending_spans: dict[tuple[int, int], str] = {}

    trs = table_tag.find_all("tr")
    for r_idx, tr in enumerate(trs):
        row: list[str] = []
        c_idx = 0

        # 이전 row의 rowspan 잔여분 먼저 채운다
        while (r_idx, c_idx) in pending_spans:
            row.append(pending_spans.pop((r_idx, c_idx)))
            c_idx += 1

        cells = tr.find_all(["th", "td"], recursive=False)
        if not cells:
            cells = tr.find_all(["th", "td"])

        for cell in cells:
            while (r_idx, c_idx) in pending_spans:
                row.append(pending_spans.pop((r_idx, c_idx)))
                c_idx += 1

            text = tag_text(cell)
            rowspan = int(cell.get("rowspan", 1) or 1)
            colspan = int(cell.get("colspan", 1) or 1)

            for dx in range(colspan):
                row.append(text if text else "")
                # 아래 row로 span 예약
                for dy in range(1, rowspan):
                    pending_spans[(r_idx + dy, c_idx + dx)] = text if text else ""
            c_idx += colspan

        # 행 끝에도 rowspan 잔여분이 있으면 채운다
        while (r_idx, c_idx) in pending_spans:
            row.append(pending_spans.pop((r_idx, c_idx)))
            c_idx += 1

        grid.append(row)

    if not grid:
        return []

    max_cols = max(len(row) for row in grid)
    normalized_grid: list[list[str]] = []
    for row in grid:
        if len(row) < max_cols:
            row = row + [""] * (max_cols - len(row))
        normalized_grid.append([normalize_space(x) for x in row])

    return normalized_grid


def table_rows_to_text(table_tag: Tag) -> str:
    lines: list[str] = []
    for tr in table_tag.find_all("tr"):
        row: list[str] = []
        for cell in tr.find_all(["th", "td"]):
            txt = tag_text(cell)
            if txt:
                row.append(txt)
        if row:
            lines.append(" | ".join(row))
    return "\n".join(lines).strip()


def table_rows_to_text_preserve_structure(table_tag: Tag) -> str:
    """
    표를 pipe text 로 변환하되,
    rowspan/colspan 을 반영한 grid 기반으로 직렬화한다.
    """
    grid = _table_to_grid(table_tag)
    if not grid:
        return ""

    lines: list[str] = []
    for row in grid:
        # 완전 빈 행은 제거
        if not any(normalize_space(cell) for cell in row):
            continue
        lines.append(" | ".join(cell if cell is not None else "" for cell in row))

    return "\n".join(lines).strip()


def detect_statement_title_from_table(table_tag: Tag) -> str | None:
    first_tr = table_tag.find("tr")
    if first_tr is None:
        return None
    cells = first_tr.find_all(["td", "th"])
    if not cells:
        return None
    for cell in cells[:3]:
        compact = normalize_compact(tag_text(cell))
        if compact in STATEMENT_TITLES:
            return compact
    table_compact = normalize_compact(table_rows_to_text(table_tag))
    for title in STATEMENT_TITLES:
        if title in table_compact[:80]:
            return title
    return None


def is_pgbrk_p(node: Tag) -> bool:
    return node.name == "p" and "PGBRK" in (node.get("class") or [])


def is_table_with_class(node: Tag, class_name: str) -> bool:
    classes = {str(c).strip().lower() for c in (node.get("class") or [])}
    return node.name == "table" and class_name.lower() in classes


def detect_statement_title_from_nb_table(table_tag: Tag) -> str | None:
    if not is_table_with_class(table_tag, "nb"):
        return None

    for td in table_tag.find_all("td"):
        compact = normalize_compact(tag_text(td))
        if compact in STATEMENT_TITLES:
            return compact

    nb_compact = normalize_compact(table_rows_to_text(table_tag))
    for title in STATEMENT_TITLES:
        if title in nb_compact:
            return title
    return None


def infer_company_name(raw_html: str, soup: BeautifulSoup) -> str:
    title = normalize_space(soup.title.get_text()) if soup.title else ""
    if "삼성전자" in title:
        return "삼성전자주식회사"
    text = normalize_space((soup.body or soup).get_text(" ", strip=True))
    if "삼성전자주식회사" in text[:5000]:
        return "삼성전자주식회사"
    return "삼성전자주식회사"


def infer_report_year(path: Path, raw_html: str) -> int | None:
    filename_match = FILENAME_YEAR_RE.search(path.stem)
    if filename_match:
        return int(filename_match.group(1))

    early_text = raw_html[:6000]
    for pattern in [
        re.compile(r"제\s*\d+\s*기.*?(20\d{2})년\s*01월\s*01일", re.S),
        re.compile(r"제\s*\d+\s*기.*?(20\d{2})년\s*12월\s*31일", re.S),
    ]:
        m = pattern.search(early_text)
        if m:
            return int(m.group(1))

    years = [int(y) for y in YEAR_RE.findall(early_text)]
    if years:
        counts: dict[int, int] = {}
        for y in years:
            counts[y] = counts.get(y, 0) + 1
        return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
    return None


def classify_major_type(title: str) -> str:
    compact = normalize_compact(title)
    if compact in {"(섹션명없음)", ""}:
        return "other"
    if "독립된감사인의감사보고서" in compact:
        return "audit_report"
    if "첨부" in compact and "재무제표" in compact:
        return "financial_statements_bundle"
    if "내부회계관리제도" in compact and ("검토보고서" in compact or "검토의견" in compact):
        return "internal_control_review"
    if "내부회계관리제도운영실태평가보고서" in compact:
        return "internal_control_management_report"
    if "외부감사실시내용" in compact:
        return "external_audit_details"
    return "other"


def _append_block(target: list[dict[str, Any]], block_type: str, text: str, **meta: Any) -> None:
    text = normalize_space(text)
    if not text:
        return
    if normalize_compact(text) in SKIP_COMPACT_TEXTS:
        return
    block = {"block_type": block_type, "text": text}
    if meta:
        block.update(meta)
    target.append(block)

def parse_numeric(value: Any) -> float | None:
    if value is None:
        return None

    s = normalize_space(str(value))
    if s in {"", "nan", "None"}:
        return None
    if s in {"-", "—", "△"}:
        return 0.0

    negative = s.startswith("(") and s.endswith(")")
    s = s.replace(",", "").replace("(", "").replace(")", "").strip()

    try:
        num = float(s)
        return -num if negative else num
    except ValueError:
        return None


def normalize_account_name(text: str) -> str:
    if not text:
        return ""
    normalized = normalize_space(text)
    normalized = re.sub(r"^\s*[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\s*[\.|\)]\s*", "", normalized)
    normalized = re.sub(r"^\s*\d+\s*[\.|\)]\s*", "", normalized)
    return re.sub(r"\s+", "", normalized)


def split_note_refs(text: Any) -> list[str]:
    if text is None:
        return []

    s = normalize_space(str(text))
    if s in {"", "-", "—", "nan", "None"}:
        return []

    return [x.strip() for x in s.split(",") if x.strip()]


def parse_statement_meta(subtitle_text: str, data_text: str = "") -> dict[str, Any]:
    combined_text = normalize_space("\n".join([subtitle_text, data_text]).strip())

    unit = None
    m = re.search(r"단위\s*[:：]\s*([^\)\n]+)", combined_text)
    if m:
        unit = normalize_space(m.group(1))

    current_label = None
    prior_label = None

    period_labels = re.findall(r"(제\s*\d+\s*\((?:당|전)\)\s*기)", combined_text)
    for label in period_labels:
        normalized = normalize_space(label)
        compact = normalize_compact(normalized)
        if "당" in compact and current_label is None:
            current_label = normalized
        elif "전" in compact and prior_label is None:
            prior_label = normalized

    if (current_label is None or prior_label is None) and data_text:
        lines = [normalize_space(line) for line in data_text.split("\n") if normalize_space(line)]
        if lines:
            header_parts = [normalize_space(x) for x in lines[0].split("|")]
            value_headers = [
                part
                for part in header_parts
                if part and "과목" not in normalize_compact(part) and "주석" not in normalize_compact(part)
            ]
            for header in value_headers:
                compact = normalize_compact(header)
                if "당" in compact and current_label is None:
                    current_label = header
                elif "전" in compact and prior_label is None:
                    prior_label = header

    return {
        "unit": unit,
        "current_period_label": current_label,
        "prior_period_label": prior_label,
    }


def _should_parse_statement_rows(statement_title: str) -> bool:
    return statement_title in {"재무상태표", "손익계산서", "포괄손익계산서", "자본변동표", "현금흐름표"}


ROMAN_SECTION_RE = re.compile(r"^\s*[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\s*[\.|\)]\s*")
LEADING_NUM_RE = re.compile(r"^\s*\d+\s*[\.|\)]\s*")


def count_display_indent(text: str) -> int:
    count = 0
    for ch in text:
        if ch in {" ", "\xa0"}:
            count += 1
        else:
            break
    return count


def clean_statement_label(raw_label: str) -> str:
    text = (raw_label or "").replace("\xa0", " ").lstrip(" ")
    text = ROMAN_SECTION_RE.sub("", text)
    text = LEADING_NUM_RE.sub("", text)
    return normalize_space(text)


def classify_statement_row_priority(raw_label: str, cleaned_label: str, has_amount: bool) -> tuple[int, str, bool, bool]:
    compact = normalize_compact(cleaned_label)
    raw_no_indent = (raw_label or "").replace("\xa0", " ").lstrip(" ")

    # 1) Roman numeral section
    if ROMAN_SECTION_RE.match(raw_no_indent):
        return 0, "section", False, False

    # 2) Total
    if "총계" in compact:
        return 0, "total", True, False

    # 3) Subtotal/profit
    if any(token in compact for token in ("이익", "소계", "합계")):
        return 0, "subtotal", False, True

    # 4) Generic row with numeric amount
    if has_amount:
        return 1, "item", False, False

    # 7) Non-amount row (non-section)
    return 0, "heading", False, False


def parse_statement_table_text(table_text: str) -> list[dict[str, Any]]:
    raw_lines = [line.replace("\xa0", " ") for line in str(table_text).split("\n")]
    lines = [line for line in raw_lines if normalize_space(line)]
    if not lines:
        return []

    header_parts = [normalize_space(x) for x in lines[0].split("|")]
    header_compacts = [normalize_compact(x) for x in header_parts]
    data_lines = lines[1:]

    has_note_column = any("주석" in compact for compact in header_compacts)
    rows: list[dict[str, Any]] = []
    current_section_row_id: str | None = None
    current_section_label: str | None = None
    row_seq = 0

    for line in data_lines:
        raw_parts = [x for x in line.split("|")]
        parts = [normalize_space(x) for x in raw_parts]
        if len(parts) < 2:
            continue

        account_name_raw = raw_parts[0]
        display_indent = count_display_indent(account_name_raw)
        account_name = clean_statement_label(account_name_raw)
        account_name_normalized = normalize_account_name(account_name)

        # 대분류/빈 행 제거
        if not account_name_normalized or account_name_normalized in {"자산", "부채", "자본"}:
            continue

        note_text = ""
        current_text = None
        prior_text = None

        if has_note_column:
            if len(parts) >= 4:
                note_text = parts[1]
                current_text = parts[2]
                prior_text = parts[3]
            elif len(parts) == 3:
                # 주석번호가 비어 있는 계정행도 있다. 예: "Ⅰ. 매출액 | 170,381,870 | 161,915,007"
                note_text = ""
                current_text = parts[1]
                prior_text = parts[2]
            else:
                continue
        else:
            if len(parts) >= 3:
                current_text = parts[1]
                prior_text = parts[2]
            else:
                continue

        parsed_current_amount = parse_numeric(current_text)
        parsed_prior_amount = parse_numeric(prior_text)
        has_amount = parsed_current_amount is not None or parsed_prior_amount is not None

        level, node_type, is_total, is_subtotal = classify_statement_row_priority(
            account_name_raw,
            account_name,
            has_amount,
        )

        # Rule 4: if no active section exists, item is treated as level 0.
        if node_type == "item" and current_section_row_id is None:
            level = 0

        row_seq += 1
        row_id = f"row_{row_seq:04d}"
        parent_row_id: str | None = None
        root_row_id = row_id

        if node_type == "section":
            current_section_row_id = row_id
            current_section_label = account_name_normalized

        if level == 1 and current_section_row_id is not None:
            parent_row_id = current_section_row_id
            root_row_id = current_section_row_id

        lineage_labels: list[str] = [account_name_normalized]
        if level == 1 and current_section_label:
            lineage_labels = [current_section_label, account_name_normalized]

        rows.append(
            {
                "row_id": row_id,
                "hierarchy_level": level,
                "node_type": node_type,
                "parent_row_id": parent_row_id,
                "root_row_id": root_row_id,
                "lineage_labels": lineage_labels,
                "display_indent": display_indent,
                "is_total": is_total,
                "is_subtotal": is_subtotal,
                "account_name": account_name,
                "account_name_normalized": account_name_normalized,
                "note_refs": split_note_refs(note_text),
                "raw_note_text": note_text,
                "current_amount": parsed_current_amount,
                "prior_amount": parsed_prior_amount,
                "raw_current_text": current_text,
                "raw_prior_text": prior_text,
            }
        )

    return rows



# -----------------------------
# hierarchical note parsing
# -----------------------------

def clean_heading_text(text: str) -> str:
    text = normalize_space(text)
    text = re.sub(r"\b계\s*속\s*:?;?\s*$", "", text).strip(" ,:-")
    return text.strip(" ,")


def normalize_heading_text(text: str) -> str:
    text = normalize_space(text)
    text = re.sub(r"\s*계\s*속\s*:?\s*$", "", text).strip(" :")
    return text.strip()


def clean_title_tail(title: str) -> str:
    if not title:
        return ""
    title = re.sub(r"\s*:\s*$", "", title)
    title = re.sub(r"\s*계\s*속\s*$", "", title)
    return title.strip()


def make_block(block_type: str, text: str) -> dict[str, str]:
    return {"block_type": block_type, "text": normalize_space(text)}


def make_section_node(code: str, title: str, level: int) -> dict[str, Any]:
    return {
        "code": code,
        "title": clean_title_tail(title),
        "level": level,
        "content_blocks": [],
        "children": [],
    }


def make_note_node(note_no: int, title: str) -> dict[str, Any]:
    return {
        "note_no": note_no,
        "note_title": clean_title_tail(title),
        "intro_blocks": [],
        "subsections": [],
        "tables": [],
        "raw_blocks": [],
    }


UNIT_LINE_RE = re.compile(r"\(\s*단위\s*[:：]\s*([^\)]+)\)")


def normalize_table_label(text: str) -> str:
    cleaned = normalize_space(str(text))
    cleaned = re.sub(r"\([^\)]*\)", "", cleaned)
    cleaned = re.sub(r"^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ\d\.\-\s]+", "", cleaned)
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned.strip()


def _row_has_numeric_values(parts: list[str], start_idx: int = 1) -> bool:
    for cell in parts[start_idx:]:
        if parse_numeric(cell) is not None:
            return True
    return False


def _split_pipe_row(line: str) -> list[str]:
    return [normalize_space(x) for x in str(line).split("|")]


def _pad_rows(rows: list[list[str]]) -> list[list[str]]:
    if not rows:
        return rows
    max_len = max(len(r) for r in rows)
    return [r + [""] * (max_len - len(r)) for r in rows]


def _is_effective_empty_row(row: list[str]) -> bool:
    return not any(normalize_space(x) for x in row)


def _count_non_empty(row: list[str]) -> int:
    return sum(1 for x in row if normalize_space(x))


def _is_group_row(row: list[str]) -> bool:
    """
    표 내부 그룹 행 판단:
    예) '단기차입금: | | | |'
        '(1) 법정적립금: | |'
        '(2) 임의적립금: | |'
    """
    non_empty = [normalize_space(x) for x in row if normalize_space(x)]
    if not non_empty:
        return False
    if len(non_empty) != 1:
        return False

    txt = non_empty[0]
    # 숫자 데이터가 아니고, 보통 ':' 또는 항목 구분 패턴
    if parse_numeric(txt) is not None:
        return False
    if txt.endswith(":"):
        return True
    if re.match(r"^\(\d+\)\s*", txt):
        return True
    if re.match(r"^[가-하]\.\s*", txt):
        return True
    return False


def _looks_like_header_row(row: list[str]) -> bool:
    """
    헤더 후보 판단:
    숫자보다 텍스트 중심이고, group row 는 제외
    """
    if _is_group_row(row):
        return False

    non_empty = [normalize_space(x) for x in row if normalize_space(x)]
    if not non_empty:
        return False

    numeric_count = sum(1 for x in non_empty if parse_numeric(x) is not None)
    # 헤더는 대체로 숫자가 거의 없다
    return numeric_count == 0


def _normalize_header_matrix(header_rows: list[list[str]]) -> list[list[str]]:
    """
    rowspan 펼침으로 인해 같은 헤더가 반복된 상태를 정리한다.
    예:
      ['구분','차입처','연이자율(%)','금액','금액']
      ['구분','차입처','당기말','당기말','전기말']
    """
    header_rows = _pad_rows(header_rows)
    normalized = []

    for r_idx, row in enumerate(header_rows):
        new_row = []
        for c_idx, cell in enumerate(row):
            cell_norm = normalize_table_label(cell)
            # 바로 위와 동일하면 하위헤더가 아니라 span 복제일 수 있음
            if r_idx > 0:
                upper = normalize_table_label(header_rows[r_idx - 1][c_idx])
                if cell_norm == upper:
                    cell_norm = ""
            new_row.append(cell_norm)
        normalized.append(new_row)

    return normalized


def _build_column_map_from_headers(header_rows: list[list[str]]) -> list[dict[str, str]]:
    """
    1단/2단 헤더를 컬럼 인덱스 기준으로 매핑한다.
    """
    if not header_rows:
        return []

    header_rows = _normalize_header_matrix(header_rows)
    col_count = max(len(r) for r in header_rows)

    col_map: list[dict[str, str]] = []
    for c_idx in range(col_count):
        parts: list[str] = []
        raw_parts: list[str] = []

        for r in header_rows:
            raw = r[c_idx] if c_idx < len(r) else ""
            raw_parts.append(raw)
            norm = normalize_table_label(raw)
            if norm and (not parts or parts[-1] != norm):
                parts.append(norm)

        if not parts:
            flat = f"col{c_idx + 1}"
            parent = ""
            child = ""
        elif len(parts) == 1:
            flat = parts[0]
            parent = parts[0]
            child = ""
        else:
            parent = parts[0]
            child = parts[-1]
            flat = normalize_table_label("".join(parts))

        col_map.append(
            {
                "parent": parent,
                "child": child,
                "flat": flat,
            }
        )

    return col_map


def _is_section_like_row(row: list[str]) -> bool:
    """
    group보다는 약하지만 section 역할 하는 row
    ex) 배당금액, 배당받을 주식
    """
    non_empty = [x for x in row if normalize_space(x)]
    if len(non_empty) == 1:
        txt = normalize_space(non_empty[0])
        if parse_numeric(txt) is None:
            return True
    return False


def parse_note_table_matrix(table_text: str, inherited_unit: str | None = None) -> dict[str, Any] | None:
    lines = [normalize_space(line) for line in str(table_text).splitlines() if normalize_space(line)]
    if not lines:
        return None

    unit = inherited_unit
    pipe_lines: list[str] = []
    for line in lines:
        unit_match = UNIT_LINE_RE.search(line)
        if unit_match:
            unit = normalize_space(unit_match.group(1))
        if "|" in line:
            pipe_lines.append(line)

    if len(pipe_lines) < 2:
        return None

    pipe_rows = [_split_pipe_row(line) for line in pipe_lines]
    pipe_rows = _pad_rows(pipe_rows)
    pipe_rows = [row for row in pipe_rows if not _is_effective_empty_row(row)]

    if len(pipe_rows) < 2:
        return None

    # header row 수를 1~2개 정도로 추정
    header_rows: list[list[str]] = []
    data_rows: list[list[str]] = []

    for idx, row in enumerate(pipe_rows):
        if idx < 2 and _looks_like_header_row(row):
            header_rows.append(row)
            continue
        data_rows = pipe_rows[idx:]
        break

    if not header_rows:
        header_rows = [pipe_rows[0]]
        data_rows = pipe_rows[1:]

    if not data_rows:
        return None

    column_map = _build_column_map_from_headers(header_rows)
    if not column_map:
        return None

    # 첫 번째 컬럼은 row label 로 쓰는 경우가 많으므로 따로 본다.
    # 다만 값 컬럼 매핑은 2번째 컬럼부터 그대로 사용한다.
    rows: list[dict[str, Any]] = []
    current_group: str | None = None
    row_stack: list[str] = []

    for row in data_rows:
        if _is_effective_empty_row(row):
            continue

        # 1️⃣ group row
        if _is_group_row(row):
            txt = next((x for x in row if normalize_space(x)), None)
            current_group = txt
            row_stack = [txt] if txt else []
            continue

        # 2️⃣ section row (중간 계층)
        if _is_section_like_row(row):
            txt = next((x for x in row if normalize_space(x)), None)
            if txt:
                # 깊이 유지하면서 append
                if len(row_stack) >= 2:
                    row_stack = row_stack[:1]  # 너무 깊어지지 않게
                row_stack.append(txt)
            continue

        # 3️⃣ 실제 데이터 row
        row_label_raw = row[0] if row else ""
        row_label = normalize_table_label(row_label_raw)
        if not row_label:
            continue

        values: dict[str, Any] = {}
        raw_values: dict[str, Any] = {}

        for c_idx in range(1, len(row)):
            value_text = row[c_idx]
            meta = column_map[c_idx] if c_idx < len(column_map) else {
                "parent": "",
                "child": "",
                "flat": f"col{c_idx + 1}",
            }

            parent = meta["parent"]
            child = meta["child"]
            flat = meta["flat"]

            if not flat:
                flat = f"col{c_idx + 1}"

            # 2단 헤더면 parent > child 구조로 저장
            if child:
                parent_key = parent or flat
                child_key = child or f"col{c_idx + 1}"

                parent_raw = raw_values.setdefault(parent_key, {})
                parent_num = values.setdefault(parent_key, {})

                if isinstance(parent_raw, dict) and isinstance(parent_num, dict):
                    parent_raw[child_key] = value_text
                    parent_num[child_key] = parse_numeric(value_text)
            else:
                raw_values[flat] = value_text
                values[flat] = parse_numeric(value_text)

        row_path = []

        if current_group:
            row_path.append(normalize_space(current_group))

        if len(row_stack) > 1:
            row_path.extend([normalize_space(x) for x in row_stack[1:]])

        row_path.append(row_label)

        row_item = {
            "row_label": row_label,
            "row_label_raw": row_label_raw,
            "row_path": row_path,
            "values": values,
            "raw_values": raw_values,
        }

        if current_group:
            row_item["row_group"] = normalize_space(current_group)
            row_item["row_group_raw"] = current_group

        rows.append(row_item)

    if not rows:
        return None

    header_raw = header_rows[0] if header_rows else []
    sub_header_raw = header_rows[1] if len(header_rows) >= 2 else None

    return {
        "table_type": "matrix",
        "unit": unit,
        "header": [normalize_table_label(h) for h in header_raw],
        "header_raw": header_raw,
        "sub_header": [normalize_table_label(h) for h in sub_header_raw] if sub_header_raw else None,
        "sub_header_raw": sub_header_raw,
        "rows": rows,
        "raw_text": "\n".join(pipe_lines),
    }


def extract_note_tables_from_blocks(raw_blocks: list[dict[str, str]]) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    current_unit: str | None = None

    for idx, block in enumerate(raw_blocks):
        if block.get("block_type") != "table":
            continue

        text = normalize_space(block.get("text", ""))
        if not text:
            continue

        unit_match = UNIT_LINE_RE.search(text)
        if unit_match and "|" not in text:
            current_unit = normalize_space(unit_match.group(1))
            continue

        parsed = parse_note_table_matrix(text, inherited_unit=current_unit)
        if parsed is None:
            continue

        parsed["source_block_index"] = idx
        tables.append(parsed)

    return tables


def append_to_current(container: Optional[dict[str, Any]], block: dict[str, str]) -> None:
    if container is not None:
        container["content_blocks"].append(block)


def append_to_note_intro(note: Optional[dict[str, Any]], block: dict[str, str]) -> None:
    if note is not None:
        note["intro_blocks"].append(block)


def classify_heading(line: str) -> dict[str, Any] | None:
    line = normalize_heading_text(line)
    if not line:
        return None

    m = DECIMAL_SECTION_RE.match(line)
    if m:
        code, title = m.groups()
        note_no = int(code.split(".")[0])
        return {
            "kind": "decimal",
            "code": code,
            "title": title,
            "level": 1,
            "note_no": note_no,
        }

    m = KOREAN_SECTION_RE.match(line)
    if m:
        code, title = m.groups()
        return {
            "kind": "korean",
            "code": f"{code}.",
            "title": title,
            "level": 2,
        }

    m = PAREN_SECTION_RE.match(line)
    if m:
        code, title = m.groups()
        return {
            "kind": "paren",
            "code": f"({code})",
            "title": title,
            "level": 3,
        }

    m = NOTE_START_RE.match(line)
    if m:
        no, title = m.groups()
        return {
            "kind": "note",
            "note_no": int(no),
            "title": title,
            "level": 0,
        }

    return None


def parse_embedded_note_heading(text: str) -> dict[str, Any] | None:
    text = normalize_space(text)

    m = re.match(r"^\s*(\d{1,3})\s*[.\)]\s*(.+?)\s*:\s*(.+)$", text)
    if m:
        note_no, title, rest = m.groups()
        return {
            "type": "note_with_rest",
            "note_no": int(note_no),
            "title": clean_title_tail(title),
            "rest": normalize_space(rest),
        }

    m = re.match(r"^\s*(\d{1,3})\s*[.\)]\s*(.+?)\s*,?\s*계\s*속\s*:\s*(.+)$", text)
    if m:
        note_no, title, rest = m.groups()
        return {
            "type": "note_continue_with_rest",
            "note_no": int(note_no),
            "title": clean_title_tail(title),
            "rest": normalize_space(rest),
        }

    return None


def split_inline_child_heading(text: str) -> dict[str, str] | None:
    text = normalize_space(text)

    m = KOREAN_SECTION_RE.match(text)
    if m:
        code, title = m.groups()
        return {"kind": "korean", "code": f"{code}.", "title": clean_title_tail(title)}

    m = PAREN_SECTION_RE.match(text)
    if m:
        code, title = m.groups()
        return {"kind": "paren", "code": f"({code})", "title": clean_title_tail(title)}

    return None


# Lookahead splits at inline sub-heading boundaries: 가. 나. ... or (1) (2) ...
_INLINE_HEADING_SPLIT_RE = re.compile(
    r"(?<![가-하])(?=[가-하]\. )|(?=\(\d{1,3}\)[\s가-힣A-Za-z])"
)


def _split_inline_headings(line: str) -> list[str]:
    """Split '(*) prefix 가. AAA 나. BBB (1) CCC' into separate heading candidates."""
    parts = _INLINE_HEADING_SPLIT_RE.split(line)
    return [p.strip() for p in parts if p.strip()]


def flatten_lines_from_block(block: dict[str, Any]) -> list[dict[str, str]]:
    block_type = block.get("block_type", "paragraph")
    text = normalize_space(block.get("text", ""))
    if not text:
        return []
    if block_type == "table":
        return [make_block("table", text)]
    lines = [normalize_space(x) for x in text.split("\n")]
    result: list[dict[str, str]] = []
    for line in lines:
        if not line:
            continue
        for part in _split_inline_headings(line):
            if part:
                result.append(make_block("paragraph", part))
    return result


def postprocess_notes(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def clean_section(node: dict[str, Any]) -> dict[str, Any] | None:
        node["title"] = clean_title_tail(node.get("title", ""))
        cleaned_children = []
        for child in node.get("children", []):
            cleaned = clean_section(child)
            if cleaned is not None:
                cleaned_children.append(cleaned)
        node["children"] = cleaned_children

        has_content = bool(node.get("content_blocks"))
        has_children = bool(node.get("children"))
        has_title = bool(node.get("title"))
        is_root = node.get("code") == "ROOT"

        if is_root and not has_content and not has_children:
            return None
        if not is_root and not has_title and not has_content and not has_children:
            return None
        return node

    result = []
    for note in notes:
        note["note_title"] = clean_title_tail(note.get("note_title", ""))
        cleaned_subs = []
        for sec in note.get("subsections", []):
            cleaned = clean_section(sec)
            if cleaned is None:
                continue
            if cleaned.get("code") == "ROOT":
                if cleaned.get("content_blocks"):
                    note["intro_blocks"].extend(cleaned["content_blocks"])
                cleaned_subs.extend(cleaned.get("children", []))
            else:
                cleaned_subs.append(cleaned)
        note["subsections"] = cleaned_subs
        note["tables"] = extract_note_tables_from_blocks(note.get("raw_blocks", []))
        result.append(note)
    return result


def parse_notes_hierarchical(content_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    current_note: Optional[dict[str, Any]] = None
    current_decimal: Optional[dict[str, Any]] = None
    current_korean: Optional[dict[str, Any]] = None
    current_paren: Optional[dict[str, Any]] = None

    def reset_lower(level: int) -> None:
        nonlocal current_decimal, current_korean, current_paren
        if level <= 1:
            current_decimal = None
        if level <= 2:
            current_korean = None
        if level <= 3:
            current_paren = None

    def start_new_note(note_no: int, title: str) -> None:
        nonlocal current_note, current_decimal, current_korean, current_paren
        current_note = make_note_node(note_no, title)
        notes.append(current_note)
        current_decimal = None
        current_korean = None
        current_paren = None

    def ensure_note(note_no: int, title: str) -> None:
        nonlocal current_note
        if current_note is None or current_note["note_no"] != note_no:
            start_new_note(note_no, title)
        elif title and not current_note["note_title"]:
            current_note["note_title"] = clean_title_tail(title)

    def start_decimal_section(code: str, title: str) -> None:
        nonlocal current_decimal, current_korean, current_paren, current_note
        if current_note is None:
            return
        node = make_section_node(code, title, level=1)
        current_note["subsections"].append(node)
        current_decimal = node
        current_korean = None
        current_paren = None

    def start_korean_section(code: str, title: str) -> None:
        nonlocal current_korean, current_paren, current_decimal, current_note
        if current_note is None:
            return
        parent = current_decimal
        if parent is None:
            parent = make_section_node("ROOT", "", level=1)
            current_note["subsections"].append(parent)
            current_decimal = parent
        node = make_section_node(code, title, level=2)
        parent["children"].append(node)
        current_korean = node
        current_paren = None

    def start_paren_section(code: str, title: str) -> None:
        nonlocal current_paren, current_korean, current_decimal, current_note
        if current_note is None:
            return
        parent = current_korean or current_decimal
        if parent is None:
            parent = make_section_node("ROOT", "", level=1)
            current_note["subsections"].append(parent)
            current_decimal = parent
        node = make_section_node(code, title, level=3)
        parent["children"].append(node)
        current_paren = node

    def get_current_target() -> Optional[dict[str, Any]]:
        if current_paren is not None:
            return current_paren
        if current_korean is not None:
            return current_korean
        if current_decimal is not None and current_decimal.get("code") != "ROOT":
            return current_decimal
        return None

    for raw_block in content_blocks:
        for block in flatten_lines_from_block(raw_block):
            text = block["text"]
            block_type = block["block_type"]

            if current_note is not None:
                current_note["raw_blocks"].append(block)

            if block_type == "table":
                target = get_current_target()
                if target is not None:
                    append_to_current(target, block)
                else:
                    append_to_note_intro(current_note, block)
                continue

            if ONLY_CONTINUATION_RE.match(text):
                continue

            # Skip pure continuation headers like "12. ... 계속 :" before heading classification.
            if CONTINUATION_RE.match(text):
                continue

            embedded = parse_embedded_note_heading(text)
            if embedded:
                ensure_note(embedded["note_no"], embedded["title"])
                reset_lower(1)
                rest = embedded["rest"]
                rest_heading = classify_heading(rest)
                if rest_heading:
                    if rest_heading["kind"] == "decimal":
                        ensure_note(rest_heading["note_no"], "")
                        start_decimal_section(rest_heading["code"], rest_heading["title"])
                    elif rest_heading["kind"] == "korean":
                        start_korean_section(rest_heading["code"], rest_heading["title"])
                    elif rest_heading["kind"] == "paren":
                        start_paren_section(rest_heading["code"], rest_heading["title"])
                    else:
                        append_to_note_intro(current_note, make_block("paragraph", rest))
                else:
                    append_to_note_intro(current_note, make_block("paragraph", rest))
                continue

            heading = classify_heading(text)
            if heading:
                if heading["kind"] == "note":
                    start_new_note(heading["note_no"], heading["title"])
                    continue
                if heading["kind"] == "decimal":
                    ensure_note(heading["note_no"], "")
                    start_decimal_section(heading["code"], heading["title"])
                    continue
                if heading["kind"] == "korean":
                    start_korean_section(heading["code"], heading["title"])
                    continue
                if heading["kind"] == "paren":
                    start_paren_section(heading["code"], heading["title"])
                    continue

            target = get_current_target()
            if target is not None:
                append_to_current(target, block)
            else:
                append_to_note_intro(current_note, block)

    return postprocess_notes(notes)


def collect_notes_nodes(major_nodes: list[Tag]) -> list[Tag]:
    notes_header = None
    for node in major_nodes:
        if is_section_header(node) and get_section_level(node) == 2 and "주석" in get_section_name(node):
            notes_header = node
            break
    if notes_header is None:
        return []

    nodes: list[Tag] = []
    for sib in notes_header.next_siblings:
        if isinstance(sib, NavigableString):
            continue
        if not isinstance(sib, Tag):
            continue
        if is_section_header(sib) and (get_section_level(sib) or 99) <= 1:
            break
        nodes.append(sib)
    return nodes


def extract_financial_statements_by_pgbrk(major_nodes: list[Tag]) -> tuple[list[dict[str, Any]], set[int], set[int]]:
    """Parse financial statement tables using PGBRK -> table.nb -> table.TABLE pattern.

    Rules:
    - A statement starts at p.PGBRK.
    - Find statement subtitle from the next table.nb.
    - Parse table data from the next table.TABLE after that.
    - After parsing a statement, ignore nodes until the next p.PGBRK.
    """
    sections: list[dict[str, Any]] = []
    used_tables: set[int] = set()
    used_nodes: set[int] = set()

    idx = 0
    while idx < len(major_nodes):
        node = major_nodes[idx]
        if not is_pgbrk_p(node):
            idx += 1
            continue

        subtitle_table_idx: int | None = None
        subtitle: str | None = None
        data_table_idx: int | None = None

        j = idx + 1
        segment_end = len(major_nodes)
        while j < len(major_nodes):
            probe = major_nodes[j]
            if is_pgbrk_p(probe):
                segment_end = j
                break

            if subtitle is None and is_table_with_class(probe, "nb"):
                detected = detect_statement_title_from_nb_table(probe)
                if detected:
                    subtitle = detected
                    subtitle_table_idx = j
                    j += 1
                    continue

            if subtitle is not None and is_table_with_class(probe, "TABLE"):
                data_table_idx = j
                break

            j += 1

        if subtitle is not None and subtitle_table_idx is not None and data_table_idx is not None:
            subtitle_table = major_nodes[subtitle_table_idx]
            data_table = major_nodes[data_table_idx]

            subtitle_text = table_rows_to_text(subtitle_table)
            data_text = table_rows_to_text(data_table)
            if data_text:
                statement_meta = parse_statement_meta(subtitle_text, data_text)
                table_rows = parse_statement_table_text(data_text) if _should_parse_statement_rows(subtitle) else []

                sections.append(
                    {
                        "title": subtitle,
                        "section_type": "financial_statement",
                        "content_blocks": [
                            {"block_type": "table", "text": subtitle_text},
                            {"block_type": "table", "text": data_text},
                        ],
                        "table_meta": statement_meta,
                        "table_rows": table_rows,
                        "notes": [],
                    }
                )
                used_tables.add(id(subtitle_table))
                used_tables.add(id(data_table))
                for k in range(idx, segment_end):
                    used_nodes.add(id(major_nodes[k]))

        idx += 1
        while idx < len(major_nodes) and not is_pgbrk_p(major_nodes[idx]):
            idx += 1

    return sections, used_tables, used_nodes


# 재무제표 파싱 함수
def extract_financial_sections(major_nodes: list[Tag]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sections: list[dict[str, Any]] = []
    major_blocks: list[dict[str, Any]] = []
    parsed_sections, used_tables, used_nodes = extract_financial_statements_by_pgbrk(major_nodes)
    sections.extend(parsed_sections)

    notes_nodes = collect_notes_nodes(major_nodes)
    if notes_nodes:
        note_blocks: list[dict[str, Any]] = []
        for node in notes_nodes:
            if node.name == "p" and "PGBRK" in (node.get("class") or []):
                continue
            if node.name == "table":
                text = table_rows_to_text_preserve_structure(node)
                if text:
                    note_blocks.append({"block_type": "table", "text": text})
            else:
                text = tag_text_br(node)  # preserve <br> as newline for inline heading detection
                if text:
                    note_blocks.append({"block_type": "paragraph", "text": text})

        notes = parse_notes_hierarchical(note_blocks)
        sections.append({
            "title": "주석",
            "section_type": "notes_section",
            "content_blocks": [],
            "notes": notes,
        })

    for node in major_nodes:
        if id(node) in used_nodes:
            continue
        if node.name == "table" and id(node) in used_tables:
            continue
        if is_section_header(node):
            continue
        if node.name == "table":
            text = table_rows_to_text(node)
            if text:
                _append_block(major_blocks, "table", text)
        else:
            text = tag_text(node)
            if text:
                _append_block(major_blocks, "paragraph", text)

    return sections, major_blocks


def _is_external_audit_subtitle(node: Tag) -> bool:
    """p.SECTION-2 used as subsection headings inside 외부감사 실시내용."""
    return node.name == "p" and "SECTION-2" in (node.get("class") or [])


def extract_external_audit_sections(major_nodes: list[Tag]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split 외부감사 실시내용 by p.SECTION-2 subtitle boundaries.

    Returns (sections, major_blocks) where each section has a title and
    content_blocks with the raw text of everything that follows it.
    Nodes before the first subtitle go into major_blocks.
    """
    sections: list[dict[str, Any]] = []
    major_blocks: list[dict[str, Any]] = []
    current_title: str | None = None
    current_blocks: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal current_title, current_blocks
        if current_title is None:
            return
        sections.append({
            "title": current_title,
            "section_type": "external_audit_subsection",
            "content_blocks": current_blocks,
        })
        current_title = None
        current_blocks = []

    for node in major_nodes:
        if is_section_header(node):
            continue
        if _is_external_audit_subtitle(node):
            flush()
            current_title = normalize_space(node.get_text(" ", strip=True))
            current_blocks = []
            continue

        if node.name == "table":
            text = table_rows_to_text(node)
            if not text:
                continue
            block = {"block_type": "table", "text": text}
        else:
            text = tag_text(node)
            if not text:
                continue
            block = {"block_type": "paragraph", "text": text}

        if current_title is None:
            major_blocks.append(block)
        else:
            current_blocks.append(block)

    flush()
    return sections, major_blocks


def extract_generic_major_content(major_nodes: list[Tag]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for node in major_nodes:
        if is_section_header(node):
            continue
        if node.name == "table":
            text = table_rows_to_text(node)
            if text:
                _append_block(blocks, "table", text)
        else:
            text = tag_text(node)
            if text:
                _append_block(blocks, "paragraph", text)
    return blocks


def has_bold_style(tag: Tag) -> bool:
    style = (tag.get("style") or "").lower().replace(" ", "")
    if "font-weight:bold" in style:
        return True
    if "font-weight:700" in style:
        return True
    return False


def is_bold_p_tag(tag: Tag) -> bool:
    return tag.name == "p" and has_bold_style(tag)


def iter_text_segments_with_bold(node: Tag) -> list[tuple[str, bool]]:
    segments: list[tuple[str, bool]] = []
    for text_node in node.find_all(string=True):
        text = normalize_space(str(text_node))
        if not text:
            continue

        is_bold = False
        parent = text_node.parent
        while isinstance(parent, Tag):
            if has_bold_style(parent):
                is_bold = True
                break
            if parent is node:
                break
            parent = parent.parent

        if segments and segments[-1][1] == is_bold:
            merged_text = normalize_space(f"{segments[-1][0]} {text}")
            segments[-1] = (merged_text, is_bold)
        else:
            segments.append((text, is_bold))
    return segments


def extract_audit_report_sections(major_nodes: list[Tag]) -> list[dict[str, Any]]:
    """Parse '독립된 감사인의 감사보고서' by bold subtitle boundaries.

    Rules:
    - Ignore everything before the first bold p subtitle.
    - Each bold p starts a new subsection.
    - Collect all following tag texts until the next bold p.
    - Stop parsing entirely when a table appears.
    """
    sections: list[dict[str, Any]] = []
    current_title: str | None = None
    current_parts: list[str] = []
    has_seen_first_subtitle = False

    def flush_current() -> None:
        nonlocal current_title, current_parts
        if not current_title:
            return
        content = "\n\n".join(part for part in current_parts if part.strip())
        if not content:
            return
        sections.append(
            {
                "title": current_title,
                "section_type": "audit_subsection",
                "content_blocks": [{"block_type": "paragraph", "text": content}],
                "notes": [],
            }
        )

    for node in major_nodes:
        if is_section_header(node):
            continue
        if node.name == "table" and has_seen_first_subtitle:
            break

        if node.name == "p":
            segments = iter_text_segments_with_bold(node)
            if not segments:
                continue

            for seg_text, seg_is_bold in segments:
                if seg_is_bold:
                    subtitle = normalize_space(seg_text)
                    if not subtitle:
                        continue
                    if has_seen_first_subtitle:
                        flush_current()
                    has_seen_first_subtitle = True
                    current_title = subtitle
                    current_parts = []
                else:
                    if has_seen_first_subtitle:
                        current_parts.append(seg_text)
            continue

        if not has_seen_first_subtitle:
            continue

        text = tag_text(node)
        if text:
            current_parts.append(text)

    flush_current()
    return sections


def parse_html_file(file_path: str | Path) -> ParsedReport:
    path = Path(file_path)
    raw_html = read_html_text(path)
    soup, parser_name = build_soup_with_fallback(raw_html)

    title = normalize_space(soup.title.get_text()) if soup.title else path.stem
    report_year = infer_report_year(path, raw_html)
    company = infer_company_name(raw_html, soup)  # 애매

    structured: dict[str, Any] = {
        "document_meta": {
            "source_file": path.name,
            "company": company,
            "report_year_guess": report_year,
            "parser_used": parser_name,
        },
        "major_sections": [],
    }

    # 후보 헤더 수집한다.
    # 문서 내 h2, h3 태그를 찾고, 그 중 class가 SECTION-1 또는 SECTION-2인 애들만 가져온다.
    headers = [tag for tag in soup.find_all(["h2", "h3"]) if is_section_header(tag)]
    # 그 중 SECTION-1인 애들만 major header 후보로 삼는다.
    major_headers = [h for h in headers if get_section_level(h) == 1]

    # 만약 major header가 없다면, 전체 문서를 하나의 major section으로 처리
    # 이번 프로젝트에서는 실행확률 없음
    if not major_headers:
        body = soup.body or soup
        major = {
            "title": "UNKNOWN",
            "major_type": "other",
            "content_blocks": [],
            "sections": [],
        }
        for child in body.find_all(recursive=False):
            if child.name == "table":
                text = table_rows_to_text(child)
                if text:
                    _append_block(major["content_blocks"], "table", text)
            else:
                text = tag_text(child)
                if text:
                    _append_block(major["content_blocks"], "paragraph", text)
        structured["major_sections"].append(major)
    else:
        for header in major_headers:
            title_text = get_section_name(header)
            major_nodes = list(iter_section_nodes(header))
            # 문서 공통적으로 등장하는 major section 유형들: 재무제표, 감사보고서, 외부감사 실시내용
            major_type = classify_major_type(title_text)

            major_section = {
                "title": title_text,
                "major_type": major_type,
                "content_blocks": [],
                "sections": [],
            }

            if major_type == "financial_statements_bundle":
                sections, major_blocks = extract_financial_sections(major_nodes)
                major_section["sections"] = sections
                major_section["content_blocks"] = major_blocks
            elif major_type == "audit_report":
                major_section["sections"] = extract_audit_report_sections(major_nodes)
                major_section["content_blocks"] = []
            elif major_type == "external_audit_details":
                sections, major_blocks = extract_external_audit_sections(major_nodes)
                major_section["sections"] = sections
                major_section["content_blocks"] = major_blocks
            else:
                major_section["content_blocks"] = extract_generic_major_content(major_nodes)

            structured["major_sections"].append(major_section)

    return ParsedReport(
        file_path=path,
        file_name=path.name,
        title=title,
        report_year=report_year,
        raw_html=raw_html,
        parser_used=parser_name,
        structured=structured,
    )


def save_parsed_json(report: ParsedReport, output_dir: str | Path) -> Path:
    output_path = Path(output_dir) / f"{report.file_path.stem}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.structured, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path