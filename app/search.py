from __future__ import annotations

from typing import Any
from app.db import get_conn

from difflib import get_close_matches
from functools import lru_cache

from dataclasses import dataclass, field
import re
import sys
from typing import Any

from app.config import settings
from app.db import get_conn
from app.embedder import Embedder
from sentence_transformers import CrossEncoder


YEAR_TOKEN_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?:\s*년(?:도)?|\s*회계연도)?")
WHITESPACE_RE = re.compile(r"\s+")
TRAILING_PARTICLE_RE = re.compile(r"[은는이가을를의에와과도만]+$")

AUDIT_SUBSECTION_HINTS = (
    "감사의견",
    "감사의견근거",
    "핵심감사사항",
    "기타사항",
    "감사인의책임",
    "재무제표감사에대한감사인의책임",
    "경영진과지배기구의책임",
)

SUB_SECTION_HINTS = {
    "감사의견근거": "감사의견근거",
    "핵심감사사항": "핵심감사사항",
    "기타사항": "기타사항",
    "감사인의책임": "감사인의 책임",
    "재무제표감사에대한감사인의책임": "재무제표감사에 대한 감사인의 책임",
    "경영진과지배기구의책임": "재무제표에 대한 경영진과 지배기구의 책임",
    "재무상태표": "재무상태표",
    "손익계산서": "손익계산서",
    "포괄손익계산서": "포괄손익계산서",
    "자본변동표": "자본변동표",
    "현금흐름표": "현금흐름표",
}

FINANCIAL_AMOUNT_QUERY_RE = re.compile(r"얼마|금액|몇|값")
FS_ACCOUNT_HINTS = {
    "매출": "손익계산서",
    "매출액": "손익계산서",
    "순매출": "손익계산서",
    "판매비와관리비": "손익계산서",
    "매출원가": "손익계산서",
    "매출총이익": "손익계산서",
    "영업이익": "손익계산서",
    "기타수익": "손익계산서",
    "기타비용": "손익계산서",
    "금융수익": "손익계산서",
    "금융비용": "손익계산서",
    "법인세비용": "손익계산서",
    "당기순이익": "손익계산서",
    "현금및현금성자산": "재무상태표",
    "매출채권": "재무상태표",
    "재고자산": "재무상태표",
}

FS_SQL_NUMERIC_HINTS = {"얼마", "금액", "값", "잔액", "각각", "비교", "증가", "감소"}
NOTE_HEAVY_HINTS = {
    "변동", "변동내역", "내역", "구성", "중", "기초", "기말",
    "상환", "상환계획", "환입", "대손", "충당부채", "리스부채",
    "사용제한금융상품", "판매보증충당부채"
}

ACCOUNT_QUERY_ALIASES = {
    "순매출": "매출액",
    "매출": "매출액",
}

GENERIC_QUERY_TERMS = {
    "감사보고서",
    "기준",
    "해당",
    "사업연도",
    "얼마",
    "무엇",
    "인가",
    "인가요",
    "입니까",
    "알려줘",
    "보여줘",
    "조회",
    "찾아줘",
}

ROW_LABEL_HINTS = {
    "기초",
    "기말",
    "당기",
    "전기",
    "사용",
    "증가",
    "감소",
    "순전입",
    "순전입환입",
    "환입",
    "잔액",
    "장부금액",
    "장부가액",
}

DOMAIN_KEYWORD_HINTS = {
    "",
}

MIN_SUPPLEMENTAL_ANCHOR_LEN = 4


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker()


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()


def has_row_label_hint(query: str) -> bool:
    compact_query = normalize_search_query(query)
    return any(hint in compact_query for hint in ROW_LABEL_HINTS)


def extract_candidate_keywords(query: str) -> list[str]:
    cleaned = remove_contextual_noise(query)
    _, cleaned = infer_report_year(cleaned)
    compact_query = normalize_search_query(cleaned)

    candidates: set[str] = set()

    for keyword in set(FS_ACCOUNT_HINTS) | set(ACCOUNT_QUERY_ALIASES):
        if keyword in compact_query:
            candidates.add(ACCOUNT_QUERY_ALIASES.get(keyword, keyword))

    token_source = re.sub(r"[^0-9A-Za-z가-힣]+", " ", cleaned)
    tokens = [tok.strip() for tok in token_source.split() if tok.strip()]
    filtered_tokens = [tok for tok in tokens if tok not in GENERIC_QUERY_TERMS and len(tok) >= 2]

    for size in (3, 2, 1):
        for idx in range(0, max(0, len(filtered_tokens) - size + 1)):
            phrase = "".join(filtered_tokens[idx:idx + size])
            normalized = normalize_search_query(phrase)
            if normalized and normalized not in GENERIC_QUERY_TERMS and len(normalized) >= 2:
                candidates.add(ACCOUNT_QUERY_ALIASES.get(normalized, normalized))

    # "X 및 Y" 형태에서 둘 다 candidates에 있으면 조합도 추가
    candidates_merged = set()
    and_split = re.split(r'\s*및\s*', cleaned)
    if len(and_split) >= 2:
        valid_parts = []
        for part in and_split:
            part_compact = normalize_search_query(part)
            if part_compact in candidates:
                valid_parts.append(part_compact)
        
        # 2개 이상의 유효한 부분이 있으면 조합 추가
        if len(valid_parts) >= 2:
            combined = "".join(valid_parts)
            candidates_merged.add(combined)
    
    candidates.update(candidates_merged)

    return sorted(candidates, key=len, reverse=True)


def split_query_keywords(candidates: list[str]) -> tuple[list[str], list[str]]:
    primary: list[str] = []
    support: list[str] = []

    support_terms = ROW_LABEL_HINTS | {"당기말", "전기말", "장기", "단기", "유동성"}
    for keyword in candidates:
        if keyword in support_terms:
            support.append(keyword)
        else:
            primary.append(keyword)

    if not primary:
        primary = support[:2]
        support = support[2:]

    return primary[:6], support[:6]


def is_table_query_intent(query: str, candidates: list[str]) -> bool:
    compact_query = normalize_search_query(query)
    table_markers = {"연이자율", "이자율", "금액", "당기말", "전기말", "상환", "리스부채", "차입금"}
    has_marker = any(marker in compact_query for marker in table_markers)
    return has_marker or has_row_label_hint(query) or (FINANCIAL_AMOUNT_QUERY_RE.search(query) is not None and len(candidates) >= 2)


def resolve_query_anchors(candidates: list[str], report_year: int | None) -> list[tuple[str, int]]:
    if not candidates:
        return []

    values_sql = ", ".join(["(%s)"] * len(candidates))
    year_filter = "WHERE c.report_year = %s" if report_year is not None else ""
    params: list[Any] = list(candidates)
    if report_year is not None:
        params.append(report_year)

    sql = f"""
        WITH candidate(keyword) AS (
            VALUES {values_sql}
        )
        SELECT
            keyword,
            MAX(
                CASE
                    WHEN regexp_replace(COALESCE(c.topic, ''), '\\s+', '', 'g') = keyword THEN 4
                    WHEN POSITION(keyword IN regexp_replace(COALESCE(c.note_title, ''), '\\s+', '', 'g')) > 0 THEN 3
                    WHEN POSITION(keyword IN regexp_replace(COALESCE(c.sub_section, ''), '\\s+', '', 'g')) > 0 THEN 2
                    WHEN POSITION(keyword IN regexp_replace(COALESCE(c.content, ''), '\\s+', '', 'g')) > 0 THEN 1
                    ELSE 0
                END
            ) AS anchor_score
        FROM candidate
        CROSS JOIN chunks c
        {year_filter}
        GROUP BY keyword
        HAVING MAX(
            CASE
                WHEN regexp_replace(COALESCE(c.topic, ''), '\\s+', '', 'g') = keyword THEN 4
                WHEN POSITION(keyword IN regexp_replace(COALESCE(c.note_title, ''), '\\s+', '', 'g')) > 0 THEN 3
                WHEN POSITION(keyword IN regexp_replace(COALESCE(c.sub_section, ''), '\\s+', '', 'g')) > 0 THEN 2
                WHEN POSITION(keyword IN regexp_replace(COALESCE(c.content, ''), '\\s+', '', 'g')) > 0 THEN 1
                ELSE 0
            END
        ) > 0
        ORDER BY anchor_score DESC, LENGTH(keyword) DESC
        LIMIT 6
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    return [(str(row["keyword"]), int(row["anchor_score"])) for row in rows]


def lookup_note_table_value(
    *,
    report_year: int | None = None,
    note_title: str | None = None,
    row_label: str | None = None,
    col_label: str | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    clauses: list[str] = ["c.section_type = 'note_table_cell'"]
    params: list[Any] = []

    if report_year is not None:
        clauses.append("c.report_year = %s")
        params.append(report_year)

    if note_title:
        clauses.append("regexp_replace(COALESCE(c.note_title, ''), '\\s+', '', 'g') LIKE %s")
        params.append(f"%{note_title.replace(' ', '')}%")

    if row_label:
        clauses.append("regexp_replace(COALESCE(c.content, ''), '\\s+', '', 'g') LIKE %s")
        params.append(f"%행:{row_label.replace(' ', '')}%")

    if col_label:
        clauses.append("regexp_replace(COALESCE(c.content, ''), '\\s+', '', 'g') LIKE %s")
        params.append(f"%열:{col_label.replace(' ', '')}%")

    where_sql = " AND ".join(clauses)

    sql = f"""
        SELECT
            d.file_name,
            c.report_year,
            c.major_section,
            c.sub_section,
            c.section_type,
            c.note_no,
            c.note_title,
            c.topic,
            c.chunk_key,
            c.content
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE {where_sql}
        ORDER BY c.report_year DESC, c.note_no ASC
        LIMIT %s
    """
    params.append(limit)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    return list(rows)


def select_anchor_keywords(resolved_anchors: list[tuple[str, int]]) -> list[str]:
    anchor_stopwords = {
        "금액",
        "지급",
        "현재",
        "보고기간종료일",
        "당기말",
        "전기말",
        "연도별",
        "상환연도",
        "계",
    }

    selected: list[str] = []
    for keyword, score in resolved_anchors:
        if score < 2:
            continue
        if keyword in GENERIC_QUERY_TERMS or keyword in ROW_LABEL_HINTS:
            continue
        if keyword in anchor_stopwords:
            continue
        if len(keyword) < 2:
            continue
        selected.append(keyword)

    for keyword, score in resolved_anchors:
        if score != 1:
            continue
        if keyword in GENERIC_QUERY_TERMS or keyword in ROW_LABEL_HINTS:
            continue
        if len(keyword) < MIN_SUPPLEMENTAL_ANCHOR_LEN:
            continue
        if any(keyword in chosen for chosen in selected if len(chosen) >= len(keyword)):
            continue
        if keyword in anchor_stopwords:
            continue
        selected.append(keyword)

    # dedupe while preserving order
    return list(dict.fromkeys(selected))

def infer_financial_row_preference(query: str) -> tuple[str | None, str | None]:
    compact_query = normalize_search_query(query)
    if not FINANCIAL_AMOUNT_QUERY_RE.search(query):
        return None, None

    candidates: list[tuple[int, str, str]] = []
    for account_name, sub_section in FS_ACCOUNT_HINTS.items():
        if account_name in compact_query:
            canonical = ACCOUNT_QUERY_ALIASES.get(account_name, account_name)
            candidates.append((len(account_name), canonical, sub_section))

    if candidates:
        _, canonical, sub_section = max(candidates, key=lambda item: item[0])
        return canonical, sub_section

    return None, None

EXACT_MATCH_BOOST = 0.35
QUERY_NOISE_PATTERNS = [
    re.compile(pattern)
    for pattern in (
        r"[?？!]+",
        r"무엇이야|무엇인가요|무엇입니까|뭐야|뭐지",
        r"어떻게설명돼|어떻게설명되나|어떻게설명돼\??|어떻게설명되어있어",
        r"설명해줘|설명해|보여줘|찾아줘|요약해줘|알려줘",
        r"뭐가적혀있어|무엇이적혀있어|어떤내용이야",
        r"관련내용|관련내역|관련사항",
        r"인가요|입니까|이야|야$",
    )
]

def fetch_fs_accounts() -> list[str]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT account_name_normalized
                FROM financial_statement_rows
                WHERE account_name_normalized IS NOT NULL
            """)
            rows = cur.fetchall()
    return [str(row["account_name_normalized"]) for row in rows if row["account_name_normalized"]]


def match_financial_accounts(query: str) -> list[str]:
    compact = normalize_search_query(query)
    accounts = fetch_fs_accounts()

    exact_matches = [acc for acc in accounts if acc and acc in compact]
    if exact_matches:
        return sorted(set(exact_matches), key=len, reverse=True)

    token_source = re.sub(r"[^0-9A-Za-z가-힣]+", " ", query)
    tokens = [normalize_search_query(tok) for tok in token_source.split() if tok.strip()]
    tokens = [tok for tok in tokens if len(tok) >= 2]

    fuzzy_matches = []
    for token in sorted(tokens, key=len, reverse=True):
        matched = get_close_matches(token, accounts, n=3, cutoff=0.92)
        fuzzy_matches.extend(matched)

    return sorted(set(fuzzy_matches), key=len, reverse=True)


def classify_query_route(query: str, report_year: int | None, accounts: list[str]) -> str:
    compact = normalize_search_query(query)

    has_note_hint = any(hint in compact for hint in NOTE_HEAVY_HINTS)
    has_numeric_hint = any(hint in compact for hint in FS_SQL_NUMERIC_HINTS)

    if has_note_hint:
        return "rag_only"

    if report_year is not None and accounts and has_numeric_hint:
        return "fs_sql_first"

    return "rag_only"


def query_financial_statement_rows(
    *,
    report_year: int,
    account_names_normalized: list[str],
    limit: int = 10,
) -> list[dict[str, Any]]:
    sql = """
        SELECT
            report_year,
            statement_type,
            account_name,
            account_name_normalized,
            parent_account_name_normalized,
            hierarchy_level,
            is_total,
            current_amount,
            prior_amount
        FROM financial_statement_rows
        WHERE report_year = %s
          AND account_name_normalized = ANY(%s)
        ORDER BY
            account_name_normalized,
            is_total DESC,
            hierarchy_level ASC NULLS LAST
        LIMIT %s
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (report_year, account_names_normalized, limit))
            rows = cur.fetchall()
    return list(rows)


def build_fs_sql_rows(fs_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []

    for row in fs_rows:
        content = (
            f"[재무제표 구조화 데이터]\n"
            f"계정명: {row.get('account_name')}\n"
            f"정규화계정명: {row.get('account_name_normalized')}\n"
            f"재무제표: {row.get('statement_type')}\n"
            f"당기금액: {row.get('current_amount')}\n"
            f"전기금액: {row.get('prior_amount')}\n"
            f"상위계정명: {row.get('parent_account_name_normalized')}\n"
            f"계층레벨: {row.get('hierarchy_level')}\n"
            f"총계여부: {row.get('is_total')}"
        )
        converted.append(
            {
                "file_name": "[financial_statement_rows]",
                "report_year": row.get("report_year"),
                "major_section": "(첨부)재무제표",
                "sub_section": row.get("statement_type"),
                "section_type": "financial_statement_sql_row",
                "note_no": None,
                "note_title": None,
                "topic": row.get("account_name_normalized"),
                "chunk_key": f"fs_sql::{row.get('report_year')}::{row.get('statement_type')}::{row.get('account_name_normalized')}",
                "content": content,
                "semantic_score": 1.0,
                "keyword_score": 1.0,
                "exact_match_score": 1.0,
                "hybrid_score": 1.0,
                "structured_match_score": 10.0,
                "keyword_coverage_score": 10.0,
            }
        )

    return converted

# 문맥 정보 제거 패턴 (임베딩의 노이즈 감소)
CONTEXTUAL_NOISE_PATTERNS = [
    re.compile(pattern)
    for pattern in (
        r"감사보고서\s?기준|기준으로|기준에|기준에서",
        r"해당\s?사업연도|사업연도\(제\d+기\)",  # Remove 사업연도(제39기) etc.
        r"제\d+기|제\d+회계연도",
        r"감사팀|감사인|감사의견",
    )
]

def remove_contextual_noise(text: str) -> str:
    """Remove context-providing phrases that don't add semantic value."""
    result = text
    for pattern in CONTEXTUAL_NOISE_PATTERNS:
        result = pattern.sub(" ", result)
    result = re.sub(r"\s+", " ", result).strip()
    return result


@dataclass(slots=True)
class SearchResult:
    original_query: str
    semantic_query: str
    report_year: int | None
    auto_year_applied: bool
    auto_section_type: str | None
    rerank_applied: bool
    candidate_keywords: list[str] = field(default_factory=list)
    resolved_anchors: list[tuple[str, int]] = field(default_factory=list)
    selected_keywords: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)


def infer_report_year(query: str) -> tuple[int | None, str]:
    """Extract a 4-digit year token and return a cleaned query string."""
    match = YEAR_TOKEN_RE.search(query)
    if not match:
        return None, query

    year = int(match.group(1))
    cleaned = YEAR_TOKEN_RE.sub(" ", query)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return year, cleaned or query


def compact_text(text: str) -> str:
    return WHITESPACE_RE.sub("", text or "")


def normalize_search_query(text: str) -> str:
    compact = compact_text(text)
    for pattern in QUERY_NOISE_PATTERNS:
        compact = pattern.sub("", compact)
    compact = TRAILING_PARTICLE_RE.sub("", compact)
    return compact or compact_text(text)


def infer_section_type_hint(query: str) -> str | None:
    compact_query = compact_text(query)
    for hint in AUDIT_SUBSECTION_HINTS:
        if hint in compact_query:
            return "audit_subsection"
    return None


def infer_sub_section_hint(query: str) -> str | None:
    compact_query = compact_text(query)
    for key, value in SUB_SECTION_HINTS.items():
        if key in compact_query:
            return key
    return None


class Reranker:
    def __init__(self) -> None:
        self.enabled = settings.use_reranker
        self.top_n = settings.rerank_top_n
        self.model_name = settings.reranker_model
        self._model: CrossEncoder | None = None

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
        if not self.enabled or not rows:
            return rows, False

        rerank_count = min(len(rows), max(1, self.top_n))
        rerank_rows = rows[:rerank_count]
        pairs = [(query, row["content"]) for row in rerank_rows]
        scores = self._get_model().predict(pairs)

        for row, score in zip(rerank_rows, scores, strict=True):
            row["rerank_score"] = float(score)

        rerank_rows.sort(key=lambda row: (row["rerank_score"], row["hybrid_score"]), reverse=True)
        return rerank_rows + rows[rerank_count:], True


def _run_hybrid_query(
    *,
    vector: list[float],
    semantic_query: str,
    compact_query: str,
    preferred_account_compact: str,
    report_year: int | None,
    sub_section: str | None,
    auto_sub_section_compact: str | None,
    section_types: list[str] | None,
    strong_note_anchor: str | None,
    primary_keywords: list[str],
    support_keywords: list[str],
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    params: list[Any] = []

    if report_year is not None:
        clauses.append("c.report_year = %s")
        params.append(report_year)

    if sub_section is not None:
        clauses.append("c.sub_section = %s")
        params.append(sub_section)

    if section_types:
        placeholders = ", ".join(["%s"] * len(section_types))
        clauses.append(f"c.section_type IN ({placeholders})")
        params.extend(section_types)

    if auto_sub_section_compact is not None:
        clauses.append("regexp_replace(COALESCE(c.sub_section, ''), '\\s+', '', 'g') = %s")
        params.append(auto_sub_section_compact)

    if section_types == ["note_table_cell"] and strong_note_anchor is not None:
        clauses.append(
            "(POSITION(%s IN regexp_replace(COALESCE(c.note_title, ''), '\\s+', '', 'g')) > 0 "
            "OR POSITION(%s IN regexp_replace(COALESCE(c.topic, ''), '\\s+', '', 'g')) > 0)"
        )
        params.append(strong_note_anchor)
        params.append(strong_note_anchor)

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    primary_kw_expr = "COALESCE(c.sub_section, '') || ' ' || COALESCE(c.note_title, '') || ' ' || COALESCE(c.topic, '') || ' ' || COALESCE(c.content, '')"
    support_kw_expr = "COALESCE(c.sub_section, '') || ' ' || COALESCE(c.note_title, '') || ' ' || COALESCE(c.content, '')"

    sql = f"""
        WITH scored AS (
            SELECT
                d.file_name,
                c.report_year,
                c.major_section,
                c.sub_section,
                c.section_type,
                c.note_no,
                c.note_title,
                c.topic,
                c.chunk_key,
                c.content,
                1 - (c.embedding <=> %s::vector) AS semantic_score,
                ts_rank_cd(
                    to_tsvector('simple', COALESCE(c.content, '')),
                    websearch_to_tsquery('simple', %s)
                ) AS keyword_score,
                CASE
                    WHEN POSITION(
                        %s IN regexp_replace(
                            COALESCE(c.sub_section, '') || COALESCE(c.note_title, '') || COALESCE(c.content, ''),
                            '\\s+',
                            '',
                            'g'
                        )
                    ) > 0
                    THEN 1.0
                    WHEN %s <> '' AND POSITION(
                        %s IN regexp_replace(COALESCE(c.content, ''), '\\s+', '', 'g')
                    ) > 0
                    THEN 1.0
                    ELSE 0.0
                END AS exact_match_score,
                (
                    SELECT COALESCE(SUM(
                        CASE
                            WHEN POSITION(
                                kw IN regexp_replace({primary_kw_expr}, '\\s+', '', 'g')
                            ) > 0 THEN 1.0
                            ELSE 0.0
                        END
                    ), 0.0)
                    FROM unnest(%s::text[]) AS kw
                ) AS primary_keyword_hits,
                (
                    SELECT COALESCE(SUM(
                        CASE
                            WHEN POSITION(
                                kw IN regexp_replace({support_kw_expr}, '\\s+', '', 'g')
                            ) > 0 THEN 1.0
                            ELSE 0.0
                        END
                    ), 0.0)
                    FROM unnest(%s::text[]) AS kw
                ) AS support_keyword_hits
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            {where_sql}
        ),
        ranked AS (
            SELECT
                *,
                (primary_keyword_hits + support_keyword_hits * 0.35) AS keyword_coverage_score,
                (primary_keyword_hits * 1.5 + support_keyword_hits * 0.7 + exact_match_score * 0.8) AS structured_match_score,
                (%s * semantic_score + %s * keyword_score + %s * exact_match_score + (primary_keyword_hits + support_keyword_hits * 0.35) * 0.22) AS hybrid_score
            FROM scored
            ORDER BY hybrid_score DESC
            LIMIT %s
        )
        SELECT * FROM ranked
        ORDER BY hybrid_score DESC
    """

    final_params = [
        vector,
        semantic_query,
        compact_query,
        preferred_account_compact,
        preferred_account_compact,
        primary_keywords,
        support_keywords,
        *params,
        settings.semantic_weight,
        settings.keyword_weight,
        EXACT_MATCH_BOOST,
        settings.hybrid_candidate_k,
    ]

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, final_params)
            rows = cur.fetchall()

    return list(rows)


def retrieve(query: str, report_year: int | None = None, sub_section: str | None = None) -> SearchResult:
    auto_year_applied = False
    auto_section_type: str | None = None
    auto_section_types: list[str] | None = None
    auto_sub_section_compact: str | None = None
    semantic_query = query

    # Remove contextual noise before year extraction for better query understanding
    semantic_query = remove_contextual_noise(semantic_query)

    if report_year is None:
        inferred_year, semantic_query = infer_report_year(semantic_query)
        if inferred_year is not None:
            report_year = inferred_year
            auto_year_applied = True

    if not semantic_query:
        semantic_query = remove_contextual_noise(query)

    matched_accounts = match_financial_accounts(query)
    route = classify_query_route(query, report_year, matched_accounts)

    if route == "fs_sql_first" and report_year is not None and matched_accounts:
        fs_rows = query_financial_statement_rows(
            report_year=report_year,
            account_names_normalized=matched_accounts,
            limit=10,
        )
        if fs_rows:
            return SearchResult(
                original_query=query,
                semantic_query=semantic_query,
                report_year=report_year,
                auto_year_applied=auto_year_applied,
                auto_section_type="financial_statement_sql",
                rerank_applied=False,
                candidate_keywords=[matched_accounts],
                resolved_anchors=[],
                selected_keywords=[matched_accounts],
                rows=build_fs_sql_rows(fs_rows),
            )

    candidate_keywords = extract_candidate_keywords(query)
    primary_keywords, support_keywords = split_query_keywords(candidate_keywords)
    resolved_anchors: list[tuple[str, int]] = []
    anchored_keywords: list[str] = []
    low_conf_anchors: list[str] = []
    strong_note_anchor: str | None = None
    row_label_query = has_row_label_hint(query)
    table_query_intent = is_table_query_intent(query, candidate_keywords)

    note_cell_keywords = [
        keyword for keyword in low_conf_anchors if keyword not in ROW_LABEL_HINTS and keyword not in GENERIC_QUERY_TERMS
    ]

    if primary_keywords:
        semantic_query = " ".join([semantic_query, *primary_keywords]).strip()

    compact_query = normalize_search_query(semantic_query)

    preferred_account, preferred_sub_section = infer_financial_row_preference(semantic_query)

    if sub_section is None:
        auto_section_type = infer_section_type_hint(compact_query)
        if auto_section_type is not None:
            auto_section_types = [auto_section_type]
        auto_sub_section_compact = infer_sub_section_hint(compact_query)

    if table_query_intent and sub_section is None:
        auto_section_type = "table_focus"
        auto_section_types = ["note_table_cell", "financial_statement_row"]

    if preferred_account is None and strong_note_anchor is not None:
        auto_section_type = "note"
        auto_section_types = ["note", "note_subsection", "note_table_cell"]

    if preferred_account is None and strong_note_anchor is not None and row_label_query:
        auto_section_type = "note_table_cell"
        auto_section_types = ["note_table_cell"]
        if note_cell_keywords:
            semantic_query = " ".join(note_cell_keywords)
            compact_query = normalize_search_query(semantic_query)

    # 금액형 질의라도 테이블 의도가 강하면 주석 테이블 셀을 함께 조회한다.
    if preferred_account is not None:
        if table_query_intent:
            auto_section_type = "table_focus"
            auto_section_types = ["note_table_cell", "financial_statement_row"]
            # note_table_cell이 재무제표 sub_section 필터에 막히지 않도록 유지
        else:
            auto_section_type = "financial_statement_row"
            auto_section_types = ["financial_statement_row"]
            if auto_sub_section_compact is None and preferred_sub_section is not None:
                auto_sub_section_compact = normalize_search_query(preferred_sub_section)

    embedder = get_embedder()
    vector = embedder.encode_texts([semantic_query])[0]

    preferred_account_compact = preferred_account or ""
    rows = _run_hybrid_query(
        vector=vector,
        semantic_query=semantic_query,
        compact_query=compact_query,
        preferred_account_compact=preferred_account_compact,
        report_year=report_year,
        sub_section=sub_section,
        auto_sub_section_compact=auto_sub_section_compact,
        section_types=auto_section_types,
        strong_note_anchor=strong_note_anchor,
        primary_keywords=primary_keywords,
        support_keywords=support_keywords,
    )

    # 1차에서 필터가 과도하게 좁아진 경우, section_type 제한을 풀어 재조회한다.
    max_structured = max((float(row.get("structured_match_score", 0.0)) for row in rows), default=0.0)
    needs_fallback = (
        auto_section_types is not None
        and (
            len(rows) < min(3, settings.top_k)
            or (table_query_intent and max_structured < 1.5)
        )
    )
    if needs_fallback:
        fallback_rows = _run_hybrid_query(
            vector=vector,
            semantic_query=semantic_query,
            compact_query=compact_query,
            preferred_account_compact=preferred_account_compact,
            report_year=report_year,
            sub_section=sub_section,
            auto_sub_section_compact=auto_sub_section_compact,
            section_types=None,
            strong_note_anchor=strong_note_anchor,
            primary_keywords=primary_keywords,
            support_keywords=support_keywords,
        )

        merged_by_chunk: dict[str, dict[str, Any]] = {}
        for row in [*rows, *fallback_rows]:
            chunk_key = str(row.get("chunk_key", ""))
            if not chunk_key:
                continue
            existing = merged_by_chunk.get(chunk_key)
            if existing is None or float(row.get("hybrid_score", 0.0)) > float(existing.get("hybrid_score", 0.0)):
                merged_by_chunk[chunk_key] = row

        rows = sorted(
            merged_by_chunk.values(),
            key=lambda row: float(row.get("hybrid_score", 0.0)),
            reverse=True,
        )[: settings.hybrid_candidate_k]

    anchored_keyword_set = {keyword for keyword, score in resolved_anchors if score >= 2 and keyword not in ROW_LABEL_HINTS}

    # 같은 점수권이면 계정명 포함 row + 기대 섹션을 위로
    if preferred_account is not None and rows:
        preferred_sub_section_compact = normalize_search_query(preferred_sub_section or "")

        def row_priority(row: dict[str, Any]) -> tuple[int, int]:
            content_compact = compact_text(str(row.get("content", "")))
            sub_compact = compact_text(str(row.get("sub_section", "")))
            is_account_match = 1 if preferred_account in content_compact else 0
            is_sub_match = 1 if preferred_sub_section_compact and preferred_sub_section_compact == sub_compact else 0
            return (is_account_match, is_sub_match)

        rows.sort(
            key=lambda row: (
                row_priority(row)[0],
                row_priority(row)[1],
                row["hybrid_score"],
            ),
            reverse=True,
        )

    if anchored_keyword_set and rows:
        def anchored_priority(row: dict[str, Any]) -> tuple[int, int, int]:
            topic_compact = compact_text(str(row.get("topic", "")))
            note_title_compact = compact_text(str(row.get("note_title", "")))
            sub_compact = compact_text(str(row.get("sub_section", "")))
            content_compact = compact_text(str(row.get("content", "")))
            topic_match = 1 if any(keyword == topic_compact for keyword in anchored_keyword_set) else 0
            note_match = 1 if any(keyword in note_title_compact for keyword in anchored_keyword_set) else 0
            section_match = 1 if any(keyword in sub_compact or keyword in content_compact for keyword in anchored_keyword_set) else 0
            return (topic_match, note_match, section_match)

        rows.sort(
            key=lambda row: (
                anchored_priority(row)[0],
                anchored_priority(row)[1],
                anchored_priority(row)[2],
                row["hybrid_score"],
            ),
            reverse=True,
        )

    if table_query_intent and rows:
        rows.sort(
            key=lambda row: (
                1 if row.get("section_type") in {"note_table_cell", "financial_statement_row"} else 0,
                float(row.get("structured_match_score", 0.0)),
                float(row.get("keyword_coverage_score", 0.0)),
                float(row.get("exact_match_score", 0.0)),
                float(row.get("hybrid_score", 0.0)),
            ),
            reverse=True,
        )

    rerank_applied = False
    if rows:
        reranker = get_reranker()
        try:
            rows, rerank_applied = reranker.rerank(semantic_query, rows)
        except Exception as exc:
            print(f"[WARN] reranker disabled due to runtime error: {exc}")

    rows = rows[: settings.top_k]

    return SearchResult(
        original_query=query,
        semantic_query=semantic_query,
        report_year=report_year,
        auto_year_applied=auto_year_applied,
        auto_section_type=auto_section_type,
        rerank_applied=rerank_applied,
        candidate_keywords=candidate_keywords,
        resolved_anchors=resolved_anchors,
        selected_keywords=anchored_keywords if anchored_keywords else low_conf_anchors,
        rows=rows,
    )
# def retrieve(query: str, report_year: int | None = None, sub_section: str | None = None) -> SearchResult:
#     auto_year_applied = False
#     auto_section_type: str | None = None
#     auto_sub_section_compact: str | None = None
#     semantic_query = query
#     if report_year is None:
#         inferred_year, semantic_query = infer_report_year(query)
#         if inferred_year is not None:
#             report_year = inferred_year
#             auto_year_applied = True

#     if not semantic_query:
#         semantic_query = query

#     compact_query = normalize_search_query(semantic_query)
#     if sub_section is None:
#         auto_section_type = infer_section_type_hint(compact_query)
#         auto_sub_section_compact = infer_sub_section_hint(compact_query)

#     embedder = Embedder()
#     vector = embedder.encode_texts([semantic_query])[0]

#     clauses = []
#     params = []
#     if report_year is not None:
#         clauses.append("c.report_year = %s")
#         params.append(report_year)
#     if sub_section is not None:
#         clauses.append("c.sub_section = %s")
#         params.append(sub_section)
#     if auto_section_type is not None:
#         clauses.append("c.section_type = %s")
#         params.append(auto_section_type)
#     if auto_sub_section_compact is not None:
#         clauses.append("regexp_replace(COALESCE(c.sub_section, ''), '\\s+', '', 'g') = %s")
#         params.append(auto_sub_section_compact)

#     where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

#     sql = f"""
#         WITH scored AS (
#             SELECT
#                 d.file_name,
#                 c.report_year,
#                 c.major_section,
#                 c.sub_section,
#                 c.section_type,
#                 c.note_no,
#                 c.note_title,
#                 c.chunk_key,
#                 c.content,
#                 1 - (c.embedding <=> %s::vector) AS semantic_score,
#                 ts_rank_cd(
#                     to_tsvector('simple', COALESCE(c.content, '')),
#                     websearch_to_tsquery('simple', %s)
#                 ) AS keyword_score,
#                 CASE
#                     WHEN POSITION(
#                         %s IN regexp_replace(
#                             COALESCE(c.sub_section, '') || COALESCE(c.note_title, '') || COALESCE(c.content, ''),
#                             '\\s+',
#                             '',
#                             'g'
#                         )
#                     ) > 0
#                     THEN 1.0
#                     ELSE 0.0
#                 END AS exact_match_score
#             FROM chunks c
#             JOIN documents d ON d.id = c.document_id
#             {where_sql}
#         ),
#         ranked AS (
#             SELECT
#                 *,
#                 (%s * semantic_score + %s * keyword_score + %s * exact_match_score) AS hybrid_score
#             FROM scored
#             ORDER BY hybrid_score DESC
#             LIMIT %s
#         )
#         SELECT * FROM ranked
#         ORDER BY hybrid_score DESC
#     """
#     final_params = [
#         vector,
#         semantic_query,
#         compact_query,
#         *params,
#         settings.semantic_weight,
#         settings.keyword_weight,
#         EXACT_MATCH_BOOST,
#         settings.hybrid_candidate_k,
#     ]

#     with get_conn() as conn:
#         with conn.cursor() as cur:
#             cur.execute(sql, final_params)
#             rows = cur.fetchall()

#     rows = list(rows)
#     rerank_applied = False
#     if rows:
#         reranker = Reranker()
#         try:
#             rows, rerank_applied = reranker.rerank(semantic_query, rows)
#         except Exception as exc:
#             print(f"[WARN] reranker disabled due to runtime error: {exc}")

#     rows = rows[: settings.top_k]

#     return SearchResult(
#         original_query=query,
#         semantic_query=semantic_query,
#         report_year=report_year,
#         auto_year_applied=auto_year_applied,
#         auto_section_type=auto_section_type,
#         rerank_applied=rerank_applied,
#         rows=rows,
#     )

def print_search_result(result: SearchResult) -> None:
    print(f"\n[Query] {result.original_query}")
    if result.auto_year_applied:
        print(f"[AutoFilter] report_year={result.report_year}")
    if result.auto_section_type is not None:
        print(f"[AutoFilter] section_type={result.auto_section_type}")
    if result.candidate_keywords:
        print(f"[QueryNorm] candidates={result.candidate_keywords}")
    if result.resolved_anchors:
        print(f"[QueryNorm] anchors={result.resolved_anchors}")
    if result.selected_keywords:
        print(f"[QueryNorm] selected={result.selected_keywords}")
    print(f"[QueryNorm] semantic_query={result.semantic_query}")
    print(
        f"[Retrieval] hybrid(semantic={settings.semantic_weight:.2f}, "
        f"keyword={settings.keyword_weight:.2f}), candidates={settings.hybrid_candidate_k}"
    )
    if result.rerank_applied:
        print(f"[Reranker] applied model={settings.reranker_model}, top_n={settings.rerank_top_n}")
    else:
        print("[Reranker] not applied")
    print()
    for idx, row in enumerate(result.rows, start=1):
        print(f"[{idx}] {row['file_name']} ({row['report_year']})")
        print(f"- chunk_key : {row['chunk_key']}")
        print(f"- major     : {row['major_section']}")
        print(f"- sub       : {row['sub_section']}")
        print(f"- note      : {row['note_no']} / {row['note_title']}")
        print(f"- semantic  : {row['semantic_score']:.4f}")
        print(f"- keyword   : {row['keyword_score']:.4f}")
        print(f"- exact     : {row['exact_match_score']:.4f}")
        print(f"- hybrid    : {row['hybrid_score']:.4f}")
        if 'rerank_score' in row:
            print(f"- rerank    : {row['rerank_score']:.4f}")
        print(f"- snippet   : {row['content'][:300]}\n")


def search(query: str, report_year: int | None = None, sub_section: str | None = None) -> None:
    result = retrieve(query, report_year=report_year, sub_section=sub_section)
    print_search_result(result)


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        raise SystemExit("사용법: poetry run python -m app.search '현금및현금성자산'")
    search(query)
