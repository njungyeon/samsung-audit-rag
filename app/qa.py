from __future__ import annotations

from functools import lru_cache
import re
import sys

from app.generator import RagAnswerGenerator
from app.search import SearchResult, retrieve


SYSTEM_PROMPT = """당신은 삼성전자 2014-2024년 감사보고서를 분석하는 금융 RAG 어시스턴트다.

답변 규칙:
1. 반드시 제공된 검색 근거만 사용한다.
2. 검색 근거에 없는 사실은 추측하지 않는다.
3. 연도별 정보가 다르면 연도별로 구분해서 답한다.
4. 먼저 질문에 대한 직접 답변을 2-4문장으로 작성한다.
5. '근거 1', '근거 2', '[출처 1]', '[출처 2]' 같은 라벨을 절대 사용하지 않는다.
6. 출처 정보나 인용 마크는 생략하고 답변만 작성한다.
7. 확인 가능한 내용만 간결하게 답한다.
"""

TEST_QUESTIONS = [
    "2020년도 감사의견근거는 무엇이야?",
    "2020년도 핵심감사사항은 무엇이야?",
    "2014년도 감사인의 책임은 어떻게 설명돼?",
    "2024년도 감사 의견은 무엇이야?",
    "2021년도 현금및현금성자산은 어떻게 설명돼?",
    "2022년도 재무활동 현금흐름 관련 내용 찾아줘",
    "2019년도 중요한 회계처리방침에서 현금및현금성자산 설명을 보여줘",
    "2023년도 독립된 감사인의 감사보고서에서 기타사항은 뭐야?",
    "2020년도 재무제표감사에 대한 감사인의 책임을 요약해줘",
    "2018년도 감사의견은 적정의견이야?",
]

FINANCIAL_QUERY_HINT_RE = re.compile(
    r"재무상태표|손익계산서|포괄손익계산서|자본변동표|현금흐름표|유동자산|비유동자산|주요항목"
)
SINGLE_VALUE_QUERY_HINT_RE = re.compile(r"얼마|금액|잔액|한도|지급액|전입액|당기말|전기말|기말|기초")
MULTI_VALUE_QUERY_HINT_RE = re.compile(r"비교|차이|각각|연도별|추이|증감|전기말과|당기말과")


def format_context(rows: list[dict[str, object]], max_sources: int = 1) -> str:
    blocks: list[str] = []
    for row in rows[:max_sources]:
        report_year = row.get("report_year")
        major_section = row.get("major_section")
        sub_section = row.get("sub_section")
        content = str(row.get("content") or "").strip()
        blocks.append(
            "\n".join(
                [
                    f"연도: {report_year}",
                    f"섹션: {major_section} > {sub_section}",
                    f"본문: {content}",
                ]
            )
        )
    return "\n\n".join(blocks)


def format_financial_statement_sql_context(result: SearchResult) -> str:
    lines: list[str] = []

    if result.report_year is not None:
        lines.append(f"{result.report_year}년 재무제표 조회 결과")

    for row in result.rows:
        account = str(row.get("topic") or row.get("account_name_normalized") or "항목")
        statement_type = str(row.get("sub_section") or "재무제표")
        content = str(row.get("content") or "")

        current_amount = None
        prior_amount = None

        for line in content.splitlines():
            line = line.strip()
            if line.startswith("당기금액:"):
                current_amount = line.replace("당기금액:", "").strip()
            elif line.startswith("전기금액:"):
                prior_amount = line.replace("전기금액:", "").strip()

        try:
            current_amount_fmt = f"{int(float(current_amount)):,}" if current_amount not in (None, "", "None") else "-"
        except Exception:
            current_amount_fmt = str(current_amount)

        try:
            prior_amount_fmt = f"{int(float(prior_amount)):,}" if prior_amount not in (None, "", "None") else "-"
        except Exception:
            prior_amount_fmt = str(prior_amount)

        lines.append(
            f"- {statement_type} | {account} | 당기 {current_amount_fmt} | 전기 {prior_amount_fmt}"
        )

    return "\n".join(lines)


def build_rag_user_prompt(result: SearchResult) -> str:
    context = format_context(result.rows, max_sources=1)
    task_lines = [
        "- 질문에 대한 최종 답변을 먼저 작성하라 (2-4문장).",
        "- '근거 1', '근거 2', '출처 1', '출처 2' 같은 라벨 나열/복붙을 금지한다.",
        "- 응답 본문에 출처, 파일명, chunk_key 같은 메타데이터를 쓰지 않는다.",
        "- 확인 가능한 정보만 답하고 과도한 단정은 피하라.",
        "- 검색 근거의 금액이 괄호 표기(예: (1,010,632)) 또는 음수(-1,010,632)면 부호를 절대 바꾸지 말고 그대로 유지하라.",
        "- 금액 단위(예: 백만원)가 근거에 있으면 최종 답변에도 동일 단위를 반드시 유지하라.",
    ]
    if SINGLE_VALUE_QUERY_HINT_RE.search(result.original_query) and not MULTI_VALUE_QUERY_HINT_RE.search(result.original_query):
        task_lines.extend(
            [
                "- 질문이 단일 수치만 묻는 경우, 요청된 값 하나만 첫 문장에서 바로 답하라.",
                "- 질문에 없는 인접 항목(예: 기초잔액, 순전입액, 증감 내역)은 덧붙이지 않는다.",
            ]
        )
    return "\n\n".join(
        [
            f"[User Question]\n{result.original_query}",
            f"[Retrieved Context]\n{context}",
            "[Assistant Task]\n" + "\n".join(task_lines),
        ]
    )


def summarize_top_content(raw: str, max_sentences: int = 3) -> str:
    text = raw.strip()
    text = re.sub(r"^\[[^\]]+\]\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    if not parts:
        return text[:360]
    return " ".join(parts[:max_sentences]).strip()


@lru_cache(maxsize=1)
def get_generator() -> RagAnswerGenerator:
    return RagAnswerGenerator()


FINANCIAL_STATEMENT_SQL_PROMPT = """
너는 재무제표 질의응답 도우미다.
반드시 아래 규칙을 지켜라.

1. 제공된 조회 결과만 사용하라.
2. 계정명과 금액을 정확히 매칭하라.
3. 추측하지 마라.
4. 질문이 여러 계정을 묻는 경우 각각 따로 답하라.
5. 당기 금액을 우선 답하라.
6. 전기 금액은 질문이 비교를 요구할 때만 사용하라.
7. 조회 결과에 없는 값은 말하지 마라.
8. 답변은 한국어로 간단명료하게 작성하라.
"""


def build_financial_statement_sql_user_prompt(result: SearchResult) -> str:
    context_lines: list[str] = []

    if result.report_year is not None:
        context_lines.append(f"{result.report_year}년 재무제표 조회 결과")

    for row in result.rows:
        account = str(row.get("topic") or row.get("account_name_normalized") or "항목")
        statement_type = str(row.get("sub_section") or "재무제표")
        content = str(row.get("content") or "")

        current_amount = None
        prior_amount = None

        for line in content.splitlines():
            line = line.strip()
            if line.startswith("당기금액:"):
                current_amount = line.replace("당기금액:", "").strip()
            elif line.startswith("전기금액:"):
                prior_amount = line.replace("전기금액:", "").strip()

        try:
            current_amount_fmt = f"{int(float(current_amount)):,}" if current_amount not in (None, "", "None") else "-"
        except Exception:
            current_amount_fmt = str(current_amount)

        try:
            prior_amount_fmt = f"{int(float(prior_amount)):,}" if prior_amount not in (None, "", "None") else "-"
        except Exception:
            prior_amount_fmt = str(prior_amount)

        context_lines.append(
            f"- {statement_type} | {account} | 당기 {current_amount_fmt} | 전기 {prior_amount_fmt}"
        )

    context_text = "\n".join(context_lines)

    targets_text = ""
    if result.selected_keywords:
        targets_text = "\n".join(f"- {kw}" for kw in result.selected_keywords)

    if targets_text:
        return f"""질문:
            {result.original_query}

            질문에서 답해야 할 계정:
            {targets_text}

            조회 결과:
            {context_text}

            위 조회 결과만 사용해서 답하라.
            계정별 금액을 정확히 매칭해서 답하라.
            """
    return f"""질문:
        {result.original_query}

        조회 결과:
        {context_text}

        위 조회 결과만 사용해서 답하라.
        계정명과 금액을 정확히 매칭해서 답하라.
        """


def generate_answer(result: SearchResult, thinking: bool = False) -> tuple[str, str, str]:
    generator = get_generator()

    if result.auto_section_type == "financial_statement_sql":
        system_prompt = FINANCIAL_STATEMENT_SQL_PROMPT.strip()
        user_prompt = build_financial_statement_sql_user_prompt(result)
    else:
        system_prompt = SYSTEM_PROMPT.strip()
        user_prompt = build_rag_user_prompt(result)

    generation = generator.generate(system_prompt, user_prompt, thinking=thinking)
    cleaned = clean_generated_answer(generation.answer)
    if not cleaned:
        cleaned = build_fallback_answer(result)
    return cleaned, generation.model_name, generation.device


def stream_answer(result: SearchResult, thinking: bool = False) -> str:
    generator = get_generator()

    if result.auto_section_type == "financial_statement_sql":
        system_prompt = FINANCIAL_STATEMENT_SQL_PROMPT.strip()
        user_prompt = build_financial_statement_sql_user_prompt(result)
    else:
        system_prompt = SYSTEM_PROMPT.strip()
        user_prompt = build_rag_user_prompt(result)

    streamer, thread = generator.stream_generate(
        system_prompt,
        user_prompt,
        thinking=thinking,
    )

    chunks: list[str] = []
    shown = ""
    for token in streamer:
        chunks.append(token)
        cleaned = trim_streaming_artifacts(clean_generated_answer("".join(chunks)))
        if len(cleaned) > len(shown):
            delta = cleaned[len(shown):]
            shown = cleaned
            if delta:
                print(delta, end="", flush=True)
    thread.join()

    cleaned = clean_generated_answer("".join(chunks))
    if not cleaned:
        cleaned = build_fallback_answer(result)

    if cleaned != shown:
        suffix = cleaned[len(shown):] if cleaned.startswith(shown) else cleaned
        if suffix:
            print(suffix, end="", flush=True)

    print()
    return cleaned


def generate_grounded_answer(result: SearchResult) -> str:
    """Return a deterministic answer synthesized only from retrieved context."""
    if not result.rows:
        return "검색 결과가 없어 답변할 수 없습니다."

    financial_answer = build_financial_statement_answer(result.semantic_query, result.rows)
    if financial_answer:
        return financial_answer

    summary_row, summary_text = synthesize_grounded_summary(result.semantic_query, result.rows)
    if summary_row is None:
        return build_fallback_answer(result)

    year = summary_row.get("report_year")
    answer_text = f"{year}년도 보고서 기준 요약: {summary_text}"

    return answer_text


def extract_table_lines(content: str) -> list[str]:
    lines = [re.sub(r"\s+", " ", line).strip() for line in content.splitlines()]
    return [line for line in lines if line and "|" in line]


def build_financial_statement_answer(query: str, rows: list[dict[str, object]]) -> str | None:
    compact_query = re.sub(r"\s+", "", query)
    has_fs_sql_rows = any(
        str(row.get("section_type", "")) == "financial_statement_sql_row"
        for row in rows
    )
    if not FINANCIAL_QUERY_HINT_RE.search(compact_query) and not has_fs_sql_rows:
        return None

    if not rows:
        return None

    section_types = {str(row.get("section_type", "")) for row in rows}

    # 새 SQL 경로 전용 처리
    if "financial_statement_sql_row" in section_types:
        year = rows[0].get("report_year")
        parts = []

        for row in rows:
            account = str(row.get("topic") or row.get("account_name_normalized") or "항목")
            content = str(row.get("content") or "")

            current_amount = None
            prior_amount = None

            for line in content.splitlines():
                line = line.strip()
                if line.startswith("당기금액:"):
                    current_amount = line.replace("당기금액:", "").strip()
                elif line.startswith("전기금액:"):
                    prior_amount = line.replace("전기금액:", "").strip()

            if current_amount not in (None, "", "None"):
                try:
                    current_amount_fmt = f"{int(float(current_amount)):,}"
                except Exception:
                    current_amount_fmt = str(current_amount)
                parts.append(f"{account}은 {current_amount_fmt}원")
            elif prior_amount not in (None, "", "None"):
                try:
                    prior_amount_fmt = f"{int(float(prior_amount)):,}"
                except Exception:
                    prior_amount_fmt = str(prior_amount)
                parts.append(f"{account}은 {prior_amount_fmt}원")

        if parts:
            if year is not None:
                return f"{year}년 " + ", ".join(parts) + "입니다."
            return ", ".join(parts) + "입니다."

        return None

    # 기존 chunks 경로 유지
    fs_row = next(
        (row for row in rows if str(row.get("section_type", "")) in {"financial_statement", "financial_statement_row"}),
        None,
    )
    if fs_row is None:
        return None

    year = fs_row.get("report_year")
    statement = str(fs_row.get("sub_section") or "재무제표")
    content = str(fs_row.get("content") or "")
    table_lines = extract_table_lines(content)
    if not table_lines:
        return None

    targets = [
        key for key in ("유동자산", "비유동자산", "현금및현금성자산", "매출채권", "재고자산", "매출액", "영업이익", "당기순이익")
        if key in compact_query
    ]
    matched = [line for line in table_lines if any(target in re.sub(r"\s+", "", line) for target in targets)]
    if not matched:
        matched = table_lines[:5]

    body = "; ".join(matched[:5])
    return f"{year}년도 {statement} 기준 주요 항목은 다음과 같습니다: {body}"


def clean_generated_answer(answer: str) -> str:
    text = answer.strip()
    text = re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    text = re.sub(r"근거\s*\d+[:\s]*", "", text)
    text = re.sub(r"\[?출처\s*\d+\]?[:\s]*", "", text)
    text = re.sub(r"\[Retrieved Context\].*?(?=\n\n|$)", "", text, flags=re.DOTALL)
    text = re.sub(r"(?:^|\n)\s*(?:file_name|report_year|major_section|sub_section|chunk_key)\s*:\s*.*", "", text)
    text = re.sub(r"(?:^|\n)\s*(?:연도|섹션|본문)\s*:\s*", "\n", text)
    text = re.sub(r"^\s*-\s+(?![\d(])", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def trim_streaming_artifacts(text: str) -> str:
    text = re.sub(r"\s*(?:\[\s*)?$", "", text)
    text = re.sub(r"\s*\[?출처\s*\d*\s*$", "", text)
    text = re.sub(r"\s*근거\s*\d*\s*$", "", text)
    text = re.sub(r"\s*(?:file_name|report_year|major_section|sub_section|chunk_key)\s*:\s*.*$", "", text)
    text = re.sub(r"\s*(?:연도|섹션|본문)\s*:\s*.*$", "", text)
    return text.rstrip()


def build_fallback_answer(result: SearchResult) -> str:
    print("!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!")
    if not result.rows:
        return "검색 결과가 없어 답변을 생성할 수 없습니다."

    # 재무제표 SQL 경로는 LLM 요약 대신 규칙 기반으로 답변 생성
    if result.auto_section_type == "financial_statement_sql":
        year = result.report_year
        parts = []

        

        for row in result.rows:
            account = row.get("topic") or row.get("account_name_normalized") or "항목"
            content = str(row.get("content", ""))

            current_amount = None
            prior_amount = None

            for line in content.splitlines():
                line = line.strip()
                if line.startswith("당기금액:"):
                    value = line.replace("당기금액:", "").strip()
                    current_amount = value
                elif line.startswith("전기금액:"):
                    value = line.replace("전기금액:", "").strip()
                    prior_amount = value

            if current_amount is not None:
                try:
                    current_amount_fmt = f"{int(float(current_amount)):,}"
                except Exception:
                    current_amount_fmt = str(current_amount)
                parts.append(f"{account}은 {current_amount_fmt}원")
            elif prior_amount is not None:
                try:
                    prior_amount_fmt = f"{int(float(prior_amount)):,}"
                except Exception:
                    prior_amount_fmt = str(prior_amount)
                parts.append(f"{account}은 {prior_amount_fmt}원")

        if parts:
            if year is not None:
                return f"{year}년 " + ", ".join(parts) + "입니다."
            return ", ".join(parts) + "입니다."

        return "재무제표 조회 결과는 찾았지만 금액을 정리하지 못했습니다."

    # 기존 주석 / 의미검색 경로는 그대로 유지
    top = result.rows[0]
    raw = str(top.get("content", ""))
    _, summary = synthesize_grounded_summary(result.semantic_query, [top])
    if not summary:
        summary = summarize_top_content(raw, max_sentences=2)

    return summary


def synthesize_grounded_summary(
    query: str,
    rows: list[dict[str, object]],
) -> tuple[dict[str, object] | None, str]:
    query_tokens = [tok for tok in re.split(r"\s+", query.strip()) if tok]
    compact_query = re.sub(r"\s+", "", query)

    best_row: dict[str, object] | None = None
    best_sentences: list[tuple[int, str]] = []
    best_score = -1

    for row in rows[:3]:
        raw = str(row.get("content", ""))
        clean = re.sub(r"^\[[^\]]+\]\s*", "", raw)
        clean = re.sub(r"\s+", " ", clean).strip()
        if not clean:
            continue

        sentence_candidates = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", clean) if len(s.strip()) >= 18
        ]
        if not sentence_candidates:
            continue

        scored: list[tuple[int, str]] = []
        total = 0
        for sentence in sentence_candidates:
            compact_sentence = re.sub(r"\s+", "", sentence)
            score = 0
            if compact_query and compact_query in compact_sentence:
                score += 3
            for tok in query_tokens:
                if len(tok) >= 2 and tok in sentence:
                    score += 1
            if sentence.endswith("."):
                score += 1
            scored.append((score, sentence))
            total += score

        if total > best_score:
            best_score = total
            best_row = row
            best_sentences = sorted(scored, key=lambda x: x[0], reverse=True)[:3]

    if best_row is None:
        return None, ""

    summary = " ".join(sentence for _, sentence in best_sentences).strip()
    summary = re.sub(r"\s+", " ", summary)
    return best_row, summary


def print_result(result: SearchResult) -> None:
    print(f"\n=== Question ===\n{result.original_query}")
    if result.candidate_keywords:
        print(f"\n=== Query Normalization ===")
        print(f"candidates: {result.candidate_keywords}")
        print(f"anchors: {result.resolved_anchors}")
        print(f"selected: {result.selected_keywords}")
        print(f"semantic_query: {result.semantic_query}")
    print(f"\n=== System Prompt ===\n{SYSTEM_PROMPT.strip()}")
    print(f"\n=== User Prompt ===\n{build_rag_user_prompt(result)}")
    print("\n=== Retrieved Top-K ===")
    for idx, row in enumerate(result.rows, start=1):
        print(f"[{idx}] {row['report_year']} | {row['major_section']} | {row['sub_section']}")
        print(
            f"    hybrid={row['hybrid_score']:.4f} semantic={row['semantic_score']:.4f} "
            f"keyword={row['keyword_score']:.4f}"
        )
        print(f"    {row['content'][:220]}\n")


def run_single(
    question: str,
    use_llm: bool = False,
    thinking: bool = False,
    stream: bool = False,
    debug: bool = False,
) -> None:
    result = retrieve(question)
    if debug:
        print_result(result)

    if use_llm:
        if stream:
            print("=== Streaming Answer ===")
            answer = stream_answer(result, thinking=thinking)
        else:
            print("=== Generated Answer ===")
            answer, _, _ = generate_answer(result, thinking=thinking)
    else:
        if stream:
            print("[안내] --stream은 --llm 모드에서만 적용됩니다.")
        print("=== Generating Grounded Answer ===")
        answer = generate_grounded_answer(result)

    if not (use_llm and stream):
        print(answer)


def run_test_suite(generate_answers: bool = False, use_llm: bool = False, thinking: bool = False) -> None:
    print("=== RAG QA Test Set (10 questions) ===")
    for idx, question in enumerate(TEST_QUESTIONS, start=1):
        result = retrieve(question)
        top = result.rows[0] if result.rows else None
        print(f"\n[{idx}] {question}")
        if top is None:
            print("- no result")
            continue
        print(f"- top_year   : {top['report_year']}")
        print(f"- top_major  : {top['major_section']}")
        print(f"- top_sub    : {top['sub_section']}")
        print(f"- top_score  : {top['hybrid_score']:.4f}")
        print(f"- top_snippet: {str(top['content'])[:180]}")
        if generate_answers:
            if use_llm:
                answer, _, _ = generate_answer(result, thinking=thinking)
            else:
                answer = generate_grounded_answer(result)
            print(f"- llm_answer : {answer[:260]}")


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        raise SystemExit(
            "사용법: poetry run python -m app.qa '질문'\n"
            "   또는: poetry run python -m app.qa --test\n"
            "   또는: poetry run python -m app.qa --test --generate\n"
            "옵션: --llm (기본은 데이터 기반 추출 답변), --thinking (LLM 모드에서만 유효), "
            "--stream (LLM 답변을 토큰 단위로 출력), --debug (검색 근거/프롬프트 디버그 출력)"
        )

    use_llm = "--llm" in args
    use_thinking = "--thinking" in args
    use_stream = "--stream" in args
    use_debug = "--debug" in args
    args = [a for a in args if a not in {"--llm", "--thinking", "--stream", "--debug"}]

    if args == ["--test"]:
        run_test_suite(generate_answers=False, use_llm=use_llm, thinking=use_thinking)
        return

    if args == ["--test", "--generate"]:
        run_test_suite(generate_answers=True, use_llm=use_llm, thinking=use_thinking)
        return

    question = " ".join(args).strip()
    if not question:
        raise SystemExit("질문이 비어 있습니다.")
    run_single(question, use_llm=use_llm, thinking=use_thinking, stream=use_stream, debug=use_debug)


if __name__ == "__main__":
    main()