from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

import streamlit as st

from app.generator import RagAnswerGenerator


AGENT_SYSTEM_PROMPT = """당신은 삼성전자 감사보고서 분석 에이전트다.

규칙:
1. 필요한 경우 tool을 호출해 근거를 먼저 확인한다.
2. 수집된 tool 결과를 근거로만 최종 답변한다.
3. tool이 이미 충분한 근거를 주면 더 이상 tool을 반복 호출하지 않는다.
4. 질문 단어를 반복한 동어반복 문장을 쓰지 않는다.
5. '기타사항에 기타사항이 포함되어' 같은 어색한 표현을 금지한다.
6. 질문이 단일 수치만 묻는 경우에는 요청된 값 하나만 먼저 답하고, 주변 표 항목은 덧붙이지 않는다.
"""

SINGLE_VALUE_QUERY_HINT_RE = re.compile(r"얼마|금액|잔액|한도|지급액|전입액|당기말|전기말|기말|기초")
MULTI_VALUE_QUERY_HINT_RE = re.compile(r"비교|차이|각각|연도별|추이|증감|전기말과|당기말과")

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
TOOL_PREFIX_RE = re.compile(
    r"^\s*(?:\[Agent Tool Trace\]\s*)?"
    r"(?:(?:\d+\.\s*)?(?:search_audit_report|get_specific_section|compare_years)\s*\(\{.*?\}\)\s*)+",
    re.DOTALL,
)


def clean_text(text: str) -> str:
    return TAG_RE.sub("", text).strip()


def clean_agent_final_text(text: str) -> str:
    cleaned = clean_text(text)
    cleaned = TOOL_PREFIX_RE.sub("", cleaned).lstrip()
    cleaned = re.sub(r"(?im)^\s*출처\s*:\s*.*$", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def build_final_answer_instruction(user_query: str) -> str:
    instruction = (
        "위까지의 tool 결과를 근거로 사용자 질문에 최종 답변을 작성하라. "
        "답변은 간결하게 2-4문장으로 작성하라. "
        "동일 단어 반복(예: '기타사항에 기타사항이')을 금지하고, "
        "핵심 사실부터 자연스러운 평서문으로 바로 시작하라."
    )
    if SINGLE_VALUE_QUERY_HINT_RE.search(user_query) and not MULTI_VALUE_QUERY_HINT_RE.search(user_query):
        instruction += " 요청된 값이 하나면 그 값만 한 문장으로 먼저 답하고, 기초잔액·증감·순전입액 같은 주변 항목은 덧붙이지 마라."
    return instruction


def format_written_content_template(result: Any) -> str:
    if not result.rows:
        return "검색 결과가 없어 답변할 수 없습니다."

    top = result.rows[0]
    year = top.get("report_year")
    sub_section = str(top.get("sub_section") or "해당 항목")
    raw = str(top.get("content") or "").strip()
    raw = re.sub(r"^\[[^\]]+\]\s*", "", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return f"{year}년도 감사보고서 {sub_section}에는 다음과 같이 기재되어 있습니다: {raw}"


@lru_cache(maxsize=1)
def get_qa_helpers() -> tuple[Any, ...]:
    from app.qa import (
        SYSTEM_PROMPT,
        FINANCIAL_STATEMENT_SQL_PROMPT,
        build_rag_user_prompt,
        build_financial_statement_sql_user_prompt,
        clean_generated_answer,
        format_context,
        generate_grounded_answer,
        trim_streaming_artifacts,
    )

    return (
        SYSTEM_PROMPT,
        FINANCIAL_STATEMENT_SQL_PROMPT,
        build_rag_user_prompt,
        build_financial_statement_sql_user_prompt,
        clean_generated_answer,
        format_context,
        generate_grounded_answer,
        trim_streaming_artifacts,
    )


@lru_cache(maxsize=1)
def get_retrieve_func() -> Any:
    from app.search import retrieve

    return retrieve


@lru_cache(maxsize=1)
def get_generator() -> RagAnswerGenerator:
    return RagAnswerGenerator()


def tool_search_audit_report(query: str, k: int = 3) -> str:
    retrieve = get_retrieve_func()
    _, _, _, _, _, format_context, _, _ = get_qa_helpers()
    result = retrieve(query)
    rows = result.rows[: max(1, min(k, 5))]
    if not rows:
        return "검색 결과가 없습니다."
    return format_context(rows, max_sources=len(rows))


def tool_get_specific_section(report_year: int, sub_section: str, k: int = 3) -> str:
    retrieve = get_retrieve_func()
    _, _, _, _, _, format_context, _, _ = get_qa_helpers()
    query = f"{report_year}년도 {sub_section}"
    result = retrieve(query, report_year=report_year, sub_section=sub_section)
    rows = result.rows[: max(1, min(k, 5))]
    if not rows:
        return "조건에 맞는 섹션을 찾지 못했습니다."
    return format_context(rows, max_sources=len(rows))


def tool_compare_years(topic: str, year_a: int, year_b: int) -> str:
    retrieve = get_retrieve_func()
    result_a = retrieve(f"{year_a}년도 {topic}", report_year=year_a)
    result_b = retrieve(f"{year_b}년도 {topic}", report_year=year_b)

    rows_a = result_a.rows[:1]
    rows_b = result_b.rows[:1]
    lines: list[str] = []

    if rows_a:
        ra = rows_a[0]
        lines.append(
            f"{year_a}: {ra['major_section']} | {ra['sub_section']} | {str(ra['content'])[:220]}"
        )
    else:
        lines.append(f"{year_a}: 결과 없음")

    if rows_b:
        rb = rows_b[0]
        lines.append(
            f"{year_b}: {rb['major_section']} | {rb['sub_section']} | {str(rb['content'])[:220]}"
        )
    else:
        lines.append(f"{year_b}: 결과 없음")

    return "\n".join(lines)


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_audit_report",
            "description": "감사보고서에서 질문 관련 근거를 검색한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "검색할 질문"},
                    "k": {"type": "integer", "description": "반환 개수, 기본 3"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_specific_section",
            "description": "특정 연도/소제목을 직접 조회한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "report_year": {"type": "integer", "description": "연도"},
                    "sub_section": {"type": "string", "description": "소제목"},
                    "k": {"type": "integer", "description": "반환 개수, 기본 3"},
                },
                "required": ["report_year", "sub_section"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_years",
            "description": "동일 주제를 두 연도에서 비교한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "비교 주제"},
                    "year_a": {"type": "integer", "description": "첫 번째 연도"},
                    "year_b": {"type": "integer", "description": "두 번째 연도"},
                },
                "required": ["topic", "year_a", "year_b"],
            },
        },
    },
]

TOOL_FUNCS = {
    "search_audit_report": tool_search_audit_report,
    "get_specific_section": tool_get_specific_section,
    "compare_years": tool_compare_years,
}


def run_agentic_react(user_query: str, max_turns: int = 4) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    generator = get_generator()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT.strip()},
        {"role": "user", "content": user_query},
    ]

    tool_traces: list[dict[str, str]] = []

    for _ in range(max_turns):
        raw = generator.generate_from_messages(
            messages,
            thinking=False,
            tools=TOOLS,
            skip_special_tokens=False,
        )

        tool_matches = TOOL_CALL_RE.findall(raw)
        if not tool_matches:
            break

        messages.append({"role": "assistant", "content": raw})

        for call_text in tool_matches:
            try:
                payload = json.loads(call_text.strip())
            except json.JSONDecodeError:
                payload = {"name": "invalid_json", "arguments": {}}

            fn_name = payload.get("name", "")
            fn_args = payload.get("arguments", {}) or {}

            if fn_name not in TOOL_FUNCS:
                tool_result = f"알 수 없는 도구: {fn_name}"
            else:
                try:
                    tool_result = str(TOOL_FUNCS[fn_name](**fn_args))
                except Exception as exc:
                    tool_result = f"도구 실행 오류: {exc}"

            tool_traces.append(
                {
                    "tool": fn_name,
                    "args": json.dumps(fn_args, ensure_ascii=False),
                    "result": tool_result,
                }
            )
            messages.append({"role": "tool", "name": fn_name, "content": tool_result})

    messages.append(
        {
            "role": "user",
            "content": build_final_answer_instruction(user_query),
        }
    )

    return messages, tool_traces


def stream_rag_answer(user_query: str, thinking: bool = False):
    retrieve = get_retrieve_func()
    (
        system_prompt,
        financial_statement_sql_prompt,
        build_rag_user_prompt,
        build_financial_statement_sql_user_prompt,
        clean_generated_answer,
        _,
        generate_grounded_answer,
        trim_streaming_artifacts,
    ) = get_qa_helpers()

    generator = get_generator()
    result = retrieve(user_query)

    if not result.rows:
        grounded = generate_grounded_answer(result)
        if grounded:
            yield grounded
        return

    # 재무제표 SQL 질의는 스트리밍에서도 전용 프롬프트 사용
    if result.auto_section_type == "financial_statement_sql":
        prompt_system = financial_statement_sql_prompt.strip()
        prompt_user = build_financial_statement_sql_user_prompt(result)
    else:
        prompt_system = system_prompt.strip()
        prompt_user = build_rag_user_prompt(result)

    streamer, thread = generator.stream_generate(
        prompt_system,
        prompt_user,
        thinking=thinking,
    )

    raw = ""
    shown = ""
    for token in streamer:
        raw += token
        cleaned = trim_streaming_artifacts(clean_generated_answer(raw))
        if not cleaned:
            continue
        if len(cleaned) > len(shown):
            delta = cleaned[len(shown):]
            shown = cleaned
            if delta:
                yield delta
    thread.join()

    final_cleaned = clean_generated_answer(raw)
    if not final_cleaned:
        grounded = generate_grounded_answer(result)
        if grounded:
            yield grounded
            return

    if final_cleaned != shown:
        suffix = final_cleaned[len(shown):] if final_cleaned.startswith(shown) else final_cleaned
        if suffix:
            yield suffix


def stream_agent_answer(user_query: str):
    _, _, _, _, clean_generated_answer, _, generate_grounded_answer, trim_streaming_artifacts = get_qa_helpers()
    retrieve = get_retrieve_func()

    compact_query = re.sub(r"\s+", "", user_query)
    if re.search(r"재무상태표|손익계산서|포괄손익계산서|자본변동표|현금흐름표|유동자산|비유동자산|주요항목", compact_query):
        grounded = generate_grounded_answer(retrieve(user_query))
        if grounded:
            yield grounded
            return

    if re.search(r"뭐가적혀|무엇이적혀|무슨내용|어떤내용|적혀있", compact_query):
        templated = format_written_content_template(retrieve(user_query))
        if templated:
            yield templated
            return

    generator = get_generator()
    messages, traces = run_agentic_react(user_query)

    streamer, thread = generator.stream_from_messages(
        messages,
        thinking=False,
        tools=None,
        skip_special_tokens=True,
    )

    raw = ""
    shown = ""
    for token in streamer:
        raw += token
        cleaned = trim_streaming_artifacts(clean_generated_answer(clean_agent_final_text(raw)))
        if not cleaned:
            continue
        if len(cleaned) > len(shown):
            delta = cleaned[len(shown):]
            shown = cleaned
            if delta:
                yield delta
    thread.join()


st.set_page_config(page_title="Samsung Audit Agent Chat", page_icon="📊", layout="wide")
st.title("Samsung Audit RAG Agent Chat")
st.caption("RAG + Streaming + Tool Calling + ReAct")
st.info("앱이 켜졌습니다. 질문을 입력하면 모델 로딩 후 스트리밍 응답을 시작합니다.")

with st.sidebar:
    mode = st.radio(
        "응답 모드",
        options=["RAG 스트리밍", "Agent(ReAct) 스트리밍"],
        index=0,
    )
    thinking = st.toggle("Thinking 모드(Qwen)", value=False, disabled=(mode != "RAG 스트리밍"))
    st.markdown("---")
    st.markdown("- RAG: 검색 컨텍스트 기반 단일 생성")
    st.markdown("- Agent: Tool Calling + ReAct 후 최종 답변 스트리밍")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("질문을 입력하세요. 예: 2023년도 기타사항에는 뭐가 적혀 있어?")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.chat_message("assistant"):
            spinner_msg = (
                "근거를 검색하고 답변을 생성하는 중입니다..."
                if mode == "RAG 스트리밍"
                else "에이전트가 도구를 사용해 근거를 수집하고 답변을 생성하는 중입니다..."
            )
            with st.spinner(spinner_msg):
                if mode == "RAG 스트리밍":
                    final_text = st.write_stream(stream_rag_answer(prompt, thinking=thinking))
                else:
                    final_text = st.write_stream(stream_agent_answer(prompt))
        st.session_state.messages.append({"role": "assistant", "content": str(final_text)})
    except Exception as exc:
        st.error(f"실행 중 오류가 발생했습니다: {exc}")
