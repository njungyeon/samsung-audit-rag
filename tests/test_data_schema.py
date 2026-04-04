"""
데이터 검증 테스트 (Step 2 – 데이터 검증)

검증 항목:
1. test_required_meta_fields  : 파싱된 JSON마다 필수 메타 필드가 모두 있는지 확인
2. test_no_empty_sections     : major_sections 리스트가 비어 있지 않고,
                                각 섹션에 title이 존재하는지 확인
3. test_report_year_range     : report_year_guess가 2014–2024 범위 내 이상값인지 확인
4. test_sections_have_content : 각 섹션의 content_blocks 또는 sections가 완전히
                                비어있지 않은지 확인 (파싱 공백 탐지)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PARSED_DIR = Path(__file__).parent.parent / "data" / "parsed"
REQUIRED_META_FIELDS = {"source_file", "report_year_guess", "parser_used"}

parsed_files = sorted(PARSED_DIR.glob("*.json"))


@pytest.mark.parametrize("json_path", parsed_files, ids=lambda p: p.name)
def test_required_meta_fields(json_path: Path) -> None:
    """document_meta에 필수 필드가 모두 존재하고 값이 비어 있지 않은지 확인."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("document_meta", {})
    missing = REQUIRED_META_FIELDS - meta.keys()
    assert not missing, f"{json_path.name}: 누락된 메타 필드 → {missing}"

    for field in REQUIRED_META_FIELDS:
        assert meta[field] not in (None, "", 0), (
            f"{json_path.name}: '{field}' 값이 비어 있습니다 (값={meta[field]!r})"
        )


@pytest.mark.parametrize("json_path", parsed_files, ids=lambda p: p.name)
def test_no_empty_sections(json_path: Path) -> None:
    """major_sections가 비어 있지 않고, 각 섹션에 title 키가 존재하는지 확인."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    sections = data.get("major_sections", [])
    assert sections, f"{json_path.name}: major_sections가 비어 있습니다"

    for i, sec in enumerate(sections):
        assert "title" in sec, (
            f"{json_path.name}: sections[{i}]에 'title' 키가 없습니다"
        )
        assert sec["title"], (
            f"{json_path.name}: sections[{i}]['title']이 빈 문자열입니다"
        )


VALID_YEAR_RANGE = range(2014, 2025)  # 2014 이상 2024 이하


@pytest.mark.parametrize("json_path", parsed_files, ids=lambda p: p.name)
def test_report_year_range(json_path: Path) -> None:
    """report_year_guess가 2014–2024 범위를 벗어나는 이상값인지 확인."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    year = data.get("document_meta", {}).get("report_year_guess")
    assert year is not None, f"{json_path.name}: report_year_guess가 None입니다"
    assert year in VALID_YEAR_RANGE, (
        f"{json_path.name}: report_year_guess={year}이 유효 범위(2014–2024)를 벗어납니다"
    )


@pytest.mark.parametrize("json_path", parsed_files, ids=lambda p: p.name)
def test_sections_have_content(json_path: Path) -> None:
    """모든 섹션이 content_blocks와 subsections 모두 비어 있는 경우를 탐지한다."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    sections = data.get("major_sections", [])
    empty_sections = [
        sec.get("title", f"index={i}")
        for i, sec in enumerate(sections)
        if not sec.get("content_blocks") and not sec.get("sections")
    ]
    assert not empty_sections, (
        f"{json_path.name}: 내용이 없는 섹션 {len(empty_sections)}개 발견 → {empty_sections[:5]}"
    )
