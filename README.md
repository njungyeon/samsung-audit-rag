# samsung-audit-rag-structured

삼성전자 감사보고서 `.htm/.html` 파일을 대상으로, **통합형 구조(PostgreSQL + pgvector)** 로

1. EUC-KR/CP949 HTML 디코딩
2. 방어적 HTML 파싱 (`lxml -> html5lib -> html.parser` fallback)
3. 구조화 JSON 생성
4. 섹션별 청킹
5. 임베딩 생성
6. PostgreSQL + pgvector 적재

까지 수행하는 최소 프로젝트입니다.

## 왜 통합형인가?

이번 데이터는 **삼성전자 감사보고서 10년치 수준**이라서, 별도 벡터 DB를 두는 분리형보다
**PostgreSQL 하나에 원문 메타데이터 + 임베딩을 같이 저장**하는 방식이 더 단순하고 관리하기 쉽습니다.

- `documents`: 원본 파일과 구조화 JSON 저장
- `chunks`: 검색 단위 text + 메타데이터 + embedding 저장

## 파싱 구조

이 프로젝트는 감사보고서 구조를 아래처럼 봅니다.

- `h2.SECTION-1`
  - 독립된 감사인의 감사보고서
  - (첨부)재무제표
- `(첨부)재무제표` 아래 table 제목 패턴
  - 재무상태표
  - 손익계산서
  - 포괄손익계산서
  - 자본변동표
  - 현금흐름표
- `h3.SECTION-2`
  - 주석
- 주석 아래 번호형 소제목
  - 1. 일반적 사항
  - 2. 중요한 회계처리방침
  - 3. 중요한 회계추정 및 가정
  - ...

## 폴더 구조

```bash
samsung-audit-rag-structured/
├─ app/
│  ├─ config.py
│  ├─ db.py
│  ├─ parser.py
│  ├─ chunker.py
│  ├─ embedder.py
│  ├─ ingest.py
│  └─ search.py
├─ data/
│  ├─ html/
│  └─ parsed/
├─ sql/
│  └─ init.sql
├─ docker-compose.yml
├─ pyproject.toml
└─ .env.example
```

## 0. macOS: pyenv / Poetry 설치

macOS 기준 설치 예시입니다.

### Homebrew 설치 확인

```bash
brew --version
```

Homebrew가 없다면 먼저 설치합니다.

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### pyenv 설치

```bash
brew install pyenv
```

쉘 설정 파일(`~/.zshrc`)에 pyenv 초기화를 추가합니다.

```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
source ~/.zshrc
```

### Poetry 설치

```bash
brew install poetry
```

설치 확인:

```bash
pyenv --version
poetry --version
```

## 1. Python/Poetry 환경 구성

요청하신 순서대로 실행하면 됩니다.

```bash
pyenv install 3.12
pyenv local 3.12.13
poetry env use $(pyenv which python)
eval $(poetry env activate)
poetry install
cp .env.example .env
```

참고:
- `pyenv install 3.12`는 최신 3.12.x를 설치합니다.
- `pyenv local 3.12.13`을 쓰려면 해당 버전이 설치되어 있어야 합니다.

예: 버전을 고정하고 싶다면 아래처럼 설치해도 됩니다.

```bash
pyenv install 3.12.13
pyenv local 3.12.13
```

## 2. PostgreSQL 실행

```bash
docker compose up -d
```

## 3. 감사보고서 파일 넣기

`data/html/` 폴더에 10년치 `.htm` 파일을 넣습니다.

예:

```bash
data/html/감사보고서_2014.htm
data/html/감사보고서_2015.htm
...
```

## 4. 적재 실행

```bash
poetry run python -m app.ingest
```

실행하면:

- `documents.parsed_json` 에 구조화 JSON 저장
- `data/parsed/*.json` 에 파싱 결과 파일 저장
- `chunks` 테이블에 청크 + 임베딩 저장

## 5. 검색 테스트

```bash
poetry run python -m app.search "현금및현금성자산"
```

## 6. RAG 프롬프트 / QA 테스트

검색 결과를 바탕으로 RAG 프롬프트를 확인하거나, 테스트 질문 10개를 한 번에 점검할 수 있습니다.

단일 질문 테스트:

```bash
poetry run python -m app.qa "2020년도 감사의견근거는 무엇이야?"
```

기본은 검색 근거에서 바로 추출한 데이터 기반 답변(`extractive`)입니다.
LLM 생성 답변을 사용하려면 `--llm` 옵션을 추가합니다.

```bash
poetry run python -m app.qa --llm "2020년도 감사의견근거는 무엇이야?"
```

LLM 답변을 토큰 단위로 실시간 출력(스트리밍)하려면 `--stream` 옵션을 함께 사용합니다.

```bash
poetry run python -m app.qa --llm --stream "2023년도 기타사항에는 뭐가 적혀 있어?"
```

검색된 Top-K 근거, 정규화된 질의, 프롬프트까지 함께 확인하려면 `--debug` 옵션을 사용합니다.

```bash
poetry run python -m app.qa --llm --debug "2020년도 감사의견근거는 무엇이야?"
poetry run python -m app.qa --llm --stream --debug "2023년도 기타사항에는 뭐가 적혀 있어?"
```

테스트 질문 10개 일괄 실행:

```bash
poetry run python -m app.qa --test
```

LLM 답변까지 포함해 단일 질문 실행:

```bash
poetry run python -m app.qa "2020년도 감사의견근거는 무엇이야?"
```

테스트 질문 10개에 대해 검색 + LLM 답변까지 함께 실행:

```bash
poetry run python -m app.qa --test --generate --llm
```

기본적으로 `app.qa`는 최종 답변만 출력합니다.

`--debug`를 함께 쓰면 다음을 추가로 출력합니다.

- 시스템 프롬프트
- 질문
- 검색된 Top-K 근거 컨텍스트
- 각 질문의 상위 검색 결과 요약
- 선택적으로 생성된 LLM 답변

## 7. Streamlit 챗봇 (RAG 스트리밍 + Agentic ReAct)

실습 5회 구조처럼 Streamlit 화면에서 질문을 던지고, 답변을 스트리밍으로 받을 수 있습니다.

```bash
poetry install
poetry run streamlit run app/streamlit_app.py
```

모드 설명:

- `RAG 스트리밍`: 검색 컨텍스트 기반으로 답변을 토큰 단위 스트리밍
- `Agent(ReAct) 스트리밍`: Tool Calling + ReAct 루프를 먼저 수행하고 최종 답변을 스트리밍

기본 Tool:

- `search_audit_report(query, k)`
- `get_specific_section(report_year, sub_section, k)`
- `compare_years(topic, year_a, year_b)`

## 청킹 전략

### 재무제표
- 표 전체를 통째로 넣지 않음
- table row를 일정 개수씩 묶어서 chunk 생성
- 예: 재무상태표 유동자산/비유동자산, 현금흐름표 영업/투자/재무활동

### 주석
- 주석 번호 기준으로 먼저 분리
- 같은 주석 번호 아래의 `p`, `table`을 논리적으로 하나로 모음
- 길면 추가 chunking

## 테이블 구조

### documents
- `file_name`
- `title`
- `report_year`
- `source_path`
- `raw_html`
- `parsed_json`

### chunks
- `chunk_key`
- `chunk_index_global`
- `chunk_index_in_section`
- `major_section`
- `sub_section`
- `section_type`
- `note_no`
- `note_title`
- `topic`
- `content`
- `embedding`

## 샘플 파일

업로드된 `감사보고서_2014.htm` 1개를 `data/html/` 에 예시로 넣어두었습니다.
나머지 9개 파일도 같은 폴더에 넣으면 같은 방식으로 적재됩니다.


## 이번 수정 사항

- 파일명 기반 연도 추출을 우선 적용해 `2014`가 `2019`로 잘못 잡히는 문제를 수정했습니다.
- `SECTION-1 / SECTION-2` 구조를 활용해 상위 섹션을 다시 분류했습니다.
- `(첨부) 재무제표` 섹션 안에서 5대 재무제표와 주석 섹션을 분리하도록 수정했습니다.
- 주석 파서는 스택 기반 계층 파싱을 적용했고, 상위 주석 번호별 병합 텍스트를 함께 저장합니다.
- 청커는 병합된 주석 텍스트를 기준으로 note chunk를 생성하도록 변경했습니다.
