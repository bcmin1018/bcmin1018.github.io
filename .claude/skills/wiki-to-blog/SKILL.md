---
name: wiki-to-blog
description: bradwiki(Obsidian 개인 위키) 페이지를 bcmin1018.github.io Jekyll 블로그 글로 변환·발행한다. 사용자가 위키 파일 경로를 주거나 "위키 X 페이지를 블로그로 옮겨줘/변환해줘/발행해줘"라고 할 때 사용.
argument-hint: <위키 페이지 절대경로 또는 페이지 제목>
---

# 위키 → 블로그 변환

입력: `$ARGUMENTS` (위키 페이지 경로 또는 제목)

- 위키 루트: `/Users/byeongcheolmin/Library/Mobile Documents/iCloud~md~obsidian/Documents/bradwiki`
  - 본문 페이지: `wiki/<카테고리>/<제목>.md`, 첨부: `wiki/assets/`, 원자료: `raw/` (**비공개**)
- 블로그 루트: `/Users/byeongcheolmin/PycharmProjects/bcmin1018.github.io`
- 참고 발행본(톤·구조의 기준): `_posts/2026-07-19-message-queue-worker-design.md`

제목만 받았다면 `wiki/` 아래에서 파일명·`aliases`로 찾는다. 후보가 여럿이면 사용자에게 고른다.

아래 단계를 순서대로 진행한다. 단계를 건너뛰었다면 최종 보고에 그 사실을 적는다.

---

## Step 0. 브랜치 준비

`master`는 라이브 사이트다. 작업 전에 `git checkout -b post-<slug>`로 브랜치를 만든다. 슬러그가 아직 없으면 Step 2에서 정한 뒤 만든다.

## Step 1. 적합성 판단 (변환 전에 반드시)

위키 페이지 전체와 그 페이지가 참조하는 `wiki/assets/` 파일을 읽는다. 다음 중 하나라도 해당하면 **변환하지 말고** 사용자에게 확인한다.

- `(작성중)` 표시, 본문 200자 미만, `TODO`가 핵심 내용을 대체하고 있음
- 사적인 메모·일지 성격 (주장이 경험 공유가 아니라 개인 할 일 목록 수준)
- `sources`가 비어 있거나, 핵심 주장이 출처로 뒷받침되지 않음
- 이미 발행됨 — frontmatter에 `published_at`이 있으면 새 글이 아니라 기존 글 **수정**인지 묻는다

`raw/` 원자료는 사실 확인용으로만 읽을 수 있다. 원문 문장·대화 로그를 블로그에 옮기지 않는다.

## Step 2. URL 결정 (발행 후 불변)

`categories`와 파일명 슬러그가 URL(`/<cat1>/<cat2>/<slug>/`)이 된다. 한번 발행하면 바꿀 수 없으므로 확정 전에 한 번 더 점검한다.

**카테고리 매핑** — 위키 frontmatter의 `category`(한국어 라벨일 수 있음) 또는 폴더 경로 기준:

| 위키 폴더 / category 라벨 | 블로그 `categories` |
|---|---|
| `ai-tech/llm/` | `[ai-tech, llm]` |
| `ai-tech/agent/` | `[ai-tech, agent]` |
| `ai-tech/rag/` | `[ai-tech, rag]` |
| `ai-tech/training/` | `[ai-tech, training]` |
| `algorithms/system-design/` · "알고리즘 / 시스템 설계" | `[algorithms, system-design]` |
| `algorithms/` (하위 폴더 없음) | `[algorithms]` |
| `infra/` | `[infra]` |
| `projects/` | `[projects]` |
| `industry/` | `[industry]` |

- 표에 없는 폴더면 `폴더/하위폴더`를 kebab-case로 옮기는 것을 제안하고 사용자 확인을 받는다.
- `archive`는 절대 쓰지 않는다.
- 같은 계열 기존 글의 `categories`를 `grep -h "^categories" _posts/*.md`로 확인해 일관성을 맞춘다.

**슬러그**: 영문 kebab-case, 주제 핵심어 2~5단어 (예: `fair-queuing-round-robin`). 한국어 제목을 음역하지 않는다.

**날짜**: 발행일 = 오늘 날짜 (위키 `created`가 아님). 파일명과 frontmatter `date`를 같게 둔다.

## Step 3. 외부 독자용 재구성

위키는 본인 메모용으로 압축돼 있다. 블로그는 **해당 스택을 모르는 일반 백엔드/AI 엔지니어**를 독자로 가정한다.

### 3.1 구조
1. **도입부** (제목 없는 첫 문단 1~3개): 독자가 겪을 법한 문제 상황으로 시작 → 이 글이 무엇을 다루는지. "누가 읽으면 좋은지" 같은 메타 문단은 넣지 않는다. 도입부 마무리는 "~한 경험을 정리해보았다" 정도의 담백한 한 문장으로 끝낸다. "이 글에서는 A와 B, 그리고 C를 정리한다"식 목차 나열이나 "문제는 X가 아니라 Y다" 같은 단정적 훅은 쓰지 않는다 (사용자 피드백).
2. **배경** — 어떤 상황에서 이 문제를 만났는지. 위키의 경험담은 살리되 프로젝트·회사·서비스 고유명사는 일반화한다 ("리딩마마 프로젝트" → "다화자 TTS 파이프라인").
3. **본문** — 위키의 문제 → 해결 → 결과 흐름을 유지하되, 위키가 "이미 안다고 가정"한 전제 개념은 처음 등장할 때 1~2문장으로 풀어 쓴다. 아직 소개하지 않은 개념을 먼저 쓰지 않았는지(전방 참조) 확인한다.
4. **정리** — 요약 표(선택) + **핵심 takeaway 3~5개** 번호 목록. 각 항목은 굵은 한 문장 + 부연 한 문장.

위키의 `## 요약`, `## 관련 페이지`, `## 출처` 섹션은 그대로 옮기지 않는다 (요약은 도입부/excerpt로, 출처는 참고문헌으로 흡수).

### 3.2 추상화 수준 (사용자 피드백으로 확정된 원칙)
문단·코드마다 "**다른 언어나 다른 브로커·프레임워크에서도 성립하는 이야기인가?**"를 자문한다.
- 아니라면 삭제하거나 보편 개념으로 격상해 서술한다.
- 언어 종속 세부(예: asyncio 태스크 정리 방식), 프로젝트 내부 추상화 이름(예: `self.uow`, 내부 enum)은 제거한다.
- 특정 도구(Redis, SQS 등)는 **예시**로 쓰고, "다른 도구에서는 무엇이 같은 역할을 하는지"를 한 줄 덧붙인다.
- 코드는 개념을 보여주는 최소한만. 명령어·API 이름은 사실 확인된 것만 쓴다.

### 3.3 문체
- 기존 발행본과 같은 평서체("~다", "~한다"). 위키 문장이 이미 좋으면 살린다.
- 핵심 문장은 **굵게**. 비유는 개념을 확실히 쉽게 만들 때만.

## Step 4. Obsidian 문법 → Jekyll 변환

| 위키 | 블로그 |
|---|---|
| `[[페이지]]`, `[[페이지\|별칭]]` — 블로그에 이미 발행된 페이지 (위키 쪽 `published_url` 또는 "블로그 발행본" 링크로 확인) | `[제목](/cat1/cat2/slug/)` 상대 링크 |
| `[[페이지]]` — 미발행 | 평문으로 풀어 쓰거나 필요한 만큼 본문에 설명. **링크 금지** |
| `[[YYYY-MM-DD]]` 일지 링크 | 제거. 필요한 사실만 본문에 녹인다 |
| `[[raw/...]]` | 제거. 공개 출처(논문·공식 문서)가 있으면 그것으로 대체 |
| `![[assets/x.png]]` 이미지 | 파일을 `assets/images/posts/YYYY-MM-DD-<설명>.png`로 복사 후 `{% include figure image_path="/assets/images/posts/..." alt="..." caption="..." %}`. alt는 그림 내용을 문장으로 서술 |
| `[[assets/x.html\|...]]` 인터랙티브 HTML | 아래 "인터랙티브 HTML 삽입" 절차를 따른다 |
| `<!-- TODO ... -->` 주석 | 해결해서 반영하거나 제거 (블로그 소스에 남기지 않음) |
| `> [!note]` 등 callout | 일반 인용문 또는 `{: .notice--info}` 블록 |
| ` ```mermaid ` | 그대로 사용 가능 (`_includes/footer/custom.html`이 렌더링) |
| 위키 URL, Obsidian 경로, 로컬 절대경로 | 절대 노출 금지 |

텍스트만 있는 핵심 흐름(전/후 비교, 순서, 구조)에는 mermaid 다이어그램 추가를 고려한다.

### 인터랙티브 HTML 삽입 (시뮬레이터 등)
위키에서는 링크로만 걸려 있어도 블로그에서는 **본문에 직접 삽입**한다.
1. 파일 전체를 읽어 비공개 정보·로컬 경로가 없는지 확인한다.
2. `assets/interactive/YYYY-MM-DD-<설명>.html`로 복사한다. (`_config.yml` `defaults`에 이 경로의 `sitemap: false`가 있는지 확인하고, 없으면 추가한다.)
3. 블로그 스킨(mint)은 라이트 전용이다. HTML에 `prefers-color-scheme: dark` 스타일이 있으면 복사본의 `<html>`에 `data-theme="light"`를 붙이거나 다크 블록을 제거해 라이트로 고정한다.
4. 본문에 높이 자동 조절 iframe을 넣고, 바로 위에 "무엇을 눌러 보면 되는지" 한 줄을 쓴다:
   ```html
   <iframe src="/assets/interactive/YYYY-MM-DD-<설명>.html" title="시뮬레이터 설명" width="100%" height="1200" style="border:0; border-radius:12px;" loading="lazy" onload="var f=this,d=f.contentDocument;if(d&&window.ResizeObserver){new ResizeObserver(function(){f.style.height=d.body.offsetHeight+'px'}).observe(d.body)}"></iframe>
   ```
   (`body`에 `height:100%` 같은 스타일이 있으면 자동 조절이 안 되므로 확인)

## Step 5. Frontmatter

```yaml
---
title: "주제 — 부제(무엇을 얻어가는지)"
date: YYYY-MM-DD
categories: [cat1, cat2]
tags: [english-kebab, ...]        # 3~6개, 위키 tags를 영문·정제
excerpt: "1~2문장. 문제 + 이 글의 답을 드러낸다."
toc: true
toc_sticky: true
---
```

- 위키의 `aliases`, `last_reviewed`, `created`, `sources`, `category`는 넣지 않는다.
- 제목·excerpt 안의 큰따옴표는 이스케이프하거나 작은따옴표로 바꿔 YAML 오류를 막는다.
- 공개 가능한 출처가 있으면 본문 끝에 `## 참고문헌` 섹션으로 둔다. 개인 경험·대화 기반 글이면 섹션을 만들지 않는다.

## Step 6. 저장 및 검증

1. `_posts/YYYY-MM-DD-<slug>.md`로 저장.
2. 금지 요소 스캔 — 아래 결과가 비어야 한다:
   ```bash
   grep -nE '\[\[|\]\]|raw/|iCloud|obsidian|bradwiki|TODO' _posts/YYYY-MM-DD-<slug>.md
   ```
3. 빌드 확인: `bundle exec jekyll build` (에러 시 `--verbose --trace`). 생성된 `_site/<cat1>/<cat2>/<slug>/index.html`이 있는지 확인한다.
4. **보고 전에 반드시 로컬 서버를 백그라운드로 띄운다** (`bundle exec jekyll serve --livereload`, run_in_background). `build`만 하고 미리보기 URL을 주면 사용자가 열 수 없다. 서버가 뜨면 글·이미지·iframe 파일 URL이 모두 200인지 curl로 확인한다. 브라우저에서 눈으로 본 것은 아니므로, 레이아웃·iframe 동작은 사용자에게 확인을 요청한다.
5. 커밋은 브랜치에만 하고, **master 머지·push는 사용자 요청이 있을 때만** 한다.

## Step 7. 위키 측 업데이트

변환을 마치면 위키 원본 페이지에:
- frontmatter에 `published_at: YYYY-MM-DD`, `published_url: https://bcmin1018.github.io/<cat1>/<cat2>/<slug>/` 추가
- `## 관련 페이지` 섹션에 `- 블로그 발행본: [블로그 글 제목](published_url)` 추가

위키 본문 내용 자체는 수정하지 않는다.

## Step 8. 보고

사용자에게 다음을 짧게 알린다.
- 파일 경로, 브랜치 이름, 미리보기 URL `http://localhost:4000/<cat1>/<cat2>/<slug>/`
- 확정한 `categories`·슬러그 (URL 불변이므로 머지 전 마지막 확인 요청)
- 위키에서 **뺀 것**과 **새로 더한 것** 요약 (예: 일지 링크 제거, 도입부·takeaway 추가, 시뮬레이터 iframe 삽입)
- 검증 결과(빌드 성공 여부, 직접 확인 못 한 항목)
