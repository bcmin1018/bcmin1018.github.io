# bcmin1018.github.io — 운영 규칙

Brad Min(AI 엔지니어)의 기술 블로그. **Jekyll + Minimal Mistakes (remote_theme)** 기반.
주된 콘텐츠 출처는 자매 레포인 [bradwiki](https://github.com/bcmin1018/bradwiki)(개인 위키). 위키 페이지를 외부 독자용으로 재가공해 블로그 글로 발행한다.

---

## 1. 레포 구조

```
bcmin1018.github.io/
├── _config.yml              # 사이트 설정 (remote_theme: mmistakes/minimal-mistakes)
├── Gemfile                  # 로컬 dev용 (jekyll, minimal-mistakes-jekyll gem)
├── _posts/                  # 발행된 글
│   ├── YYYY-MM-DD-제목.md     # 신규 글
│   └── archive/             # 2022년 이전 옛 글 (URL 보존, 댓글 off, related off)
├── _drafts/                 # 미발행 초안 (jekyll serve --drafts 로만 보임)
├── _pages/                  # about, archive index 등 정적 페이지
├── _project/                # 프로젝트 별도 컬렉션
├── _data/navigation.yml     # 상단 메뉴
├── assets/avatar.jpg        # 작성자 아바타
├── assets/css/main.scss     # SCSS 엔트리 (Noto Sans KR 폰트 import)
├── categories/              # 카테고리 인덱스 페이지
├── index.html               # 홈 (layout: home)
└── google*.html             # Google site verification
```

**`master`(현 default) = 라이브 사이트**. 작업은 별도 브랜치(`renewal` 등)에서 하고 검증 후 머지.

---

## 2. 새 글 작성 규칙

### 2.1 파일명
- 형식: `_posts/YYYY-MM-DD-슬러그.md`
- 슬러그는 영문 kebab-case (URL 안정성). 예: `2026-05-23-llm-eval-best-practices.md`
- 한국어 제목은 frontmatter `title:`에.

### 2.2 Frontmatter 표준

```yaml
---
title: "글 제목"
date: 2026-05-23
categories: [ai-tech, llm]    # 위키 카테고리 매핑 (1~2개)
tags: [evaluation, gpt, rag]  # 검색용 (3~6개)
excerpt: "본문 위에 보일 1~2줄 요약"
toc: true
toc_sticky: true
header:
  teaser: /assets/images/posts/2026-05-23-teaser.jpg  # 선택
---
```

**중요**: `categories`가 URL 경로가 된다 (`/ai-tech/llm/슬러그/`). 한번 발행 후 변경 금지(링크 깨짐).

### 2.3 카테고리 (위키 ↔ 블로그 매핑)

| bradwiki 카테고리 | 블로그 categories |
|------------------|-------------------|
| `ai-tech/llm/`    | `[ai-tech, llm]`  |
| `ai-tech/agent/`  | `[ai-tech, agent]` |
| `ai-tech/rag/`    | `[ai-tech, rag]`   |
| `ai-tech/training/` | `[ai-tech, training]` |
| `algorithms/`     | `[algorithms]`     |
| `infra/`          | `[infra]`          |
| `projects/`       | `[projects]`       |
| `industry/`       | `[industry]`       |

`archive`는 2022년 글 전용. 신규 글에 사용 금지.

---

## 3. 위키 → 블로그 변환 워크플로우

사용자가 "위키의 X 페이지를 블로그로 변환해줘"라고 하면 다음 절차:

### Step 1. 변환 적합성 판단
다음 경우 변환하지 말고 사용자에게 확인:
- 위키 페이지가 미완성(`(작성중)` 또는 본문 < 200자)
- 사적인 노트·미정리 메모 성격
- 출처가 raw에 없거나 검증되지 않은 주장

### Step 2. 외부 독자용 재가공
위키 페이지는 본인 메모용으로 압축적. 블로그로 옮길 때:
- **도입부 추가** — 왜 이 주제를 다루는지, 누가 읽으면 좋은지
- **전제 지식 명시** — 위키는 "이미 안다고 가정", 블로그는 친절하게
- **결론 강화** — 핵심 takeaway 3~5개
- **이미지·다이어그램** — 위키엔 텍스트 위주, 블로그엔 시각화 추가 고려

### Step 3. 링크·참조 변환
- **`[[위키링크]]`** → 다음 중 하나로:
  - 해당 페이지가 이미 블로그에도 발행됐다면 → 일반 마크다운 링크 `[제목](/category/slug/)`
  - 아직 블로그에 없으면 → 평문 텍스트 또는 외부 출처로 대체
  - **위키 URL을 그대로 노출 금지** (위키는 비공개)
- **`[[raw/파일명]]`** → 평문 출처 표기 ("arxiv: XXXX" 등) 또는 제거

### Step 4. Frontmatter 변환

| 위키 frontmatter | 블로그 frontmatter |
|------------------|--------------------|
| `title`          | `title` (그대로)   |
| `category: ai-tech/llm` | `categories: [ai-tech, llm]` |
| `tags`           | `tags` (영문화·정제) |
| `created`        | `date`             |
| `last_reviewed`  | (제거 — 블로그엔 불필요) |
| `aliases`        | (제거)             |
| `sources`        | 본문 끝 "참고문헌" 섹션으로 |

### Step 5. 저장
- 파일: `_posts/YYYY-MM-DD-슬러그.md` (YYYY-MM-DD는 발행일)
- 저장 후 사용자에게 보고: 파일 경로 + 미리보기 URL (`http://localhost:4000/.../`)

### Step 6. 위키 측 업데이트
- 위키 페이지 frontmatter에 `published_at: 2026-05-23` 추가 (어디로 발행됐는지 추적)
- 위키 페이지의 "관련 페이지" 섹션에 블로그 링크 추가 검토

---

## 4. 로컬 미리보기·배포

### 로컬 실행
```bash
cd /Users/byeongcheolmin/PycharmProjects/bcmin1018.github.io
bundle exec jekyll serve --livereload
# http://localhost:4000 에서 확인
# 초안 포함: bundle exec jekyll serve --drafts
```

첫 실행 시 `bundle install` 필요. Ruby는 rbenv로 관리 (현재 3.3.6).

### 배포
- `git push origin master` → GitHub가 자동 빌드 (1~2분 후 라이브)
- GitHub Pages Settings에서 빌드 상태 확인 가능
- 빌드 실패 시 GitHub가 이메일로 알림

### 디버깅
- 로컬 빌드 에러: `bundle exec jekyll build --verbose --trace`
- frontmatter 검증: Jekyll은 잘못된 YAML에 친절하지 않음. 변경 후 항상 로컬 실행으로 확인.

---

## 5. 행동 원칙 (Claude용)

- **사용자가 변환을 요청하면 Step 1(적합성 판단)부터 시작** — 무조건 변환하지 말 것.
- **archive 카테고리에 새 글 추가 금지** — 2022년 글 전용.
- **`master` 브랜치에 직접 푸시 금지** — 변경은 항상 별도 브랜치 → 검증 → 머지.
- **URL은 한 번 정해지면 불변** — `categories`와 슬러그(파일명) 신중히 결정.
- **위키 raw 파일을 블로그에 직접 노출 금지** — 저작권·미정제 정보 위험.
- **giscus 댓글 설정**: `_config.yml`의 `comments.giscus.repo_id` 등이 `TODO`로 남아있음. 사용자가 giscus.app에서 값을 받아 채워야 댓글 작동.
