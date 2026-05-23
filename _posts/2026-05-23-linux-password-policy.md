---
title: "Linux 패스워드 정책 — KISA 기준 vs NIST 권고"
date: 2026-05-23
categories: [infra]
tags: [linux, security, password, pam]
excerpt: "리눅스 패스워드 정책은 PAM 모듈로 구성된다. KISA 점검 기준은 복잡도·주기 만료를 강조하지만, NIST는 정반대 — 길이·MFA·블랙리스트를 권한다. 두 기준이 충돌할 때 실무에서 어떻게 정리할지."
toc: true
toc_sticky: true
---

내부 서버 보안 점검 시즌이 오면 매번 같은 질문을 받는다. "KISA 기준 통과는 어떻게 맞추지?" 그런데 막상 [NIST SP 800-63B](https://pages.nist.gov/800-63-3/sp800-63b.html)를 보면 KISA가 강조하는 항목들 중 상당수를 **하지 말라**고 한다. 두 기준이 충돌하는 상황에서 리눅스 패스워드 정책을 어떻게 정리해야 하는지 한 번 정리한다.

대상은 RHEL/CentOS 계열 서버를 운영하면서 보안 점검 요구사항에 부딪히는 엔지니어다. PAM 구조에 익숙하지 않다면 마지막 "구현" 섹션부터 봐도 된다.

---

## 정책의 세 축

리눅스 패스워드 정책은 PAM(Pluggable Authentication Modules) 모듈로 구현되며 크게 세 가지로 나뉜다.

1. **복잡도** — 길이, 문자 종류 조합 (CentOS 7: `pam_cracklib`, RHEL 8+: `pam_pwquality`)
2. **잠금/실패 카운트** — 연속 실패 시 계정 잠금 (CentOS 7: `pam_tally2`, RHEL 8+: `pam_faillock`)
3. **유효 기간** — 패스워드 만료/변경 주기 (`/etc/login.defs` + `chage` 명령)

설정 파일은 대부분 `/etc/pam.d/system-auth`에 모여있고, sshd 등 일부 서비스는 `/etc/pam.d/password-auth`를 추가로 본다.

## KISA 권고 (한국 컴플라이언스)

KISA 주요정보통신기반시설 점검 기준에서 흔히 보는 복잡도 규칙:

- **2종류 조합** (영문 대소문자, 숫자, 특수문자 중 2종) → **최소 10자**
- **3종류 조합** → **최소 8자**

여기서 한 가지 함정이 있다. KISA가 말하는 "영문 대소문자"는 **한 묶음(letter)**으로 본다. 반면 PAM의 `pam_cracklib`은 대문자(`ucredit`)와 소문자(`lcredit`)를 **별도 클래스**로 다룬다. 그래서 KISA 정책을 PAM 옵션으로 옮길 때 1:1 매핑이 안 되고, 실무에선 안전 측에 서서 "4종 모두 강제"로 더 엄격하게 가는 경우가 많다.

## NIST SP 800-63B (현대 베스트 프랙티스)

NIST 2017 개정판은 KISA와 결이 꽤 다르다.

- **길이 ≥ 12자** 권장 (복잡도보다 길이가 안전성에 더 기여)
- **유출된 패스워드 블랙리스트 검사** ([Have I Been Pwned](https://haveibeenpwned.com/Passwords) 등 활용)
- **강제 주기 만료 폐지** (주기 변경은 `password1!` → `password2!` 같은 단순 변형을 유도해 오히려 약해짐)
- **힌트/비밀 질문 폐지**
- **MFA 병행 권장**

요약하면 *"복잡한 규칙으로 사용자를 고문하지 말고, 길이 + MFA + 블랙리스트로 갈음하라"*다.

## 충돌하는 두 기준, 어떻게 정리할까

실무에서 권장:

1. **KISA 기준은 컴플라이언스 통과용 최저선**으로 본다 (점검 무사 통과가 목적).
2. **실제 보안은 길이 + MFA + 블랙리스트 조합**으로 가져간다.
3. 강제 주기 만료(`PASS_MAX_DAYS`)는 컴플라이언스 요구가 있으면 켜되, 짧게(예: 30일) 잡지 말 것 — NIST가 경고하는 약화 패턴이 발생.

## 구현 — PAM 모듈 매핑

| 정책 | CentOS 7 | RHEL 8+ |
|------|----------|---------|
| 복잡도 | `pam_cracklib` | `pam_pwquality` |
| 실패 시 잠금 | `pam_tally2` | `pam_faillock` |
| 유닉스 인증 핵심 | `pam_unix` | (동일) |

설정 파일: `/etc/pam.d/system-auth` (대부분), `/etc/pam.d/password-auth` (sshd 등 일부 서비스)

### 만료 정책

PAM과 별개로 `/etc/login.defs`에서 시스템 전역 관리:

```
PASS_MAX_DAYS  90
PASS_MIN_DAYS  1
PASS_MIN_LEN   8
PASS_WARN_AGE  7
```

기존 계정에는 `chage` 명령으로 적용:

```bash
# 최대 90일, 최소 1일, 만료 7일 전 경고
chage -M 90 -m 1 -W 7 alice

# 현재 만료 정책 확인
chage -l alice
```

`/etc/shadow`의 3·4·5·6·7번 컬럼이 각각 마지막 변경일, 최소·최대 일수, 경고일, 비활성화 일수다.

### 해시 알고리즘

`pam_unix` 옵션으로 지정:

```
password sufficient pam_unix.so sha512 shadow nullok try_first_pass use_authtok
```

- `sha512` — SHA-512 해시 (CentOS 7 기본, 권장)
- 과거 `md5`는 약하므로 사용 금지
- `bcrypt`/`yescrypt`는 RHEL 8+ 일부에서 지원

## 정리

- 리눅스 패스워드 정책 = **PAM 복잡도 + 실패 잠금 + 만료** 세 축.
- KISA는 점검 통과용 최저선, NIST는 현대 보안 권고. **목적이 다르다.**
- 가능하면 길이 + MFA + 유출 블랙리스트로 본질 보안을 챙기고, KISA 항목은 컴플라이언스 요건만 맞추자.
- 패스워드 정책을 강화할 때는 사용자가 우회 패턴(메모, 단순 변형)을 만들지 않도록 UX도 같이 고려해야 한다.

## 참고문헌

- NIST SP 800-63B Digital Identity Guidelines: <https://pages.nist.gov/800-63-3/sp800-63b.html>
- KISA 주요정보통신기반시설 기술적 취약점 분석·평가 방법 상세가이드
- man pages: `pam_cracklib(8)`, `pam_tally2(8)`, `chage(1)`, `login.defs(5)`
