# API 계약

모든 경로는 `/api`로 시작한다. JSON의 ID는 UUID 문자열, 날짜는 시간대가 포함된 ISO 8601이다.
요청·응답 예시의 ID와 토큰은 설명용이다. 이 문서는 T5 인증·기록 계약이며 번역·음성 API는 T10에서 추가한다.

## 인증

가입 후 로그인하여 토큰을 받는다. 보호된 경로에는 `Authorization: Bearer <access_token>`을 보낸다.
access 토큰은 30분, refresh 토큰은 7일이다. access와 refresh는 서로 대신 사용할 수 없다.
화면은 access를 메모리, refresh를 `sessionStorage`에 보관한다. 401 응답을 받으면 한 번 갱신하고
실패하면 로그인 화면으로 이동한다. 탭을 닫거나 로그아웃하면 저장된 토큰을 지운다.

토큰은 HS256으로 서명한다. 서명·만료·필수 클레임·토큰 종류·사용자 존재 여부를 검증한다.
refresh 성공 시 새 토큰 쌍을 발급한다. 현재는 서버의 토큰 폐기 목록이나 일회용 refresh 회전 기능이 없어서
이전 토큰도 원래 만료 시각까지 유효하다. 로그아웃은 클라이언트에서만 처리한다.
토큰 응답에는 `Cache-Control: no-store`, `Pragma: no-cache`가 붙는다.

### POST /api/auth/register

`Content-Type: application/json`

```json
{"email": "person@example.com", "password": "example-password"}
```

이메일 형식을 검증하고 전체를 소문자로 저장·비교한다. 비밀번호는 8자 이상, UTF-8 인코딩 기준
72바이트 이하여야 한다(한글 `가` 24개는 72바이트). 공백을 자르거나 비밀번호를 정규화하지 않는다.
DB에는 bcrypt 해시만 저장하며 응답에 비밀번호나 해시를 넣지 않는다.

응답 `201 Created`:

```json
{
  "id": "d74c1fe2-4a06-4ea3-8f90-1e7ff1f5a113",
  "email": "person@example.com",
  "created_at": "2026-09-13T00:00:00Z"
}
```

오류: `409` 이메일 중복(대소문자만 다른 경우 포함), `422` 이메일·비밀번호 검증 실패 또는 필수 필드 누락.

### POST /api/auth/login

OAuth2 password 폼을 사용한다. `username`에 이메일을 보낸다.
`Content-Type: application/x-www-form-urlencoded`

```text
username=person%40example.com&password=example-password
```

선택 필드 `grant_type`을 보낼 때는 `password`로 지정한다. 응답 `200 OK`:

```json
{
  "access_token": "<access JWT>",
  "refresh_token": "<refresh JWT>",
  "token_type": "bearer",
  "expires_in": 1800
}
```

오류: `401` 이메일 또는 비밀번호 불일치, 이메일 형식 오류, 비밀번호 72바이트 초과.
`422` 폼 필드 누락·잘못된 `grant_type`. 로그인에 JSON을 보내면 필수 폼 필드가 없어 `422`가 된다.

### POST /api/auth/refresh

access 헤더 없이 refresh 토큰을 JSON으로 보낸다.

```json
{"refresh_token": "<refresh JWT>"}
```

응답 `200 OK`:

```json
{
  "access_token": "<new access JWT>",
  "refresh_token": "<new refresh JWT>",
  "token_type": "bearer",
  "expires_in": 1800
}
```

화면은 두 토큰을 새 값으로 교체한다. 오류: `401` 잘못된 서명·만료·access 토큰 사용·없는 사용자,
`422` 필드 누락 또는 길이 제한 위반(refresh 문자열 1~4096자).

### GET /api/auth/me

요청: `GET /api/auth/me`, `Authorization: Bearer <access_token>`.
본문 없음. 응답 `200 OK`:

```json
{
  "id": "d74c1fe2-4a06-4ea3-8f90-1e7ff1f5a113",
  "email": "person@example.com",
  "created_at": "2026-09-13T00:00:00Z"
}
```

오류: `401` access 토큰 누락·유효하지 않음·사용자 삭제됨.

## 번역 기록

아래 경로 모두 access 토큰이 필요하다. 본인 기록만 조회·삭제할 수 있다.
다른 사용자 기록과 존재하지 않는 기록은 동일한 `404`를 반환한다.
T5에는 기록 생성 API가 없다. T6 번역 파이프라인에서 성공한 번역을 저장한다.

### GET /api/history?limit=20&offset=0

본문 없음. `limit` 기본값 20, 범위 1~100. `offset` 기본값 0, 0 이상 정수.
`created_at` 내림차순이며 시각이 같으면 `id` 내림차순으로 정렬한다.
`total`은 페이지 크기와 무관한 본인 기록 전체 개수다.

응답 `200 OK`:

```json
{
  "items": [
    {
      "id": "080b161c-53d6-4460-992b-f778a5e348cd",
      "mode": "text",
      "source_lang": "ko",
      "target_lang": "en",
      "source_text": "안녕하세요",
      "translated_text": "Hello",
      "stt_model": null,
      "mt_model": "example-mt",
      "tts_model": null,
      "stt_ms": null,
      "mt_ms": 120,
      "tts_ms": null,
      "audio_id": null,
      "created_at": "2026-09-13T00:00:00Z"
    }
  ],
  "total": 1
}
```

기록이 없으면 `{"items": [], "total": 0}`. offset이 끝을 넘으면 items만 빈 배열이고 total은 유지된다.
오류: `401` 인증 실패, `422` 페이지 인자의 형식·범위 오류.

### GET /api/history/{id}

요청 예: `GET /api/history/080b161c-53d6-4460-992b-f778a5e348cd`. 본문 없음.
응답 `200 OK`(목록의 항목과 같은 형태):

```json
{
  "id": "080b161c-53d6-4460-992b-f778a5e348cd",
  "mode": "speech",
  "source_lang": "en",
  "target_lang": "ko",
  "source_text": "Hello",
  "translated_text": "안녕하세요",
  "stt_model": "example-stt",
  "mt_model": "example-mt",
  "tts_model": "example-tts",
  "stt_ms": 200,
  "mt_ms": 120,
  "tts_ms": 300,
  "audio_id": "a17a7bf4-ce62-4d8b-9309-a220b1868299",
  "created_at": "2026-09-13T00:00:00Z"
}
```

`mode`는 `text` 또는 `speech`, 언어는 `en` 또는 `ko`이며 출발·도착 언어는 서로 다르다.
모델명은 실제 사용한 모델 식별자다. 시간 필드 단위는 밀리초이며, 실행하지 않은 STT/TTS는 `null`이다.
합성 음성 메타데이터가 없으면 `audio_id`는 `null`이다. 파일 시스템 경로는 응답에 노출하지 않는다.
서버 음성 조회는 T6의 `/api/audio/{id}`에서 Bearer 인증 후 blob URL로 재생한다.

오류: `401` 인증 실패, `404` 본인 기록 없음, `422` UUID 형식 오류.

### DELETE /api/history/{id}

요청 예: `DELETE /api/history/080b161c-53d6-4460-992b-f778a5e348cd`. 본문 없음.
응답 `204 No Content`, 응답 본문 없음. 다시 삭제하면 `404`이다.

번역 행과 연결된 `audio_files` 행을 같은 DB 트랜잭션에서 삭제한다.
T5에서는 디스크 파일을 삭제하지 않는다. 파일 저장소가 생기는 T6에서 디스크 삭제를 연결한다.
오류: `401` 인증 실패, `404` 본인 기록 없음, `422` UUID 형식 오류.

## 오류 응답

`401`에는 `WWW-Authenticate: Bearer` 헤더가 붙는다. 인증 정보가 틀린 이유를 세분화해
계정 존재 여부를 알리지 않는다. 주요 응답 본문:

| 상태 | 본문 |
|---|---|
| 401 (헤더 누락 또는 Bearer 아님) | `{"detail": "Not authenticated"}` |
| 401 (자격 증명·토큰 오류) | `{"detail": "Could not validate credentials"}` |
| 409 | `{"detail": "Email already registered"}` |
| 404 | `{"detail": "History not found"}` |

`422`는 FastAPI 검증 오류 형식인 `{"detail": [...]}`이며 각 항목의 `loc`, `msg`, `type`으로
위치와 이유를 확인한다. 입력에 따라 `input`, `ctx`가 추가될 수 있다.

## 실행 설정과 DB

백엔드는 기본값과 환경 변수만 사용하며 `.env`를 자동으로 읽지 않는다. `.env.example`은 설정 예시다.
`DATABASE_URL` 기본값은 compose 개발 DB(`localhost:55442/voicetranslator`)다.
backend 폴더에서 `uv sync` 후 `uv run alembic upgrade head`를 실행해 테이블을 만든다.

`JWT_SECRET_KEY`가 없으면 프로세스마다 임의 키를 생성하므로 서버 재시작 시 기존 토큰이 무효가 된다.
재시작 후 로그인 유지 또는 여러 worker를 사용할 때는 모든 프로세스에 같은 32바이트 이상의 비밀 키를
환경 변수로 설정한다. 키는 저장소에 기록하지 않는다.
`ACCESS_TOKEN_EXPIRE_MINUTES`(기본 30), `REFRESH_TOKEN_EXPIRE_DAYS`(기본 7)로 유효 기간을 바꿀 수 있다.

`uv run pytest`는 conftest가 고유한 `vt_test_<uuid>` DB를 생성하고 Alembic을 적용하여 실제 PostgreSQL로
테스트한 뒤 해당 DB만 삭제한다. 접속 계정에 CREATE DATABASE 권한이 필요하다.
`TEST_DATABASE_ADMIN_URL`로 테스트 서버 접속을 바꿀 수 있다(미설정 시 DATABASE_URL 사용).
이 URL의 DB 이름을 테스트 대상으로 사용하지 않으며 개발 DB의 행을 지우지 않는다.

구현 참고: [SQLAlchemy 비동기 세션](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html),
[PyJWT 검증 API](https://pyjwt.readthedocs.io/en/stable/api.html), [bcrypt](https://github.com/pyca/bcrypt).
