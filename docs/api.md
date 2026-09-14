# API 계약

모든 경로는 `/api`로 시작한다. JSON의 ID는 UUID 문자열, 날짜는 시간대가 포함된 ISO 8601이다.
요청·응답 예시의 ID와 토큰은 설명용이다. 인증·기록 계약은 T5, 번역·음성 계약은 T10에서 정했다.

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
기록을 만드는 API는 따로 없다. 번역 API(`/api/translate/text`, `/api/translate/speech`)가 성공한 번역을 저장한다.

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
서버 음성은 번역 절의 `GET /api/audio/{id}`로 받아 blob URL로 재생한다. `audio_id`가 `null`이면 화면은 브라우저 내장 음성으로 읽는다.

오류: `401` 인증 실패, `404` 본인 기록 없음, `422` UUID 형식 오류.

### DELETE /api/history/{id}

요청 예: `DELETE /api/history/080b161c-53d6-4460-992b-f778a5e348cd`. 본문 없음.
응답 `204 No Content`, 응답 본문 없음. 다시 삭제하면 `404`이다.

번역 행과 연결된 `audio_files` 행을 같은 DB 트랜잭션에서 삭제한 뒤 디스크의 음성 파일도 지운다.
파일 삭제가 실패해도 DB 삭제는 되돌리지 않고 서버 로그에 남긴다(번역 절의 "저장과 삭제").
오류: `401` 인증 실패, `404` 본인 기록 없음, `422` UUID 형식 오류.

## 번역

아래 경로 모두 access 토큰이 필요하다. 번역이 성공하면 기록을 하나 만들고, 그 기록과 `tts_error`를 돌려준다.
모델 호출은 서버의 모델 전용 스레드에서 실행되므로 번역 중에도 다른 요청(로그인, 기록 조회)은 막히지 않는다.

### POST /api/translate/text

`Content-Type: application/json`

```json
{"text": "안녕하세요", "source_lang": "ko", "target_lang": "en"}
```

- `text`: 앞뒤 공백을 자른 뒤 1~500자(유니코드 문자 기준). 자른 값을 번역하고 저장한다
- `source_lang`, `target_lang`: `en` 또는 `ko`, 서로 달라야 한다

응답 `201 Created`, `Location: /api/history/{id}`. 본문은 기록 항목(`GET /api/history/{id}`와 같은 형태)에 `tts_error`를 더한 것:

```json
{
  "id": "080b161c-53d6-4460-992b-f778a5e348cd",
  "mode": "text",
  "source_lang": "ko",
  "target_lang": "en",
  "source_text": "안녕하세요",
  "translated_text": "Hello",
  "stt_model": null,
  "mt_model": "example-mt",
  "tts_model": "example-tts",
  "stt_ms": null,
  "mt_ms": 120,
  "tts_ms": 300,
  "audio_id": "a17a7bf4-ce62-4d8b-9309-a220b1868299",
  "created_at": "2026-09-13T00:00:00Z",
  "tts_error": null
}
```

### POST /api/translate/speech

`Content-Type: multipart/form-data`

| 필드 | 내용 |
|---|---|
| `audio` | 녹음 파일. 브라우저 MediaRecorder의 `audio/webm`(Opus)을 기본으로 하고, 서버가 풀 수 있는 형식(wav, ogg, mp3, m4a 등)이면 받는다. 파일의 Content-Type은 믿지 않고 실제로 풀어 본다 |
| `source_lang` | 말한 언어. `en` 또는 `ko` |
| `target_lang` | 번역할 언어. `source_lang`과 달라야 한다 |

- 제한: 파일 10MB 이하, 음성 길이 30초 이하
- 음성 인식 언어는 `source_lang`으로 고정한다 (언어 자동 판별을 하지 않음)
- 녹음 원본은 저장하지 않는다. 인식된 글자만 `source_text`로 저장한다

응답 `201 Created`, 본문은 `/api/translate/text`와 같은 형태이며 `mode`가 `speech`, `stt_model`과 `stt_ms`가 채워진다.

### 실패했을 때

| 상황 | 상태 | 본문 `detail` | 기록 |
|---|---|---|---|
| 필드 누락, 언어 값 오류, 같은 언어, 글자 수 초과·빈 글자 | 422 | FastAPI 검증 오류 형식 (`[...]`) | 없음 |
| multipart 형식 자체가 잘못됨(경계 없음 등) | 400 | `"Invalid multipart request"` | 없음 |
| 파일이 10MB(10 × 1024 × 1024바이트) 초과 | 413 | `"Audio file is larger than 10 MB"` | 없음 |
| 파일을 음성으로 풀 수 없음 | 422 | `"Audio could not be decoded"` | 없음 |
| 음성이 30초 초과 | 422 | `"Audio is longer than 30 seconds"` | 없음 |
| 음성에서 말을 찾지 못함 | 422 | `"No speech was recognized"` | 없음 |
| 음성 인식·번역 모델 오류 | 503 | `"Translation service is unavailable"` | 없음 |
| 음성 합성만 실패 | 201 | 정상 응답, `audio_id: null`, `tts_error`에 이유 | 있음 |

- 화면은 상태 코드와 위 표의 고정 문자열로 안내 문구를 고른다. 이 문자열은 계약이므로 바꾸려면 이 문서와 화면을 함께 고친다
- `tts_error`: 음성 합성을 하지 못했을 때의 이유. `"Speech synthesis failed"`(합성 오류) 또는 `"Speech synthesis is not available"`(서버에 합성 모델이 없음). 성공하면 `null`. 합성하지 못했으면 `tts_model`, `tts_ms`도 `null`이다
- `tts_error`는 번역 응답에만 있고 기록에는 저장하지 않는다. 기록에서 `audio_id`가 `null`이면 화면은 브라우저 내장 음성(speechSynthesis)으로 읽는다
- 모델 오류의 자세한 내용은 서버 로그에만 남기고 응답에 넣지 않는다

### GET /api/audio/{id}

요청 예: `GET /api/audio/a17a7bf4-ce62-4d8b-9309-a220b1868299`, `Authorization: Bearer <access_token>`. 본문 없음.

응답 `200 OK`, `Content-Type: audio/wav`(16비트 PCM, 모노, 표본 추출률은 합성 모델에 따름), `Cache-Control: private, no-store`.
`<audio src>`는 인증 헤더를 붙일 수 없으므로 화면은 이 경로를 `fetch`로 받아 blob URL을 만들어 재생하고, 다 쓰면 `URL.revokeObjectURL`로 해제한다.

오류: `401` 인증 실패, `404` 본인 음성 없음(다른 사용자 음성, 없는 ID, 기록 삭제로 지워진 음성 모두 같은 `{"detail": "Audio not found"}`), `422` UUID 형식 오류.

### 저장과 삭제

- 합성 음성은 서버의 음성 폴더(`AUDIO_DIR`)에 파일로 두고 `audio_files` 행이 경로를 가진다. 경로는 응답에 노출하지 않는다
- `DELETE /api/history/{id}`는 기록, `audio_files` 행, 디스크의 파일을 함께 지운다. 파일 삭제가 실패해도 DB 삭제는 되돌리지 않고 서버 로그에 남긴다
- 시간 필드(`stt_ms`, `mt_ms`, `tts_ms`)는 각 단계의 모델 호출 시간(밀리초)이다. 모델 스레드를 기다린 시간과 네트워크 시간은 넣지 않는다

### WS /api/translate/live (동시통역)

말하는 동안 원문·번역문 자막을 받고, 말을 멈추면 대화 모드와 같은 최종 결과를 받는 웹소켓 (docs/experiments.md 8절). 한 마디의 경계는 브라우저가 대화 모드처럼 정한다(Silero VAD, 1초 조용하면 끝).

브라우저 WebSocket은 인증 헤더를 붙일 수 없고 주소의 토큰은 로그에 남으므로, 토큰은 연결 뒤 첫 메시지로 보낸다. 글 메시지는 JSON, 소리는 바이너리 메시지다.

**브라우저 → 서버**

| 메시지 | 뜻 |
|---|---|
| `{"type": "start", "token": "<access_token>", "source_lang": "ko", "target_lang": "en"}` | 연결 뒤 10초 안에 보내는 첫 메시지. 성공하면 서버가 `{"type": "ready"}` |
| `{"type": "utterance", "id": 1}` | 새 마디 시작. `id`는 브라우저가 정하는 정수. 진행 중인 마디가 있으면 그 마디는 버린다 |
| 바이너리 | 지금 마디의 소리. 16kHz 모노 16비트 little-endian PCM, 말 앞 192ms부터. 마디 밖의 소리는 버린다 |
| `{"type": "pause", "id": 1}` | 말 끝 뒤 192ms 조용함. 지금까지 보낸 소리가 대화 모드가 올릴 소리와 같아, 서버가 최종 인식·번역·합성을 미리 시작한다 |
| `{"type": "resume", "id": 1}` | 1초가 되기 전에 다시 말함. 서버는 미리 한 결과를 버리고 자막 갱신을 이어 간다 |
| `{"type": "end", "id": 1}` | 1초 조용함 확인(또는 29초 강제 끊김). 서버가 기록을 저장하고 `final`을 보낸다. `pause` 없이 오면 받은 소리 전체로 처리한다 |
| `{"type": "cancel", "id": 1}` | 그 마디를 버린다(짧은 소리, 번역 음성 재생 시작, 멈춤) |

지금 마디가 아닌 `id`의 `pause`·`resume`·`end`·`cancel`은 무시한다.

**서버 → 브라우저**

| 메시지 | 뜻 |
|---|---|
| `{"type": "source", "id": 1, "text": "...", "stable": 12}` | 그 마디의 지금까지 소리를 인식한 원문. `stable`은 앞에서부터 진하게(확정) 보일 글자 수: 바로 앞 결과와 같은 앞부분을 단어 경계까지 자른 것(대소문자·띄어쓰기·문장부호는 무시). 나머지는 흐리게 보인다. 진한 부분도 다음 결과에서 바뀔 수 있다 |
| `{"type": "translation", "id": 1, "text": "...", "stable": 0}` | 위 원문의 번역. 번역은 원문이 늘 때마다 앞부분까지 자주 바뀌어(측정에서 진하게 보일 부분의 16~20%가 나중에 바뀜) `stable`은 늘 0이다: 최종 결과 전까지 모두 흐리게 보인다(docs/experiments.md 8절) |
| `{"type": "final", "id": 1, "result": {...}}` | 최종 결과. `result`는 `POST /api/translate/speech`의 201 응답과 같은 필드이고 같은 기록이 저장된다. 음성은 `GET /api/audio/{audio_id}`로 받는다 |
| `{"type": "error", "id": 1, "detail": "..."}` | 그 마디만 실패, 기록 없음. `detail`은 음성 번역 API의 422·503 문구와 같다(`"Audio could not be decoded"`, `"Audio is longer than 30 seconds"`, `"No speech was recognized"`, `"Translation service is unavailable"`), 저장이 실패하면 `"The translation could not be saved"`. 연결은 유지된다 |

**닫힘 코드**: `4401` 토큰이 없거나 틀림, 없는 사용자, 또는 연결 중 토큰 만료(만료 뒤 첫 메시지에서 닫는다. 화면은 토큰을 새로 받아 다시 연결한다), `4422` 첫 메시지가 없거나 틀림, 언어 오류, 계약에 없는 메시지, `4503` 서버에 모델이 없음.

- 자막 갱신: 서버는 한 마디에서 한 번에 한 갱신만 돌리고, 끝나면 그때까지 받은 소리 전체를 다시 인식한다. 두 갱신의 시작은 `LIVE_UPDATE_MS` 이상 떨어진다. 원문은 인식이 끝나는 대로, 번역문은 번역이 끝나는 대로 보낸다. 원문이 바뀌지 않으면 번역하지 않는다
- 자막 갱신의 인식 설정은 `LIVE_BEAM_SIZE`·`LIVE_TEMPERATURE_FALLBACK`을 쓰고, 최종 결과는 음성 번역 API와 같은 설정·같은 WAV로 처리한다. 모델 호출은 HTTP 요청과 같은 모델 스레드 대기열에서 돈다
- 한 마디 소리가 30초를 넘으면 그 마디는 `end`에서 `"Audio is longer than 30 seconds"` 오류가 된다
- 연결이 끊기면 진행 중인 마디와 아직 저장하지 않은 최종 결과는 버린다

## 상태 확인

### GET /api/health

인증이 필요 없다. 응답 `200 OK`, 본문 `{"status": "ok"}`.

API 프로세스가 요청을 받고 있다는 뜻일 뿐, DB 연결이나 모델 준비는 확인하지 않는다. `LOAD_MODELS=true`면 앱은 모델을 올리고 준비 단계를 마친 뒤에 요청을 받기 시작하므로, 그때부터 200이 나온다(Docker compose의 api 상태 검사가 이 경로를 쓴다). 오타 교정 모델은 기다리지 않고 뒤에서 준비하므로 200이어도 아직 교정이 켜지지 않았을 수 있다.

## 오류 응답

`401`에는 `WWW-Authenticate: Bearer` 헤더가 붙는다. 인증 정보가 틀린 이유를 세분화해
계정 존재 여부를 알리지 않는다. 주요 응답 본문:

| 상태 | 본문 |
|---|---|
| 401 (헤더 누락 또는 Bearer 아님) | `{"detail": "Not authenticated"}` |
| 401 (자격 증명·토큰 오류) | `{"detail": "Could not validate credentials"}` |
| 409 | `{"detail": "Email already registered"}` |
| 404 | `{"detail": "History not found"}` |
| 404 (음성) | `{"detail": "Audio not found"}` |

번역 경로의 413·422·503 고정 문자열은 번역 절의 "실패했을 때" 표에 있다.

`422`는 FastAPI 검증 오류 형식인 `{"detail": [...]}`이며 각 항목의 `loc`, `msg`, `type`으로
위치와 이유를 확인한다. 입력에 따라 `input`, `ctx`가 추가될 수 있다.

## 실행 설정과 DB

백엔드는 기본값과 환경 변수만 사용하며 `.env`를 자동으로 읽지 않는다. `.env.example`은 설정 예시다.
`DATABASE_URL` 기본값은 compose 개발 DB(`127.0.0.1:55442/voicetranslator`)다. DB는 IPv4로만 열리므로 `localhost` 대신 `127.0.0.1`을 쓴다(Windows에서 `localhost`는 IPv6를 먼저 시도해 연결마다 약 2초가 늦었다).
backend 폴더에서 `uv sync` 후 `uv run alembic upgrade head`를 실행해 테이블을 만든다.

`JWT_SECRET_KEY`가 없으면 프로세스마다 임의 키를 생성하므로 서버 재시작 시 기존 토큰이 무효가 된다.
재시작 후 로그인 유지 또는 여러 worker를 사용할 때는 모든 프로세스에 같은 32바이트 이상의 비밀 키를
환경 변수로 설정한다. 키는 저장소에 기록하지 않는다.
`ACCESS_TOKEN_EXPIRE_MINUTES`(기본 30), `REFRESH_TOKEN_EXPIRE_DAYS`(기본 7)로 유효 기간을 바꿀 수 있다.

모델 설정(모두 환경 변수, 아래 이름 그대로). 모델을 올리려면 backend에서 `uv sync --group gpu --group tts`로
음성 합성 의존성까지 설치한다. Docker 이미지는 `LOAD_MODELS=true`, `CT2_DIR=/models/ct2`, `AUDIO_DIR=/data/audio`로 뜬다.

| 변수 | 기본값 | 내용 |
|---|---|---|
| `LOAD_MODELS` | `false` | 앱 시작 때 세 모델을 올린다. 끄면 번역 요청은 503 (테스트·CI는 끈 채로 돈다) |
| `MODEL_DEVICE` | `cuda` | `cuda` 또는 `cpu`. GPU는 float16, CPU는 int8 |
| `STT_MODEL` | `large-v3-turbo` | faster-whisper 모델 이름 |
| `CT2_DIR` | `<프로젝트>/data/models/ct2` | 변환한 번역 모델 폴더 (`eval.mt_convert`) |
| `TTS_ENABLED` | `true` | 끄면 음성 합성 없이 뜨고 응답에 `tts_error`가 들어간다 |
| `STT_VAD_FILTER` | `true` | 말소리 구간만 인식 모델에 넘긴다 (docs/experiments.md 1-1) |
| `STT_OWN_DECODE` | `false` | 업로드를 faster-whisper 대신 앱에서 디코딩한다. 비교용으로만 남긴 설정 (5절 후보 C) |
| `LIVE_UPDATE_MS` | `1000` | 동시통역 자막 갱신 사이의 최소 간격(밀리초). 한 마디의 두 갱신 시작이 이만큼 떨어진다. 측정으로 고름: 짧을수록 빨리 보이지만 더 흔들린다 (docs/experiments.md 8절) |
| `LIVE_BEAM_SIZE` | `5` | 동시통역 자막 갱신의 인식 beam 크기. 최종 결과는 늘 음성 번역 API와 같은 설정을 쓴다 |
| `LIVE_TEMPERATURE_FALLBACK` | `true` | 자막 갱신 인식이 결과가 나쁠 때 temperature를 올려 다시 풀지. 끄면 temperature 0으로 한 번만 푼다 |
| `LIVE_UPDATE_THREAD` | `false` | 자막 갱신을 모델 스레드 대기열 대신 전용 스레드에서 돌린다. 인식 호출끼리 실제로 동시에 돌려면 `STT_NUM_WORKERS`도 2 이상이어야 한다 (docs/experiments.md 8절 T61) |
| `STT_NUM_WORKERS` | `1` | 인식 모델 복제 수(faster-whisper num_workers). 여러 스레드의 인식 호출이 이 수만큼 동시에 돈다. 복제마다 GPU 메모리를 더 쓴다 |
| `WARM_UP` | `true` | 모델을 올린 뒤 번역·합성·인식을 한 번씩 돌려 첫 요청의 지연을 없앤다 |
| `TYPO_CORRECTION` | `true` | 글자로 입력한 영어를 번역 전에 Ollama의 작은 LLM으로 오타·띄어쓰기만 고친다 (docs/experiments.md 6절). 한국어 입력과 음성 인식 결과는 고치지 않는다. Ollama가 응답하지 않거나, 정상 종료로 답하지 않았거나(교정문은 `done`이 true이고 `done_reason`이 "stop"일 때만 쓴다), 한글이 섞였거나, 입력과 글자가 절반 넘게 다르면(거절문·설명을 붙인 답 등, docs/experiments.md 6절) 입력한 그대로 번역한다. 글자는 거의 같은데 뜻만 바뀐 교정까지 막지는 못한다. 서버 시작은 Ollama를 기다리지 않는다: 교정 모델은 뒤에서 준비되고(첫 시작의 내려받기 포함, 실패하면 30초마다 다시 시도), 준비되기 전에는 입력 그대로 번역한다. 기록의 `source_text`는 입력한 그대로, `mt_model`에는 교정 모델이 붙고(`... + ollama/...`), `mt_ms`는 교정 시간을 포함한다 |
| `OLLAMA_URL` | `http://localhost:11434` | 교정 모델을 돌리는 Ollama 주소. Docker compose는 `http://ollama:11434` |
| `CORRECTION_MODEL` | `qwen2.5:1.5b-instruct` | 교정 모델. Ollama에 없으면 시작할 때 받는다(약 1GB). 올린 뒤 계속 올려 둔다(`ollama stop <모델>`로 내림) |
| `CORRECTION_TIMEOUT_S` | `10` | 교정 요청 제한 시간(초). 넘으면 입력한 그대로 번역한다 |
| `CORRECTION_PREPARE_TIMEOUT_S` | `600` | 교정 모델 준비(내려받기·적재·첫 교정) 요청마다의 제한 시간(초). 넘거나 실패하면 30초 뒤 다시 시도한다 |
| `GC_FREEZE` | `true` | 모델을 올린 뒤 `gc.freeze()`. 번역 중 다른 요청이 막히지 않게 한다 (5절) |
| `MODEL_THREADS` | `1` | 모델 호출 스레드 수 (docs/experiments.md 4절) |
| `AUDIO_DIR` | `<프로젝트>/work/audio` | 번역 음성 파일을 두는 폴더 |

`uv run pytest`는 conftest가 고유한 `vt_test_<uuid>` DB를 생성하고 Alembic을 적용하여 실제 PostgreSQL로
테스트한 뒤 해당 DB만 삭제한다. 접속 계정에 CREATE DATABASE 권한이 필요하다.
`TEST_DATABASE_ADMIN_URL`로 테스트 서버 접속을 바꿀 수 있다(미설정 시 DATABASE_URL 사용).
이 URL의 DB 이름을 테스트 대상으로 사용하지 않으며 개발 DB의 행을 지우지 않는다.

구현 참고: [SQLAlchemy 비동기 세션](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html),
[PyJWT 검증 API](https://pyjwt.readthedocs.io/en/stable/api.html), [bcrypt](https://github.com/pyca/bcrypt).
