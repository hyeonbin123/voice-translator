# [파일 역할] 앱이 쓰는 비동기 DB 엔진(연결 풀)과 세션 팩토리를 모듈을 불러올 때 한 번 만든다.
# [연결] dependencies.py의 get_db(요청마다 세션 하나)와 get_sessions(동시통역 웹소켓이 필요할 때만 세션을 여는
# 용도)가 SessionLocal을 쓴다. 테스트(tests/conftest.py, test_live_api.py)는 ENGINE_OPTIONS를 가져와 테스트 DB
# 엔진도 같은 옵션으로 만든다.

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import get_settings

# pool_pre_ping=True: 풀에서 꺼낸 연결이 DB 재시작 등으로 끊겼는지 쓰기 전에 확인해, 끊긴 연결로 요청이
# 실패하지 않게 한다.
# hide_parameters=True: 이것이 없으면 DB 오류 메시지가 바인드된 값을 나열하는데, 그 값에 사용자가 입력한 글과
# 번역문이 들어 있어 로그에 남게 된다(tests/test_log_privacy.py가 확인). 테스트도 이 옵션으로 엔진을 만든다.
ENGINE_OPTIONS = {"pool_pre_ping": True, "hide_parameters": True}
# asyncpg 드라이버의 비동기 엔진. expire_on_commit=False: commit 뒤에도 객체의 값이 남아 있어, 응답을 만들 때
# 다시 조회(추가 쿼리)하지 않는다. dependencies.get_current_user가 사용자 조회 직후 commit해도 user 객체를
# 계속 쓸 수 있는 것도 이 설정 덕분이다.
engine = create_async_engine(get_settings().database_url, **ENGINE_OPTIONS)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
