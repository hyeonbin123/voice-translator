# [파일 역할] ORM 모델 세 개를 한곳에서 내보내는 패키지 입구다. 다른 모듈은 `from app.models import User`처럼
# 가져온다.
# [왜 필요한가] 이 패키지를 한 번 불러오면 세 모델이 모두 정의되어 Base.metadata에 등록되고, 서로를 문자열로
# 가리키는 relationship(Translation ↔ AudioFile)도 풀린다. alembic/env.py가 마이그레이션 비교 전에 이 패키지를
# 불러오는 것도 그 때문이다.

from app.models.audio_file import AudioFile
from app.models.translation import Translation
from app.models.user import User

# `from app.models import *`와 린터에 공개 이름을 알려 준다.
__all__ = ["AudioFile", "Translation", "User"]
