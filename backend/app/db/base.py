# [파일 역할] 모든 ORM 모델(app/models/)이 상속하는 SQLAlchemy 선언 기반 클래스와 공용 MetaData를 정의한다.
# [설계 이유] 제약·인덱스 이름 규칙(naming convention)을 한곳에 정해 두면 모델과 Alembic 마이그레이션이 같은
# 이름을 만들어, `alembic check`(tests/test_migrations.py)가 이름 차이로 어긋나지 않고 나중에 제약을 이름으로
# 지우기도 쉽다.
# [연결] app/models/user.py·translation.py·audio_file.py가 Base를 상속하고, alembic/env.py가 Base.metadata를
# 마이그레이션 비교 기준(target_metadata)으로 쓴다.

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase


# SQLAlchemy 2.0 방식의 선언 기반 클래스. 이름 규칙: ix(인덱스)·uq(유일)·fk(외래 키)는 테이블과 첫 열
# 이름으로, ck(검사 제약)는 테이블 이름과 코드에서 준 제약 이름으로, pk는 테이블 이름으로 만든다. 예:
# CheckConstraint(name="mode") → ck_translations_mode.
class Base(DeclarativeBase):
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(table_name)s_%(column_0_name)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )
