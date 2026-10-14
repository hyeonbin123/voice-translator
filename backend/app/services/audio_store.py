"""합성한 WAV 파일만 다룬다. 업로드한 녹음 원본은 여기에 절대 남기지 않는다."""

# [파일 역할] 합성한 번역 음성(WAV)을 AUDIO_DIR 폴더에 파일로 저장·읽기·삭제하는 작은 저장소다.
# [설계 이유] 음성 바이트는 DB 대신 디스크에 두고 DB(audio_files.path)에는 파일 이름만 둔다. 파일 입출력은
# 막는(blocking) 작업이라 run_in_threadpool로 이벤트 루프 밖에서 한다. 경로는 늘 AUDIO_DIR 안으로 제한해, DB
# 값이 잘못되더라도 다른 파일을 읽거나 지우지 못하게 한다.
# [연결] services/pipeline.save가 save·delete를, routers/audio.py가 read를, routers/history.py가 delete를
# 부른다. dependencies.get_audio_store가 설정의 audio_dir로 만든다.

import logging
from pathlib import Path
from uuid import UUID

from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)


# directory는 resolve()로 절대 경로(심볼릭 링크 해석)로 바꿔 두어, 아래 경로 비교가 정확하게 되게 한다.
class AudioStore:
    def __init__(self, directory: Path):
        self.directory = directory.resolve()

    # 저장된 상대 경로를 실제 파일 경로로 바꾸되, `..`이나 절대 경로로 AUDIO_DIR 밖을 가리키거나 폴더 자체를
    # 가리키면 ValueError(경로 조작 방지).
    def resolve(self, path: str) -> Path:
        candidate = (self.directory / path).resolve()
        if not candidate.is_relative_to(self.directory) or candidate == self.directory:
            raise ValueError("Audio path is outside AUDIO_DIR")
        return candidate

    # 새 음성을 "<uuid>.wav"로 쓰고 DB에 넣을 상대 이름을 돌려준다. 'xb' 모드는 같은 이름의 파일이 이미 있으면
    # 덮어쓰지 않고 실패한다. 쓰는 도중 실패(디스크 가득 참, 취소 등)하면 반쯤 쓴 파일을 지우고 예외를 다시
    # 올려, 기록 없는 파일이 남지 않게 한다.
    async def save(self, id: UUID, wav: bytes) -> str:
        name = f"{id}.wav"

        def write():
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.resolve(name)
            # 새로 만든 UUID 이름이므로 기존 파일을 덮어쓰지 않는다.
            output = path.open("xb")
            try:
                with output:
                    output.write(wav)
            except BaseException:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Could not remove incomplete audio file", exc_info=True)
                raise

        await run_in_threadpool(write)
        return name

    # 파일 내용을 읽는다. 파일이 없거나 경로가 잘못되면 예외 대신 None을 돌려줘 routers/audio.py가 404로
    # 답하게 한다.
    async def read(self, path: str) -> bytes | None:
        def read():
            try:
                return self.resolve(path).read_bytes()
            except (OSError, ValueError):
                logger.warning("Audio file is unavailable", exc_info=True)
                return None

        return await run_in_threadpool(read)

    # 파일을 지운다. 이미 없으면 그냥 넘어가고(missing_ok), 실패해도 예외를 올리지 않고 경고 로그만 남긴다.
    # 기록 삭제는 DB 쪽이 기준이라 파일 삭제 실패가 요청을 실패시키면 안 되기 때문이다(docs/api.md).
    async def delete(self, path: str) -> None:
        def remove():
            try:
                self.resolve(path).unlink(missing_ok=True)
            except (OSError, ValueError):
                logger.warning("Could not delete audio file", exc_info=True)

        await run_in_threadpool(remove)
