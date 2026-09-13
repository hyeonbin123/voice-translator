"""Generated WAV files only. Original uploads are never retained here."""

import logging
from pathlib import Path
from uuid import UUID

from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)


class AudioStore:
    def __init__(self, directory: Path):
        self.directory = directory.resolve()

    def resolve(self, path: str) -> Path:
        candidate = (self.directory / path).resolve()
        if not candidate.is_relative_to(self.directory) or candidate == self.directory:
            raise ValueError("Audio path is outside AUDIO_DIR")
        return candidate

    async def save(self, id: UUID, wav: bytes) -> str:
        name = f"{id}.wav"

        def write():
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.resolve(name)
            # A generated UUID never overwrites an existing file.
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

    async def read(self, path: str) -> bytes | None:
        def read():
            try:
                return self.resolve(path).read_bytes()
            except (OSError, ValueError):
                logger.warning("Audio file is unavailable", exc_info=True)
                return None

        return await run_in_threadpool(read)

    async def delete(self, path: str) -> None:
        def remove():
            try:
                self.resolve(path).unlink(missing_ok=True)
            except (OSError, ValueError):
                logger.warning("Could not delete audio file", exc_info=True)

        await run_in_threadpool(remove)
