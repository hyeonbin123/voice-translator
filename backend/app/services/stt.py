"""faster-whisper(CTranslate2)로 돌리는 Whisper 음성 인식."""

# [파일 역할] 음성 인식(STT) 구현. faster-whisper(CTranslate2로 돌리는 Whisper)로 업로드 파일
# 인식(transcribe), 동시통역 자막용 인식(transcribe_live), 두 사람 대화의 언어 판별(detect_language)을
# 제공한다.
# [설계 이유] 측정으로 large-v3-turbo + VAD를 골랐다(README: 한국어 CER 4.57%, 영어 WER 4.95%,
# docs/experiments.md 1절·1-1절). 엔진 안에서 나는 모든 예외는 ModelError로 바꿔 API가 500이 아니라 503으로
# 답하게 하고, 디코딩 실패·말 없음은 따로 구분해 422로 답하게 한다.
# [연결] services/models.load_models가 만들고, services/pipeline.prepare(transcribe),
# routers/live.py(transcribe_live, detect_language), routers/translate.py(detect_language)가 쓴다. 측정 도구
# eval/stt_eval.py 등도 같은 클래스를 쓴다.

import io

# av(PyAV): 업로드 음성 디코딩. faster_whisper.audio의 밑줄 함수들은 faster-whisper 내부 도우미로, 아래
# decode_audio가 원래 구현과 같은 단계를 밟도록 그대로 가져다 쓴다(공개 API가 아니라 faster-whisper 버전이
# 바뀌면 깨질 수 있음).
import av
import numpy as np
from av.error import FFmpegError
from faster_whisper import WhisperModel
from faster_whisper.audio import _group_frames, _ignore_invalid_frames, _resample_frames

from app.services.cuda import add_cuda_dll_dirs
from app.services.interfaces import Language, ModelError, NoSpeechError, Transcript, UndecodableAudioError

# Whisper 입력 규격인 16kHz.
SAMPLE_RATE = 16_000


def decode_audio(audio: bytes) -> np.ndarray:
    """faster-whisper 1.2의 decode_audio에서 gc.collect()만 뺀 것(docs/experiments.md 5절, 후보 C).

    faster-whisper는 디코딩할 때마다 resampler 객체를 풀려고 전체 가비지 컬렉션을 부른다(faster-whisper
    이슈 390). 모델을 올린 프로세스에서는 그 수집이 GIL을 약 0.4초 쥐어 이벤트 루프가 멈춘다. 나머지 단계는
    같다: PyAV로 16kHz 모노 s16으로 푼 뒤 float32(-1~1)로 바꾼다(정수 16비트 표본을 32768로 나눔).
    기본 설정(STT_OWN_DECODE=false)의 transcribe에서는 쓰이지 않고, own_decode를 켠 비교용 설정과
    detect_language가 쓴다.
    """
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    raw = io.BytesIO()
    dtype = None
    with av.open(io.BytesIO(audio), mode="r", metadata_errors="ignore") as container:
        frames = _group_frames(_ignore_invalid_frames(container.decode(audio=0)), 500000)
        for frame in _resample_frames(frames, resampler):
            array = frame.to_ndarray()
            dtype = array.dtype
            raw.write(array)
    del resampler
    return np.frombuffer(raw.getbuffer(), dtype=dtype).astype(np.float32) / 32768.0


class WhisperSpeechToText:
    """app.services.interfaces.SpeechToText(그리고 LiveSpeechToText·LanguageDetector)를 구현한다.

    vad_filter와 no_speech_threshold는 faster-whisper에 그대로 넘긴다(faster-whisper 기본값: False, 0.6).
    retry_without_no_speech를 켜면 빈 결과가 나온 입력을 무음 판정을 끄고 한 번 더 인식한다
    (docs/experiments.md 1-1절의 후보들). own_decode를 켜면 업로드를 faster-whisper 안이 아니라 위의
    decode_audio로 디코딩한다(docs/experiments.md 5절, 후보 C).

    transcribe_live는 동시통역 자막용이다: 같은 옵션을 쓰되 beam 크기는 live_beam_size이고,
    live_temperature_fallback이 꺼져 있으면 temperature 0만 쓴다(docs/experiments.md 8절).
    생성자 기본값과 서비스 설정은 다르다: 서비스에서는 services/models.py가 Settings 값으로 VAD를 켜고,
    자막 인식도 beam 5·temperature fallback 켬(최종 인식과 같은 설정)으로 넘긴다.
    """

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "float16",
        beam_size: int = 5,
        vad_filter: bool = False,
        no_speech_threshold: float | None = 0.6,
        retry_without_no_speech: bool = False,
        own_decode: bool = False,
        live_beam_size: int = 1,
        live_temperature_fallback: bool = False,
        num_workers: int = 1,
    ) -> None:
        # Windows에서 CTranslate2가 pip로 설치된 CUDA DLL을 찾게 한다(services/cuda.py). model_name은
        # 기록·상태 확인에 남는 이름이다.
        add_cuda_dll_dirs()
        self.model_name = f"faster-whisper/{model_size}"
        self.options = {
            "beam_size": beam_size,
            "vad_filter": vad_filter,
            "no_speech_threshold": no_speech_threshold,
        }
        # 자막용 옵션은 최종 인식 옵션에서 beam 크기만 바꾸고, temperature fallback을 끄면 temperature를 0
        # 하나로 고정해 결과가 나빠도 다시 풀지 않는다(빠름).
        self.live_options = {**self.options, "beam_size": live_beam_size}
        if not live_temperature_fallback:
            self.live_options["temperature"] = 0.0
        self.retry_without_no_speech = retry_without_no_speech
        self.own_decode = own_decode
        # num_workers: 여러 스레드의 호출은 그 수만큼 모델 복제가 있어야 나란히 돈다. faster-whisper 자체
        # 기본값 1과 다를 때만 넘긴다.
        replicas = {"num_workers": num_workers} if num_workers != 1 else {}
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type, **replicas)

    # 업로드 파일 인식. 언어는 사용자가 고른 것으로 고정한다. 빈 업로드는 디코딩 불가(422), 인식 결과가 비면
    # 말 없음(422 "No speech was recognized").
    # retry_without_no_speech는 docs/experiments.md 1-1의 후보 중 하나로, 빈 결과면 무음 판정을 끄고 한 번 더
    # 인식한다(서비스 설정에서는 쓰지 않음). duration은 faster-whisper가 알려 준 음성 길이(초)다.
    def transcribe(self, audio: bytes, language: Language) -> Transcript:
        if not audio:
            raise UndecodableAudioError("empty audio")
        source = self._decode(audio) if self.own_decode else audio
        text, duration = self._recognize(source, language, self.options)
        if not text and self.retry_without_no_speech:
            text, duration = self._recognize(source, language, {**self.options, "no_speech_threshold": None})
        if not text:
            raise NoSpeechError("no speech was recognized")
        return Transcript(text=text, language=language, duration_ms=round(duration * 1000))

    # 동시통역 자막용: 브라우저가 보낸 16비트 little-endian PCM을 바로 float32 배열로 바꿔 넘긴다(파일
    # 디코딩·gc.collect 없음). 홀수 바이트는 잘라 반쪽 표본을 없앤다. 말이 없으면 예외 대신 빈 문자열.
    def transcribe_live(self, pcm: bytes, language: Language) -> str:
        samples = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
        if not samples.size:
            return ""
        text, _ = self._recognize(samples, language, self.live_options)
        return text

    def detect_language(self, audio: bytes) -> tuple[Language, float]:
        """한국어와 영어만 남긴 Whisper 언어 판별(docs/experiments.md 10절).

        Whisper의 언어 확률 중 한국어·영어 두 값만 남겨 큰 쪽을 고르고, 확신은 큰 쪽 ÷ 둘의 합(0.5~1.0)으로
        낸다. 이 서비스가 두 언어만 다루므로 다른 언어의 확률은 버린다. 두 확률이 모두 0이면 판단할 수 없어
        확신 0.5로 돌려주고, 그러면 직전 마디가 있을 때 번갈아 처리로 넘어간다. VAD 설정은 서버 인식과 같게
        해 말소리 구간으로만 판별한다.
        """
        if not audio:
            raise UndecodableAudioError("empty audio")
        samples = self._decode(audio)
        try:
            _, _, probabilities = self._model.detect_language(samples, vad_filter=self.options["vad_filter"])
        except Exception as exc:  # noqa: BLE001 - _recognize처럼 엔진을 감싸는 예외 경계
            raise ModelError(f"language detection failed: {type(exc).__name__}") from exc
        found = dict(probabilities)
        ko, en = found.get("ko", 0.0), found.get("en", 0.0)
        if ko + en == 0:
            return "ko", 0.5
        return ("ko" if ko >= en else "en"), max(ko, en) / (ko + en)

    # decode_audio를 감싸 FFmpeg 오류는 디코딩 불가로, 표본이 하나도 없으면 말 없음으로 바꾼다.
    def _decode(self, audio: bytes) -> np.ndarray:
        try:
            samples = decode_audio(audio)
        except FFmpegError as exc:
            raise UndecodableAudioError("the audio could not be decoded") from exc
        if not samples.size:
            raise NoSpeechError("the audio has no samples")
        return samples

    # 실제 인식 한 번. 바이트면 파일처럼 BytesIO로 감싸 faster-whisper가 직접 디코딩하게 하고, 배열이면 그대로
    # 넘긴다. 구간(segment) 텍스트를 공백으로 이어 붙인다.
    # 예외 경계: FFmpeg 오류 → UndecodableAudioError, 이미 ModelError면 그대로, 그 밖의 모든 예외(CUDA 오류
    # 등) → ModelError. 오류 메시지에는 예외 종류 이름만 넣어 입력 내용이 섞이지 않게 한다.
    def _recognize(self, source: bytes | np.ndarray, language: Language, options: dict) -> tuple[str, float]:
        model_input = io.BytesIO(source) if isinstance(source, bytes) else source
        try:
            segments, info = self._model.transcribe(model_input, language=language, **options)
            # segments는 제너레이터다: 모델은 여기서 실제로 돌므로 모델 오류도 여기서 나온다.
            text = " ".join(segment.text.strip() for segment in segments).strip()
        except FFmpegError as exc:
            raise UndecodableAudioError("the audio could not be decoded") from exc
        except ModelError:
            raise
        except Exception as exc:  # noqa: BLE001 - 엔진을 감싸는 예외 경계(CUDA 오류 등)
            # 전에는 500이 됐다. API는 모델 실패를 503으로 답한다.
            raise ModelError(f"the speech recognition model failed: {type(exc).__name__}") from exc
        return text, info.duration
