"""Does swapping onnxruntime for onnxruntime-gpu change the server's speech recognition? (T82,
docs/experiments.md 14)

faster-whisper runs the server's Silero VAD (STT_VAD_FILTER, T14) on ONNX Runtime's CPU provider. The GPU
trial of Supertonic replaces the onnxruntime package with onnxruntime-gpu in a trial image, so before the
server is measured in that image this compares it with the image it was built from:

    vad         CPU only: faster-whisper's speech timestamps (the server's VAD options) for each clip and for
                the T14 no-speech inputs (3 s of silence, 3 s of quiet noise)
    transcribe  GPU: the server's recognition (its default settings) of the same clips, and T14's four
                no-speech checks (silence and noise, in Korean and in English)
    compare     two records of one kind: the same timestamps or texts for every clip, and for transcribe the
                T14 rule that the four no-speech checks come back empty (4/4)

The clips are the e2e clips (data/e2e/<split>/<language>_<id>.wav, eval/e2e_eval.py). It runs inside the API
images, which have no pyarrow or jiwer. From backend/ (or /app in a container):

    python -m eval.ort_swap_check vad --clips ../data/e2e/validation --out ../work/t82/vad_base.json
    python -m eval.ort_swap_check compare ../work/t82/vad_base.json ../work/t82/vad_gpu.json
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import wave
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
NO_SPEECH_INPUTS = ("silence", "noise")


def no_speech_wav(kind: str, seconds: float = 3.0) -> bytes:
    """T14's audio without speech (eval/stt_eval.py, which needs the eval group): digital silence, or quiet
    white noise (peak amplitude 0.01, fixed seed)."""
    count = int(SAMPLE_RATE * seconds)
    samples = np.zeros(count) if kind == "silence" else np.random.default_rng(0).uniform(-0.01, 0.01, count)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes((samples * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def clips(folder: Path) -> list[tuple[str, bytes, str]]:
    """(file name, audio, language) for each <language>_<id>.wav, in name order."""
    return [
        (path.name, path.read_bytes(), path.name.split("_", 1)[0])
        for path in sorted(Path(folder).glob("*_*.wav"))
    ]


def runtime_info() -> dict:
    """Which ONNX Runtime package is installed, its build and the providers it offers."""
    from importlib import metadata

    import onnxruntime

    packages = []
    for name in ("onnxruntime", "onnxruntime-gpu"):
        try:
            packages.append(f"{name}=={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            pass
    return {
        "package": ", ".join(packages),
        "version": onnxruntime.__version__,
        "build_info": onnxruntime.get_build_info(),
        "available_providers": onnxruntime.get_available_providers(),
    }


def speech_timestamps(audio: bytes) -> list[list[int]]:
    """What faster-whisper's transcribe does with vad_filter=True and no vad_parameters (the server)."""
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    samples = decode_audio(io.BytesIO(audio), sampling_rate=SAMPLE_RATE)
    return [[chunk["start"], chunk["end"]] for chunk in get_speech_timestamps(samples, VadOptions())]


def _write(record: dict, out: Path) -> None:
    record["written"] = datetime.now(UTC).isoformat()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"written to {out}")


def vad(args: argparse.Namespace) -> None:
    record = {
        "kind": "vad",
        "runtime": runtime_info(),
        "clips": {name: speech_timestamps(audio) for name, audio, _ in clips(args.clips)},
        "no_speech": {kind: speech_timestamps(no_speech_wav(kind)) for kind in NO_SPEECH_INPUTS},
    }
    _write(record, args.out)


def server_recognizer():
    """The server's speech recognition as app/services/models.py builds it from the default settings."""
    from app.config import Settings
    from app.services.stt import WhisperSpeechToText

    settings = Settings()
    device = settings.model_device
    return WhisperSpeechToText(
        settings.stt_model,
        vad_filter=settings.stt_vad_filter,
        own_decode=settings.stt_own_decode,
        live_beam_size=settings.live_beam_size,
        live_temperature_fallback=settings.live_temperature_fallback,
        num_workers=settings.stt_num_workers,
        device=device,
        compute_type="float16" if device == "cuda" else "int8",
    )


def transcribe(args: argparse.Namespace) -> None:
    from app.services.interfaces import InvalidAudioError

    stt = server_recognizer()

    def text(audio: bytes, language: str) -> str:
        try:  # no speech found is an empty result, as the server reports it (T14)
            return stt.transcribe(audio, language).text
        except InvalidAudioError:
            return ""

    record = {
        "kind": "transcribe",
        "runtime": runtime_info(),
        "model_name": stt.model_name,
        "options": stt.options,
        "clips": {name: text(audio, language) for name, audio, language in clips(args.clips)},
        "no_speech": {
            f"{language}_{kind}": text(no_speech_wav(kind), language)
            for language in ("ko", "en")
            for kind in NO_SPEECH_INPUTS
        },
    }
    _write(record, args.out)


def compare(before: dict, after: dict) -> dict:
    """Pass when every clip has the same result and so do the no-speech inputs; for transcribe, the no-speech
    checks must also all be empty after the swap (T14: 4/4)."""
    if before["kind"] != after["kind"]:
        raise ValueError(f"the records are of a different kind: {before['kind']} and {after['kind']}")
    if set(before["clips"]) != set(after["clips"]) or set(before["no_speech"]) != set(after["no_speech"]):
        raise ValueError("the two records do not cover the same clips")
    differs = sorted(name for name in before["clips"] if before["clips"][name] != after["clips"][name])
    no_speech_differs = sorted(
        name for name in before["no_speech"] if before["no_speech"][name] != after["no_speech"][name]
    )

    def empty(record: dict) -> int:
        return sum(1 for value in record["no_speech"].values() if not value)

    result = {
        "kind": before["kind"],
        "packages": [before["runtime"].get("package"), after["runtime"].get("package")],
        "clips": len(before["clips"]),
        "same": len(before["clips"]) - len(differs),
        "differs": differs,
        "no_speech_differs": no_speech_differs,
        "no_speech_empty": {"before": empty(before), "after": empty(after), "of": len(after["no_speech"])},
    }
    passed = not differs and not no_speech_differs
    if before["kind"] == "transcribe":
        passed = passed and empty(after) == len(after["no_speech"])
    result["pass"] = passed
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    phases = parser.add_subparsers(dest="phase", required=True)
    for phase in ("vad", "transcribe"):
        run = phases.add_parser(phase)
        run.add_argument("--clips", type=Path, required=True, help="folder of <language>_<id>.wav clips")
        run.add_argument("--out", type=Path, required=True)
    check = phases.add_parser("compare", help="exit status 1 unless the two records agree")
    check.add_argument("before", type=Path)
    check.add_argument("after", type=Path)
    args = parser.parse_args()
    if args.phase == "compare":
        result = compare(
            *(json.loads(path.read_text(encoding="utf-8")) for path in (args.before, args.after))
        )
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(0 if result["pass"] else 1)
    {"vad": vad, "transcribe": transcribe}[args.phase](args)


if __name__ == "__main__":
    main()
