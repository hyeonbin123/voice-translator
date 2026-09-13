# Conversation detection assets

- `silero_vad_v6.onnx`: byte-for-byte copy of `faster-whisper` 1.2.1's `faster_whisper/assets/silero_vad_v6.onnx` (1,245,151 bytes).
- SHA-256: `4cbf549b8326f60f80f2536d9eefeb450a9abe83365a098031c89719f1be17d2`.
- Model author: Silero Team, [snakers4/silero-vad](https://github.com/snakers4/silero-vad), MIT. Full notice in `SILERO-LICENSE.txt`.
- Browser inference: `onnxruntime-web` 1.29.0, Microsoft, MIT. Full notice from the [v1.29.0 release](https://github.com/microsoft/onnxruntime/blob/v1.29.0/LICENSE) in `ONNXRUNTIME-LICENSE.txt`.
- Vite imports the ONNX file and the installed package's WASM/MJS as local, hashed assets. It emits both license files beside them. No runtime CDN or model download endpoint is used.
- The runtime and model are initialized on the first conversation start, once per page. Frames and recurrent state reset on stop/playback. Inference runs only in the browser; it does not send audio to a VAD service.

Timing follows `backend/eval/eos_eval.py`'s frame rounding: 32 ms frames, two consecutive speech frames at probability >= 0.5, 1000 ms silence rounded to 31 frames (992 ms), 200 ms padding rounded to six frames (192 ms), 29 seconds including pre-roll rounded to 906 frames (28.992 s). Fewer than eight speech frames (256 ms) are discarded. These are the D9 measurement settings; there is no silence or detector selector in the UI.
