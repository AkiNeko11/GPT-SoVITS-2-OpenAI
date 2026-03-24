"""
GPT-SoVITS OpenAI-compatible TTS API Plugin

Provides /v1/audio/speech endpoint that directly loads models and runs inference,
mimicking the OpenAI TTS API interface.
"""

import sys
import os
import io
import time
import logging
import threading

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import numpy as np
import soundfile as sf
import yaml
from pydub import AudioSegment

import inference_cli as tts

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

API_KEY = os.environ.get("API_KEY")

_current_gpt = None
_current_sovits = None
_infer_lock = threading.Lock()

DEFAULT_TOP_K = int(os.environ.get("TOP_K", "15"))
DEFAULT_TOP_P = float(os.environ.get("TOP_P", "1"))
DEFAULT_TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.45"))
DEFAULT_SPEED = float(os.environ.get("SPEED", "1.0"))


def _resolve(p: str) -> str:
    """Resolve a path relative to the project root."""
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(_PROJECT_ROOT, p))


def _config_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def load_voices() -> dict:
    try:
        with open(_config_path(), "r", encoding="utf-8") as f:
            voices = yaml.safe_load(f).get("voices", {})
        for cfg in voices.values():
            m = cfg.get("models", {})
            r = cfg.get("refer", {})
            for k in ("gpt_model_path", "sovits_model_path"):
                if k in m:
                    m[k] = _resolve(m[k])
            if "refer_wav_path" in r:
                r["refer_wav_path"] = _resolve(r["refer_wav_path"])
        return voices
    except Exception as e:
        logger.error("Failed to load config.yaml: %s", e)
        return {}


VOICES = load_voices()
logger.info("Loaded voices: %s", list(VOICES.keys()))
logger.info(
    "Defaults: top_k=%d, top_p=%.2f, temperature=%.2f, speed=%.2f",
    DEFAULT_TOP_K, DEFAULT_TOP_P, DEFAULT_TEMPERATURE, DEFAULT_SPEED,
)


# ─── Helpers ─────────────────────────────────────────────────────────────


def _check_auth():
    """Return an error response tuple if authentication fails, else None."""
    if not API_KEY:
        return None
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer ") and auth[7:] == API_KEY:
        return None
    return jsonify({
        "error": {"message": "Invalid or missing API key", "type": "authentication_error", "code": 401}
    }), 401


def _ensure_models(gpt_path: str, sovits_path: str):
    """Load GPT / SoVITS weights if they differ from the currently loaded ones."""
    global _current_gpt, _current_sovits
    if _current_gpt != gpt_path:
        logger.info("Loading GPT model: %s", gpt_path)
        tts.change_gpt_weights(gpt_path)
        _current_gpt = gpt_path
    if _current_sovits != sovits_path:
        logger.info("Loading SoVITS model: %s", sovits_path)
        tts.change_sovits_weights(sovits_path)
        _current_sovits = sovits_path


def _export_audio(audio_data: np.ndarray, sr: int, fmt: str):
    """Convert numpy audio to the requested format. Returns (BytesIO, mimetype, filename)."""
    wav_buf = io.BytesIO()
    sf.write(wav_buf, audio_data, sr, format="WAV")
    wav_buf.seek(0)

    if fmt == "wav":
        return wav_buf, "audio/wav", "speech.wav"

    if fmt == "pcm":
        pcm = io.BytesIO()
        pcm.write((audio_data * 32767).astype(np.int16).tobytes())
        pcm.seek(0)
        return pcm, "audio/pcm", "speech.pcm"

    codec_map = {
        "mp3":  ("mp3",  "audio/mpeg", "speech.mp3"),
        "opus": ("opus", "audio/opus", "speech.opus"),
        "aac":  ("adts", "audio/aac",  "speech.aac"),
        "flac": ("flac", "audio/flac", "speech.flac"),
    }
    codec, mime, name = codec_map.get(fmt, codec_map["mp3"])
    seg = AudioSegment.from_wav(wav_buf)
    out = io.BytesIO()
    seg.export(out, format=codec)
    out.seek(0)
    return out, mime, name


# ─── Routes ──────────────────────────────────────────────────────────────


@app.route("/v1/audio/speech", methods=["POST"])
def speech():
    """
    OpenAI-compatible TTS endpoint.

    Request body (JSON):
        input            (str, required)  – text to synthesize
        voice            (str, required)  – voice name defined in config.yaml
        speed            (float, optional, default from env)
        response_format  (str, optional, default "mp3") – mp3/wav/flac/opus/aac/pcm
        lang             (str, optional, default "auto") – auto/zh/en/ja/ko/yue
    """
    auth_err = _check_auth()
    if auth_err:
        return auth_err

    data = request.json or {}
    text = data.get("input", "").strip()
    voice = data.get("voice", "")
    speed = float(data.get("speed", DEFAULT_SPEED))
    response_format = data.get("response_format", "mp3")
    lang = data.get("lang", "auto")

    if not text:
        return jsonify({"error": {"message": "'input' is required", "type": "invalid_request_error"}}), 400

    if voice not in VOICES:
        return jsonify({
            "error": {
                "message": f"Unknown voice '{voice}'. Available: {list(VOICES.keys())}",
                "type": "invalid_request_error",
            }
        }), 400

    vcfg = VOICES[voice]
    gpt_path = vcfg["models"]["gpt_model_path"]
    sovits_path = vcfg["models"]["sovits_model_path"]
    refer_wav = vcfg["refer"]["refer_wav_path"]

    with _infer_lock:
        try:
            t0 = time.time()
            _ensure_models(gpt_path, sovits_path)

            refers = [
                tts.get_spepc(tts.hps, refer_wav)
                .to(tts.vq_precision)
                .to(tts.device)
            ]

            result = tts.get_tts_wav(
                text=text,
                top_k=DEFAULT_TOP_K,
                top_p=DEFAULT_TOP_P,
                temperature=DEFAULT_TEMPERATURE,
                speed=speed,
                refer=refers,
                lang=lang,
            )

            if not result:
                return jsonify({"error": {"message": "Synthesis produced no audio", "type": "server_error"}}), 500

            audio = np.concatenate(result, axis=0)
            sr = int(tts.hps.data.sampling_rate)
            buf, mime, fname = _export_audio(audio, sr, response_format)

            elapsed = time.time() - t0
            logger.info("Synthesized %d chars in %.2fs | voice=%s fmt=%s", len(text), elapsed, voice, response_format)
            return send_file(buf, mimetype=mime, as_attachment=True, download_name=fname)

        except Exception as e:
            logger.exception("Synthesis error")
            return jsonify({"error": {"message": str(e), "type": "server_error"}}), 500


@app.route("/v1/models", methods=["GET"])
def list_models():
    """List available voice models (OpenAI-compatible format)."""
    data = [{"id": v, "object": "model", "owned_by": "gpt-sovits"} for v in VOICES]
    return jsonify({"object": "list", "data": data})


@app.route("/v1/voices", methods=["GET"])
def list_voices():
    """List available voices with metadata."""
    data = [{"voice_id": name, "name": name} for name in VOICES]
    return jsonify({"object": "list", "data": data})


@app.route("/v1/config/reload", methods=["POST"])
def reload_config():
    """Hot-reload voice configuration from config.yaml."""
    auth_err = _check_auth()
    if auth_err:
        return auth_err
    global VOICES
    VOICES = load_voices()
    logger.info("Config reloaded, voices: %s", list(VOICES.keys()))
    return jsonify({"status": "ok", "voices": list(VOICES.keys())})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    logger.info("Starting GPT-SoVITS OpenAI-compatible API on %s:%d", host, port)
    app.run(host=host, port=port)
