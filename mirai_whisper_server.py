#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import ipaddress
import os
import re
import socket
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
import uvicorn
import whisper
import numpy as np
import webrtcvad
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from whisper.audio import SAMPLE_RATE
from whisper.model import LayerNorm


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "large-v3-turbo"
DEFAULT_MAX_UPLOAD_MIB = 25
MIN_AUDIO_SECONDS = 0.25
VAD_FRAME_MS = 30
MIN_VOICED_FRAMES = 3
MIN_VOICED_RATIO = 0.12
REPETITION_MIN_COUNT = 6


def default_whisper_model_dir() -> Path:
    """Use colocated models when present, otherwise Whisper's standard cache."""
    configured = (
        os.getenv("MIRAI_WHISPER_MODEL_DIR", "").strip()
        or os.getenv("WHISPER_ASR_MODELS", "").strip()
    )
    if configured:
        return Path(configured).expanduser().resolve()

    local_models = APP_ROOT / "models"
    if local_models.is_dir():
        return local_models.resolve()

    cache_root = os.getenv("XDG_CACHE_HOME", "").strip()
    if cache_root:
        return (Path(cache_root).expanduser() / "whisper").resolve()
    return (Path.home() / ".cache" / "whisper").resolve()


class ServerSettings:
    host = "0.0.0.0"
    port = 8000
    model = DEFAULT_MODEL
    model_dir = default_whisper_model_dir()
    device = "auto"
    max_upload_bytes = DEFAULT_MAX_UPLOAD_MIB * 1024 * 1024
    api_key = ""
    glossary_path = APP_ROOT / "glossary.txt"


settings = ServerSettings()


def discover_lan_ipv4_addresses() -> list[str]:
    """Return usable local IPv4 addresses, with the primary route first."""
    addresses: list[str] = []
    seen: set[str] = set()

    def add(address: str) -> None:
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            return
        if (
            parsed.version != 4
            or parsed.is_loopback
            or parsed.is_link_local
            or parsed.is_unspecified
        ):
            return
        normalized = str(parsed)
        if normalized not in seen:
            seen.add(normalized)
            addresses.append(normalized)

    # A UDP connect selects the interface Windows would use for this route.
    # It does not transmit application data and works without a reachable peer.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route_socket:
            route_socket.connect(("192.0.2.1", 9))
            add(str(route_socket.getsockname()[0]))
    except OSError:
        pass

    try:
        host_name = socket.gethostname()
        for entry in socket.getaddrinfo(host_name, None, socket.AF_INET):
            add(str(entry[4][0]))
    except OSError:
        pass

    return addresses


def print_connection_urls() -> None:
    host = settings.host.strip()
    if host in {"0.0.0.0", ""}:
        addresses = discover_lan_ipv4_addresses()
        print("", flush=True)
        print("MiRAI Head connection URL(s):", flush=True)
        if addresses:
            for address in addresses:
                print(f"  http://{address}:{settings.port}", flush=True)
        else:
            print("  No LAN IPv4 address was detected. Run ipconfig to check it.", flush=True)
        print("", flush=True)
        return

    print("", flush=True)
    print(f"Server URL: http://{host}:{settings.port}", flush=True)
    print("", flush=True)


class GlossaryProvider:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cache: dict[Path, tuple[tuple[int, int], list[str]]] = {}

    def language_path(self, language: str | None) -> Path | None:
        if not language or not re.fullmatch(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", language):
            return None

        base_path = settings.glossary_path
        return base_path.with_name(
            f"{base_path.stem}.{language}{base_path.suffix}"
        )

    def select_path(self, language: str | None) -> Path | None:
        localized_path = self.language_path(language)
        if localized_path is not None and localized_path.is_file():
            return localized_path
        if settings.glossary_path.is_file():
            return settings.glossary_path
        return None

    def terms(self, language: str | None = None) -> tuple[list[str], Path | None]:
        path = self.select_path(language)
        if path is None:
            return [], None

        try:
            stat = path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
        except FileNotFoundError:
            return [], None

        with self._lock:
            cached = self._cache.get(path)
            if cached is not None and signature == cached[0]:
                return list(cached[1]), path

            terms: list[str] = []
            seen: set[str] = set()
            for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
                term = raw_line.strip()
                if not term or term.startswith("#"):
                    continue
                key = term.casefold()
                if key not in seen:
                    seen.add(key)
                    terms.append(term)

            self._cache[path] = (signature, terms)
            print(f"Glossary loaded: {len(terms)} term(s) from {path}", flush=True)
            return list(terms), path

    def compose_prompt(
        self,
        request_prompt: str | None,
        language: str | None,
    ) -> tuple[str | None, int, str]:
        terms, path = self.terms(language)
        parts: list[str] = []
        if terms:
            parts.append("、".join(terms))
        if request_prompt and request_prompt.strip():
            parts.append(request_prompt.strip())
        return ("\n".join(parts) or None), len(terms), path.name if path else ""


glossary = GlossaryProvider()


def audio_contains_speech(audio_path: Path) -> tuple[bool, dict[str, Any]]:
    """Run a cheap speech gate before the expensive Whisper inference."""
    audio = whisper.load_audio(str(audio_path))
    duration = float(len(audio)) / SAMPLE_RATE
    if duration < MIN_AUDIO_SECONDS:
        return False, {"duration": duration, "reason": "too_short"}

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    if peak < 0.01 or rms < 0.0015:
        return False, {
            "duration": duration,
            "peak": peak,
            "rms": rms,
            "reason": "low_energy",
        }

    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16).tobytes()
    frame_bytes = (
        SAMPLE_RATE * VAD_FRAME_MS // 1000 * 2
    )
    vad = webrtcvad.Vad(2)
    total_frames = len(pcm) // frame_bytes
    voiced_frames = 0
    for offset in range(0, total_frames * frame_bytes, frame_bytes):
        if vad.is_speech(
            pcm[offset : offset + frame_bytes],
            SAMPLE_RATE,
        ):
            voiced_frames += 1

    voiced_ratio = voiced_frames / max(1, total_frames)
    accepted = (
        voiced_frames >= MIN_VOICED_FRAMES
        and voiced_ratio >= MIN_VOICED_RATIO
    )
    return accepted, {
        "duration": duration,
        "peak": peak,
        "rms": rms,
        "voiced_frames": voiced_frames,
        "total_frames": total_frames,
        "voiced_ratio": voiced_ratio,
        "reason": "speech" if accepted else "vad_rejected",
    }


def normalize_for_repetition(text: str) -> str:
    return "".join(character.casefold() for character in text if character.isalnum())


def has_excessive_repetition(text: str) -> bool:
    words = re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)
    if len(words) >= REPETITION_MIN_COUNT:
        most_common = max(words.count(word) for word in set(words))
        if most_common >= REPETITION_MIN_COUNT and most_common / len(words) >= 0.6:
            return True

    compact = normalize_for_repetition(text)
    if len(compact) < REPETITION_MIN_COUNT:
        return False
    max_unit_length = min(16, len(compact) // REPETITION_MIN_COUNT)
    for unit_length in range(1, max_unit_length + 1):
        for start in range(0, min(unit_length, len(compact))):
            unit = compact[start : start + unit_length]
            if not unit:
                continue
            repeats = 1
            cursor = start + unit_length
            while compact[cursor : cursor + unit_length] == unit:
                repeats += 1
                cursor += unit_length
            if repeats >= REPETITION_MIN_COUNT and (
                repeats * unit_length >= len(compact) * 0.6
            ):
                return True
    return False


def result_diagnostics(result: dict[str, Any]) -> dict[str, Any]:
    text = str(result.get("text", "")).strip()
    segments = result.get("segments") or []
    compression_ratios = [
        float(segment.get("compression_ratio", 0.0)) for segment in segments
    ]
    average_logprobs = [
        float(segment.get("avg_logprob", 0.0)) for segment in segments
    ]
    no_speech_probs = [
        float(segment.get("no_speech_prob", 0.0)) for segment in segments
    ]
    maximum_compression = max(compression_ratios, default=0.0)
    minimum_logprob = min(average_logprobs, default=0.0)
    maximum_no_speech = max(no_speech_probs, default=0.0)
    repeated = has_excessive_repetition(text)
    low_confidence = bool(segments) and minimum_logprob < -1.0
    likely_no_speech = bool(segments) and (
        maximum_no_speech > 0.6 and minimum_logprob < -0.8
    )
    return {
        "repeated": repeated,
        "maximum_compression_ratio": maximum_compression,
        "minimum_avg_logprob": minimum_logprob,
        "maximum_no_speech_prob": maximum_no_speech,
        "low_confidence": low_confidence,
        "likely_no_speech": likely_no_speech,
        "retry_recommended": (
            not repeated
            and not likely_no_speech
            and (maximum_compression > 2.4 or low_confidence)
        ),
        "reject": repeated or likely_no_speech,
    }


def normalize_model_name(value: str | None) -> str:
    name = (value or settings.model).strip()
    if not name:
        name = settings.model
    if name.lower() == "whisper-1":
        return settings.model
    return name


def resolve_model(value: str) -> tuple[str, str]:
    """Return the load argument and stable model id without allowing arbitrary paths."""
    name = normalize_model_name(value)
    candidate_name = Path(name).name
    candidates = [settings.model_dir / candidate_name]
    if not candidate_name.lower().endswith(".pt"):
        candidates.append(settings.model_dir / f"{candidate_name}.pt")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve()), candidate.stem

    if name in whisper.available_models():
        return name, name

    available = ", ".join(sorted(whisper.available_models()))
    raise ValueError(
        f"Unknown model '{name}'. Install its .pt file in {settings.model_dir} "
        f"or use one of: {available}"
    )


class ModelManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._model: Any = None
        self._model_id = ""
        self._device = "cpu"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def device(self) -> str:
        return self._device

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def choose_device(self) -> str:
        requested = settings.device.lower()
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but PyTorch cannot access the GPU")
        return requested

    def load_cuda_model(self, load_argument: str, model_id: str) -> Any:
        checkpoint_path = Path(load_argument)
        if checkpoint_path.is_file():
            kwargs = {"weights_only": True} if torch.__version__ >= "1.13" else {}
            with checkpoint_path.open("rb") as checkpoint_file:
                checkpoint = torch.load(checkpoint_file, map_location="cpu", **kwargs)
            dimensions = whisper.ModelDimensions(**checkpoint["dims"])
            previous_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.float16)
                model = whisper.Whisper(dimensions)
            finally:
                torch.set_default_dtype(previous_dtype)
            model.load_state_dict(checkpoint["model_state_dict"])
            del checkpoint

            alignment_heads = getattr(whisper, "_ALIGNMENT_HEADS", {}).get(model_id)
            if alignment_heads is not None:
                model.set_alignment_heads(alignment_heads)
        else:
            model = whisper.load_model(
                load_argument,
                device="cpu",
                download_root=str(settings.model_dir),
            )

        model = model.half()
        for layer in model.modules():
            if isinstance(layer, LayerNorm):
                layer.float()
        return model.to("cuda")

    def load(self, requested_model: str | None) -> Any:
        load_argument, model_id = resolve_model(requested_model or settings.model)
        with self._lock:
            if self._model is not None and self._model_id == model_id:
                return self._model

            device = self.choose_device()
            print(f"Loading Whisper model '{model_id}' on {device} ...", flush=True)
            previous = self._model
            self._model = None
            self._model_id = ""
            if previous is not None:
                del previous
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if device == "cuda":
                # Build local checkpoints as FP16 on CPU. This avoids the
                # temporary FP32 copy that can exhaust RAM or WDDM's GPU
                # allocation budget.
                self._model = self.load_cuda_model(load_argument, model_id)
            else:
                self._model = whisper.load_model(
                    load_argument,
                    device=device,
                    download_root=str(settings.model_dir),
                )
            self._model_id = model_id
            self._device = device
            print(f"Whisper model '{model_id}' is ready.", flush=True)
            return self._model

    def transcribe(
        self,
        audio_path: Path,
        requested_model: str,
        language: str | None,
        prompt: str | None,
        temperature: float,
    ) -> tuple[dict[str, Any], str]:
        with self._lock:
            model = self.load(requested_model)
            result = model.transcribe(
                str(audio_path),
                language=language,
                task="transcribe",
                fp16=(self._device == "cuda"),
                verbose=False,
                initial_prompt=prompt or None,
                temperature=temperature,
                condition_on_previous_text=False,
            )
            return result, self._model_id


models = ModelManager()


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    models.load(settings.model)
    print_connection_urls()
    yield


app = FastAPI(
    title="MiRAI Whisper OpenAI-Compatible Server",
    version="1.0.0",
    lifespan=lifespan,
)


def authorize(authorization: str | None) -> None:
    if not settings.api_key:
        return
    expected = f"Bearer {settings.api_key}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


@app.get("/health")
def health() -> dict[str, Any]:
    glossary_terms, glossary_path = glossary.terms()
    return {
        "status": "ok" if models.loaded else "loading",
        "model": models.model_id or settings.model,
        "device": models.device,
        "glossary_terms": len(glossary_terms),
        "glossary_file": glossary_path.name if glossary_path else "",
    }


@app.get("/v1/models")
def list_models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    authorize(authorization)
    local_models = sorted(path.stem for path in settings.model_dir.glob("*.pt"))
    model_ids = sorted(set(local_models + list(whisper.available_models())))
    return {
        "object": "list",
        "data": [
            {"id": model_id, "object": "model", "owned_by": "local"}
            for model_id in model_ids
        ],
    }


def serializable_segments(result: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for segment in result.get("segments") or []:
        output.append(
            {
                "id": int(segment.get("id", len(output))),
                "start": float(segment.get("start", 0.0)),
                "end": float(segment.get("end", 0.0)),
                "text": str(segment.get("text", "")),
                "avg_logprob": float(segment.get("avg_logprob", 0.0)),
                "compression_ratio": float(segment.get("compression_ratio", 0.0)),
                "no_speech_prob": float(segment.get("no_speech_prob", 0.0)),
            }
        )
    return output


@app.post("/v1/audio/transcriptions")
def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(default=DEFAULT_MODEL),
    language: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    prompt: str | None = Form(default=None),
    temperature: float = Form(default=0.0),
    authorization: str | None = Header(default=None),
):
    authorize(authorization)
    response_format = response_format.strip().lower()
    if response_format not in {"json", "text", "verbose_json"}:
        raise HTTPException(
            status_code=400,
            detail="response_format must be json, text, or verbose_json",
        )
    if not 0.0 <= temperature <= 1.0:
        raise HTTPException(status_code=400, detail="temperature must be between 0 and 1")

    normalized_language = (language or "").strip().lower()
    if normalized_language in {"", "auto"}:
        normalized_language = None

    suffix = Path(file.filename or "speech.wav").suffix.lower() or ".wav"
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)
            copied = 0
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                copied += len(chunk)
                if copied > settings.max_upload_bytes:
                    raise HTTPException(status_code=413, detail="Audio file is too large")
                temp_file.write(chunk)

        if copied == 0:
            raise HTTPException(status_code=400, detail="Audio file is empty")

        try:
            combined_prompt, glossary_count, glossary_file = glossary.compose_prompt(
                prompt,
                normalized_language,
            )
            speech_detected, vad_diagnostics = audio_contains_speech(temp_path)
            result: dict[str, Any] = {}
            diagnostics: dict[str, Any] = {}
            used_model = models.model_id or normalize_model_name(model)
            retry_used = False
            suppressed_reason = ""

            if not speech_detected:
                suppressed_reason = str(vad_diagnostics.get("reason", "vad_rejected"))
            else:
                result, used_model = models.transcribe(
                    temp_path,
                    model,
                    normalized_language,
                    combined_prompt,
                    temperature,
                )
                diagnostics = result_diagnostics(result)

                if diagnostics["reject"]:
                    suppressed_reason = (
                        "repetition"
                        if diagnostics["repeated"]
                        else "likely_no_speech"
                    )
                elif diagnostics["retry_recommended"]:
                    # Keep normal requests to one inference. Only uncertain output
                    # gets one retry, without glossary bias, at a moderate temperature.
                    if temperature <= 0.2:
                        retry_used = True
                        retry_result, used_model = models.transcribe(
                            temp_path,
                            model,
                            normalized_language,
                            None,
                            0.4,
                        )
                        retry_diagnostics = result_diagnostics(retry_result)
                        if (
                            retry_diagnostics["reject"]
                            or retry_diagnostics["retry_recommended"]
                        ):
                            result = retry_result
                            diagnostics = retry_diagnostics
                            suppressed_reason = "retry_failed_quality_check"
                        else:
                            result = retry_result
                            diagnostics = retry_diagnostics
                    else:
                        suppressed_reason = "low_quality"
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            print(f"Transcription failed: {type(exc).__name__}: {exc}", flush=True)
            raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc

        text = "" if suppressed_reason else str(result.get("text", "")).strip()
        if suppressed_reason:
            print(
                "Transcription suppressed: "
                f"reason={suppressed_reason}, retry={retry_used}, "
                f"duration={vad_diagnostics.get('duration', 0.0):.2f}s, "
                f"voiced_ratio={vad_diagnostics.get('voiced_ratio', 0.0):.2f}",
                flush=True,
            )
        if response_format == "text":
            return PlainTextResponse(text)
        if response_format == "verbose_json":
            return JSONResponse(
                {
                    "task": "transcribe",
                    "language": str(result.get("language", normalized_language or "")),
                    "text": text,
                    "model": used_model,
                    "glossary_terms": glossary_count,
                    "glossary_file": glossary_file,
                    "segments": [] if suppressed_reason else serializable_segments(result),
                    "suppressed": bool(suppressed_reason),
                    "suppressed_reason": suppressed_reason,
                    "retry_used": retry_used,
                    "vad": vad_diagnostics,
                    "quality": diagnostics,
                }
            )
        return JSONResponse({"text": text})
    finally:
        try:
            file.file.close()
        except Exception:
            pass
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible local Whisper server for MiRAI Head"
    )
    parser.add_argument("--host", default=os.getenv("MIRAI_WHISPER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MIRAI_WHISPER_PORT", "8000")))
    parser.add_argument("--model", default=os.getenv("MIRAI_WHISPER_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--model-dir",
        default=None,
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default=os.getenv("MIRAI_WHISPER_DEVICE", "auto"),
    )
    parser.add_argument(
        "--max-upload-mib",
        type=int,
        default=int(os.getenv("MIRAI_WHISPER_MAX_UPLOAD_MIB", str(DEFAULT_MAX_UPLOAD_MIB))),
    )
    parser.add_argument(
        "--glossary",
        default=os.getenv("MIRAI_WHISPER_GLOSSARY", str(APP_ROOT / "glossary.txt")),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings.host = args.host
    settings.port = args.port
    settings.model = normalize_model_name(args.model)
    settings.model_dir = (
        Path(args.model_dir).expanduser().resolve()
        if args.model_dir
        else default_whisper_model_dir()
    )
    settings.device = args.device
    settings.max_upload_bytes = max(1, args.max_upload_mib) * 1024 * 1024
    settings.glossary_path = Path(args.glossary).expanduser().resolve()
    settings.api_key = os.getenv("MIRAI_WHISPER_API_KEY", "").strip()
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
