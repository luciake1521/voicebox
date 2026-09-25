"""
MOSS-TTS-Nano ONNX backend implementation.

Uses OpenMOSS's ONNX CPU runtime for multilingual zero-shot voice cloning.
The upstream package is installed without dependencies so Voicebox keeps
control of the shared torch/transformers stack; only ONNX Runtime and
SentencePiece are added explicitly.

Model assets are two Hugging Face repositories (TTS + audio tokenizer), so
they are downloaded into one persistent Voicebox-managed bundle under the
configured Hugging Face cache directory.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .base import combine_voice_prompts as _combine_voice_prompts
from .base import model_load_progress

logger = logging.getLogger(__name__)

MOSS_TTS_REPO = "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX"
MOSS_CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX"
MOSS_CACHE_SUBDIR = "voicebox-moss-tts-nano"
MOSS_COMPLETE_SENTINEL = ".voicebox-complete"

_TTS_REQUIRED = (
    "MOSS-TTS-Nano-100M-ONNX/browser_poc_manifest.json",
    "MOSS-TTS-Nano-100M-ONNX/tts_browser_onnx_meta.json",
    "MOSS-TTS-Nano-100M-ONNX/tokenizer.model",
    "MOSS-TTS-Nano-100M-ONNX/moss_tts_global_shared.data",
    "MOSS-TTS-Nano-100M-ONNX/moss_tts_local_shared.data",
)
_CODEC_REQUIRED = (
    "MOSS-Audio-Tokenizer-Nano-ONNX/codec_browser_onnx_meta.json",
    "MOSS-Audio-Tokenizer-Nano-ONNX/moss_audio_tokenizer_encode.data",
    "MOSS-Audio-Tokenizer-Nano-ONNX/moss_audio_tokenizer_decode_shared.data",
)


class MossTTSNanoBackend:
    """CPU-first MOSS-TTS-Nano backend using the upstream ONNX runtime."""

    def __init__(self):
        self.model = None
        self.model_size = "default"
        self._model_load_lock = asyncio.Lock()
        self._generation_lock = asyncio.Lock()

    def is_loaded(self) -> bool:
        return self.model is not None

    def _get_cache_root(self) -> Path:
        from huggingface_hub import constants as hf_constants

        return Path(hf_constants.HF_HUB_CACHE) / MOSS_CACHE_SUBDIR

    def _get_model_path(self, model_size: str = "default") -> str:
        return MOSS_TTS_REPO

    def _is_model_cached(self, model_size: str = "default") -> bool:
        root = self._get_cache_root()
        if not (root / MOSS_COMPLETE_SENTINEL).is_file():
            return False
        for relative_path in (*_TTS_REQUIRED, *_CODEC_REQUIRED):
            if not (root / relative_path).is_file():
                return False
        if any(root.rglob("*.incomplete")):
            return False
        return True

    def _download_assets_sync(self) -> None:
        from huggingface_hub import snapshot_download

        root = self._get_cache_root()
        root.mkdir(parents=True, exist_ok=True)
        sentinel = root / MOSS_COMPLETE_SENTINEL
        sentinel.unlink(missing_ok=True)

        snapshot_download(
            repo_id=MOSS_TTS_REPO,
            local_dir=str(root / "MOSS-TTS-Nano-100M-ONNX"),
            token=None,
            allow_patterns=["*.onnx", "*.data", "*.json", "tokenizer.model"],
        )
        snapshot_download(
            repo_id=MOSS_CODEC_REPO,
            local_dir=str(root / "MOSS-Audio-Tokenizer-Nano-ONNX"),
            token=None,
            allow_patterns=["*.onnx", "*.data", "*.json"],
        )

        missing = [
            relative_path
            for relative_path in (*_TTS_REQUIRED, *_CODEC_REQUIRED)
            if not (root / relative_path).is_file()
        ]
        if missing:
            raise RuntimeError(
                "MOSS-TTS-Nano download completed but required files are missing: "
                + ", ".join(missing)
            )

        sentinel.write_text("complete\n", encoding="utf-8")

    async def load_model(self, model_size: str = "default") -> None:
        if self.model is not None:
            return

        async with self._model_load_lock:
            if self.model is not None:
                return
            await asyncio.to_thread(self._load_model_sync)

    def _load_model_sync(self) -> None:
        is_cached = self._is_model_cached()

        with model_load_progress("moss-tts-nano", is_cached):
            if not is_cached:
                self._download_assets_sync()

            # These are top-level modules shipped by the upstream
            # moss-tts-nano package.
            from onnx_tts_runtime import OnnxTtsRuntime

            root = self._get_cache_root()
            output_dir = root / ".runtime-output"
            output_dir.mkdir(parents=True, exist_ok=True)
            thread_count = max(1, min(os.cpu_count() or 4, 4))

            logger.info(
                "Loading MOSS-TTS-Nano ONNX runtime on CPU (%d threads)...",
                thread_count,
            )
            self.model = OnnxTtsRuntime(
                model_dir=root,
                thread_count=thread_count,
                sample_mode="fixed",
                execution_provider="cpu",
                output_dir=output_dir,
            )

        logger.info("MOSS-TTS-Nano ONNX runtime loaded successfully")

    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            logger.info("MOSS-TTS-Nano unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        if not Path(audio_path).is_file():
            raise FileNotFoundError(f"Reference audio not found: {audio_path}")
        # MOSS performs reference-audio tokenization at synthesis time.
        return {
            "ref_audio": str(audio_path),
            "ref_text": reference_text,
        }, False

    async def combine_voice_prompts(
        self,
        audio_paths: List[str],
        reference_texts: List[str],
    ) -> Tuple[np.ndarray, str]:
        # profiles.py persists the combined result as 24 kHz WAV, so make the
        # returned array match that assumption.
        return await _combine_voice_prompts(
            audio_paths,
            reference_texts,
            sample_rate=24000,
        )

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        await self.load_model()

        ref_audio = voice_prompt.get("ref_audio")
        if not ref_audio or not Path(ref_audio).is_file():
            raise FileNotFoundError(f"Reference audio not found: {ref_audio}")

        async with self._generation_lock:
            return await asyncio.to_thread(
                self._generate_sync,
                text,
                str(ref_audio),
                seed,
            )

    def _generate_sync(
        self,
        text: str,
        ref_audio: str,
        seed: Optional[int],
    ) -> Tuple[np.ndarray, int]:
        assert self.model is not None

        runtime_output_dir = self._get_cache_root() / ".runtime-output"
        runtime_output_dir.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.NamedTemporaryFile(
            prefix="moss-",
            suffix=".wav",
            dir=runtime_output_dir,
            delete=False,
        )
        tmp_path = Path(tmp.name)
        tmp.close()

        try:
            result = self.model.synthesize(
                text=text,
                prompt_audio_path=ref_audio,
                output_audio_path=tmp_path,
                sample_mode="fixed",
                do_sample=True,
                streaming=True,
                enable_wetext=False,
                enable_normalize_tts_text=True,
                seed=seed,
            )

            audio = np.asarray(result["waveform"], dtype=np.float32)
            sample_rate = int(result["sample_rate"])

            # Voicebox's chunk crossfader expects a 1-D waveform. MOSS emits
            # native stereo, so downmix cleanly for the shared pipeline.
            if audio.ndim == 2:
                if audio.shape[1] <= 2:
                    audio = audio.mean(axis=1)
                elif audio.shape[0] <= 2:
                    audio = audio.mean(axis=0)
                else:
                    raise RuntimeError(f"Unexpected MOSS waveform shape: {audio.shape}")
            elif audio.ndim != 1:
                audio = np.squeeze(audio)
                if audio.ndim != 1:
                    raise RuntimeError(f"Unexpected MOSS waveform shape: {audio.shape}")

            return audio.astype(np.float32, copy=False), sample_rate
        finally:
            tmp_path.unlink(missing_ok=True)
