import numpy as np
import pytest

from backend.backends import get_model_config
from backend.backends import moss_tts_nano_backend as moss_backend
from backend.backends.moss_tts_nano_backend import MossTTSNanoBackend


def test_moss_model_is_registered():
    config = get_model_config("moss-tts-nano")
    assert config is not None
    assert config.engine == "moss_tts_nano"
    assert config.cache_subdir == "voicebox-moss-tts-nano"
    assert ".voicebox-complete" in config.cache_required_files


def test_moss_phrase_split_removes_problematic_punctuation():
    assert moss_backend._split_moss_phrases(
        "Hey, you alright? Take your time. There is no hurry."
    ) == [
        "Hey",
        "you alright",
        "Take your time",
        "There is no hurry",
    ]
    assert moss_backend._split_moss_phrases("Hey you alright") == ["Hey you alright"]


def test_moss_cache_requires_complete_bundle(monkeypatch, tmp_path):
    backend = MossTTSNanoBackend()
    root = tmp_path / "moss"
    monkeypatch.setattr(backend, "_get_cache_root", lambda: root)

    for relative_path in (*moss_backend._TTS_REQUIRED, *moss_backend._CODEC_REQUIRED):
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
    (root / moss_backend.MOSS_COMPLETE_SENTINEL).write_text("complete\n", encoding="utf-8")

    assert backend._is_model_cached()

    (root / moss_backend._TTS_REQUIRED[0]).unlink()
    assert not backend._is_model_cached()


@pytest.mark.asyncio
async def test_moss_generation_downmixes_stereo(monkeypatch, tmp_path):
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"wav")

    class FakeRuntime:
        def synthesize(self, **kwargs):
            assert kwargs["prompt_audio_path"] == str(reference)
            assert kwargs["streaming"] is True
            assert kwargs["enable_wetext"] is False
            return {
                "waveform": np.asarray(
                    [
                        [1.0, -1.0],
                        [0.5, 0.5],
                        [-0.25, 0.25],
                    ],
                    dtype=np.float32,
                ),
                "sample_rate": 48000,
            }

    backend = MossTTSNanoBackend()
    backend.model = FakeRuntime()
    monkeypatch.setattr(backend, "_get_cache_root", lambda: tmp_path / "cache")

    audio, sample_rate = await backend.generate(
        "Hello there.",
        {"ref_audio": str(reference)},
        seed=123,
    )

    assert sample_rate == 48000
    assert audio.dtype == np.float32
    np.testing.assert_allclose(audio, np.asarray([0.0, 0.5, 0.0], dtype=np.float32))