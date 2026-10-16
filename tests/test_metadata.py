from __future__ import annotations

from core.models.metadata import ModelInfo, ModelMetadata

MEASURED = {"large-v3", "large-v3-turbo", "distil-whisper-large-v3"}


def test_models_the_spike_measured_are_good():
    for name in MEASURED:
        assert ModelMetadata.get_word_timestamp_quality(name) == "good"
        info = ModelMetadata.get_model_info(name)
        assert info is not None
        assert info.word_timestamp_quality == "good"


def test_every_other_model_is_unverified():
    others = set(ModelMetadata.get_all_model_names()) - MEASURED

    assert others
    for name in others:
        assert ModelMetadata.get_word_timestamp_quality(name) == "unverified"


def test_an_unknown_model_is_unverified():
    assert ModelMetadata.get_word_timestamp_quality("not-a-model") == "unverified"


def test_a_model_info_defaults_to_unverified():
    assert ModelInfo("tiny", True).word_timestamp_quality == "unverified"


def test_existing_metadata_is_unchanged():
    assert ModelMetadata.supports_translation("large-v3") is True
    assert ModelMetadata.supports_translation("large-v3-turbo") is False
    assert ModelMetadata.get_quantization_options(
        "large-v3-turbo", "cuda", {"cuda": ["int8"]}
    ) == ["float16", "bfloat16", "float32"]
    assert ModelMetadata.get_quantization_options(
        "large-v3", "cpu", {"cpu": ["float16", "float32"]}
    ) == ["float32"]
