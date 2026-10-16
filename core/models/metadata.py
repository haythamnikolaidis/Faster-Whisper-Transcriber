from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set

@dataclass
class ModelInfo:
    name: str
    supports_translation: bool
    quantization_overrides: Dict[str, List[str]] | None = None
    # How far the model's word-level timestamps can be trusted: "good" or
    # "unverified". A model that has not been measured is "unverified".
    word_timestamp_quality: str = "unverified"

class ModelMetadata:

    # word_timestamp_quality is seeded from VerseSync's
    # spikes/word_timestamps/RESULTS.md. Section 4 measured large-v3,
    # large-v3-turbo and distil-large-v3 on real sermon audio with word
    # timestamps on, and section 1 recommends distil-large-v3. The remaining
    # error comes from paraphrased mentions, not from the model, so all three
    # are "good" and none is rated higher. The spike's distil-large-v3 is the
    # distil-whisper-large-v3 listed here; section 5 says to re-check its
    # ranking on a second sermon. No other model was measured, so the rest
    # stay "unverified".
    _MODELS: List[ModelInfo] = [
        ModelInfo("tiny", True),
        ModelInfo("tiny.en", False),
        ModelInfo("base", True),
        ModelInfo("base.en", False),
        ModelInfo("small", True),
        ModelInfo("small.en", False),
        ModelInfo("medium", True),
        ModelInfo("medium.en", False),
        ModelInfo("large-v3", True, word_timestamp_quality="good"),
        ModelInfo("large-v3-turbo", False, {"cpu": ["float32"], "cuda": ["float16", "bfloat16", "float32"]}, word_timestamp_quality="good"),
        ModelInfo("distil-whisper-small.en", False, {"cpu": ["float32"], "cuda": ["float16", "bfloat16", "float32"]}),
        ModelInfo("distil-whisper-medium.en", False, {"cpu": ["float32"], "cuda": ["float16", "bfloat16", "float32"]}),
        ModelInfo("distil-whisper-large-v3", False, {"cpu": ["float32"], "cuda": ["float16", "bfloat16", "float32"]}, word_timestamp_quality="good"),
    ]
    
    _MODEL_MAP: Dict[str, ModelInfo] = {m.name: m for m in _MODELS}
    
    @classmethod
    def get_all_model_names(cls) -> List[str]:
        return [m.name for m in cls._MODELS]
    
    @classmethod
    def get_model_info(cls, name: str) -> ModelInfo | None:
        return cls._MODEL_MAP.get(name)
    
    @classmethod
    def supports_translation(cls, name: str) -> bool:
        info = cls._MODEL_MAP.get(name)
        return info.supports_translation if info else False
    
    @classmethod
    def get_word_timestamp_quality(cls, name: str) -> str:
        info = cls._MODEL_MAP.get(name)
        return info.word_timestamp_quality if info else "unverified"

    @classmethod
    def get_quantization_options(cls, model_name: str, device: str, supported_quantizations: Dict[str, List[str]]) -> List[str]:
        info = cls._MODEL_MAP.get(model_name)
        
        if info and info.quantization_overrides:
            options = info.quantization_overrides.get(device, [])
        else:
            options = supported_quantizations.get(device, [])
        
        if device == "cpu":
            options = [opt for opt in options if opt not in ["float16", "bfloat16"]]
        
        return options