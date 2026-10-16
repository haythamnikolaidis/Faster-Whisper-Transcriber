"""Headless transcription engine.

A plain function over a faster-whisper model: no Qt, no signals, no thread pool.
It returns word-level timestamps, which ``TranscriptionService`` discards.

``faster_whisper`` is imported lazily, so importing this module neither loads
ctranslate2 nor requires ``core.cuda_setup`` to have run. The caller still has
to run ``core.cuda_setup`` before it loads the model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.exceptions import TranscriberError


class Cancelled(TranscriberError):
    """The ``cancel`` callback reported True, so the transcription was abandoned."""


@dataclass
class Word:
    start: float
    end: float
    # Exactly as faster-whisper returns it, leading space included, so that
    # "".join(word.text for word in words) rebuilds the segment text.
    text: str
    probability: float


@dataclass
class Transcript:
    words: list[Word]
    language: str
    duration: float


def _raise_if_cancelled(cancel: Callable[[], bool] | None) -> None:
    if cancel is not None and cancel():
        raise Cancelled("Transcription cancelled")


def transcribe_with_words(
    model: Any,
    audio_path: str | Path,
    *,
    batch_size: int | None = None,
    vad_filter: bool = True,
    vad_parameters: dict | None = None,
    initial_prompt: str | None = None,
    language: str | None = None,
    temperature: float = 0.0,
    condition_on_previous_text: bool = True,
    progress: Callable[[float], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> Transcript:
    """Transcribe ``audio_path`` and return every word with its timing.

    ``progress`` is called after each segment with the fraction of the audio
    covered so far, from 0.0 to 1.0, never decreasing. ``cancel`` is polled
    before the model is touched and again after each segment; when it returns
    True, ``Cancelled`` is raised and no further segment is decoded.

    A ``batch_size`` above 1 runs the model through ``BatchedInferencePipeline``.
    """
    _raise_if_cancelled(cancel)

    options = {
        "language": language,
        "word_timestamps": True,
        "vad_filter": vad_filter,
        "vad_parameters": vad_parameters,
        "initial_prompt": initial_prompt,
        "temperature": temperature,
        "condition_on_previous_text": condition_on_previous_text,
    }

    if batch_size is not None and int(batch_size) > 1:
        from faster_whisper import BatchedInferencePipeline

        segments, info = BatchedInferencePipeline(model=model).transcribe(
            str(audio_path), batch_size=int(batch_size), **options
        )
    else:
        segments, info = model.transcribe(str(audio_path), **options)

    words: list[Word] = []
    reported = 0.0
    for segment in segments:
        for word in segment.words or ():
            words.append(
                Word(
                    start=word.start,
                    end=word.end,
                    text=word.word,
                    probability=word.probability,
                )
            )

        if progress is not None and info.duration > 0:
            # The clamp stops rounding at the tail of the audio from reporting
            # more than 100%. The max() keeps the sequence non-decreasing even
            # if a segment ever ends earlier than the one before it.
            reported = max(reported, min(1.0, segment.end / info.duration))
            progress(reported)

        _raise_if_cancelled(cancel)

    return Transcript(words=words, language=info.language, duration=info.duration)
