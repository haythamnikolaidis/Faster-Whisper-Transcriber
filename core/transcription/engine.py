"""Headless transcription engine.

A plain function over a faster-whisper model: no Qt, no signals, no thread pool.
``transcribe_with_words`` returns word-level timestamps; ``transcribe_segments``
returns the same words grouped under their segments, with each segment's text.

``faster_whisper`` is imported lazily, so importing this module neither loads
ctranslate2 nor requires ``core.cuda_setup`` to have run. The caller still has
to run ``core.cuda_setup`` before it loads the model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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


@dataclass
class Segment:
    start: float
    end: float
    # The segment text exactly as faster-whisper returns it. It is kept apart
    # from ``words`` because a segment can have text and no words.
    text: str
    words: list[Word]


@dataclass
class SegmentedTranscript:
    segments: list[Segment]
    language: str
    duration: float


def _raise_if_cancelled(cancel: Callable[[], bool] | None) -> None:
    if cancel is not None and cancel():
        raise Cancelled("Transcription cancelled")


def transcribe_segments(
    model: Any,
    audio_path: str | Path,
    *,
    task: str = "transcribe",
    batch_size: int | None = None,
    vad_filter: bool = True,
    vad_parameters: dict | None = None,
    initial_prompt: str | None = None,
    language: str | None = None,
    temperature: float | Sequence[float] = 0.0,
    condition_on_previous_text: bool = True,
    progress: Callable[[float], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> SegmentedTranscript:
    """Transcribe ``audio_path`` and return its segments, each with its words.

    ``task`` is ``"transcribe"`` or ``"translate"``. ``progress`` is called
    after each segment with the fraction of the audio covered so far, from 0.0
    to 1.0, never decreasing. ``cancel`` is polled before the model is touched
    and again after each segment; when it returns True, ``Cancelled`` is raised
    and no further segment is decoded.

    A ``batch_size`` above 1 runs the model through ``BatchedInferencePipeline``.
    """
    _raise_if_cancelled(cancel)

    options: dict[str, Any] = {
        "language": language,
        "word_timestamps": True,
        "vad_filter": vad_filter,
        "vad_parameters": vad_parameters,
        "initial_prompt": initial_prompt,
        "temperature": temperature,
        "condition_on_previous_text": condition_on_previous_text,
    }
    if task != "transcribe":
        # faster-whisper's own default is "transcribe", so leaving the option
        # out is the same request. It keeps the call identical to the one made
        # before ``task`` existed.
        options["task"] = task

    if batch_size is not None and int(batch_size) > 1:
        from faster_whisper import BatchedInferencePipeline

        segments, info = BatchedInferencePipeline(model=model).transcribe(
            str(audio_path), batch_size=int(batch_size), **options
        )
    else:
        segments, info = model.transcribe(str(audio_path), **options)

    collected: list[Segment] = []
    reported = 0.0
    for segment in segments:
        collected.append(
            Segment(
                start=segment.start,
                end=segment.end,
                text=segment.text,
                words=[
                    Word(
                        start=word.start,
                        end=word.end,
                        text=word.word,
                        probability=word.probability,
                    )
                    for word in segment.words or ()
                ],
            )
        )

        if progress is not None and info.duration > 0:
            # The clamp stops rounding at the tail of the audio from reporting
            # more than 100%. The max() keeps the sequence non-decreasing even
            # if a segment ever ends earlier than the one before it.
            reported = max(reported, min(1.0, segment.end / info.duration))
            progress(reported)

        _raise_if_cancelled(cancel)

    return SegmentedTranscript(
        segments=collected, language=info.language, duration=info.duration
    )


def transcribe_with_words(
    model: Any,
    audio_path: str | Path,
    *,
    task: str = "transcribe",
    batch_size: int | None = None,
    vad_filter: bool = True,
    vad_parameters: dict | None = None,
    initial_prompt: str | None = None,
    language: str | None = None,
    temperature: float | Sequence[float] = 0.0,
    condition_on_previous_text: bool = True,
    progress: Callable[[float], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> Transcript:
    """Transcribe ``audio_path`` and return every word with its timing.

    Takes the same arguments as ``transcribe_segments`` and flattens its
    segments into one word list. A segment without words contributes nothing.
    """
    transcript = transcribe_segments(
        model,
        audio_path,
        task=task,
        batch_size=batch_size,
        vad_filter=vad_filter,
        vad_parameters=vad_parameters,
        initial_prompt=initial_prompt,
        language=language,
        temperature=temperature,
        condition_on_previous_text=condition_on_previous_text,
        progress=progress,
        cancel=cancel,
    )
    return Transcript(
        words=[word for segment in transcript.segments for word in segment.words],
        language=transcript.language,
        duration=transcript.duration,
    )
