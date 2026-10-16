from __future__ import annotations

import os
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from core.transcription.engine import (
    Cancelled,
    Segment,
    SegmentedTranscript,
    Transcript,
    Word,
    transcribe_segments,
    transcribe_with_words,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class FakeWord:
    start: float
    end: float
    word: str
    probability: float


@dataclass
class FakeSegment:
    start: float
    end: float
    text: str
    words: list[FakeWord] | None


@dataclass
class FakeInfo:
    language: str
    duration: float


class FakeModel:
    """Stands in for WhisperModel: transcribe() returns a lazy segment generator."""

    def __init__(self, segments, duration=10.0, language="en"):
        self._segments = segments
        self.info = FakeInfo(language=language, duration=duration)
        self.calls = []
        self.pulled = 0

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return self._generate(), self.info

    def _generate(self):
        for segment in self._segments:
            self.pulled += 1
            yield segment


def two_segments():
    return [
        FakeSegment(
            0.0,
            2.5,
            " In the beginning",
            [
                FakeWord(0.0, 0.4, " In", 0.98),
                FakeWord(0.4, 0.7, " the", 0.91),
                FakeWord(0.7, 2.5, " beginning", 0.64),
            ],
        ),
        FakeSegment(
            2.5,
            5.0,
            " God created",
            [
                FakeWord(2.5, 3.1, " God", 0.88),
                FakeWord(3.1, 5.0, " created", 0.73),
            ],
        ),
    ]


def test_words_keep_timing_text_and_probability():
    model = FakeModel(two_segments(), duration=5.0, language="en")

    transcript = transcribe_with_words(model, "sermon.wav")

    assert isinstance(transcript, Transcript)
    assert transcript.language == "en"
    assert transcript.duration == 5.0
    assert transcript.words == [
        Word(0.0, 0.4, " In", 0.98),
        Word(0.4, 0.7, " the", 0.91),
        Word(0.7, 2.5, " beginning", 0.64),
        Word(2.5, 3.1, " God", 0.88),
        Word(3.1, 5.0, " created", 0.73),
    ]
    assert "".join(word.text for word in transcript.words) == " In the beginning God created"


def test_segment_without_words_contributes_nothing():
    segments = [
        FakeSegment(0.0, 1.0, " ...", None),
        FakeSegment(1.0, 2.0, " Amen", [FakeWord(1.0, 2.0, " Amen", 0.9)]),
    ]

    transcript = transcribe_with_words(FakeModel(segments, duration=2.0), "a.wav")

    assert transcript.words == [Word(1.0, 2.0, " Amen", 0.9)]


def test_no_segments_gives_empty_transcript():
    transcript = transcribe_with_words(FakeModel([], duration=0.0), "silence.wav")

    assert transcript == Transcript(words=[], language="en", duration=0.0)


def test_model_is_called_with_word_timestamps_and_defaults():
    model = FakeModel(two_segments())

    transcribe_with_words(model, Path("sermon.wav"))

    ((audio, kwargs),) = model.calls
    assert audio == str(Path("sermon.wav"))
    assert kwargs == {
        "language": None,
        "word_timestamps": True,
        "vad_filter": True,
        "vad_parameters": None,
        "initial_prompt": None,
        "temperature": 0.0,
        "condition_on_previous_text": True,
    }


def test_options_are_passed_through():
    model = FakeModel(two_segments())

    transcribe_with_words(
        model,
        "sermon.wav",
        vad_filter=False,
        vad_parameters={"min_silence_duration_ms": 500},
        initial_prompt="Genesis Exodus",
        language="en",
        temperature=0.2,
        condition_on_previous_text=False,
    )

    ((_, kwargs),) = model.calls
    assert kwargs["vad_filter"] is False
    assert kwargs["vad_parameters"] == {"min_silence_duration_ms": 500}
    assert kwargs["initial_prompt"] == "Genesis Exodus"
    assert kwargs["language"] == "en"
    assert kwargs["temperature"] == 0.2
    assert kwargs["condition_on_previous_text"] is False
    assert kwargs["word_timestamps"] is True


def test_progress_is_the_fraction_of_the_duration_after_each_segment():
    seen = []

    transcribe_with_words(
        FakeModel(two_segments(), duration=10.0), "a.wav", progress=seen.append
    )

    assert seen == [0.25, 0.5]


def test_progress_is_monotonic_and_stays_within_zero_and_one():
    segments = [
        FakeSegment(0.0, 4.0, " a", []),
        FakeSegment(4.0, 3.0, " b", []),
        FakeSegment(3.0, 10.4, " c", []),
        FakeSegment(10.4, 10.4, " d", []),
    ]
    seen = []

    transcribe_with_words(
        FakeModel(segments, duration=10.0), "a.wav", progress=seen.append
    )

    assert seen == [0.4, 0.4, 1.0, 1.0]
    assert seen == sorted(seen)


def test_progress_is_optional():
    transcript = transcribe_with_words(FakeModel(two_segments()), "a.wav")

    assert len(transcript.words) == 5


def test_cancel_stops_before_the_next_segment_is_decoded():
    model = FakeModel(two_segments() + two_segments())
    seen = []

    with pytest.raises(Cancelled):
        transcribe_with_words(
            model,
            "a.wav",
            progress=seen.append,
            cancel=lambda: len(seen) >= 2,
        )

    assert len(seen) == 2
    assert model.pulled == 2


def test_cancel_set_before_start_never_touches_the_model():
    model = FakeModel(two_segments())

    with pytest.raises(Cancelled):
        transcribe_with_words(model, "a.wav", cancel=lambda: True)

    assert model.calls == []


def test_cancel_that_stays_false_lets_the_run_finish():
    transcript = transcribe_with_words(
        FakeModel(two_segments()), "a.wav", cancel=lambda: False
    )

    assert len(transcript.words) == 5


@pytest.fixture
def fake_batched_pipeline(monkeypatch):
    created = {}

    class FakeBatchedInferencePipeline:
        def __init__(self, model):
            created["model"] = model

        def transcribe(self, audio, **kwargs):
            created["audio"] = audio
            created["kwargs"] = kwargs
            return iter(two_segments()), FakeInfo(language="en", duration=5.0)

    module = types.ModuleType("faster_whisper")
    module.BatchedInferencePipeline = FakeBatchedInferencePipeline
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return created


def test_batch_size_above_one_uses_the_batched_pipeline(fake_batched_pipeline):
    model = FakeModel([])

    transcript = transcribe_with_words(model, "a.wav", batch_size=8)

    assert fake_batched_pipeline["model"] is model
    assert fake_batched_pipeline["audio"] == "a.wav"
    assert fake_batched_pipeline["kwargs"]["batch_size"] == 8
    assert fake_batched_pipeline["kwargs"]["word_timestamps"] is True
    assert model.calls == []
    assert len(transcript.words) == 5


@pytest.mark.parametrize("batch_size", [None, 0, 1])
def test_small_batch_size_calls_the_model_directly(fake_batched_pipeline, batch_size):
    model = FakeModel(two_segments())

    transcribe_with_words(model, "a.wav", batch_size=batch_size)

    assert fake_batched_pipeline == {}
    ((_, kwargs),) = model.calls
    assert "batch_size" not in kwargs


def test_task_defaults_to_transcribe_and_is_not_sent_to_the_model():
    model = FakeModel(two_segments())

    transcribe_with_words(model, "a.wav")
    transcribe_with_words(model, "a.wav", task="transcribe")

    assert all("task" not in kwargs for _, kwargs in model.calls)
    assert len(model.calls) == 2


@pytest.mark.parametrize("function", [transcribe_with_words, transcribe_segments])
def test_translate_task_reaches_the_model(function):
    model = FakeModel(two_segments())

    function(model, "a.wav", task="translate")

    ((_, kwargs),) = model.calls
    assert kwargs["task"] == "translate"
    assert kwargs["word_timestamps"] is True


def test_translate_task_reaches_the_batched_pipeline(fake_batched_pipeline):
    transcribe_segments(FakeModel([]), "a.wav", task="translate", batch_size=4)

    assert fake_batched_pipeline["kwargs"]["task"] == "translate"
    assert fake_batched_pipeline["kwargs"]["batch_size"] == 4


def test_transcribe_task_is_not_sent_to_the_batched_pipeline(fake_batched_pipeline):
    transcribe_segments(FakeModel([]), "a.wav", batch_size=4)

    assert "task" not in fake_batched_pipeline["kwargs"]


def test_temperature_accepts_a_fallback_sequence():
    fallback = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    model = FakeModel(two_segments())

    transcribe_segments(model, "a.wav", temperature=fallback)

    ((_, kwargs),) = model.calls
    assert kwargs["temperature"] == fallback


def test_segments_keep_text_timing_and_words():
    model = FakeModel(two_segments(), duration=5.0, language="en")

    transcript = transcribe_segments(model, "sermon.wav")

    assert isinstance(transcript, SegmentedTranscript)
    assert transcript.language == "en"
    assert transcript.duration == 5.0
    assert transcript.segments == [
        Segment(
            0.0,
            2.5,
            " In the beginning",
            [
                Word(0.0, 0.4, " In", 0.98),
                Word(0.4, 0.7, " the", 0.91),
                Word(0.7, 2.5, " beginning", 0.64),
            ],
        ),
        Segment(
            2.5,
            5.0,
            " God created",
            [
                Word(2.5, 3.1, " God", 0.88),
                Word(3.1, 5.0, " created", 0.73),
            ],
        ),
    ]


@pytest.mark.parametrize("no_words", [None, []])
def test_segment_without_words_keeps_its_text(no_words):
    segments = [
        FakeSegment(0.0, 1.0, " ...", no_words),
        FakeSegment(1.0, 2.0, " Amen", [FakeWord(1.0, 2.0, " Amen", 0.9)]),
    ]

    transcript = transcribe_segments(FakeModel(segments, duration=2.0), "a.wav")

    assert [(segment.text, len(segment.words)) for segment in transcript.segments] == [
        (" ...", 0),
        (" Amen", 1),
    ]
    assert "\n".join(segment.text for segment in transcript.segments) == " ...\n Amen"


def test_segment_text_is_not_rebuilt_from_the_words():
    segments = [
        FakeSegment(0.0, 1.0, " Hello, world.", [FakeWord(0.0, 1.0, " Hello", 0.9)]),
    ]

    transcript = transcribe_segments(FakeModel(segments, duration=1.0), "a.wav")

    assert transcript.segments[0].text == " Hello, world."


def test_no_segments_gives_an_empty_segmented_transcript():
    transcript = transcribe_segments(FakeModel([], duration=0.0), "silence.wav")

    assert transcript == SegmentedTranscript(segments=[], language="en", duration=0.0)


def test_segments_use_the_same_model_call_as_the_word_list():
    for_words = FakeModel(two_segments())
    for_segments = FakeModel(two_segments())

    transcribe_with_words(for_words, "a.wav", initial_prompt="Genesis")
    transcribe_segments(for_segments, "a.wav", initial_prompt="Genesis")

    assert for_words.calls == for_segments.calls


def test_the_word_list_is_the_flattened_segments():
    flat = transcribe_with_words(FakeModel(two_segments(), duration=5.0), "a.wav")
    grouped = transcribe_segments(FakeModel(two_segments(), duration=5.0), "a.wav")

    assert flat.words == [
        word for segment in grouped.segments for word in segment.words
    ]
    assert (flat.language, flat.duration) == (grouped.language, grouped.duration)


def test_segment_progress_and_cancel_behave_like_the_word_list():
    seen = []
    transcribe_segments(
        FakeModel(two_segments(), duration=10.0), "a.wav", progress=seen.append
    )
    assert seen == [0.25, 0.5]

    model = FakeModel(two_segments() + two_segments())
    seen = []
    with pytest.raises(Cancelled):
        transcribe_segments(
            model, "a.wav", progress=seen.append, cancel=lambda: len(seen) >= 2
        )
    assert model.pulled == 2

    untouched = FakeModel(two_segments())
    with pytest.raises(Cancelled):
        transcribe_segments(untouched, "a.wav", cancel=lambda: True)
    assert untouched.calls == []


def test_importing_the_engine_does_not_import_qt():
    code = (
        "import sys\n"
        "import core.transcription.engine\n"
        "qt = sorted(name for name in sys.modules\n"
        "            if name.split('.')[0] in {'PyQt5', 'PyQt6', 'PySide2', 'PySide6'})\n"
        "assert not qt, qt\n"
        "assert 'faster_whisper' not in sys.modules\n"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
