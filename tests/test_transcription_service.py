from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass

import pytest

SERVICE = "core.transcription.service"


class _BoundSignal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def emit(self, *args):
        for slot in list(self.slots):
            slot(*args)


class _Signal:
    """Stands in for PySide6.QtCore.Signal: one bound signal per instance."""

    def __init__(self, *types):
        self.key = ""

    def __set_name__(self, owner, name):
        self.key = f"_signal_{name}"

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return instance.__dict__.setdefault(self.key, _BoundSignal())


class _QObject:
    def __init__(self, *args, **kwargs):
        pass


class _QRunnable:
    def __init__(self):
        pass

    def setAutoDelete(self, flag):
        pass


class _ImmediatePool:
    """Runs each runnable on the calling thread, as soon as it is started."""

    def start(self, runnable):
        runnable.run()

    def waitForDone(self, msecs):
        return True


class _QThreadPool:
    @staticmethod
    def globalInstance():
        return _ImmediatePool()


@pytest.fixture
def service_module(monkeypatch):
    """core.transcription.service imported against a minimal stand-in for Qt."""
    qtcore = types.ModuleType("PySide6.QtCore")
    qtcore.QObject = _QObject
    qtcore.QRunnable = _QRunnable
    qtcore.QThreadPool = _QThreadPool
    qtcore.Signal = _Signal
    package = types.ModuleType("PySide6")
    package.QtCore = qtcore
    monkeypatch.setitem(sys.modules, "PySide6", package)
    monkeypatch.setitem(sys.modules, "PySide6.QtCore", qtcore)
    sys.modules.pop(SERVICE, None)

    module = importlib.import_module(SERVICE)
    yield module

    sys.modules.pop(SERVICE, None)
    transcription = sys.modules.get("core.transcription")
    if transcription is not None and hasattr(transcription, "service"):
        delattr(transcription, "service")


class _RecordingTempFiles:
    def __init__(self):
        self.released = []

    def release(self, path):
        self.released.append(path)


@pytest.fixture
def temp_files(service_module, monkeypatch):
    recorder = _RecordingTempFiles()
    monkeypatch.setattr(service_module, "temp_file_manager", recorder)
    return recorder


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
    language: str = "en"
    duration: float = 10.0


class FakeModel:
    def __init__(self, segments, error=None):
        self._segments = segments
        self._error = error
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        if self._error is not None:
            raise self._error
        return iter(self._segments), FakeInfo()


def sermon_segments():
    return [
        FakeSegment(0.0, 2.5, " In the beginning", [FakeWord(0.0, 2.5, " In", 0.9)]),
        FakeSegment(2.5, 5.0, " God created", [FakeWord(2.5, 5.0, " God", 0.9)]),
    ]


def transcribe(
    module,
    model,
    *,
    task_mode="transcribe",
    batch_size=None,
    expected_id=None,
    is_temp_file=False,
):
    service = module.TranscriptionService(task_mode=task_mode)
    completed, errors = [], []
    service.transcription_completed.connect(completed.append)
    service.transcription_error.connect(errors.append)
    service.transcribe_file(
        model,
        id(model) if expected_id is None else expected_id,
        "clip.wav",
        is_temp_file=is_temp_file,
        batch_size=batch_size,
    )
    return completed, errors


def test_each_segment_becomes_one_line(service_module, temp_files):
    completed, errors = transcribe(service_module, FakeModel(sermon_segments()))

    assert completed == ["In the beginning\nGod created"]
    assert errors == []


def test_a_segment_without_words_keeps_its_text_and_its_line(
    service_module, temp_files
):
    segments = [
        FakeSegment(0.0, 1.0, " Welcome", None),
        FakeSegment(1.0, 2.0, " Amen", [FakeWord(1.0, 2.0, " Amen", 0.9)]),
        FakeSegment(2.0, 3.0, " Go in peace", []),
    ]

    completed, _ = transcribe(service_module, FakeModel(segments))

    assert completed == ["Welcome\nAmen\nGo in peace"]


def test_no_segments_gives_empty_text(service_module, temp_files):
    completed, errors = transcribe(service_module, FakeModel([]))

    assert completed == [""]
    assert errors == []


def test_transcribe_mode_asks_the_model_to_transcribe(service_module, temp_files):
    model = FakeModel(sermon_segments())

    transcribe(service_module, model)

    ((audio, kwargs),) = model.calls
    assert audio == "clip.wav"
    assert kwargs.get("task", "transcribe") == "transcribe"


def test_translate_mode_asks_the_model_to_translate(service_module, temp_files):
    model = FakeModel(sermon_segments())

    transcribe(service_module, model, task_mode="translate")

    ((_, kwargs),) = model.calls
    assert kwargs["task"] == "translate"


def test_set_task_mode_changes_the_next_transcription(service_module, temp_files):
    model = FakeModel(sermon_segments())
    service = service_module.TranscriptionService()

    service.set_task_mode("translate")
    service.transcribe_file(model, id(model), "clip.wav", is_temp_file=False)

    ((_, kwargs),) = model.calls
    assert kwargs["task"] == "translate"


def test_the_plain_model_runs_with_the_options_it_ran_with_before(
    service_module, temp_files
):
    model = FakeModel(sermon_segments())

    transcribe(service_module, model)

    ((_, kwargs),) = model.calls
    assert kwargs == {
        "language": None,
        "word_timestamps": True,
        "vad_filter": False,
        "vad_parameters": None,
        "initial_prompt": None,
        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        "condition_on_previous_text": True,
    }


@pytest.fixture
def fake_batched_pipeline(monkeypatch):
    created = {}

    class FakeBatchedInferencePipeline:
        def __init__(self, model):
            created["model"] = model

        def transcribe(self, audio, **kwargs):
            created["audio"] = audio
            created["kwargs"] = kwargs
            return iter(sermon_segments()), FakeInfo()

    module = types.ModuleType("faster_whisper")
    module.BatchedInferencePipeline = FakeBatchedInferencePipeline
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return created


def test_a_batch_size_above_one_uses_the_batched_pipeline(
    service_module, temp_files, fake_batched_pipeline
):
    model = FakeModel([])

    completed, _ = transcribe(
        service_module, model, task_mode="translate", batch_size=8
    )

    assert completed == ["In the beginning\nGod created"]
    assert model.calls == []
    assert fake_batched_pipeline["model"] is model
    assert fake_batched_pipeline["audio"] == "clip.wav"
    kwargs = fake_batched_pipeline["kwargs"]
    assert kwargs["batch_size"] == 8
    assert kwargs["task"] == "translate"
    assert kwargs["language"] is None
    assert kwargs["vad_filter"] is True
    assert kwargs["temperature"] == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


@pytest.mark.parametrize("batch_size", [None, 0, 1])
def test_a_small_batch_size_uses_the_plain_model(
    service_module, temp_files, fake_batched_pipeline, batch_size
):
    model = FakeModel(sermon_segments())

    transcribe(service_module, model, batch_size=batch_size)

    assert fake_batched_pipeline == {}
    ((_, kwargs),) = model.calls
    assert kwargs["vad_filter"] is False


def test_a_model_that_changed_is_not_used(service_module, temp_files):
    model = FakeModel(sermon_segments())

    completed, errors = transcribe(
        service_module, model, expected_id=id(model) + 1, is_temp_file=True
    )

    assert model.calls == []
    assert completed == []
    assert errors == []
    assert [str(path) for path in temp_files.released] == ["clip.wav"]


def test_a_model_failure_is_reported_as_an_error(service_module, temp_files):
    model = FakeModel([], error=RuntimeError("boom"))

    completed, errors = transcribe(service_module, model)

    assert completed == []
    assert errors == ["Transcription failed: boom"]


def test_a_temp_file_is_released_after_success_and_after_failure(
    service_module, temp_files
):
    transcribe(service_module, FakeModel(sermon_segments()), is_temp_file=True)
    transcribe(
        service_module, FakeModel([], error=RuntimeError("boom")), is_temp_file=True
    )
    transcribe(service_module, FakeModel(sermon_segments()), is_temp_file=False)

    assert [str(path) for path in temp_files.released] == ["clip.wav", "clip.wav"]


def test_no_model_is_reported_as_an_error(service_module, temp_files):
    service = service_module.TranscriptionService()
    errors = []
    service.transcription_error.connect(errors.append)

    service.transcribe_file(None, 0, "clip.wav", is_temp_file=True)

    assert errors == ["No model available for transcription"]
    assert [str(path) for path in temp_files.released] == ["clip.wav"]
