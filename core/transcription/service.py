from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal, QRunnable, QThreadPool

from core.temp_file_manager import temp_file_manager
from core.logging_config import get_logger
from core.transcription.engine import transcribe_segments

logger = get_logger(__name__)

# faster-whisper's own default for ``temperature``. The service never passed one
# before it went through the engine, so the model ran with this fallback list;
# the engine's default is the single value 0.0, which would drop the fallback.
_FASTER_WHISPER_TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


class _TranscriberSignals(QObject):
    transcription_done = Signal(str)
    error_occurred = Signal(str)


class _TranscriptionRunnable(QRunnable):
    def __init__(
        self,
        model,
        expected_id: int,
        audio_file: str | Path,
        task_mode: str = "transcribe",
        is_temp_file: bool = True,
        batch_size: int | None = None,
    ) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self.model = model
        self.expected_id = expected_id
        self.audio_file = Path(audio_file)
        self.task_mode = task_mode
        self.is_temp_file = is_temp_file
        self.batch_size = batch_size
        self.signals = _TranscriberSignals()

    def run(self) -> None:
        try:
            if id(self.model) != self.expected_id:
                logger.warning("Model changed during transcription setup")
                return

            logger.info(f"Starting transcription: {self.audio_file}")

            batched = self.batch_size is not None and int(self.batch_size) > 1
            transcript = transcribe_segments(
                self.model,
                self.audio_file,
                task=self.task_mode,
                batch_size=self.batch_size,
                # Left unset before the engine, so faster-whisper's own default
                # applied, and that default differs by path: the batched
                # pipeline filters with VAD, the plain model does not.
                vad_filter=batched,
                temperature=_FASTER_WHISPER_TEMPERATURES,
            )

            text = "\n".join(segment.text for segment in transcript.segments)
            logger.info("Transcription completed successfully")
            self.signals.transcription_done.emit(text)

        except Exception as e:
            logger.exception("Transcription failed")
            self.signals.error_occurred.emit(f"Transcription failed: {e}")
        finally:
            if self.is_temp_file:
                temp_file_manager.release(self.audio_file)


class TranscriptionService(QObject):
    transcription_started = Signal()
    transcription_completed = Signal(str)
    transcription_error = Signal(str)

    def __init__(self, curate_text_enabled: bool = False, task_mode: str = "transcribe"):
        super().__init__()
        self.curate_enabled = curate_text_enabled
        self.task_mode = task_mode
        self._thread_pool = QThreadPool.globalInstance()

    def transcribe_file(
        self,
        model,
        expected_id: int,
        audio_file: str | Path,
        is_temp_file: bool = True,
        batch_size: int | None = None,
    ) -> None:
        if not model:
            error_msg = "No model available for transcription"
            logger.error(error_msg)
            self.transcription_error.emit(error_msg)
            if is_temp_file:
                temp_file_manager.release(Path(audio_file))
            return

        try:
            runnable = _TranscriptionRunnable(
                model=model,
                expected_id=expected_id,
                audio_file=str(audio_file),
                task_mode=self.task_mode,
                is_temp_file=is_temp_file,
                batch_size=batch_size,
            )
            runnable.signals.transcription_done.connect(self._on_transcription_done)
            runnable.signals.error_occurred.connect(self._on_transcription_error)
            self._thread_pool.start(runnable)
            self.transcription_started.emit()
        except Exception as e:
            logger.exception("Failed to start transcription")
            self.transcription_error.emit(f"Failed to start transcription: {e}")
            if is_temp_file:
                temp_file_manager.release(Path(audio_file))

    def _on_transcription_done(self, text: str) -> None:
        if self.curate_enabled:
            try:
                from core.text.curation import curate_text
                text = curate_text(text)
            except Exception as e:
                logger.warning(f"Text curation failed: {e}")

        text = "\n".join(line.lstrip() for line in text.splitlines())
        self.transcription_completed.emit(text)

    def _on_transcription_error(self, error: str) -> None:
        logger.error(f"Transcription error: {error}")
        self.transcription_error.emit(error)

    def set_task_mode(self, mode: str) -> None:
        self.task_mode = mode
        logger.debug(f"Task mode set to: {mode}")

    def set_curation_enabled(self, enabled: bool) -> None:
        self.curate_enabled = enabled

    def cleanup(self) -> None:
        self._thread_pool.waitForDone(5000)
        logger.debug("TranscriptionService cleanup complete")
