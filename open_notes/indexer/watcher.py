from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.observers import Observer


class NoteFileHandler(FileSystemEventHandler):
    def __init__(self, callback: Callable[[Path], None], debounce_ms: int = 500):
        self.callback = callback
        self.debounce_ms = debounce_ms / 1000.0
        self.last_events: dict[Path, float] = {}

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._handle_event(Path(event.src_path))

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._handle_event(Path(event.src_path))

    def on_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix == ".md":
            self.callback(path)

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.dest_path)
        if path.suffix == ".md":
            self.callback(path)

    def _handle_event(self, path: Path) -> None:
        if path.suffix != ".md":
            return

        current_time = time.time()
        last_time = self.last_events.get(path, 0)

        if current_time - last_time < self.debounce_ms:
            return

        self.last_events[path] = current_time
        self.callback(path)


class NoteWatcher:
    def __init__(self, watch_path: Path, callback: Callable[[Path], None]):
        self.watch_path = watch_path
        self.callback = callback
        self.observer: Observer | None = None
        self.handler = NoteFileHandler(callback)

    def start(self) -> None:
        self.observer = Observer()
        self.observer.schedule(self.handler, str(self.watch_path), recursive=True)
        self.observer.start()

    def stop(self) -> None:
        if self.observer:
            self.observer.stop()
            self.observer.join()

    def __enter__(self) -> "NoteWatcher":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
