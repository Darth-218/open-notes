from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.observers import Observer


# FIX: Path conversion
class NoteFileHandler(FileSystemEventHandler):
    """File system event handler for Markdown note files.

    Handles file system events for .md files with debouncing to prevent
    duplicate events from being processed in quick succession.

    Attributes:
        callback: Function to call when a note file event occurs.
        debounce_ms: Debounce time in milliseconds.
    """

    def __init__(self, callback: Callable[[Path], None], debounce_ms: int = 500):
        """Initialize the NoteFileHandler.

        Args:
            callback: Function to call when a note file event occurs.
                Receives the file Path as an argument.
            debounce_ms: Debounce time in milliseconds. Defaults to 500.
        """
        self.callback = callback
        self.debounce_ms = debounce_ms / 1000.0
        self.last_events: dict[Path, float] = {}

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events.

        Args:
            event: The file system event.
        """
        if event.is_directory:
            return
        self._handle_event(Path(event.src_path))

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events.

        Args:
            event: The file system event.
        """
        if event.is_directory:
            return
        self._handle_event(Path(event.src_path))

    def on_deleted(self, event: FileSystemEvent) -> None:
        """Handle file deletion events.

        Args:
            event: The file system event.
        """
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix == ".md":
            self.callback(path)

    def on_moved(self, event: FileSystemEvent) -> None:
        """Handle file move/rename events.

        Args:
            event: The file system event.
        """
        if event.is_directory:
            return
        path = Path(event.dest_path)
        if path.suffix == ".md":
            self.callback(path)

    def _handle_event(self, path: Path) -> None:
        """Process a file event with debouncing.

        Only processes .md files and applies debouncing to prevent
        duplicate callbacks.

        Args:
            path: The file path that triggered the event.
        """
        if path.suffix != ".md":
            return

        current_time = time.time()
        last_time = self.last_events.get(path, 0)

        if current_time - last_time < self.debounce_ms:
            return

        self.last_events[path] = current_time
        self.callback(path)


# FIX: Types
class NoteWatcher:
    """Watches a directory for changes to Markdown note files.

    Uses watchdog to monitor a directory recursively and triggers
    callbacks when .md files are created, modified, moved, or deleted.
    Supports use as a context manager for automatic startup and shutdown.

    Attributes:
        watch_path: The path to watch for changes.
        callback: Function to call when a note file event occurs.
    """

    def __init__(self, watch_path: Path, callback: Callable[[Path], None]):
        """Initialize the NoteWatcher.

        Args:
            watch_path: The directory path to watch for changes.
            callback: Function to call when a note file event occurs.
                Receives the file Path as an argument.
        """
        self.watch_path = watch_path
        self.callback = callback
        self.observer: Observer | None = None
        self.handler = NoteFileHandler(callback)

    def start(self) -> None:
        """Start watching the directory for file changes."""
        self.observer = Observer()
        self.observer.schedule(self.handler, str(self.watch_path), recursive=True)
        self.observer.start()

    def stop(self) -> None:
        """Stop watching the directory.

        Blocks until the observer thread has fully stopped.
        """
        if self.observer:
            self.observer.stop()
            self.observer.join()

    def __enter__(self) -> "NoteWatcher":
        """Enter the context manager, starting the watcher.

        Returns:
            Self.
        """
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context manager, stopping the watcher."""
        self.stop()
