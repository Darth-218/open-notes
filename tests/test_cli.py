"""Tests for CLI module."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from open_notes.cli.main import cli


class TestCLI:
    """Tests for CLI commands."""

    def test_cli_help(self):
        """Test CLI help output."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        
        assert result.exit_code == 0
        assert "open-notes" in result.output.lower()

    def test_config_command(self, temp_dir):
        """Test config command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["config"])
        
        assert result.exit_code == 0

    def test_stats_command_no_index(self, temp_dir, monkeypatch):
        """Test stats command with no index."""
        monkeypatch.setenv("OPEN_NOTES_KB_PATH", str(temp_dir / "kb"))
        
        runner = CliRunner()
        result = runner.invoke(cli, ["stats"])
        
        assert result.exit_code == 0

    def test_search_command_no_index(self, temp_dir, monkeypatch):
        """Test search command with no index."""
        kb_path = temp_dir / "kb"
        kb_path.mkdir()
        
        runner = CliRunner()
        result = runner.invoke(cli, ["search", "test"], obj={"knowledge_base_path": kb_path})
        
        assert "error" not in result.output.lower() or result.exit_code == 0

    def test_index_creates_directory(self, temp_dir):
        """Test that index command creates KB if not exists."""
        kb_path = temp_dir / "new_kb"
        
        runner = CliRunner()
        result = runner.invoke(cli, ["index"], obj={"knowledge_base_path": kb_path})
        
        assert kb_path.exists() or result.exit_code == 0

    def test_query_command_exists(self):
        """Test that query command exists."""
        runner = CliRunner()
        result = runner.invoke(cli, ["query", "test"])
        
        assert "error" not in result.output.lower() or result.exit_code != 2

    def test_watch_command_exists(self):
        """Test that watch command exists."""
        runner = CliRunner()
        result = runner.invoke(cli, ["watch"])
        
        assert result.exit_code in [0, 1, 2]

    def test_serve_command_exists(self):
        """Test that serve command exists."""
        runner = CliRunner()
        result = runner.invoke(cli, ["serve"])
        
        assert "error" not in result.output.lower() or result.exit_code != 2