"""Tests for stockscreen.config module."""

import os
import tempfile
import shutil

import pytest


class TestDefaultPaths:
    def test_data_path_uses_env_var(self, monkeypatch, tmp_path):
        """When STOCKSCREEN_DATA_PATH is set, it should be used."""
        custom_path = str(tmp_path / "custom_data")
        monkeypatch.setenv("STOCKSCREEN_DATA_PATH", custom_path)

        # Re-import to pick up env var
        import importlib
        import stockscreen.config as config_mod
        importlib.reload(config_mod)

        assert config_mod.DEFAULT_DATA_PATH == custom_path

    def test_data_path_defaults_to_script_relative(self, monkeypatch):
        """When no env var is set, data path should be relative to the package."""
        monkeypatch.delenv("STOCKSCREEN_DATA_PATH", raising=False)

        import importlib
        import stockscreen.config as config_mod
        importlib.reload(config_mod)

        # Should end with /stockscreen/data (relative to the config.py file)
        assert config_mod.DEFAULT_DATA_PATH.endswith(os.path.join("stockscreen", "data"))

    def test_log_path_relative_to_package(self):
        """Log path should be relative to the package directory."""
        from stockscreen.config import DEFAULT_LOG_PATH

        assert DEFAULT_LOG_PATH.endswith("stockscreen_v1.log")


class TestMigrateLegacyData:
    def test_no_migration_when_env_var_set(self, monkeypatch, tmp_path):
        """Should skip migration if STOCKSCREEN_DATA_PATH is explicitly set."""
        old_data = tmp_path / "old" / "data"
        old_data.mkdir(parents=True)
        (old_data / "test.json").write_text("{}")

        monkeypatch.setenv("STOCKSCREEN_DATA_PATH", str(tmp_path / "custom"))
        monkeypatch.setattr("os.getcwd", lambda: str(tmp_path / "old"))

        from stockscreen.config import migrate_legacy_data
        migrate_legacy_data()

        # Old data should still be there (not moved)
        assert old_data.exists()

    def test_migration_moves_data(self, monkeypatch, tmp_path):
        """Should move data from old CWD-based location to new location."""
        old_cwd = tmp_path / "old_cwd"
        old_data = old_cwd / "data"
        old_data.mkdir(parents=True)
        (old_data / "watchlist.json").write_text('["AAPL"]')

        new_data = tmp_path / "new_data"

        monkeypatch.delenv("STOCKSCREEN_DATA_PATH", raising=False)
        monkeypatch.setattr("os.getcwd", lambda: str(old_cwd))

        from stockscreen.config import migrate_legacy_data
        migrate_legacy_data(target_path=str(new_data))

        assert (new_data / "watchlist.json").exists()
        assert not old_data.exists()

    def test_no_migration_when_old_path_missing(self, monkeypatch, tmp_path):
        """Should do nothing if old data directory doesn't exist."""
        monkeypatch.delenv("STOCKSCREEN_DATA_PATH", raising=False)
        monkeypatch.setattr("os.getcwd", lambda: str(tmp_path / "nonexistent"))

        from stockscreen.config import migrate_legacy_data
        # Should not raise
        migrate_legacy_data(target_path=str(tmp_path / "new_data"))

    def test_no_migration_when_target_exists(self, monkeypatch, tmp_path):
        """Should not overwrite existing target directory."""
        old_cwd = tmp_path / "old_cwd"
        old_data = old_cwd / "data"
        old_data.mkdir(parents=True)
        (old_data / "old.json").write_text("{}")

        new_data = tmp_path / "new_data"
        new_data.mkdir(parents=True)
        (new_data / "existing.json").write_text("{}")

        monkeypatch.delenv("STOCKSCREEN_DATA_PATH", raising=False)
        monkeypatch.setattr("os.getcwd", lambda: str(old_cwd))

        from stockscreen.config import migrate_legacy_data
        migrate_legacy_data(target_path=str(new_data))

        # Target should keep its existing file, old data untouched
        assert (new_data / "existing.json").exists()
        assert not (new_data / "old.json").exists()
        assert old_data.exists()


class TestGetLogger:
    def test_returns_named_logger(self):
        from stockscreen.config import get_logger
        logger = get_logger()
        assert logger.name == "stockscreen-server-v1"
