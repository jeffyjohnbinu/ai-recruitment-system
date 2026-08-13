import logging

from utils.logger import get_logger


def test_get_logger_returns_namespaced_logger():
    logger = get_logger("tests.demo")
    assert logger.name == "ai_recruitment_system.tests.demo"


def test_logger_writes_to_log_file(tmp_path, monkeypatch):
    from config import settings as settings_module

    # Point logging at a temp directory so the test doesn't pollute logs/
    monkeypatch.setattr(settings_module.settings, "log_dir", tmp_path)

    logging.getLogger("ai_recruitment_system").handlers.clear()
    import utils.logger as logger_module
    logger_module._configured = False

    logger = get_logger("tests.file_write")
    logger.info("test log message")

    for handler in logging.getLogger("ai_recruitment_system").handlers:
        handler.flush()

    log_file = tmp_path / "app.log"
    assert log_file.exists()
    assert "test log message" in log_file.read_text(encoding="utf-8")
