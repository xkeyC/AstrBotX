"""Codex's retry and warning events are logged with the chat they belong to:
they are the only record of why a request was retried."""

from astrbot.core.agent.runners.codex import codex_agent_runner as runner_mod


def test_a_stream_error_is_logged_with_its_status_and_details(monkeypatch):
    lines = []
    monkeypatch.setattr(
        runner_mod.logger, "warning", lambda fmt, *args: lines.append(fmt % args)
    )
    runner_mod._log_diagnostic(
        "qq:GroupMessage:1",
        "thread-1",
        {
            "type": "stream_error",
            "message": "Reconnecting... 2/5",
            "codex_error_info": {
                "response_stream_disconnected": {"http_status_code": 504}
            },
            "additional_details": "idle timeout waiting for SSE",
        },
        42.4,
    )
    (line,) = lines
    assert "stream_error" in line and "qq:GroupMessage:1" in line
    assert "thread-1" in line and "at 42 s" in line
    assert "Reconnecting... 2/5" in line and "504" in line
    assert "idle timeout waiting for SSE" in line


def test_retry_and_warning_events_are_the_ones_logged():
    assert set(runner_mod._DIAGNOSTIC_EVENTS) == {
        "stream_error",
        "warning",
        "stream_info",
    }
