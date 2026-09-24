"""Cognito/S3/SQS clients must not hang a Prefect task forever."""

from auraframes.aws.awsclient import SESSION_CONFIG


def test_session_config_sets_connect_and_read_timeouts():
    assert SESSION_CONFIG.connect_timeout == 10
    assert SESSION_CONFIG.read_timeout == 60
    assert SESSION_CONFIG.retries.get("max_attempts") == 3
