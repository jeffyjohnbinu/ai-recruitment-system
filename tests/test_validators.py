from utils.validators import clamp_score, is_valid_email, is_valid_phone


def test_valid_email_accepted():
    assert is_valid_email("jordan.lee@example.com") is True


def test_invalid_email_rejected():
    assert is_valid_email("not-an-email") is False
    assert is_valid_email("") is False


def test_valid_phone_accepted():
    assert is_valid_phone("+1 555-123-4567") is True


def test_invalid_phone_rejected():
    assert is_valid_phone("call me maybe") is False


def test_clamp_score_within_bounds():
    assert clamp_score(0.5) == 0.5
    assert clamp_score(1.5) == 1.0
    assert clamp_score(-0.5) == 0.0
