from settings_validate import validate_settings


def test_validate_settings_ok():
    v = validate_settings()
    assert v.ok, v.errors
