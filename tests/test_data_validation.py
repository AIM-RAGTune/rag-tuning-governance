import pytest

from square_sim.data.validate import validate_columns


def test_data_validation_expected_targets():
    result = validate_columns(["a", "target", "in_pocket"], ["target", "target_real", "in_pocket"])
    assert result["present_targets"] == ["target", "in_pocket"]
    assert "target_real" in result["missing_targets"]


def test_data_validation_fails_clear_when_absent():
    with pytest.raises(ValueError, match="None of the expected"):
        validate_columns(["a", "b"], ["target", "target_real", "in_pocket"])

