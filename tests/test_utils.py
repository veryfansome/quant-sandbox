import calendar
import pytest
from datetime import date

from quant_sandbox import utils


@pytest.mark.parametrize("year, month, expected_day", [
    (2024, 1, 5), (2024, 2, 2), (2024, 3, 1), (2025, 2, 7), (2025, 12, 5),
])
def test_get_first_friday_of_month(year, month, expected_day):
    """Verifies the function returns the correct 1st Friday of the month."""
    result = utils.get_first_friday_of_month(year, month)
    assert result == expected_day
    assert date(year, month, result).weekday() == calendar.FRIDAY


@pytest.mark.parametrize("year, month, expected_day", [
    (2024, 1, 26), (2024, 2, 23), (2024, 3, 22), (2025, 2, 28), (2025, 12, 26),
])
def test_get_fourth_friday_of_month(year, month, expected_day):
    """Verifies the function returns the correct 4th Friday of the month."""
    result = utils.get_fourth_friday_of_month(year, month)
    assert result == expected_day
    assert date(year, month, result).weekday() == calendar.FRIDAY


@pytest.mark.parametrize("year, month, expected_day", [
    (2024, 1, 10), (2024, 2, 14), (2024, 3, 13), (2025, 6, 11), (2025, 12, 10),
])
def test_get_second_wednesday_of_month(year, month, expected_day):
    """Verifies the function returns the correct 2nd Wednesday of the month."""
    result = utils.get_second_wednesday_of_month(year, month)
    assert result == expected_day
    assert date(year, month, result).weekday() == calendar.WEDNESDAY
