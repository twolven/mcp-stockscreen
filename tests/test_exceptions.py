"""Tests for stockscreen.exceptions module."""

from stockscreen.exceptions import APIError, StockscreenError, ValidationError


class TestExceptionHierarchy:
    def test_stockscreen_error_is_exception(self):
        assert issubclass(StockscreenError, Exception)

    def test_validation_error_is_stockscreen_error(self):
        assert issubclass(ValidationError, StockscreenError)

    def test_api_error_is_stockscreen_error(self):
        assert issubclass(APIError, StockscreenError)

    def test_stockscreen_error_catchable(self):
        with pytest.raises(StockscreenError):
            raise StockscreenError("test")

    def test_validation_error_caught_by_parent(self):
        with pytest.raises(StockscreenError):
            raise ValidationError("bad input")

    def test_api_error_caught_by_parent(self):
        with pytest.raises(StockscreenError):
            raise APIError("api down")

    def test_error_message_preserved(self):
        err = ValidationError("invalid symbol")
        assert str(err) == "invalid symbol"


import pytest
