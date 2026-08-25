"""Tests for EPA data ingestion module."""

from co2sat.data.epa import parse_capacities, categorize_fuel, is_active


class TestParseCapacities:
    def test_single_generator(self):
        assert parse_capacities("5 (788.8)") == 788.8

    def test_multiple_generators(self):
        assert parse_capacities("A1ST (191.8), A1CT (170.1)") == 361.9

    def test_none_returns_zero(self):
        assert parse_capacities(None) == 0.0

    def test_empty_string_returns_zero(self):
        assert parse_capacities("") == 0.0


class TestCategorizeFuel:
    def test_coal(self):
        assert categorize_fuel("Coal") == "coal"

    def test_gas_variants(self):
        assert categorize_fuel("Pipeline Natural Gas") == "gas"
        assert categorize_fuel("Process Gas") == "gas"

    def test_oil(self):
        assert categorize_fuel("Diesel Oil") == "oil"

    def test_unknown_is_other(self):
        assert categorize_fuel("Wood") == "other"

    def test_none_is_other(self):
        assert categorize_fuel(None) == "other"


class TestIsActive:
    def test_bare_operating(self):
        assert is_active("Operating") is True

    def test_operating_with_retirement_annotation(self):
        assert is_active("Operating (Retired 06/01/2022)") is True

    def test_operating_with_start_annotation(self):
        assert is_active("Operating (Started 03/10/2022)") is True

    def test_retired(self):
        assert is_active("Retired") is False

    def test_cold_storage(self):
        assert is_active("Long-term Cold Storage") is False
