"""Tests for static features — zenith angle against paper eqs 1-3."""

import numpy as np

from co2sat.data.static import satellite_zenith_angle, SAT_LON_DEG


class TestZenithAngle:
    def test_subsatellite_point_is_zero(self):
        assert abs(satellite_zenith_angle(0.0, SAT_LON_DEG)) < 1e-9

    def test_reference_values(self):
        """Cross-check against an independent implementation of eqs 1-3."""
        cases = [
            (29.4839, -95.6314, 41.033),  # W A Parish
            (25.8, -80.2, 30.668),  # Miami area
            (47.6, -122.3, 70.885),  # Seattle area
            (39.7, -105.0, 55.245),  # Denver area
        ]
        for lat, lon, expected in cases:
            got = satellite_zenith_angle(lat, lon)
            assert abs(got - expected) < 0.01, f"({lat},{lon}): {got:.3f} != {expected}"

    def test_increases_away_from_subsatellite(self):
        near = satellite_zenith_angle(30.0, -80.0)
        far = satellite_zenith_angle(45.0, -120.0)
        assert far > near

    def test_east_west_symmetry(self):
        """Same |lon offset| east or west of the satellite -> same angle."""
        west = satellite_zenith_angle(35.0, SAT_LON_DEG - 20.0)
        east = satellite_zenith_angle(35.0, SAT_LON_DEG + 20.0)
        assert abs(west - east) < 1e-9

    def test_vectorized_matches_scalar(self):
        lats = np.array([29.4839, 47.6])
        lons = np.array([-95.6314, -122.3])
        vec = satellite_zenith_angle(lats, lons)
        assert abs(vec[0] - satellite_zenith_angle(29.4839, -95.6314)) < 1e-12
        assert abs(vec[1] - satellite_zenith_angle(47.6, -122.3)) < 1e-12
