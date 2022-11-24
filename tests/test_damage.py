from collections import OrderedDict
import pandas as pd
import numpy as np
import os
from armageddon.damage import damage_zones
from pytest import fixture, mark


@fixture(scope='module')
def armageddon():
    import armageddon
    return armageddon


# lat, lon conversion taken from
# https://www.gpsvisualizer.com/calculators for checks
outcome1 = {'burst_altitude': 0, 'burst_energy': 7e2,
            'burst_distance': 50e3, 'burst_peak_dedz': 1e3,
            'outcome': 'Crater'}
outcome2 = {'burst_altitude': 10e3, 'burst_energy': 7e4,
            'burst_distance': 90e2, 'burst_peak_dedz': 1e3,
            'outcome': 'Airburst'}
pressures1 = [1e3, 5e3]
pressures2 = [1e2, 5e2, 1e3]
@mark.parametrize(
    '''outcome, lat, lon, bearing, pressures, expected_lat, expected_lon,
       shape_expected_damrad''',
    [
        (outcome1, 50, 50, 45, pressures1, 50.31635229845, 50.49708288034,
         len(pressures1)),
        (outcome2, 100, 65, 30, pressures2, 79.9299029, -115.2311903,
         len(pressures2)),
         (outcome2, 120, 70, 20, pressures2, 59.9240159, 110.0551768,
         len(pressures2))
    ]
)
def test_damage_zones(outcome, lat, lon, bearing,
                                  pressures, expected_lat,
                                  expected_lon, shape_expected_damrad):
    blat, blon, damrad = damage_zones(outcome, lat, lon, bearing, pressures)
    damrad = np.array(damrad, dtype=float)
    assert int(blat) == int(expected_lat)
    if expected_lon > 90:
        assert int(blon) == int(180 - expected_lon)
    elif expected_lon < -90:
        assert int(blon) == 180 + int(expected_lon)
    else:
        assert int(blon) == int(expected_lon)
    assert len(damrad) == shape_expected_damrad