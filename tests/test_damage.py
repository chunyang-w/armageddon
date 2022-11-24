# from collections import OrderedDict
import pandas as pd
import numpy as np
import os
from armageddon.damage import damage_zones, impact_risk
from armageddon import Planet
from pytest import mark


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
        (outcome2, 120, 70, 20, pressures2, 59.9240159, -110.0551768,
         len(pressures2))
    ]
)
def test_damage_zones_coord(outcome, lat, lon, bearing,
                            pressures, expected_lat,
                            expected_lon, shape_expected_damrad):
    blat, blon, damrad = damage_zones(outcome, lat, lon, bearing, pressures)
    damrad = np.array(damrad, dtype=float)
    assert int(blat) == int(expected_lat)
    if expected_lon > 90:
        assert int(blon) == int(180 - expected_lon)
    elif expected_lon < -90:
        assert int(blon) == int(180 + expected_lon)
    else:
        assert int(blon) == int(expected_lon)
    assert len(damrad) == shape_expected_damrad


@mark.parametrize(
    'pressures, outcome',
    [
        (pressures1, outcome1),
        (pressures1, outcome2),
        (pressures2, outcome1),
        (pressures2, outcome2)
    ]
)
def test_damage_zones_damrad(pressures, outcome):
    blat, blon, damrad = damage_zones(outcome, 120, 70, 20, pressures)
    Ek = outcome['burst_energy']
    zb = outcome['burst_altitude']
    for p in range(len(pressures)):
        def f(r):
            return 3.14e11*(((r**2 + zb**2) / (Ek**(2/3)))**(-1.3))\
                    + 1.8e7*(((r**2 + zb**2) / (Ek**(2/3)))**(-0.565))\
                    - pressures[p]
        assert np.isclose(f(damrad[p]), 0)


fiducial_means = {'radius': 35, 'angle': 45, 'strength': 1e7,
                  'density': 3000, 'velocity': 19e3,
                  'lat': 53.0, 'lon': -2.5, 'bearing': 115.}
fiducial_stdevs = {'radius': 1, 'angle': 1, 'strength': 5e6,
                   'density': 500, 'velocity': 1e3,
                   'lat': 0.025, 'lon': 0.025, 'bearing': 0.5}


def test_impact_risk():
    impact_risk_df = impact_risk(Planet(), fiducial_means,
                                 fiducial_stdevs,
                                 pressure=27.e3, nsamples=1,
                                 sector=True)
    highest_val_df = pd.read_csv(os.sep.join((
                                    os.path.dirname(__file__), '..',
                                    'resources',
                                    'population_by_postcode_sector.csv')))
    hstr = 'Variable: All usual residents; measures: Value'
    highest_val = highest_val_df[hstr].max()
    assert (impact_risk_df['risk'].index >= 0).all()
    assert (impact_risk_df['risk'].index <= highest_val).all()
    assert type(impact_risk_df) == pd.DataFrame
