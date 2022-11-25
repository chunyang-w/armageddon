import pandas as pd
from numpy import sin, cos, arcsin, arctan
import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve
from armageddon.locator import PostcodeLocator


def damage_zones(outcome, lat, lon, bearing, pressures):
    """
    Calculate the latitude and longitude of the surface zero location and the
    list of airblast damage radii (m) for a given impact scenario.
    Parameters
    ----------
    outcome: Dict
        the outcome dictionary from an impact scenario
    lat: float
        latitude of the meteoroid entry point (degrees)
    lon: float
        longitude of the meteoroid entry point (degrees)
    bearing: float
        Bearing (azimuth) relative to north of meteoroid trajectory (degrees)
    pressures: float, arraylike
        List of threshold pressures to define airblast damage levels
    Returns
    -------
    blat: float
        latitude of the surface zero point (degrees)
    blon: float
        longitude of the surface zero point (degrees)
    damrad: arraylike, float
        List of distances specifying the blast radii
        for the input damage levels
    Examples
    --------
    >>> import armageddon
    >>> import numpy as np
    >>> outcome = {'burst_altitude': 8e3, 'burst_energy': 7e3,\
                   'burst_distance': 90e3, 'burst_peak_dedz': 1e3,\
                   'outcome': 'Airburst'}
    >>> result = armageddon.damage_zones(outcome, 52.79, -2.95, 135,\
                                         pressures=[1e3, 3.5e3, 27e3, 43e3])
    >>> np.isclose(result[0], 52.21396905216966)
    True
    >>> np.isclose(result[1], -2.015908861677074)
    True
    >>> np.isclose(result[2][0], 115971.31673025587)
    True
    >>> np.isclose(result[2][1], 42628.36651535611)
    True
    >>> np.isclose(result[2][2], 9575.214234120966)
    True
    >>> np.isclose(result[2][3], 5835.983452079387)
    True
    """
    r_h = outcome['burst_distance']
    Ek = outcome['burst_energy']
    zb = outcome['burst_altitude']
    Rp = 6371e3
    if type(pressures) == float:
        pressures = [float(pressures)]
    pressures = np.array(pressures)
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    bearing = np.deg2rad(bearing)

    sin_blat = ((sin(lat) * cos(r_h / Rp)) +
                (cos(lat) * sin(r_h / Rp) * cos(bearing)))
    blat = arcsin(sin_blat)
    tan_blon_diff = ((sin(bearing) * sin(r_h / Rp) * cos(lat)) /
                     (cos(r_h / Rp) - (sin(lat) * sin_blat)))
    blon = arctan(tan_blon_diff) + lon
    blon = float(np.rad2deg(blon))
    blat = float(np.rad2deg(blat))

    discriminant = np.sqrt((3.24e14 + (1.256e12 * pressures)))
    pre_sol = (((((-1.8e7 + discriminant) / 6.28e11)**(-2/1.3)) *
                (Ek**(2/3))) - (zb**2))
    initial = np.sqrt(np.abs(pre_sol))
    damrad = np.zeros(len(pressures))

    for index in range(len(pressures)):
        p = pressures[index]
        f = lambda r: 3.14e11*(((r**2 + zb**2) / (Ek**(2/3)))**(-1.3))\
            + 1.8e7*(((r**2 + zb**2) / (Ek**(2/3)))**(-0.565)) - p # noqa
        damrad[index] = fsolve(f, initial[index])

    return blat, blon, np.abs(damrad).tolist()


fiducial_means = {'radius': 35, 'angle': 45, 'strength': 1e7,
                  'density': 3000, 'velocity': 19e3,
                  'lat': 53.0, 'lon': -2.5, 'bearing': 115.}
fiducial_stdevs = {'radius': 1, 'angle': 1, 'strength': 5e6,
                   'density': 500, 'velocity': 1e3,
                   'lat': 0.025, 'lon': 0.025, 'bearing': 0.5}


def impact_risk(planet, means=fiducial_means, stdevs=fiducial_stdevs,
                pressure=27.e3, nsamples=10, sector=True):
    """
    Perform an uncertainty analysis to calculate the risk for each affected
    UK postcode or postcode sector

    Parameters
    ----------
    planet: armageddon.Planet instance
        The Planet instance from which to solve the atmospheric entry

    means: dict
        A dictionary of mean input values for the uncertainty analysis. This
        should include values for ``radius``, ``angle``, ``strength``,
        ``density``, ``velocity``, ``lat``, ``lon`` and ``bearing``

    stdevs: dict
        A dictionary of standard deviations for each input value. This
        should include values for ``radius``, ``angle``, ``strength``,
        ``density``, ``velocity``, ``lat``, ``lon`` and ``bearing``

    pressure: float
        A single pressure at which to calculate the damage zone for each impact

    nsamples: int
        The number of iterations to perform in the uncertainty analysis

    sector: logical, optional
        If True (default) calculate the risk for postcode sectors, otherwise
        calculate the risk for postcodes

    Returns
    -------
    risk: DataFrame
        A pandas DataFrame with columns for postcode (or postcode sector) and
        the associated risk. These should be called ``postcode`` or ``sector``,
        and ``risk``.
    Examples:
    ---------
    >>> import armageddon
    >>> import pandas as pd
    >>> planet = armageddon.Planet()
    >>> type_of_risk = type(armageddon.impact_risk(\
                                    planet,\
                                    means=fiducial_means,\
                                    stdevs=fiducial_stdevs,\
                                    pressure=27.e3,\
                                    nsamples=10,\
                                    sector=True))
    >>> type_of_risk == pd.DataFrame
    True
    """
    locator = PostcodeLocator()
    params = list(zip(means.values(), stdevs.values()))
    postcodes = []
    for i in range(nsamples):
        radius, angle, strength, density, velocity, lat, lon, bearing = [
            norm.rvs(*param, 1)[0] for param in params
        ]
        result = planet.solve_atmospheric_entry(
            radius, velocity, density, strength, angle
        )
        result = planet.calculate_energy(result)
        analysis = planet.analyse_outcome(result)
        blat, blon, damrad = damage_zones(
            analysis, lat, lon, bearing, pressure
        )
        damcode = locator.get_postcodes_by_radius(
            (blat, blon), damrad, sector)[0]
        # print('blast data', blat, blon, damrad)
        postcodes = postcodes + damcode
    postcode_sq = pd.Series(data=np.array(postcodes))
    postcode_sq = postcode_sq.value_counts()
    prob = postcode_sq.values / nsamples
    popu = locator.get_population_of_postcode([postcode_sq.index], sector)[0]
    risk = popu * prob
    col = 'postcode sector' if sector is True else 'postcode'
    risk = pd.DataFrame({
        col: postcode_sq.index,
        'risk': risk
    })
    return risk
