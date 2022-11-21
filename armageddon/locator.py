"""Module dealing with postcode information."""

import numpy as np
import pandas as pd

__all__ = ['PostcodeLocator', 'great_circle_distance']


def great_circle_distance(latlon1, latlon2):
    """
    Calculate the great circle distance (in metres) between pairs of
    points specified as latitude and longitude on a spherical Earth
    (with radius 6371 km).

    Parameters
    ----------

    latlon1: arraylike
        latitudes and longitudes of first point (as [n, 2] array for n points)
    latlon2: arraylike
        latitudes and longitudes of second point (as [m, 2] array for m points)

    Returns
    -------

    numpy.ndarray
        Distance in metres between each pair of points (as an n x m array)

    Examples
    --------

    >>> import numpy
    >>> fmt = lambda x: numpy.format_float_scientific(x, precision=3)}
    >>> with numpy.printoptions(formatter={'all', fmt}):
        print(great_circle_distance([[54.0, 0.0], [55, 0.0]], [55, 1.0]))
    [1.286e+05 6.378e+04]
    """
    latlon1 = np.array(latlon1)
    latlon2 = np.array(latlon2)
    if (latlon1.ndim == 1):
        latlon1 = latlon1.reshape(1, *latlon1.shape)
    if (latlon2.ndim == 1):
        latlon2 = latlon2.reshape(1, *latlon2.shape)
    distance = np.empty((len(latlon1), len(latlon2)), float)
    m, n = distance.shape
    for i in range(m):
        for j in range(n):
            ll1 = latlon1[i]
            ll2 = latlon2[j]
            lat1 = ll1[0] * np.pi / 180
            lat2 = ll2[0] * np.pi / 180
            lon1 = ll1[1] * np.pi / 180
            lon2 = ll2[1] * np.pi / 180
            distance[i, j] = np.arccos(
                np.sin(lat1) * np.sin(lat2) +
                np.cos(lat1) * np.cos(lat2) *
                np.cos(np.abs(lon1 - lon2))
            ) * 6371e3
    return distance


class PostcodeLocator(object):
    """Class to interact with a postcode database file."""

    def __init__(self, postcode_file='',
                 census_file='',
                 norm=great_circle_distance):
        """
        Parameters
        ----------

        postcode_file : str, optional
            Filename of a .csv file containing geographic
            location data for postcodes.

        census_file :  str, optional
            Filename of a .csv file containing census data by postcode sector.

        norm : function
            Python function defining the distance between points in
            latitude-longitude space.

        """
        self.postcode_df = pd.read_csv(postcode_file)
        self.postcode_df['Sector_Postcode'] = self.postcode_df.apply(
            lambda row: (row['Postcode'][:4].strip()), axis=1
        )
        self.census_df = pd.read_csv(census_file)
        self.norm = norm

    def get_postcodes_by_radius(self, X, radii, sector=False):
        """
        Return (unit or sector) postcodes within specific distances of
        input location.

        Parameters
        ----------
        X : arraylike
            Latitude-longitude pair of centre location
        radii : arraylike
            array of radial distances from X
        sector : bool, optional
            if true return postcode sectors, otherwise postcode units

        Returns
        -------
        list of lists
            Contains the lists of postcodes closer than the elements
            of radii to the location X.


        Examples
        --------

        >>> locator = PostcodeLocator()
        >>> locator.get_postcodes_by_radius((51.4981, -0.1773), [0.13e3])
        >>> locator.get_postcodes_by_radius((51.4981, -0.1773),
                                            [0.4e3, 0.2e3], True)
        """
        place_list = []
        selector = 'Sector_Postcode' if sector is True else 'Postcode'
        df = self.postcode_df
        df['Distance'] = self.norm(
            np.stack(
                [df['Latitude'].to_numpy(), df['Longitude'].to_numpy()],
                axis=1), X
        )
        for r in radii:
            place_list = df[df['Distance'] < r][selector].to_list() +\
                place_list
        return list(set(place_list))

    def get_population_of_postcode(self, postcodes, sector=False):
        """
        Return populations of a list of postcode units or sectors.

        Parameters
        ----------
        postcodes : list of lists
            list of postcode units or postcode sectors
        sector : bool, optional
            if true return populations for postcode sectors,
            otherwise returns populations for postcode units

        Returns
        -------
        list of lists
            Contains the populations of input postcode units or sectors


        Examples
        --------

        >>> locator = PostcodeLocator()
        >>> locator.get_population_of_postcode([['SW7 2AZ', 'SW7 2BT',
                                                 'SW7 2BU', 'SW7 2DD']])
        >>> locator.get_population_of_postcode([['SW7  2']], True)
        """

        return [[]]

    def in_area(self, lat, lon, radii, point):
        if (((lat - radii) < point[0]) and ((lat + radii) > point[0])):
            if ((lon - radii < point[1]) and (lon + radii > point[1])):
                return True
        return False
