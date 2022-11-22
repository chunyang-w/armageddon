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
    >>> fmt = lambda x: numpy.format_float_scientific(x, precision=3)
    >>> with numpy.printoptions(formatter={'all': fmt}):
    >>> print(great_circle_distance([[54.0, 0.0], [55, 0.0]], [55, 1.0]))
    [1.286e+05 6.378e+04]
    """

    Rp = 6371000

    latlon1 = np.array(latlon1)*np.pi/180
    latlon2 = np.array(latlon2)*np.pi/180

    if latlon1.ndim == 1:
        latlon1 = latlon1.reshape(1, 2)

    if latlon2.ndim == 1:
        latlon2 = latlon2.reshape(1, 2)

    distance = np.empty((len(latlon1), len(latlon2)), float)

    lat1 = latlon1[:, 0]
    lat2 = latlon2[:, 0]
    lon1 = latlon1[:, 1]
    lon2 = latlon2[:, 1]

    for i in range(len(latlon1)):

        for j in range(len(latlon2)):
            num = np.sqrt((np.cos(lat2[j]) *
                           np.sin(abs(lon1[i] - lon2[j])))**2 +
                          (np.cos(lat1[i]) * np.sin(lat2[j]) -
                           np.sin(lat1[i]) * np.cos(lat2[j]) *
                           np.cos(abs(lon1[i] - lon2[j])))**2)
            den = np.sin(lat1[i]) * np.sin(lat2[j]) + np.cos(lat1[i]) *\
                np.cos(lat2[j]) * np.cos(abs(lon1[i] - lon2[j]))
            dis = Rp * np.arctan(num / den)

            distance[i][j] = dis

    return distance
# pnts1 = np.array([[54.0, 0.0], [55.0, 1.0], [54.2, -3.0]])
# pnts2 = np.array([[55.0, 1.0], [56.0, -2.1], [54.001, -0.003]])
# print(great_circle_distance(pnts1, pnts2))
# fmt = lambda x: np.format_float_scientific(x, precision=3)
# with np.printoptions(formatter={'all': fmt}):
#     print(great_circle_distance([[54.0, 0.0], [55, 0.0]], [55, 1.0]))
    # print(great_circle_distance(pnts1, pnts2))


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

        >>> locator = PostcodeLocator('resources/full_postcodes.csv', 'resources/population_by_postcode_sector.csv')
        >>> locator.get_postcodes_by_radius((51.4981, -0.1773), [0.13e3])
        >>> locator.get_postcodes_by_radius((51.4981, -0.1773), [0.4e3, 0.2e3], True)
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

        >>> locator = PostcodeLocator('resources/full_postcodes.csv', 'resources/population_by_postcode_sector.csv')
        >>> pop1 = locator.get_population_of_postcode([['SW7 2AZ', 'SW7 2BT', 'SW7 2BU', 'SW7 2DD']])
        >>> pop1
        [[19.0, 19.0, 19.0, 19.0]]
        >>> pop2 = locator.get_population_of_postcode([['SW7  2']], True)
        >>> pop2
        [[2283.0]]
        """
        pc = np.array(postcodes)
        m, n = pc.shape
        result = np.zeros(pc.shape)

        for i in range(m):
            for j in range(n):
                district, sec = pc[i][j].split()

                if len(district) == 3:
                    searchSector = district+'  '+sec[0]
                else:
                    searchSector = district+' '+sec[0]

                col = 'Variable: All usual residents; measures: Value'
                try:
                    sectorPopulation = int(self.census_df.loc[self.census_df
                                           ['geography']
                                           == searchSector][col])
                except TypeError:
                    print('Sector not in list')
                    return [[0]]

                if sector is True:
                    result[i][j] = sectorPopulation
                else:
                    if len(district) == 3:
                        search = district+' '+sec[0]
                    else:
                        search = district+sec[0]

                    count = self.postcode_df['Postcode']\
                        .str.contains(search, na=False).sum()

                    unitPopulation = sectorPopulation/float(count)
                    result[i][j] = np.ceil(unitPopulation)

        result = result.tolist()
        return result
locator = PostcodeLocator('resources/full_postcodes.csv', 'resources/population_by_postcode_sector.csv')
print(locator.get_population_of_postcode([['BS99 72R']]))
# print(locator.get_postcodes_by_radius((51.4981, -0.1773), [0.13e3]))



