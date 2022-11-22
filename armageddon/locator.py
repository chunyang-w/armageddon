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
    R_p = 6371e3
    latlon1 = np.array(latlon1) * np.pi / 180
    latlon2 = np.array(latlon2) * np.pi / 180
    if (latlon1.ndim == 1):
        latlon1 = latlon1.reshape(1, *latlon1.shape)
    if (latlon2.ndim == 1):
        latlon2 = latlon2.reshape(1, *latlon2.shape)
    lat1 = latlon1[:, 0]
    lat2 = latlon2[:, 0]
    lon1 = latlon1[:, 1]
    lon2 = latlon2[:, 1]
    lon_diff = np.abs(
        (lon1.reshape(len(lon1), 1)) -
        (lon2.reshape(1, len(lon2)))
    )
    distance = np.arccos(
        np.sin(lat1).reshape(len(lat1), 1)*np.sin(lat2).reshape(1, len(lat2)) +
        np.cos(lat1).reshape(len(lat1), 1)*np.cos(lat2).reshape(1, len(lat2)) *
        np.cos(lon_diff)
    ) * R_p
    return distance


class PostcodeLocator(object):
    """Class to interact with a postcode database file."""

    def __init__(self, postcode_file='../resources/full_postcodes.csv',
                 census_file='../resources/population_by_postcode_sector.csv',
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
        pc = np.array(postcodes)
        m,n = pc.shape
        result = np.zeros(pc.shape)

        for i in range(m):
            for j in range(n):
                district, sector = pc[i][j].split()

                if len(district) == 3:
                    searchSector = district+'  '+sector[0]
                else:
                    searchSector = district+' '+sector[0]
                    
                sectorPopulation = self.census_df.loc[self.census_df['geography'] == searchSector]['Variable: All usual residents; measures: Value']

                if sector == True:
                    result[i][j] = sectorPopulation
                else:
                    if len(district) == 3:
                        search = district+' '+sector[0]
                    else:
                        search = district+sector[0]
   
                    count = self.postcode_df['Postcode'].str.contains(search, na=False).sum()

                    unitPopulation = sectorPopulation/float(count)
                    result[i][j] = np.ceil(unitPopulation)

        result = result.tolist()
        return result
