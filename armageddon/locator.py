"""Module dealing with postcode information."""

import numpy as np
import pandas as pd
import os

__all__ = ['PostcodeLocator', 'great_circle_distance', 'get_sector_code']


def get_sector_code(code):
    code = code[:-2]
    code = code.replace(' ', '')
    code = code.replace(' ', '')
    code = code[:-1] + ' ' * (4 - len(code[:-1])) + code[-1]
    return code


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
    distance: numpy.ndarray
        Distance in metres between each pair of points (as an n x m array)
    Examples
    --------
    >>> import numpy
    >>> fmt = lambda x: numpy.format_float_scientific(x, precision=3)
    >>> with numpy.printoptions(formatter={'all': fmt}):\
    print(great_circle_distance([[54.0, 0.0], [55, 0.0]], [55, 1.0]))
    [[1.286e+05]
     [6.378e+04]]
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

    def __init__(self, postcode_file=os.sep.join((
                                     os.path.dirname(__file__), '..',
                                     'resources',
                                     'full_postcodes.csv')),
                 census_file=os.sep.join((
                             os.path.dirname(__file__), '..',
                             'resources',
                             'population_by_postcode_sector.csv')),
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
            lambda row: get_sector_code(row['Postcode']), axis=1
        )
        self.census_df = pd.read_csv(census_file)
        self.sector_sq = self.postcode_df.value_counts('Sector_Postcode')
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
        >>> locator = PostcodeLocator(\
                        'resources/full_postcodes.csv',\
                        'resources/population_by_postcode_sector.csv')
        >>> postcodes = locator.get_postcodes_by_radius(\
                                (51.4981, -0.1773), [0.13e3])
        >>> postcode_dictionaries = [dict.fromkeys(postcodes[i], "risk")\
                                        for i in range(len(postcodes))]
        >>> ans1 = [{'SW7 5HG': 'risk','SW7 2BU': 'risk','SW7 5HQ': 'risk',\
                    'SW7 2BT': 'risk','SW7 5HF': 'risk','SW7 2DD': 'risk',\
                    'SW7 2AZ': 'risk'}]
        >>> postcode_dictionaries == ans1
        True
        >>> postcodes = locator.get_postcodes_by_radius(\
                                (51.4981, -0.1773), [0.4e3, 0.2e3], True)
        >>> postcode_dictionaries = [dict.fromkeys(postcodes[i], "risk")\
                                        for i in range(len(postcodes))]
        >>> ans2 = [{'SW7 4': 'risk','SW7 5': 'risk','SW7 3': 'risk',\
                    'SW7 1': 'risk','SW7 9': 'risk', 'SW7 2': 'risk'},\
                   {'SW7 4': 'risk','SW7 5': 'risk','SW7 3': 'risk',\
                    'SW7 1': 'risk','SW7 9': 'risk','SW7 2': 'risk'}]
        >>> postcode_dictionaries == ans2
        True
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
            place_list.append(list(
                set(df[df['Distance'] < r][selector].to_list())
            ))
        return place_list

    def get_postcode_count(self, sec_code):
        return self.sector_sq[sec_code]

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
        global_pc: list of lists
            Contains the populations of input postcode units or sectors
        Examples
        --------
        >>> locator = PostcodeLocator('resources/full_postcodes.csv',\
                                      'resources/population_by_postcode_sector.csv')
        >>> pop1 = locator.get_population_of_postcode(\
                            [['SW7 2AZ', 'SW7 2BT', 'SW7 2BU', 'SW7 2DD']])
        >>> pop1
        [[19.0, 19.0, 19.0, 19.0]]
        >>> pop2 = locator.get_population_of_postcode([['SW7  2']], True)
        >>> pop2
        [[2283.0]]
        """
        col = 'Variable: All usual residents; measures: Value'
        global_pc = []
        for pc_list in postcodes:
            nested_pc = []
            for pc in pc_list:
                outcode = None
                remainder = None
                if (len(pc) == 7):  # postcode
                    outcode = pc[:-3].strip()
                    remainder = pc[-3:]
                else:  # sector code
                    outcode = pc[:-1].strip()
                    remainder = pc[-1]
                sec_digit = remainder[0]
                outcode = outcode + ' ' * (5 - len(outcode))
                sec_code = outcode + sec_digit
                target = self.census_df[
                    self.census_df['geography code'] == sec_code
                ]
                if (target.shape[0] == 0):
                    nested_pc.append(0)
                else:
                    if (sector is True):
                        nested_pc.append(float(target[col].values[0]))
                    else:
                        outcode = outcode.strip()
                        outcode = outcode + ' ' * (4 - len(outcode))
                        sec_code = outcode + sec_digit
                        pc_count = self.get_postcode_count(sec_code)
                        nested_pc.append(float(round(
                            target[col].values[0] / pc_count)))
            global_pc.append(nested_pc)
        return global_pc
