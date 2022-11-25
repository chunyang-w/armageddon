import folium


def plot_circle(lat, lon, radius, map=None, **kwargs):
    """
    Plot a circle on a map (creating a new folium map instance if necessary).

    Parameters
    ----------

    lat: float
        latitude of circle to plot (degrees)
    lon: float
        longitude of circle to plot (degrees)
    radius: float
        radius of circle to plot (m)
    map: folium.Map
        existing map object

    Returns
    -------

    map: Folium map object

    Examples
    --------

    >>> import folium
    >>> armageddon.plot_circle(52.79, -2.95, 1e3, map=None)
    """

    if not map:
        map = folium.Map(location=[lat, lon], control_scale=True)

    folium.Circle([lat, lon], radius, fill=True,
                  fillOpacity=0.6, **kwargs).add_to(map)

    return map


def damage_map(blat, blon, damrad, lat, lon):
    """
    Plot circles on a map as needed
    (creating a new folium map instance if necessary).

    Parameters
    ----------

    blat: float
        latitude of circle to plot (degrees)
    blon: float
        longitude of circle to plot (degrees)
    damrad: arraylike
    lat: float
        the entry point of the meteorite as a latitude (degrees)
    lon: float
        the entry point of the meteorite as a longitude (degrees)

    Returns
    -------

    map: Folium map object

    Examples
    --------

    >>> import folium
    >>> outcome = {'burst_altitude': 8e3, 'burst_energy': 7e3,\
                   'burst_distance': 90e3, 'burst_peak_dedz': 1e3,\
                   'outcome': 'Airburst'}
    >>> lat = 52.79
    >>> lon = -2.95
    >>> blat, blon, damrad = armageddon.damage_zones(\
                                        outcome, lat, lon,\
                                        135,\
                                        pressures=[1e3, 3.5e3, 27e3, 43e3])
    >>> armageddon.damage_map(blat, blon, damrad, lat, lon)
    """
    damrad = damrad[::-1]
    for rad_index in range(len(damrad)):
        if rad_index == 0:
            map = plot_circle(
                blat, blon,
                damrad[rad_index],
                map=None,
                color='red'
            )
            folium.PolyLine(
                [[lat, lon], [blat, blon]], color='black').add_to(map)
        else:
            map = plot_circle(
                blat, blon,
                damrad[rad_index],
                map
            )
    return map
