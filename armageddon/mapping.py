import folium
import numpy as np


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

    Folium map object

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
    colors = ['yellow', 'blue', 'white']
    for rad_index in range(len(damrad)):
        if rad_index == 0:
            map = plot_circle(
                blat, blon,
                damrad[rad_index],
                map=None,
                color='red'
            )
            folium.PolyLine([[lat, lon], [blat, blon]], color='black').add_to(map)
        else:
            map = plot_circle(
                blat, blon,
                damrad[rad_index],
                map,
                colors=np.roll(colors, rad_index)
            )
    return map
