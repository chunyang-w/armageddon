from .solver import Planet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def findstrengthradius(
        density=3300, angle=18.3, velocity=19200, init_altitude=1e5,
        dt=0.05, data_file='./resources/ChelyabinskEnergyAltitude.csv',
        backend="FE", radians=False):
    """
        Find the optimal radius and strength of the input dataset that
        minimize the error with respect to the given density, angle, velocity
        and radians

        Parameters
        ----------
        velocity : float
            The entery speed of the asteroid in meters/second

        density : float
            The density of the asteroid in kg/m^3

        angle : float
            The initial trajectory angle of the asteroid to the horizontal
            By default, input is in degrees. If 'radians' is set to True, the
            input should be in radians

        init_altitude : float, optional
            Initial altitude in m

        dt : float, optional
            The output timestep, in s

        data_file: str, optional
            file contains the data of interest

        radians : logical, optional
            Whether angles should be given in degrees or radians. Default=False
            Angles returned in the dataframe will have the same units as the
            input

        backend : str, optional
            Which solving method to use. Default='FE'

        Returns
        -------
        best_radius : float
            best radius that minimize the error

        minerror : float
            minimum error found

        beststrength: float
            best strength that minimize the error
    """
    planet_instance = Planet()
    data2 = pd.read_csv(data_file)
    data2.columns = ['h', 'energy']
    data2['h'] = data2['h']*1000
    data2['energy'] = data2['energy']
    target_ind = data2['energy'].idxmax()
    target_peak = data2['energy'][target_ind]
    target_alti = data2['h'][target_ind]
    maxstrength = 5e7
    minstrength = 1
    strengthrange = [minstrength, maxstrength]
    tol = 0.01
    maxradius = 100
    minradius = 1
    radiusrange = [minradius, maxradius]
    right = radiusrange[1]
    left = radiusrange[0]
    tao = (5**0.5 - 1) / 2
    x1 = left + (1 - tao) * (right - left)
    beststrength, f1 = searchstrength(planet_instance, x1, velocity,
                                      density, angle, target_peak,
                                      target_alti, strengthrange,
                                      tol, init_altitude, dt, backend,
                                      radians=radians)
    x2 = left + tao * (right - left)
    beststrength, f2 = searchstrength(planet_instance, x2, velocity,
                                      density, angle, target_peak,
                                      target_alti, strengthrange,
                                      tol, init_altitude, dt, backend,
                                      radians=radians)

    while right - left > tol:
        print(right, left)
        print("f:", f1, f2)
        print("x:", x1, x2)
        print()
        if f1 > f2:
            left = x1
            x1 = x2
            f1 = f2
            x2 = left + tao * (right - left)
            beststrength, f2 = searchstrength(planet_instance, x2, velocity,
                                              density, angle, target_peak,
                                              target_alti, strengthrange,
                                              tol, init_altitude, dt, backend,
                                              radians=radians)
        else:
            right = x2
            x2 = x1
            f2 = f1
            x1 = left + (1 - tao) * (right - left)
            beststrength, f1 = searchstrength(planet_instance, x1, velocity,
                                              density, angle, target_peak,
                                              target_alti, strengthrange,
                                              tol, init_altitude, dt, backend,
                                              radians=radians)
    print(x2, f2, beststrength)
    best_radius = x2
    minerror = f2
    return best_radius, minerror, beststrength


def plot_against(radius, strength, density=3300, angle=18.3, velocity=19200,
                 init_altitude=1e5, dt=0.05, backend='FE',
                 data_file='./resources/ChelyabinskEnergyAltitude.csv',
                 radians=False):
    """
        Plot the difference between the simulation using given input and the
        given data

        Parameters
        ----------
        radius: float
            The redius of the asteroid in meters

        strength: float
            The strength of the asteroid in N/m^2

        density : float
            The density of the asteroid in kg/m^3

        angle : float
            The initial trajectory angle of the asteroid to the horizontal
            By default, input is in degrees. If 'radians' is set to True, the
            input should be in radians

        velocity : float
            The entery speed of the asteroid in meters/second

        init_altitude : float, optional
            Initial altitude in m

        dt : float, optional
            The output timestep, in s

        data_file: str, optional
            file contains the data of interest

        radians : logical, optional
            Whether angles should be given in degrees or radians. Default=False
            Angles returned in the dataframe will have the same units as the
            input

        backend : str, optional
            Which solving method to use. Default='FE'

        Returns
        -------
        None
    """
    data2 = pd.read_csv(data_file)
    data2.columns = ['h', 'energy']
    data2['h'] = data2['h'] * 1000
    data2['energy'] = data2['energy']
    planet_instance = Planet()
    simresult = planet_instance.solve_atmospheric_entry(
        radius, velocity, density, strength, angle,
        init_altitude=init_altitude,
        dt=dt, radians=radians, backend=backend, hard=True)
    energy = planet_instance.calculate_energy(simresult)
    range_of_interest = [max(data2['h']), min(data2['h'])]
    temp_energy = energy.loc[energy['altitude'] <= range_of_interest[0]]
    temp_energy = temp_energy.loc[energy['altitude'] >= range_of_interest[1]]
    temp_index1 = temp_energy.index[-1]+1
    temp_index2 = temp_energy.index[0]-1
    temp_energy.loc[temp_index1] = energy.loc[temp_index1]
    temp_energy.loc[temp_index2] = energy.loc[temp_index2]
    temp_energy = temp_energy.sort_index().reset_index()
    plt.plot(data2['energy'], data2['h'], label="True")
    plt.plot(temp_energy['dedz'], temp_energy['altitude'], label="sim")
    plt.legend()
    plt.show()


def getfunctionvalue(planet_instance, radius, velocity, density, strength,
                     angle, target_peak, target_alti, init_altitude, dt,
                     backend, radians=False):
    """
        Calculate peak dedz, burst altitude and distance between
        burst point and true data

        Parameters
        ----------
        planet_instance: object
            The Planet object

        radius: float
            The redius of the asteroid in meters

        velocity : float
            The entery speed of the asteroid in meters/second

        density : float
            The density of the asteroid in kg/m^3

        strength: float
            The strength of the asteroid in N/m^2

        angle : float
            The initial trajectory angle of the asteroid to the horizontal
            By default, input is in degrees. If 'radians' is set to True, the
            input should be in radians

        target_peak: float
            The peak dedz value of the true data

        target_alti: float
            The altitude of the peak dedz of the true data

        init_altitude : float, optional
            Initial altitude in m

        dt : float, optional
            The output timestep, in s

        radians : logical, optional
            Whether angles should be given in degrees or radians. Default=False
            Angles returned in the dataframe will have the same units as the
            input

        backend : str, optional
            Which solving method to use. Default='FE'

        Returns
        -------
        peak: float
            The peak dedz value of the simulated solution

        alti: float
            The altitude where the peak dedz of the simulated solution happens

        dist: float
            The distance between the simulation peak and true peak
    """
    planet_instance.burstpoint = -1
    result = planet_instance.solve_atmospheric_entry(
                radius, velocity,
                density, strength, angle,
                init_altitude=init_altitude,
                dt=dt, radians=radians,
                backend=backend, hard=True)
    energy = planet_instance.calculate_energy(result)
    outcome = planet_instance.analyse_outcome(energy)
    peak = outcome['burst_peak_dedz']
    alti = outcome['burst_altitude']
    rate = target_alti/target_peak
    dist = np.sqrt((rate * (target_peak - peak))**2 + (target_alti - alti)**2)
    return peak, alti, dist


def searchstrength(planet_instance, radius, velocity, density,
                   angle, target_peak, target_alti, strengthrange,
                   tol, init_altitude, dt, backend, radians=False):
    """
        Find the best strength that minimize the error
        for a given configuration

        Parameters
        ----------
        planet_instance: object
            The Planet object

        radius: float
            The redius of the asteroid in meters

        velocity : float
            The entery speed of the asteroid in meters/second

        density : float
            The density of the asteroid in kg/m^3

        angle : float
            The initial trajectory angle of the asteroid to the horizontal
            By default, input is in degrees. If 'radians' is set to True, the
            input should be in radians

        target_peak: float
            The peak dedz value of the true data

        target_alti: float
            The altitude of the peak dedz of the true data

        strengthrange: list
            The search range of strength

        tol: float
            The tolarence of the stopping criteria

        init_altitude : float, optional
            Initial altitude in m

        dt : float, optional
            The output timestep, in s

        radians : logical, optional
            Whether angles should be given in degrees or radians. Default=False
            Angles returned in the dataframe will have the same units as the
            input

        backend : str, optional
            Which solving method to use. Default='FE'

        Returns
        -------
        best_strength: float
            best strength that minimize the distance

        best_dist: float
            The minimum distance between the simulation peak and true peak
    """
    right = strengthrange[1]
    left = strengthrange[0]
    tao = (5**0.5 - 1) / 2
    x1 = left + (1 - tao) * (right - left)

    _, _, f1 = getfunctionvalue(planet_instance, radius, velocity,
                                density, x1, angle, target_peak,
                                target_alti, init_altitude, dt,
                                backend, radians=radians)
    x2 = left + tao * (right - left)
    _, _, f2 = getfunctionvalue(planet_instance, radius, velocity,
                                density, x2, angle, target_peak,
                                target_alti, init_altitude, dt,
                                backend, radians=radians)
    while right - left > tol:
        # print(radius, left, right)
        if f1 > f2:
            left = x1
            x1 = x2
            f1 = f2
            x2 = left + tao * (right - left)
            _, _, f2 = getfunctionvalue(planet_instance, radius, velocity,
                                        density, x2, angle, target_peak,
                                        target_alti, init_altitude, dt,
                                        backend, radians=radians)
        else:
            right = x2
            x2 = x1
            f2 = f1
            x1 = left + (1 - tao) * (right - left)
            _, _, f1 = getfunctionvalue(planet_instance, radius, velocity,
                                        density, x1, angle, target_peak,
                                        target_alti, init_altitude, dt,
                                        backend, radians=radians)
    best_strength = x2
    best_dist = f2
    return best_strength, best_dist
