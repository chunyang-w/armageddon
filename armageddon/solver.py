import os
import numpy as np
import pandas as pd


class Planet():
    """
    The class called Planet is initialised with constants appropriate
    for the given target planet, including the atmospheric density profile
    and other constants
    """

    def __init__(self, atmos_func='exponential',
                 atmos_filename=os.sep.join((os.path.dirname(__file__), '..',
                                             'resources',
                                             'AltitudeDensityTable.csv')),
                 Cd=1., Ch=0.1, Q=1e7, Cl=1e-3, alpha=0.3,
                 Rp=6371e3, g=9.81, H=8000., rho0=1.2):
        """
        Set up the initial parameters and constants for the target planet

        Parameters
        ----------
        atmos_func : string, optional
            Function which computes atmospheric density, rho, at altitude, z.
            Default is the exponential function rho = rho0 exp(-z/H).
            Options are 'exponential', 'tabular' and 'constant'

        atmos_filename : string, optional
            Name of the filename to use with the tabular atmos_func option

        Cd : float, optional
            The drag coefficient

        Ch : float, optional
            The heat transfer coefficient

        Q : float, optional
            The heat of ablation (J/kg)

        Cl : float, optional
            Lift coefficient

        alpha : float, optional
            Dispersion coefficient

        Rp : float, optional
            Planet radius (m)

        rho0 : float, optional
            Air density at zero altitude (kg/m^3)

        g : float, optional
            Surface gravity (m/s^2)

        H : float, optional
            Atmospheric scale height (m)

        """

        # Input constants
        self.Cd = Cd
        self.Ch = Ch
        self.Q = Q
        self.Cl = Cl
        self.alpha = alpha
        self.Rp = Rp
        self.g = g
        self.H = H
        self.rho0 = rho0
        self.atmos_filename = atmos_filename
        self.velocity = -1
        self.mass = -1
        self.angle = -1
        self.altitude = -1
        self.distance = -1
        self.radius = -1
        self.burstpoint = -1
        try:
            # set function to define atmoshperic density
            if atmos_func == 'exponential':
                self.rhoa = lambda x: rho0 * np.exp(-x / H)
            elif atmos_func == 'tabular':
                self.rhoa = self.create_tabular_density(
                                filename=atmos_filename)
            elif atmos_func == 'constant':
                self.rhoa = lambda x: rho0
            else:
                raise NotImplementedError(
                    "atmos_func must be 'exponential', 'tabular' or 'constant'"
                    )
        except NotImplementedError:
            print("atmos_func {} not implemented yet.".format(atmos_func))
            print("Falling back to constant density atmosphere for now")
            self.rhoa = lambda x: rho0

    def solve_atmospheric_entry(
            self, radius, velocity, density, strength, angle,
            init_altitude=100e3, dt=0.05, radians=False,
            backend="RK4", hard=False):
        """
        Solve the system of differential equations for a given impact scenario

        Parameters
        ----------
        radius : float
            The radius of the asteroid in meters

        velocity : float
            The entery speed of the asteroid in meters/second

        density : float
            The density of the asteroid in kg/m^3

        strength : float
            The strength of the asteroid (i.e. the maximum pressure it can
            take before fragmenting) in N/m^2

        angle : float
            The initial trajectory angle of the asteroid to the horizontal
            By default, input is in degrees. If 'radians' is set to True, the
            input should be in radians

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

        hard : bool, optional
            if True, the solver will use the passed in stepsize.

        Returns
        -------
        result : DataFrame
            A pandas dataframe containing the solution to the system.
            Includes the following columns:
            'velocity', 'mass', 'angle', 'altitude',
            'distance', 'radius', 'time'
        """

        # Enter your code here to solve the differential equations
        if not radians:
            angle = angle/180 * np.pi
        self.strength = strength
        self.density = density
        self.burstpoint = -1
        if backend == "FE":
            solver = self.solve_atmospheric_entry_FE
        elif backend == "RK4":
            solver = self.solve_atmospheric_entry_RK4
        else:
            try:
                raise NotImplementedError(
                        "backend must be 'FE' or 'RK4' "
                        )
            except NotImplementedError:
                print("solving method {} not implemented yet.".format(backend))
                print("Falling back to FE for now")
                solver = self.solve_atmospheric_entry_FE
        if dt >= 0.02:
            tempdt = 0.02
            if hard:
                tempdt = dt
            solver(radius, velocity, angle,
                   init_altitude, tempdt, dt)
        else:
            self.solve_atmospheric_entry_FE(radius, velocity, angle,
                                            init_altitude, dt, dt)
        if not radians:
            all_angle = [i/np.pi * 180 for i in self.angle]
        else:
            all_angle = self.angle
        result = pd.DataFrame({'velocity': self.velocity,
                               'mass': self.mass,
                               'angle': all_angle,
                               'altitude': self.altitude,
                               'distance': self.distance,
                               'radius': self.radius,
                               'time': self.alltimestep})
        n = len(result)
        if self.stopping(
                result.loc[n-1, "velocity"],
                result.loc[n-1, "mass"],
                result.loc[n-1, "altitude"],
                result.loc[n-1, 'radius']):
            if result.loc[n-1, "altitude"] <= 0:
                pass
        return result

    def calculate_energy(self, result):
        """
        Function to calculate the kinetic energy lost per unit altitude in
        kilotons TNT per km, for a given solution.

        Parameters
        ----------
        result : DataFrame
            A pandas dataframe with columns for the velocity, mass, angle,
            altitude, horizontal distance and radius as a function of time

        Returns
        -------
        result : DataFrame
            Returns the dataframe with additional column ``dedz`` which is the
            kinetic energy lost per unit altitude

        """

        # Replace these lines with your code to add the dedz column to
        # the result DataFrame
        mass = result["mass"]
        velocity = result["velocity"]
        altitude = np.array(result["altitude"])
        dedz = np.array(0.5 * mass / (10**5) * velocity /
                        (10**4) * velocity / 4.184)
        temp = (dedz[1:] - dedz[:-1]) / (altitude[:-1] - altitude[1:])
        temp = temp
        temp = np.insert(temp, 0, 0)
        result.insert(len(result.columns),
                      'dedz', -temp)
        return result

    def analyse_outcome(self, result):
        """
        Inspect a pre-found solution to calculate the impact and airburst stats
        Parameters
        ----------
        result : DataFrame
            pandas dataframe with velocity, mass, angle, altitude, horizontal
            distance, radius and dedz as a function of time

        Returns
        -------
        outcome : Dict
            dictionary with details of the impact event, which should contain
            the key:
                ``outcome`` (which should contain one of the
                following strings: ``Airburst`` or ``Cratering``),
            as well as the following 4 keys:
                ``burst_peak_dedz``, ``burst_altitude``,
                ``burst_distance``, ``burst_energy``
        """
        burstidx = result['dedz'].idxmax()
        initial_energy = (0.5 * result["mass"][0]
                          * result["velocity"][0]**2 / (4.184*10**12))
        burstenergy = (0.5 * result["mass"][burstidx]
                       * result["velocity"][burstidx]**2 / (4.184*10**12))
        outcome = "Airburst"
        if burstidx == len(result) - 1 and result['altitude'][burstidx] <= 100:
            outcome = "Cratering"
            burstenergy = max(burstenergy, initial_energy - burstenergy)
            burst_altitude = 0
        else:
            burstenergy = initial_energy - burstenergy
            burst_altitude = result["altitude"][burstidx]
        outcome = {'outcome': outcome,
                   'burst_peak_dedz': result['dedz'][burstidx],
                   'burst_altitude': burst_altitude,
                   'burst_distance': result["distance"][burstidx],
                   'burst_energy': burstenergy}
        return outcome

    def create_tabular_density(
            self,
            filename="./resources/AltitudeDensityTable.csv"):
        """
        Create a function given altitude return the density of atomosphere
        using tabulated value
        Parameters
        ----------
        filename : str, optional
            Path to the tabular. default="./resources/AltitudeDensityTable.csv"
        Returns
        -------
        tabular_density : function
            A function that takes altitude as input and return the density of
            atomosphere density at given altitude.
        """
        X = []
        Y = []
        data = pd.read_csv(filename)
        for i in data[data.keys()[0]]:
            temp = i.split()
            X.append(eval(temp[0]))
            Y.append(eval(temp[1]))

        def tabular_density(x):
            if x > 100e3:
                return 0
            if x > X[-1]:
                pressure = (x - X[-1])/(100e3 - X[-1]) * (0 - Y[-1]) + Y[-1]
            for i in range(len(X)):
                if X[i] >= x:
                    break
            pressure = (x - X[i-1])/(X[i] - X[i-1]) * (Y[i] - Y[i-1]) + Y[i-1]

            return pressure
        return tabular_density

    def solve_atmospheric_entry_RK4(
            self, radius, velocity, angle,
            init_altitude, dt, actualdt):
        """
        Solve the system of differential equations for a given impact scenario
        using RK4 method

        Parameters
        ----------
        radius : float
            The radius of the asteroid in meters

        velocity : float
            The entery speed of the asteroid in meters/second

        angle : float
            The initial trajectory angle of the asteroid to the horizontal
            By default, input is in degrees. If 'radians' is set to True, the
            input should be in radians

        init_altitude : float, optional
            Initial altitude in m

        dt : float, optional
            The output timestep, in s

        Returns
        -------
        None
        """

        # Enter your code here to solve the differential equations
        self.velocity = [velocity]
        self.mass = [4/3 * np.pi * radius**3 * self.density]
        self.angle = [angle]
        self.altitude = [init_altitude]
        self.distance = [0]
        self.radius = [radius]
        self.solver_param = np.array([angle,
                                      radius,
                                      init_altitude,
                                      velocity,
                                      4/3 * np.pi * radius**3 * self.density,
                                      0])
        self.alltimestep = [0]
        timestep = dt
        iter_num = 0
        acumulated_step = 0
        while True:
            cpara = self.RK4_helper(timestep)
            flag = np.isclose(acumulated_step + timestep, actualdt)
            if acumulated_step + timestep >= actualdt or flag:
                rate = (actualdt - acumulated_step)/timestep
                output_para = rate * cpara + self.solver_param
                self.angle.append(output_para[0])
                self.radius.append(output_para[1])
                self.altitude.append(output_para[2])
                self.velocity.append(output_para[3])
                self.mass.append(output_para[4])
                self.distance.append(output_para[5])
                self.alltimestep.append(actualdt + self.alltimestep[-1])
                acumulated_step -= actualdt
            self.solver_param += cpara
            if self.stopping(self.solver_param[3], self.solver_param[4],
                             self.solver_param[2], self.solver_param[1]):
                break
            iter_num += 1
            acumulated_step += timestep
            if flag:
                acumulated_step = 0

    def RK4_helper(self, timestep):
        """
        Helper function for RK4 method

        Parameters
        ----------
        timestep : float
            The stepsize of iteration

        Returns
        -------
        change : ndarray
            A numpy array containing the change of each variable.
            Includes the following variables:
            'angle', 'radius', 'altitude',
            'velocity', 'mass', 'distance'
        """
        variables = self.solver_param
        k1 = self.calculator_rk4(variables)
        k2 = self.calculator_rk4(variables + 0.5 * timestep * k1)
        k3 = self.calculator_rk4(variables + 0.5 * timestep * k2)
        k4 = self.calculator_rk4(variables + timestep * k3)
        change = (k1 + 2 * k2 + 2 * k3 + k4) * timestep / 6
        return change

    def calculator_rk4(self, variables):
        """
        Calculate the change of variables at given point

        Parameters
        ----------
        variables : float
            Angle, radius, altitude, velocity, mass, distance at currenty step

        Returns
        -------
        result : ndarray
            A numpy array containing the change of each variable.
            Includes the following variables:
            'angle', 'radius', 'altitude',
            'velocity', 'mass', 'distance'
        """
        angle, radius, altitude, velocity, mass, _ = variables
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        area = np.pi * radius**2
        rhoa = self.rhoa(altitude)
        rhoAv = rhoa * area * velocity
        dvdt = -self.Cd * rhoAv * velocity / (2 * mass) + self.g * sin_theta
        dmdt = -self.Ch * rhoAv * velocity**2 / (2 * self.Q)
        dthetadt = (-self.Cl * rhoAv / (2 * mass)
                    + self.g * cos_theta / velocity
                    - velocity * cos_theta / (self.Rp + altitude))
        dzdt = -velocity * sin_theta
        dxdt = velocity * cos_theta / (1 + altitude / self.Rp)
        ram = self.rhoa(altitude) * velocity**2
        drdt = 0
        if ram > self.strength:
            if self.burstpoint == -1:
                self.burstpoint = len(self.distance)
            drdt = (7 * rhoa * self.alpha/2/self.density)**0.5 * velocity
        result = np.array([dthetadt, drdt, dzdt, dvdt, dmdt, dxdt])
        return result

    def solve_atmospheric_entry_FE(
            self, radius, velocity, angle,
            init_altitude, dt, dtextra):
        """
        Solve the system of differential equations for a given impact scenario
        using forward Eular method

        Parameters
        ----------
        radius : float
            The radius of the asteroid in meters

        velocity : float
            The entery speed of the asteroid in meters/second

        angle : float
            The initial trajectory angle of the asteroid to the horizontal
            By default, input is in degrees. If 'radians' is set to True, the
            input should be in radians

        init_altitude : float, optional
            Initial altitude in m

        dt : float, optional
            The output timestep, in s

        Returns
        -------
        None
        """

        # Enter your code here to solve the differential equations
        self.velocity = [velocity]
        self.mass = [4/3 * np.pi * radius**3 * self.density]
        self.angle = [angle]
        self.altitude = [init_altitude]
        self.distance = [0]
        self.radius = [radius]
        timestep = dt
        self.alltimestep = [0]
        while True:
            cos_theta = np.cos(self.angle[-1])
            sin_theta = np.sin(self.angle[-1])
            area = np.pi * self.radius[-1]**2
            rhoa = self.rhoa(self.altitude[-1])
            rhoAv = rhoa * area * self.velocity[-1]
            dvdt = (-self.Cd * rhoAv * self.velocity[-1] / (2 * self.mass[-1])
                    + self.g * sin_theta)
            dmdt = -self.Ch * rhoAv * self.velocity[-1]**2 / (2 * self.Q)
            dthetadt = (-self.Cl * rhoAv / (2 * self.mass[-1])
                        + self.g * cos_theta / self.velocity[-1]
                        - self.velocity[-1] * cos_theta
                        / (self.Rp + self.altitude[-1]))
            dzdt = -self.velocity[-1] * sin_theta
            dxdt = (self.velocity[-1] * cos_theta
                    / (1 + self.altitude[-1] / self.Rp))
            ram = self.rhoa(self.altitude[-1]) * self.velocity[-1]**2
            drdt = 0
            if ram > self.strength:
                if self.burstpoint == -1:
                    self.burstpoint = len(self.distance)
                drdt = ((7 * rhoa * self.alpha/2/self.density)**0.5
                        * self.velocity[-1])
            self.velocity.append(dvdt * timestep + self.velocity[-1])
            self.mass.append(dmdt * timestep + self.mass[-1])
            self.altitude.append(dzdt * timestep + self.altitude[-1])
            self.angle.append(dthetadt * timestep + self.angle[-1])
            self.distance.append(dxdt * timestep + self.distance[-1])
            self.radius.append(drdt * timestep + self.radius[-1])
            self.alltimestep.append(timestep + self.alltimestep[-1])
            if (self.altitude[-1] <= 0 or self.mass[-1] <= 0 or
                    self.radius[-1] <= 0 or self.velocity[-1] <= 0 or
                    self.altitude[-1] >= init_altitude):
                break

    def stopping(self, newv, newm, newal, newradius):
        """
        Decide whether to stop the solver

        Parameters
        ----------
        newv : float
            The current velocity

        newm : float
            The current mass

        newal : float
            The current altitude

        newradius : float
            The current radius

        Returns
        -------
        Result: bool
            True if solver should stop,
            False if solver should continue
        """
        if (newal <= 0 or newm <= 0 or
                newradius <= 0 or newv <= 0 or
                newal >= self.altitude[0]):
            return True
        else:
            return False
