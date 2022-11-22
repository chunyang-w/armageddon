from .solver import *
import matplotlib.pyplot as plt
import numpy as np

def findstrengthradius(density=3300, angle=18.3, velocity=19.2,
                        data_file='./resources/ChelyabinskEnergyAltitude.csv', radians=False):
    planet_instance = Planet()
    data2 = pd.read_csv("./resources/ChelyabinskEnergyAltitude.csv")
    data2.columns = ['h','energy']
    data2['h'] = data2['h']*1000
    data2['energy'] = data2['energy']*4.184*10**12
    range_of_interest = [max(data2['h']), min(data2['h'])]
    strength = np.arange(1000,10000,10)
    radius = np.arange(1000,10000,10)
    minmse = -1
    for i in strength:
        for j in radius:
            result = planet_instance.solve_atmospheric_entry(j, velocity, 
                                                                density, i, angle, radians=radians)
            energy = planet_instance.calculate_energy(result)
            temp_energy = energy[energy['altitude']<= range_of_interest[0]][energy['altitude']>= range_of_interest[1]]
            temp_energy.loc[temp_energy.index[-1]+1] = energy.loc[temp_energy.index[-1]]
            temp_energy.loc[temp_energy.index[0]-1] = energy.loc[temp_energy.index[0]-1]
            print(temp_energy)
            quit()
            mse = calculate_error(energy['dedz'], result)

            if minmse == -1 or minmse > mse:
                minmse = mse
                minconfig = (i,j)

def calculate_error(array1, array2):
    return ((array1 - array2)**2).sum()/len(array1)