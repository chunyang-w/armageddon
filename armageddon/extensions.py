from .solver import *
import matplotlib.pyplot as plt
import numpy as np

def findstrengthradius(density=3300, angle=18.3, velocity=19200,
                        data_file='./resources/ChelyabinskEnergyAltitude.csv', radians=False):
    planet_instance = Planet()
    data2 = pd.read_csv("./resources/ChelyabinskEnergyAltitude.csv")
    data2.columns = ['h','energy']
    data2['h'] = data2['h']*1000
    data2['energy'] = data2['energy']*4.184*10**12
    range_of_interest = [max(data2['h']), min(data2['h'])]
    strength = np.arange(0,100000, 1000)
    radius = np.arange(15,23,1)
    minmse = -1
    minconfig = (0,0)

    for i in strength:
        for j in radius:
            print(i,j, 'cur_best', minconfig)
            result = planet_instance.solve_atmospheric_entry(j, velocity, 
                                                                density, i, angle, radians=radians)
            energy = planet_instance.calculate_energy(result)
            temp_energy = energy.loc[energy['altitude']<= range_of_interest[0]].loc[energy['altitude']>= range_of_interest[1]]
            temp_energy.loc[temp_energy.index[-1]+1] = energy.loc[temp_energy.index[-1]+1]
            temp_energy.loc[temp_energy.index[0]-1] = energy.loc[temp_energy.index[0]-1]
            # print(temp_energy)
            temp_energy = temp_energy.sort_index().reset_index()
            energy_with_matched_index = np.zeros([len(data2)], dtype=float)
            curind_temp = 0
            curind_data = 0
            curval = data2['h'][0]
            while curind_temp < len(temp_energy):
                cur_alti = temp_energy['altitude'][curind_temp]
                while cur_alti <= curval:
                    prev_alti = temp_energy['altitude'][curind_temp - 1]
                    prev_energy = temp_energy['dedz'][curind_temp - 1]
                    cur_energy = temp_energy['dedz'][curind_temp]
                    inter = (curval - cur_alti) / (prev_alti - cur_alti) * (prev_energy - cur_energy)
                    energy_with_matched_index[curind_data] = inter
                    curind_data += 1
                    if curind_data >= len(data2):
                        break
                    curval = data2['h'][curind_data]
                curind_temp += 1
            # print(len(energy_with_matched_index), len(data2))

            # quit()
            mse = calculate_error(energy_with_matched_index, np.array(data2['energy']))
            if minmse == -1 or minmse > mse:
                minmse = mse
                minconfig = (i,j)
    print(minmse, minconfig)

def calculate_error(array1, array2):
    return ((array1 - array2)**2).sum()/len(array1)

def plot_against(radius, strength, density=3300, angle=18.3, velocity=19200,
                        data_file='./resources/ChelyabinskEnergyAltitude.csv', radians=False):
    data2 = pd.read_csv(data_file)
    data2.columns = ['h','energy']
    data2['h'] = data2['h']*1000
    data2['energy'] = data2['energy']*4.184*10**12
    planet_instance = Planet()
    simresult = planet_instance.solve_atmospheric_entry(radius, velocity, 
                                                                density, strength, angle, radians=radians)
    energy = planet_instance.calculate_energy(simresult)
    print(energy)
    range_of_interest = [max(data2['h']), min(data2['h'])]
    temp_energy = energy.loc[energy['altitude']<= range_of_interest[0]].loc[energy['altitude']>= range_of_interest[1]]
    temp_energy.loc[temp_energy.index[-1]+1] = energy.loc[temp_energy.index[-1]+1]
    temp_energy.loc[temp_energy.index[0]-1] = energy.loc[temp_energy.index[0]-1]
    # print(temp_energy)
    temp_energy = temp_energy.sort_index().reset_index()
    print(temp_energy)
    plt.plot(data2['energy'], data2['h'], label="True")
    plt.plot(temp_energy['dedz'], temp_energy['altitude'], label="sim")
    plt.legend()
    plt.show()