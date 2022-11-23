from .solver import Planet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial



def findstrengthradius(
        density=3300, angle=18.3, velocity=19200,
        data_file='./resources/ChelyabinskEnergyAltitude.csv',
        radians=False):
    planet_instance = Planet()
    data2 = pd.read_csv(data_file)
    data2.columns = ['h', 'energy']
    data2['h'] = data2['h']*1000
    data2['energy'] = data2['energy']
    target_ind = data2['energy'].idxmax()
    target_peak = data2['energy'][target_ind]
    target_alti = data2['h'][target_ind]
    # strength = np.arange(1000000, 30000000, 100000)
    maxstrength = 5e7
    minstrength = 1
    strengthrange = [minstrength, maxstrength]
    tol = 0.01

    radius = np.arange(1, 50, 1)
    # templist = []
    # for i in radius:
    #     print(i)
    #     beststrength, f1 = searchstrenght(planet_instance, i, velocity, density,
    #                                    angle, target_peak, target_alti, strengthrange,
    #                                    tol, radians=radians)
    #     templist.append(f1)
    # plt.plot(radius, templist)
    # plt.show()
    maxradius = 100
    minradius = 1
    radiusrange = [minradius, maxradius]
    right = radiusrange[1]
    left = radiusrange[0]
    tao = (5**0.5 - 1) / 2
    x1 = left + (1 - tao) * (right - left)
    beststrength, f1 = searchstrenght(planet_instance, x1, velocity, density,
                                       angle, target_peak, target_alti, strengthrange,
                                       tol, radians=radians)
    x2 = left + tao * (right - left)
    beststrength, f2 = searchstrenght(planet_instance, x2, velocity, density,
                                       angle, target_peak, target_alti, strengthrange,
                                       tol, radians=radians)
                                
    while right - left > tol:
        print(right, left)
        print("f:", f1, f2)
        print("x:", x1, x2)
        print()
        # print(, left, right)
        if f1 > f2:
            left = x1
            x1 = x2  
            f1 = f2
            x2 = left + tao * (right - left)
            beststrength, f2 = searchstrenght(planet_instance, x2, velocity, density,
                                               angle, target_peak, target_alti, strengthrange,
                                               tol, radians=radians)
        else:
            right = x2
            x2 = x1
            f2 = f1
            x1 = left + (1 - tao) * (right - left)
            beststrength, f1 = searchstrenght(planet_instance, x1, velocity, density,
                                               angle, target_peak, target_alti, strengthrange,
                                               tol, radians=radians)
            # cur_x, cur_f = searchstrenght(planet_instance, x1, velocity, density,
            #                             angle, target_peak, target_alti, strengthrange,
            #                             tol, radians=radians)
        # templist3.append(f2)
        # if cur_f < minmse or minmse == -1:
        #     minmse = cur_f
        #     minconfig = (cur_x, j)
        # continue
            # energy = planet_instance.calculate_energy(result)
            # temp_energy = energy
            # .loc[energy['altitude']<= range_of_interest[0]]
            # .loc[energy['altitude']>= range_of_interest[1]]
            # temp_energy.loc[temp_energy.index[-1]+1] =
            # energy.loc[temp_energy.index[-1]+1]
            # temp_energy.loc[temp_energy.index[0]-1] =
            # energy.loc[temp_energy.index[0]-1]
            # print(temp_energy)
            # temp_energy = temp_energy.sort_index().reset_index()
            # energy_with_matched_index = np.zeros([len(data2)], dtype=float)
            # curind_temp = 0
            # curind_data = 0
            # curval = data2['h'][0]
            # while curind_temp < len(temp_energy):
            #     cur_alti = temp_energy['altitude'][curind_temp]
            #     while cur_alti <= curval:
            #         prev_alti = temp_energy['altitude'][curind_temp - 1]
            #         prev_energy = temp_energy['dedz'][curind_temp - 1]
            #         cur_energy = temp_energy['dedz'][curind_temp]
            #         inter = (curval - cur_alti)
            #                 / (prev_alti - cur_alti)
            #                 * (prev_energy - cur_energy)
            #         energy_with_matched_index[curind_data] = inter
            #         curind_data += 1
            #         if curind_data >= len(data2):
            #             break
            #         curval = data2['h'][curind_data]
            #     curind_temp += 1
            # print(len(energy_with_matched_index), len(data2))

            # quit()
            # mse = calculate_error(
            #       energy_with_matched_index,
            #       np.array(data2['energy']))
            # if minmse == -1 or minmse > mse:
            #     minmse = mse
            #     minconfig = (i,j)
        # strengthlist.append(templist)
        # strengthlist2.append(templist2)
        # strengthlist3.append(templist3)
    # plt.plot(radius, templist3)
    # plt.savefig("./ragainstmse")
    # plt.close()
    print(x2, f2, beststrength)
    return x2, f2, beststrength


def calculate_error(array1, array2):
    return ((array1 - array2)**2).sum()/len(array1)

def get_rmse():
    x = [21.6192,22.3258,23.3290,25.0314,26.7576,29.3710,31.7921,34.3910,36.6733,38.0784,39.9311,40,42.1922]
    y = [0.556991,2.889170,5.341250,2.61324,23.89990,80.0292,54.5100,21.35330,5.31833,1.85465,1.095450,1.07,1.070700]
    poly = lagrange(x, y)
    coefx = Polynomial(poly.coef[::-1])

    def rmse(result):
        actu_vals = np.array(result.dedz)
        pred_vals = np.array(coefx(result.altitude))
        difference = np.subtract(actu_vals, pred_vals)
        sqre_err = np.square(difference)
        rslt_meansqre_err = sqre_err.mean()
        root_meansqre_err = np.sqrt(rslt_meansqre_err)
        return root_meansqre_err
    return rmse
rmse = get_rmse()
def plot_against(radius, strength, density=3300, angle=18.3, velocity=19200,
                 data_file='./resources/ChelyabinskEnergyAltitude.csv',
                 radians=False):
    data2 = pd.read_csv(data_file)
    data2.columns = ['h', 'energy']
    data2['h'] = data2['h'] * 1000
    data2['energy'] = data2['energy']
    planet_instance = Planet()
    simresult = planet_instance.solve_atmospheric_entry(radius, velocity,
                                                        density, strength,
                                                        angle, radians=radians)
    energy = planet_instance.calculate_energy(simresult)
    print(energy)
    range_of_interest = [max(data2['h']), min(data2['h'])]
    temp_energy = energy.loc[energy['altitude'] <= range_of_interest[0]]
    temp_energy = temp_energy.loc[energy['altitude'] >= range_of_interest[1]]
    temp_index1 = temp_energy.index[-1]+1
    temp_index2 = temp_energy.index[0]-1
    temp_energy.loc[temp_index1] = energy.loc[temp_index1]
    temp_energy.loc[temp_index2] = energy.loc[temp_index2]
    print(temp_energy)
    temp_energy = temp_energy.sort_index().reset_index()
    print(temp_energy)
    plt.plot(data2['energy'], data2['h'], label="True")
    plt.plot(temp_energy['dedz'], temp_energy['altitude'], label="sim")
    plt.legend()
    plt.show()

# data2 = pd.read_csv('./resources/ChelyabinskEnergyAltitude.csv')
# data2.columns = ['h', 'energy']
# data2['h'] = data2['h']*1000
# data2['energy'] = data2['energy']
# range_of_interest = [max(data2['h']),min(data2['h'])]
def getfunctionvalue(planet_instance, radius, velocity, density, strength,
                     angle, target_peak, target_alti, radians=False):
    planet_instance.burstpoint = -1
    result = planet_instance.solve_atmospheric_entry(radius, velocity,
                                                     density, strength,
                                                     angle, radians=radians)
    energy = planet_instance.calculate_energy(result)
    outcome = planet_instance.analyse_outcome(energy)
    peak = outcome['burst_peak_dedz']
    alti = outcome['burst_altitude']
    rate = target_alti/target_peak
    # temp_energy = energy.loc[energy['altitude']<= range_of_interest[0]].loc[energy['altitude']>= range_of_interest[1]]
    # dist = rmse(temp_energy)
    dist = np.sqrt((rate * (target_peak - peak))**2 + (target_alti - alti)**2)
    return peak, alti, dist


    

def searchstrenght(planet_instance, radius, velocity, density,
                     angle, target_peak, target_alti, strengthrange,
                     tol, radians=False):
    right = strengthrange[1]
    left = strengthrange[0]
    tao = (5**0.5 - 1) / 2
    x1 = left + (1 - tao) * (right - left)
    _, _, f1 = getfunctionvalue(planet_instance, radius, velocity,
                                density, x1, angle, target_peak,
                                target_alti, radians=radians)
    x2 = left + tao * (right - left)
    _, _, f2 = getfunctionvalue(planet_instance, radius, velocity,
                                density, x2, angle, target_peak,
                                target_alti, radians=radians)
    while right - left > tol:
        # print(radius, left, right)
        if f1 > f2:
            left = x1
            x1 = x2
            f1 = f2
            x2 = left + tao * (right - left)
            _, _, f2 = getfunctionvalue(planet_instance, radius, velocity,
                                        density, x2, angle, target_peak,
                                        target_alti, radians=radians)
        else:
            right = x2
            x2 = x1
            f2 = f1
            x1 = left + (1 - tao) * (right - left)
            _, _, f1 = getfunctionvalue(planet_instance, radius, velocity,
                                        density, x1, angle, target_peak,
                                        target_alti, radians=radians)
    return x2, f2
