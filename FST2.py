############################################################################################################################
# Freeing Surface Temperature (FST)
# Author : Amaury Laridon
# Course : LPHYS2269 - Remote Sensing of Climate Change
# Goal : Second part of the TSIM model. Modelisation of the evolution of sea-ice thickness with a dynamic surface temperature
#        More information on the GitHub Page of the project : https://github.com/AmauryLaridon/TSIM.git
# Date : 10/03/23
############################################################################################################################
#################################################### Packages ##############################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
######################################## 2 Freeing surface temperature ############################################
################################################### Parameters #####################################################
### Physical Constant ###
epsilon = 0.99  # surface emissivity
sigma = 5.67e-8  # Stefan-Boltzman constant
Kelvin = 273.15  # Conversion form Celsius to Kelvin
alb = 0.8  # surface albedo
ki = 2.2  # sea ice thermal conductivity [W/m/K]
### Simulation parameters ###
N_days = 365  # number of days in the simulation
h = 0.5  # sea ice thickness
# temperature at the freezing point of sea water with a salinity of 34g/kg
T_bo = -1.8 + Kelvin
Day_0 = 1
### Display Parameters ###
plt.rcParams['text.usetex'] = True
save_dir = "/home/amaury/Bureau/LPHYS2265 - Sea ice ocean atmosphere interactions in polar regions/Projet/Figures/"
figure = plt.figure(figsize=(16, 10))

########################################### Parameterization ######################################################


def solar_flux(day):
    """Definition of the atmospheric solar heat flux Q_sol for a given day in the year. Conversion from Fletcher(1965)"""
    Q_sol = 314*np.exp((-(day-164)**2)/4608)
    return Q_sol


def non_solar_flux(day):
    """Definition of the atmospheric non solar heat flux Q_nsol for a given day in the year. Conversion from Fletcher(1965)"""
    Q_nsol = 118*np.exp((-0.5*(day-206)**2)/(53**2)) + 179
    return Q_nsol


def surface_temp(T_su_0):
    """Compute the evolution of the surface temperature with respect to the variation of the atmospheric
    heat fluxes"""

    # array collecting the surface temperature for each day
    T_su = np.zeros(N_days)
    # T_su_init = T_su_0 + Kelvin
    # T_su[0] = T_su_init
    solar_flux_ar = np.zeros(N_days)
    non_solar_flux_ar = np.zeros(N_days)

    for day in range(N_days):
        solar_flux_ar[day] = solar_flux(day)
        non_solar_flux_ar[day] = non_solar_flux(day)

    """ def energ_bal_fun(T_su, day):
        f = -epsilon*sigma*(T_su**4) - (ki/h)*T_su + \
            solar_flux_ar[day]*(1-alb) + non_solar_flux_ar[day]+(ki/h)*T_bo
        return f """

    for day in range(0, N_days):
        # print(T_su)
        # print(day)
        # finding the surface temperature using roots function from numpy. As required in 2.1.2 temperature is not
        # physically sensible for ice in summer so we cap the surface temperature to 273,15°K.
        root = min([273.15, np.roots([-epsilon * sigma, 0, 0, -ki/h, ki /
                                      h * T_bo + solar_flux_ar[day] * (1-alb) + non_solar_flux_ar[day]]).real[3]])
        # root = opt.newton(energ_bal_fun(day), T_su[day-1]) ## root computation with newton method from scipy
        T_su[day] = root

    return T_su


########################################### Cases of Simulations ######################################################

### 2.1 Surface heat fluxes ###


def exo2_1_surf_heat_flux():
    """Parameterisation of the surface heat fluxes.
    Section 2.1 of the Exercise_part_1.pdf file available on the GitHub."""
    ##### 2.1.1 #####
    ### Instancing ###
    year = np.arange(1, N_days, 1)
    Q_sol = solar_flux(year)
    Q_nsol = non_solar_flux(year)

    ### Display ###
    plt.plot(year, Q_sol, label="Q_sol")
    plt.plot(year, Q_nsol, label="Q_nsol")
    plt.title("Evolution of the atmospheric solar heat flux", size=26)
    plt.xlabel("Days", size=20)
    plt.ylabel("Q_sol [J]", size=20)
    plt.legend(fontsize=18)
    plt.grid()
    plt.savefig(save_dir + "2.1.1.png", dpi=300)
    # plt.show()
    plt.clf()


### 2.2 Calculate surface temperature ###

def exo2_2_surf_temp():
    """Computation of the surface temperature.
    Section 2.2 of the Exercise_part_1.pdf file available on the GitHub."""

    ##### 2.2.1 #####
    ### Instancing ###
    T_su = surface_temp(T_su_0=-10.)
    year = np.arange(Day_0, N_days + 1)
    ### Display ###
    plt.plot(year, T_su - Kelvin, label="T_su")
    plt.title("Evolution of the surface temperature", size=26)
    plt.xlabel("Days", size=20)
    plt.ylabel("T_su [°C]", size=20)
    plt.legend(fontsize=18)
    plt.grid()
    plt.savefig(save_dir + "2.1.2.png", dpi=300)
    # plt.show()
    plt.clf()

### 2.3 Couple temperature and thickness ###


if __name__ == "__main__":
    exo2_1_surf_heat_flux()
    exo2_2_surf_temp()
