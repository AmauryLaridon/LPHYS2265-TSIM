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
### 2.1 Addition of snow on top of the ice ###
########################################### Parameterization ######################################################


def solar_flux(day):
    """Definition of the atmospheric solar heat flux Q_sol for a given day in the year. Conversion from Fletcher(1965)"""
    Q_sol = 314*np.exp((-(day-164)**2)/4608)
    return Q_sol


def non_solar_flux(day):
    """Definition of the atmospheric non solar heat flux Q_nsol for a given day in the year. Conversion from Fletcher(1965)"""
    Q_nsol = 118*np.exp((-0.5*(day-206)**2)/(53**2)) + 179
    return Q_nsol


################################################### Parameters #####################################################
### Physical Constant ###
epsilon = 0.99  # surface emissivity
sigma = 5.67e-8  # Stefan-Boltzman constant
Kelvin = 237.15  # Conversion form Celsius to Kelvin
alb = 0.8  # surface albedo
ki = 2.2  # sea ice thermal conductivity [W/m/K]
h = 0.5  # sea ice thickness

### Instancing ###
# 2.1.1

N_days = 365
year = np.arange(1, N_days, 1)
Q_sol = solar_flux(year)
Q_nsol = non_solar_flux(year)

### Display ###
plt.plot(year, Q_sol, label="Q_sol")
plt.plot(year, Q_nsol, label="Q_nsol")
plt.title("Evolution of the atmospheric solar heat flux", size=26)
plt.xlabel("Days", size=20)
plt.ylabel("Q_sol [J]", size=20)
plt.legend()
plt.grid()
plt.show()

### 2.2 Calculate surface temperature ###
# temperature at the freezing point of sea water with a salinity of 34g/kg
T_bo = -1.8 + Kelvin

########################################### Model ######################################################


def surface_temp(T_su_0):
    """Compute the evolution of the surface temperature with respect to the variation of the atmospheric
       heat fluxes."""
    T_su = np.zeros(N_days)
    T_su_init = T_su_0 + Kelvin

    def energ_bal_fun(T_su, days):
        f = -epsilon*sigma*(T_su**4) - (ki/h)*T_su + \
            solar_flux(year)[days]*(1-alb) + \
            non_solar_flux(year)[days]+(ki/h)*T_bo
        return f

    for days in year:
        root = opt.newton(energ_bal_fun(days), T_su_init)
        T_su[days] = root
    return T_su


### Instancing ###
# 2.2.1
T_su = surface_temp(T_su_0=-10.)

### Display ###
plt.plot(year, T_su, label="T_su")
plt.title("Evolution of the surface temperature", size=26)
plt.xlabel("Days", size=20)
plt.ylabel("T_su [K]", size=20)
plt.legend()
plt.grid()
plt.show()
