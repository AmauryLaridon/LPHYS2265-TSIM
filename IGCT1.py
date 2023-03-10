############################################################################################################################
# Ice Growth with Constant Temperature (IGCT)
# Author : Amaury Laridon
# Course : LPHYS2269 - Remote Sensing of Climate Change
# Goal : First part of the TSIM model. Modelisation of the evolution of sea-ice thickness using constant
#        temperature and the Stefan's law. An ocean heat flux can be simulate.
#        More information on the GitHub Page of the project : https://github.com/AmauryLaridon/TSIM.git
# Date : 10/03/23
############################################################################################################################
#################################################### Packages ##############################################################
import numpy as np
import matplotlib.pyplot as plt
################################################### Parameters #############################################################
### Physical Constant ###
L_fus = 3.35e5  # Latent heat of fusion for water [J/kg]
rhoi = 917  # Sea ice density [kg/m³]
ki = 2.2  # Sea ice thermal conductivity [W/m/K]
ks = 0.31  # Snow thermal conductivity [W/m/K]
sec_per_day = 86400  # Seconds in one day [s/day]
### Bottom boundary conditions ###
T_bo = -1.8 + 273.15  # Bottom temperature [K]

######################################## 1 Ice growth with constant temperature ############################################
##################################################### Model ################################################################
### 1.1 Stefan and numerical solution for ice growth ###
# arry with the values of the sea ice thickness for the 30 days when we want to compute
N_days = 30
h_i = np.zeros(N_days)
h_i[0] = 0.1  # initial condition for the sea ice thickness for the first day [m]
T_air = -10 + 273.15  # air temperature [K]


def fourier_cond_flux(h, T_f, T_su):
    """Computation of the conductive heat flux trough the ice using the Fourier-Fick's law"""
    F_c = ((T_f - T_su)/(h))*ki
    return F_c


def ice_thick(h_i, ocean_heat_flux=False, F_w=0):
    """Computation of the evolution of the sea ice thickness using Stefan's law. An option gives the possibility to add an Oceanic heat flux"""
    print("Evolution of the sea ice thickness")
    print("-----------------------------")
    for t in range(1, N_days):
        print("Day = {}/{}".format(t, N_days))
        print("h_i[t-1] = ", h_i[t-1])
        if ocean_heat_flux:
            Q_c = fourier_cond_flux(h_i[t-1], T_bo, T_air)
        else:
            Q_c = fourier_cond_flux(h_i[t-1], T_bo, T_air) + F_w
        print("Q_c = ", Q_c)
        delta_h = Q_c * (1/rhoi*L_fus)
        print("delta_h = ", delta_h)
        h_new = h_i[t-1] + delta_h
        print("h_new = ", h_new)
        h_i[t] = h_new
        print("-----------------------------")
    return h_i


def stefan_law(t_0, T, h_0, k, deltaT, rhoi, L):
    time = np.arange(t_0, T, 1)
    H_t = np.zeros(N_days)
    for t in time:
        h_t = np.sqrt(h_0**2 + (2*k*deltaT*t)/(rhoi*L))
        H_t[t] = h_t
    return H_t


evo_h_i = ice_thick(h_i)  # instanciation modèle numérique
evo_h_i_law = stefan_law(1, 30, h_i[0], ki, T_bo - T_air, rhoi, L_fus)

## Affichage ##
# 1.1.1
plt.plot(np.arange(0, 30, 1), evo_h_i, label="Numerical model")
plt.plot(np.arange(0, 30, 1), evo_h_i_law, label="Stefan's law")
plt.title("Ice thickness evolution for 30 days", size=26)
plt.xlabel("Days", size=20)
plt.ylabel("Ice thickness [m]", size=20)
plt.legend()
plt.grid()
plt.show()
# 1.1.2
print("Thickness of the ice at the end of the simulation : ",
      evo_h_i[-1], "m.")
# 1.1.3
print("Thickness of the ice at the end the period using Stefan's law : ",
      evo_h_i_law[-1], "m.")

### 1.2 Addition of an ocean heat flux ###
F_w = 5  # oceanic heat flux[W/m²]

evo_h_i_F_w = ice_thick(h_i, ocean_heat_flux=True,
                        F_w=5)  # instanciation with F_w
evo_h_i_F_w180 = ice_thick(h_i, ocean_heat_flux=True,
                           F_w=180.4)  # instanciation with F_w
# 1.2.1
plt.plot(np.arange(0, 30, 1), evo_h_i, label="Numerical model without F_w")
plt.plot(np.arange(0, 30, 1), evo_h_i_F_w, label="Numerical model with F_w")
plt.plot(np.arange(0, 30, 1), evo_h_i_law, label="Stefan's law")
plt.title("Ice thickness evolution for 30 days", size=26)
plt.xlabel("Days", size=20)
plt.ylabel("Ice thickness [m]", size=20)
plt.legend()
plt.grid()
plt.show()
# 1.2.2
print("Thickness of the ice at the end of the simulation : ",
      evo_h_i_F_w[-1], "m.")
# 1.2.3
print("Thickness of the ice at the end of the simulation with F_w = 180.4W/m²: ",
      evo_h_i_F_w180[-1], "m.")
# 1.2.4
h_1 = np.zeros(N_days)
h_1[0] = 1

### 1.3 Addition of snow on top of the ice ###
