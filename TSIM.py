############################################################################################################################
# Thermodynamic Sea Ice Model (TSIM)
# Author : Amaury Laridon
# Course : LPHYS2269 - Remote Sensing of Climate Change
# Goal : Build a thermodynamic sea ice model that can be able to simulate a full seasonal cycle of sea ice growth and melt.
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

##################################################### Model ################################################################
### 1.1 Stefan and numerical solution for ice growth ###
# arry with the values of the sea ice thickness for the 30 days when we want to compute
h_i = np.zeros(30)
h_i[0] = 0.1  # initial condition for the sea ice thickness for the first day [m]
T_air = -10 + 273.15  # air temperature [K]
N_days = 30


def fourier_cond_flux(h, T_f, T_su):
    """Computation of the conductive heat flux trough the ice using the Fourier-Fick's law"""
    F_c = (ki*(T_f - T_su))/(h)
    return F_c


def ice_thick(h_i):
    """Computation of the evolution of the sea ice thickness using Stefan's law"""
    print("Evolution of the sea ice thickness")
    print("-----------------------------")
    for t in range(1, N_days):
        print("Day = {}/{}".format(t, N_days))
        print("h_i[t-1] = ", h_i[t-1])
        Q_c = fourier_cond_flux(h_i[t-1], T_bo, T_air)
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
plt.plot(np.arange(0, 30, 1), evo_h_i, label="Numerical model")
plt.plot(np.arange(0, 30, 1), evo_h_i_law, label="Stefan's law")
plt.title("Ice thickness evolution for 30 days", size=26)
plt.xlabel("Days", size=20)
plt.ylabel("Ice thickness [m]", size=20)
plt.legend()
plt.grid()
plt.show()
