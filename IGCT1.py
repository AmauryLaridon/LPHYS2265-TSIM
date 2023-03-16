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
rho_i = 917  # Sea ice density [kg/m³]
ki = 2.2  # Sea ice thermal conductivity [W/m/K]
ks = 0.31  # Snow thermal conductivity [W/m/K]
sec_per_day = 86400  # Seconds in one day [s/day]
### Bottom boundary conditions ###
kelvin = 273.15  # Conversion degre K -> C
T_bo = -1.8 + kelvin  # Bottom temperature [K]
### Parameters ###
N_days = 30  # number of days of integration
h_i0 = 0.1  # initial sea ice thickness [m]
T_su = -10 + kelvin  # air temperature [K]
######################################## 1 Ice growth with constant temperature ############################################
##################################################### Model ################################################################
### 1.1 Stefan and numerical solution for ice growth ###


def fourier_cond_flux(h, T_bo, T_su):
    """Computation of the conductive heat flux Q_c trough the ice using the Fourier-Fick's law (upward positive)
    [W/m^2]"""
    Q_c = ((T_bo - T_su)/(h))*ki
    return Q_c


def E_loss_bottom(ice_thick, T_su, T_bott):
    """ Compute the total energy loss in one day at the bottom of the sea ice layer
        [J/m^2]
    """
    E_loss_through_ice = fourier_cond_flux(ice_thick, T_bo, T_su) * sec_per_day
    return E_loss_through_ice


def ice_thick(integration_range=N_days, T_su=T_su, T_bo=T_bo, h_i0=h_i0, ocean_heat_flux=False):
    """Computation of the evolution of the sea ice thickness using Stefan's law. 
    An option gives the possibility to add an Oceanic heat flux. 
    This function returns an array with the sea ice thickness
     and an array with the time of integration"""
    print("------------------------------------------------------------------")
    print("                   ICGT SEA ICE THIKNESS MODEL")
    print("------------------------------------------------------------------")
    print(
        "Evolution of the sea ice thickness using numerical Stefan's law.\nintegration range = {} days, T_su = {:.2f} °C, T_bo = {:.2f} °C,\nh_i0 = {:.2f} m, ocean_heat_flux = {}".format(N_days, T_su - kelvin, T_bo-kelvin, h_i0, ocean_heat_flux))
    print("------------------------------------------------------------------")
    # array colecting the values of the sea ice thickness for each day
    h_i = np.zeros(N_days)
    h_i[0] = h_i0  # initial condition for the sea ice thickness for the first day [m]
    time_range = range(0, integration_range)  # integration range in days
    for day in range(1, integration_range):
        print("Day {}/{}".format(day, integration_range))
        print("Sea ice thickness at begining of Day {} = {:.2f} m".format(
            day, h_i[day-1]))
        # Energy lost at the bottom during one day due to flux from water to ice.[J/m^2]
        E_loss = E_loss_bottom(h_i[day-1], T_su, T_bo)
        # Mass of water freezed at the bottom of the ice layer at the end of one day [kg/m^2]
        Freezing_water_mass = E_loss/L_fus
        # We obtain [m] as needed
        delta_h = Freezing_water_mass / rho_i
        h_i[day] = h_i[day-1] + delta_h
        print(
            "Energy lost at the bottom during Day {} = {:.2f} MJ/m²".format(day, E_loss/1e6))
        print("Variation of sea-ice thickness during Day {} = {:.2f} m".format(
            day, delta_h))
        print("Sea ice thickness at the end of Day {} = {:.2f} m".format(
            day, h_i[day]))
        print("------------------------------------------------------------------")
    return h_i, time_range


def stefan_law(integration_range=N_days, T_su=T_su, T_bo=T_bo, h_i0=h_i0):
    time_range = range(0, integration_range)  # integration range in days
    # array colecting the values of the sea ice thickness for each day
    H_t = np.zeros(N_days)
    deltaT = T_bo - T_su
    for day in range(1, integration_range):
        day_sec = day * sec_per_day
        h_day = np.sqrt(h_i0**2 + (2*ki*deltaT*day_sec)/(rho_i*L_fus))
        H_t[day] = h_day
    return H_t, time_range


h_i, time_range = ice_thick()  # instanciation of the numerical model
h_i_law, time_range_law = stefan_law()

## Display ##
# 1.1.1
figure = plt.figure(figsize=(16, 10))
plt.plot(time_range, h_i, label="Numerical model")
plt.title("ICGT Ice thickness evolution for {} days".format(N_days), size=26)
plt.xlabel("Days", size=20)
plt.ylabel("Ice thickness [m]", size=20)
plt.legend(fontsize=18)
plt.grid()
save_dir = "/home/amaury/Bureau/LPHYS2265 - Sea ice ocean atmosphere interactions in polar regions/Projet/Figures/"
plt.savefig(
    save_dir + "1.1.1.png", dpi=300)
# plt.show()
# 1.1.2
print(
    "Thickness of the ice at the end of the simulation : {:.4f} m".format(h_i[-1]))
# 1.1.3
figure = plt.figure(figsize=(16, 10))
plt.plot(time_range, h_i, label="Numerical model")
plt.plot(time_range, h_i_law, label="Stefan's law")
plt.title("ICGT Ice thickness evolution for {} days".format(N_days), size=26)
plt.xlabel("Days", size=20)
plt.ylabel("Ice thickness [m]", size=20)
plt.legend(fontsize=18)
plt.grid()
save_dir = "/home/amaury/Bureau/LPHYS2265 - Sea ice ocean atmosphere interactions in polar regions/Projet/Figures/"
plt.savefig(
    save_dir + "1.1.3.png", dpi=300)
# plt.show()
print("Thickness of the ice at the end the period using Stefan's law : {:.4f} m ".format(
    h_i_law[-1]))


"""
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
"""
