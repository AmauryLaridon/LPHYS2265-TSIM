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
# Binary condition in order to implement ocean heat flux in the model. Will be used in 1.2
""" ocean_heat_flux = False
Q_w = 5  # oceanic heat flux[W/m²] """
# Binary condition in order to implement a layer of snow in the model. Will be used in 1.3
""" snow = False
h_s0 = 0  # initial snow thickness [m] """
######################################## 1 Ice growth with constant temperature ############################################
##################################################### Model ################################################################

### Functions of the model ###


def fourier_cond_flux(h_i, snow, h_s, T_bo=T_bo, T_su=T_su):
    """Computation of the conductive heat flux Q_c trough the ice using the Fourier-Fick's law (upward positive)
    [W/m^2]"""
    if snow == False:
        Q_c = ((T_bo - T_su)/(h_i))*ki
    else:
        k_eff = (ki*ks)/(ki * h_s + ks * h_i)  # [W/m²/K]
        Q_c = (T_bo - T_su)*k_eff  # [W/m²]
    print("Fourier-Thick conductive flux = {:.2f}".format(Q_c))
    return Q_c


def E_net_bottom(ice_thick, T_su, T_bott, ocean_heat, Q_w, snow, h_s):
    """ Compute the total energy loss in one day at the bottom of the sea ice layer (upward positive)
        [J/m^2]
    """
    if ocean_heat:
        E_loss_through_ice = fourier_cond_flux(
            ice_thick, T_bo, T_su, snow, h_s) * sec_per_day
        E_gain_ocean_flux = Q_w * sec_per_day
        E_net_bot = E_loss_through_ice - E_gain_ocean_flux
    else:
        E_loss_through_ice = fourier_cond_flux(
            ice_thick, T_bo, T_su, snow, h_s) * sec_per_day
        E_net_bot = E_loss_through_ice
    return E_net_bot


def ice_thick(h_i0, ocean_heat, Q_w, snow, h_s, integration_range=N_days, T_su=T_su, T_bo=T_bo):
    """Computation of the evolution of the sea ice thickness using Stefan's law.
    An option gives the possibility to add an Oceanic heat flux.
    This function returns an array with the sea ice thickness
     and an array with the time of integration"""
    print("------------------------------------------------------------------")
    print("                   ICGT SEA ICE THIKNESS MODEL")
    print("------------------------------------------------------------------")
    print(
        "Evolution of the sea ice thickness using numerical Stefan's law.\nintegration range = {} days, T_su = {:.2f} °C, T_bo = {:.2f} °C,\nh_i0 = {:.2f} m, ocean_heat_flux = {}, Q_w = {:.2f} W/m²\nsnow = {}, h_s0 = {:.2f} m".format(N_days, T_su - kelvin, T_bo-kelvin, h_i0, ocean_heat, Q_w, snow, h_s))
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
        E_net_bot = E_net_bottom(
            h_i[day-1], T_su, T_bo, ocean_heat, Q_w, snow, h_s)
        # Mass of water freezed at the bottom of the ice layer at the end of one day [kg/m^2]
        Freezing_water_mass = E_net_bot/L_fus
        # We obtain [m] as needed
        delta_h = Freezing_water_mass / rho_i
        h_i[day] = h_i[day-1] + delta_h
        print(
            "Energy balance at the bottom during Day {} = {:.2f} MJ/m²".format(day, E_net_bot/1e6))
        print("Variation of sea-ice thickness during Day {} = {:.2f} m".format(
            day, delta_h))
        print("Sea ice thickness at the end of Day {} = {:.2f} m".format(
            day, h_i[day]))
        print("------------------------------------------------------------------")
    return h_i, time_range


def stefan_law(h_i0, integration_range=N_days, T_su=T_su, T_bo=T_bo):
    """Analytical solution of the sea ice thickness evolution following Stefan's law."""
    time_range = range(0, integration_range)  # integration range in days
    # array colecting the values of the sea ice thickness for each day
    H_t = np.zeros(N_days)
    deltaT = T_bo - T_su
    for day in range(1, integration_range):
        day_sec = day * sec_per_day
        h_day = np.sqrt(h_i0**2 + (2*ki*deltaT*day_sec)/(rho_i*L_fus))
        H_t[day] = h_day
    return H_t, time_range


## Display Param ##
plt.rcParams['text.usetex'] = True

save_dir = "/home/amaury/Bureau/LPHYS2265 - Sea ice ocean atmosphere interactions in polar regions/Projet/Figures/"
figure = plt.figure(figsize=(16, 10))


def exo_1_1_Stef_law():
    """Computation of the evolution of the sea ice thickness using Stefan's law. 
    Correspond to the section 1.2 in Exercise_part_1.pdf on GitHub"""

    ##### 1.1.1 #####
    ## Instanciation ##
    # instanciation of the numerical model
    h_i, time_range = ice_thick(
        h_i0=h_i0, ocean_heat=False, Q_w=0, snow=False, h_s=0)
    h_i_law, time_range_law = stefan_law(h_i0=h_i0)

    ## Display ##
    plt.plot(time_range, h_i, label="Numerical model")
    plt.title("ICGT Ice thickness evolution for {} days".format(N_days), size=26)
    plt.xlabel("Days", size=20)
    plt.ylabel("Ice thickness [m]", size=20)
    plt.legend(fontsize=18)
    plt.grid()
    plt.savefig(
        save_dir + "1.1.1.png", dpi=300)
    # plt.show()
    plt.clf()
    ##### 1.1.2 #####
    print(
        "Thickness of the ice at the end of the simulation : {:.4f} m".format(h_i[-1]))
    ##### 1.1.3 #####
    plt.plot(time_range, h_i, label="Numerical model")
    plt.plot(time_range, h_i_law, label="Stefan's law")
    plt.title("ICGT Ice thickness evolution for {} days".format(N_days), size=26)
    plt.xlabel("Days", size=20)
    plt.ylabel("Ice thickness [m]", size=20)
    plt.legend(fontsize=18)
    plt.grid()
    plt.savefig(
        save_dir + "1.1.3.png", dpi=300)
    # plt.show()
    plt.clf()
    print("Thickness of the ice at the end of the period using Stefan's law : {:.4f} m ".format(
        h_i_law[-1]))


def exo1_2_add_OHF():
    """Addition of Ocean Heat Flux (OHF) in the model. 
    Correspond to the section 1.2 in Exercise_part_1.pdf on GitHub"""

    ##### 1.2.1 #####
    ## Instanciation ##
    # Binary condition in order to implement ocean heat flux in the model
    # instanciation of the numerical model

    h_i, time_range = ice_thick(
        h_i0=0.1, ocean_heat=False, Q_w=0, snow=False, h_s=0)
    ocean_heat_flux = True
    Q_w = 5  # oceanic heat flux[W/m²]
    h_i_F_w, time_range = ice_thick(h_i0=0.1,
                                    ocean_heat=ocean_heat_flux, Q_w=Q_w, snow=False, h_s=0)  # instanciation with F_w

    ## Display ##
    plt.plot(time_range, h_i_F_w, label="Numerical model with OHF")
    plt.plot(time_range, h_i, label="Numerical model without OHF")
    plt.title('ICGT Ice thickness evolution for {} days\nwith oceanic heat flux Q_w = {:.2f}W/m²'.format(
        N_days, Q_w), size=26)
    plt.xlabel("Days", size=20)
    plt.ylabel("Ice thickness [m]", size=20)
    plt.legend(fontsize=18)
    plt.grid()
    plt.savefig(
        save_dir + "1.2.1.png", dpi=300)
    # plt.show()
    plt.clf()

    ##### 1.2.2 #####
    print(
        "Thickness of the ice at the end of the simulation with OHF = {:.2f} : {:.4f} m".format(Q_w, h_i_F_w[-1]))

    ##### 1.2.3 #####
    ## Instanciation ##
    h_i0 = 0.1
    # For the following value of the physical parameter the fourier_cond_flux function gives us the value of the conductive flux. In order to counterbalance this
    # value we have the put the same amount of radiative forcing with the OHF.
    Q_w = fourier_cond_flux(h_i=h_i0, T_bo=T_bo,
                            T_su=T_su, snow=snow, h_s=h_s0)
    h_i_F_w180, time_range = ice_thick(h_i0=h_i0,
                                       ocean_heat=ocean_heat_flux, Q_w=Q_w, snow=False, h_s=0)  # instanciation with F_w

    ## Display ##
    plt.plot(time_range, h_i_F_w180, label="Numerical model with OHF")
    # plt.plot(time_range, h_i, label="Numerical model without OHF")
    plt.title('ICGT Ice thickness evolution for {} days\nwith oceanic heat flux Q_w = {:.2f}W/m²'.format(
        N_days, Q_w), size=26)
    plt.xlabel("Days", size=20)
    plt.ylabel("Ice thickness [m]", size=20)
    plt.legend(fontsize=18)
    plt.grid()
    plt.savefig(
        save_dir + "1.2.3.png", dpi=300)
    # plt.show()
    plt.clf()

    print(
        "Thickness of the ice at the end of the simulation with OHF = {:.2f} : {:.4f} m".format(Q_w, h_i_F_w180[-1]))
    # Analysis of the result : the ice thickness remains at its orignal value since we have here an OHC exactly conterbalancing the conductive flux in the ice i.e we have a net
    # energy balance of 0 so no gain or lose of ice during the all simulation.

    ##### 1.2.4 #####
    ## Instanciation ##
    Q_w = fourier_cond_flux(1, T_bo=T_bo, T_su=T_su)
    # For the following value of the physical parameter the fourier_cond_flux function gives us a value of 18,039W/m^2. In order to counterbalance this
    # value we have the put the same amount of radiative forcing with the OHF.
    h_i0 = 1
    ocean_heat_flux = True
    h_i0_0, time_range = ice_thick(h_i0=h_i0,
                                   ocean_heat=ocean_heat_flux, Q_w=Q_w, snow=False, h_s=0)

    ## Display ##
    plt.plot(time_range, h_i0_0, label="Numerical model with OHF")
    plt.title('ICGT Ice thickness evolution for {} days\nwith oceanic heat flux Q_w = {:.2f}W/m²'.format(
        N_days, Q_w), size=26)
    plt.xlabel("Days", size=20)
    plt.ylabel("Ice thickness [m]", size=20)
    plt.legend(fontsize=18)
    plt.grid()
    plt.savefig(
        save_dir + "1.2.4.png", dpi=300)
    # plt.show()
    plt.clf()


def exo1_3_add_snow():
    """Addition of snow in the model. 
    Correspond to the section 1.3 in Exercise_part_1.pdf on GitHub"""
    ## Physical parameters ##
    h_i0 = 0.1
    h_s0 = 0.05
    Q_w = 0
    ###### 1.3.1 #####
    ## Instanciation ##
    h_i, time_range = ice_thick(
        h_i0=h_i0, ocean_heat=False, Q_w=0, snow=False, h_s=0)
    h_i_wt_snow, time_range = ice_thick(
        h_i0=h_i0, ocean_heat=False, Q_w=0, snow=True, h_s=h_s0)

    ## Display ##
    plt.plot(time_range, h_i_wt_snow, label="With Snow")
    plt.plot(time_range, h_i, label="Without Snow")
    plt.title('ICGT Ice thickness evolution for {} days\nwith oceanic heat flux Q_w = {:.2f}W/m²\nwith a layer of snow h_s0 = {:.2f}m'.format(
        N_days, Q_w, h_s0), size=22)
    plt.xlabel("Days", size=20)
    plt.ylabel("Ice thickness [m]", size=20)
    plt.legend(fontsize=18)
    plt.grid()
    plt.savefig(
        save_dir + "1.3.1.png", dpi=300)
    # plt.show()
    plt.clf()
    print(
        "Thickness of the ice at the end of the simulation without but with a snow layer = {:.2f}m : {:.4f}m".format(h_s0, h_i_wt_snow[-1]))

    ##### 1.3.2 #####
    ## Instanciation ##
    ocean_heat_flux = True
    h_i0 = 1
    snow = True
    # For the following value of the physical parameter the fourier_cond_flux function gives us the value of the conductive flux. In order to counterbalance this
    # value we have the put the same amount of radiative forcing with the OHF.
    Q_w = fourier_cond_flux(h_i=h_i0, T_bo=T_bo,
                            T_su=T_su, snow=snow, h_s=h_s0)
    h_i_wt_ice_cst, time_range = ice_thick(
        h_i0=h_i0, ocean_heat=ocean_heat_flux, Q_w=Q_w, snow=snow, h_s=h_s0)
    print(
        "Ocean heat flux needed to keep the ice thickness constat at {:.2f}m with a {:.2f}m layer of snow : {:.2f} W/m²".format(h_i0, h_s0, Q_w))

    ## Display ##
    plt.plot(time_range, h_i_wt_ice_cst, label="Numerical model")
    plt.title('ICGT Ice thickness evolution for {} days\nwith oceanic heat flux Q_w = {:.2f}W/m²\nwith a layer of snow h_s0 = {:.2f}m'.format(
        N_days, Q_w, h_s0), size=22)
    plt.xlabel("Days", size=20)
    plt.ylabel("Ice thickness [m]", size=20)
    plt.legend(fontsize=18)
    plt.grid()
    plt.savefig(
        save_dir + "1.3.2.png", dpi=300)
    # plt.show()
    plt.clf()


# exo_1_1_Stef_law()
# exo1_2_add_OHF()
exo1_3_add_snow()
