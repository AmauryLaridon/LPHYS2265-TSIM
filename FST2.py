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
kelvin = 273.15  # Conversion form Celsius to Kelvin
alb = 0.8  # surface albedo
ki = 2.2  # sea ice thermal conductivity [W/m/K]
ks = 0.31  # Snow thermal conductivity [W/m/K]
sec_per_day = 86400  # Seconds in one day [s/day]
L_fus = 3.35e5  # Latent heat of fusion for water [J/kg]
rho_i = 917  # Sea ice density [kg/m³]
### Simulation parameters ###
N_years = 1
N_days = 365 * N_years  # number of days in the simulation
h = 0.5  # sea ice thickness
# temperature at the freezing point of sea water with a salinity of 34g/kg
T_bo = -1.8 + kelvin
Day_0 = 1
### Display Parameters ###
plt.rcParams['text.usetex'] = True
save_dir = "/home/amaury/Bureau/LPHYS2265 - Sea ice ocean atmosphere interactions in polar regions/Projet/Figures/"
figure = plt.figure(figsize=(16, 10))


########################################### Model of Surface Temperature ####################################################

######################### Parameterization of atmospheric fluxes ####################################


def solar_flux(day):
    """Definition of the atmospheric solar heat flux Q_sol for a given day in the year. Conversion from Fletcher(1965)"""
    doy = day % 365
    Q_sol = 314*np.exp((-(doy-164)**2)/4608)
    return Q_sol


def non_solar_flux(day):
    """Definition of the atmospheric non solar heat flux Q_nsol for a given day in the year. Conversion from Fletcher(1965)"""
    doy = day % 365
    Q_nsol = 118*np.exp((-0.5*(doy-206)**2)/(53**2)) + 179
    return Q_nsol

######################### Model of surface temperature evolution ####################################


def surface_temp(h):
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
        # finding the surface temperature using roots function from numpy. As required in 2.1.2 temperature is not
        # physically sensible for ice in summer so we cap the surface temperature to 273,15°K.
        root = min([273.15, np.roots([-epsilon * sigma, 0, 0, -ki/h, ki /
                                      h * T_bo + solar_flux_ar[day] * (1-alb) + non_solar_flux_ar[day]]).real[3]])
        # root = opt.newton(energ_bal_fun(day), T_su[day-1]) ## root computation with newton method from scipy
        T_su[day] = root

    return T_su

########################################### Model of Sea ice thickness ####################################################


def fourier_cond_flux(h_i, T_su, snow, h_s):
    """Computation of the conductive heat flux Q_c trough the ice using the Fourier-Fick's law (upward positive)
    [W/m^2]"""
    if snow == False:
        Q_c = ((T_bo - T_su)/(h_i))*ki
    else:
        k_eff = (ki*ks)/(ki * h_s + ks * h_i)  # [W/m²/K]
        Q_c = (T_bo - T_su)*k_eff  # [W/m²]
    print("Fourier-Thick conductive flux = {:.2f} W/m²".format(Q_c))
    return Q_c


def E_net_bottom(ice_thick,  ocean_heat, Q_w, snow, h_s, T_su, T_bo=T_bo):
    """ Compute the total energy loss in one day at the bottom of the sea ice layer (upward positive)
        [J/m^2]
    """
    if ocean_heat:
        E_loss_through_ice = fourier_cond_flux(
            ice_thick, T_su, snow, h_s) * sec_per_day
        E_gain_ocean_flux = Q_w * sec_per_day
        E_net_bot = E_loss_through_ice - E_gain_ocean_flux
    else:
        E_loss_through_ice = fourier_cond_flux(
            ice_thick, T_su, snow, h_s) * sec_per_day
        E_net_bot = E_loss_through_ice
    return E_net_bot


def ice_thick(h_i0, ocean_heat, Q_w, snow, h_s, integration_range=N_days, T_bo=T_bo):
    """Computation of the evolution of the sea ice thickness using Stefan's law.
    An option gives the possibility to add an Oceanic heat flux.
    This function returns an array with the sea ice thickness
     and an array with the time of integration"""
    print("------------------------------------------------------------------")
    print("                    FST SEA ICE THIKNESS MODEL")
    print("------------------------------------------------------------------")
    print(
        "Evolution of the sea ice thickness using numerical Stefan's law.\nintegration range = {} days, T_bo = {:.2f} °C,\nh_i0 = {:.2f} m, ocean_heat_flux = {}, Q_w = {:.2f} W/m²\nsnow = {}, h_s0 = {:.2f} m".format(N_days, T_bo-kelvin, h_i0, ocean_heat, Q_w, snow, h_s))
    print("------------------------------------------------------------------")
    # array colecting the values of the sea ice thickness for each day
    h_i = np.zeros(N_days)
    h_i[0] = h_i0  # initial condition for the sea ice thickness for the first day [m]
    time_range = range(0, integration_range)  # integration range in days
    for day in range(1, integration_range):
        print("Day {}/{}".format(day, integration_range))
        print("----------")
        print("Sea ice thickness at begining of Day {} = {:.2f} m".format(
            day, h_i[day-1]))
        # Energy lost at the bottom during one day due to flux from water to ice.[J/m^2]
        T_su_ar = surface_temp(h_i[day-1])
        T_su = T_su_ar[day]
        E_net_bot = E_net_bottom(
            h_i[day-1], ocean_heat, Q_w, snow, h_s, T_su)
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


########################################### Cases of Simulations ######################################################

### 2.1 Surface heat fluxes ###


def exo2_1_surf_heat_flux():
    """Parameterisation of the surface heat fluxes.
    Section 2.1 of the Exercise_part_1.pdf file available on the GitHub."""
    ##### 2.1.1 #####
    ### Instancing ###
    year = np.arange(Day_0, N_days + 1)
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


## year = np.arange(1, N_days, 1)# 2.2 Calculate surface temperature ###

def exo2_2_surf_temp():
    """Computation of the surface temperature.
    Section 2.2 of the Exercise_part_1.pdf file available on the GitHub."""

    ##### 2.2.1 #####
    ### Instancing ###
    T_su = surface_temp(h=h)
    year = np.arange(Day_0, N_days + 1)
    ### Display ###
    plt.plot(year, T_su - kelvin, label="T_su")
    plt.title("Evolution of the surface temperature", size=26)
    plt.xlabel("Days", size=20)
    plt.ylabel("T_su [°C]", size=20)
    plt.legend(fontsize=18)
    plt.grid()
    plt.savefig(save_dir + "2.1.2.png", dpi=300)
    # plt.show()
    plt.clf()

### 2.3 Couple temperature and thickness ###


def exo2_3_coupl_temp_thick():
    ##### 2.3.1 #####
    ### Instancing ###
    h_coupl, time_range = ice_thick(
        h_i0=0.1, ocean_heat=True, Q_w=2, snow=False, h_s=0)
    ### Display ###
    Q_w = 2
    h_s0 = 0
    plt.plot(time_range, h_coupl, label="h_coupl")
    plt.title('FST Ice thickness evolution for {} days\nwith oceanic heat flux Q_w = {:.2f}W/m²\nwith a layer of snow h_s0 = {:.2f}m'.format(
        N_days, Q_w, h_s0), size=22)
    plt.xlabel("Days", size=20)
    plt.ylabel("Ice Thickness [m]", size=20)
    plt.legend(fontsize=18)
    plt.grid()
    plt.savefig(save_dir + "2.3.1.png", dpi=300)
    # plt.show()
    plt.clf()

    ##### 2.3.2 #####


if __name__ == "__main__":
    # exo2_1_surf_heat_flux()
    # exo2_2_surf_temp()
    exo2_3_coupl_temp_thick()
