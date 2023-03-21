############################################################################################################################
# Thermodynamic Sea Ice Model (TSIM)
# Author : Amaury Laridon
# Course : LPHYS2269 - Remote Sensing of Climate Change
# Goal : Third of the TSIM model. Modelisation of the evolution of sea-ice thickness with a dynamic surface temperature
#        Free Surface Temperature (FST) and an Addition of Ocean and Snow (AOO)
#        More information on the GitHub Page of the project : https://github.com/AmauryLaridon/TSIM.git
# Date : 19/03/23
############################################################################################################################
#################################################### Packages ##############################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
######################################## 2 Freeing surface temperature ############################################
################################################### Parameters #####################################################
### Physical Constant ###
epsilon = 0.99  # surface emissivity [Adim]
sigma = 5.67e-8  # Stefan-Boltzman constant [J/°K]
kelvin = 273.15  # Conversion form Celsius to Kelvin [Adim]
alb_sur = 0.8  # surface albedo [Adim]
ki = 2.2  # sea ice thermal conductivity [W/m/K]
ks = 0.31  # Snow thermal conductivity [W/m/K]
sec_per_day = 86400  # seconds in one day [s/day]
L_fus = 3.35e5  # latent heat of fusion for water [J/kg]
rho_i = 917  # sea ice density [kg/m³]
rho_s = 330  # snow density [kg/m³]
c = 4000  # heat capacity of water [J/(kg.°K)]
alb_wat = 0.1  # albedo of water [Adim]
rho_w = 1025  # density of sea water [kg/m³]
### Simulation parameters ###
N_years = 50  # number of years in the simulation [Adim]
N_days = 365 * N_years  # number of days in the simulation [Adim]
h = 0.5  # sea ice thickness [m]
h_w = 50  # depth of the ocean mix layer [m]
M_w = rho_w*h_w  # masse of water in the mixed layer [kg/m^2]
# temperature at the freezing point of sea water with a salinity of 34g/kg
T_bo = -1.8 + kelvin
Day_0 = 1  # set the first day of the simulation [Adim]
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


def surface_temp(h_i, h_s, day):
    """Compute the evolution of the surface temperature with respect to the variation of the atmospheric
    heat fluxes and return a single value for a given day and a given thickness of the ice and snow"""

    # finding the surface temperature using roots function from numpy. As required in 2.1.2 temperature is not
    # physically sensible for ice in summer so we cap the surface temperature to 273,15°K.
    if h_i > 0:
        k_eff = (ki*ks)/(ki * h_s + ks * h_i)  # [W/m²/K]
        root = min([273.15, np.roots([-epsilon * sigma, 0, 0, -k_eff, k_eff *
                   T_bo + solar_flux(day) * (1-alb_sur) + non_solar_flux(day)]).real[3]])
    else:
        root = min([273.15, np.roots([-epsilon * sigma, 0, 0, 0,
                   solar_flux(day) * (1-alb_sur) + non_solar_flux(day)]).real[3]])
    T_su = root

    def net_surf_flux(h_i, h_s, day, T_su):
        """Compute the net solar flux for a given day with a given sea ice and snow thickness"""
        k_eff = (ki*ks)/(ki * h_s + ks * h_i)  # [W/m²/K]
        nsf = solar_flux(day)*(1-alb_sur) + non_solar_flux(day) - \
            epsilon*sigma*(T_su**4) - k_eff*(T_su - T_bo)
        return nsf

    nsf = net_surf_flux(h_i, h_s, day, T_su)

    if nsf > 0:
        # If the net solar flux is positive, this energy is available for melting and will be stored in a variable efm (energy for melting)
        efm = nsf
    else:
        efm = 0  # If the net solar flux is negative or egal to zero, the efm = 0

    return T_su, efm

########################################### Model of the Ocean Mix Layer ####################################################


def E_gain_mixed_layer(T_w, day, Q_w):
    """ Compute the free water energy gain in one day. [J/m^2]
    Incoming:  Q_sol: solar flux,  Q_nsol : non solar flux,  Q_w: flux from deep water
    Outgoing: Blackbody radiation with temperature T_w"""
    E_gain_mix_lay = (solar_flux(day)*(1-alb_wat) +
                      non_solar_flux(day) + Q_w - epsilon*sigma*(T_w**4))*sec_per_day
    return E_gain_mix_lay

########################################### Model of Sea ice thickness ######################################################


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
    """ Compute the total energy loss in one day at the bottom of the sea ice layer (upward positive)[J/m^2] """
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


def E_net_surf(efm):
    """ Compute the energy gain at the surface of the sea ice in one day due to a non-equilibrium between the solar and non-solar
    flux incoming, the flux coming from the water through the ice and the ice blackbody radiation. This disequilibrium
    is because the ice can't go over 273.15K without melting. This energy will hence be used to melt the ice during summer.
    [J/m^2]
    """
    E_net_surf = efm * sec_per_day
    return E_net_surf


def snow_fall(day):
    """ Function that modelise the snowfall in [m]. The values are given in the Exercise_part_1.pdf file available on the GitHub.
        30 cm between 20 August and October, 5 cm between November and april, 5 cm in May. We use an uniform distribution of those snowfall 
        during these three different periods."""
    doy = day % 365
    if doy >= 232 and doy <= 304:  # between 20 August and October (included)
        return 0.3/(304-232)
    elif doy >= 305 or doy <= 120:  # between November and April (included)
        return 0.05/(365-305+120)
    elif doy >= 121 and doy <= 151:  # May
        return 0.05/(151-121)
    else:
        return 0


def ice_thick(h_i0, ocean_heat, Q_w, snow, h_s0, integration_range=N_days, T_bo=T_bo):
    """Computation of the evolution of the sea ice and snow thickness using Stefan's law.
    An option gives the possibility to add an Oceanic heat flux.
    This function returns an array with the sea ice thickness, snow thickness, surface temperature, mixed layer temperature and
    and an array with the time of integration"""

    ### Output Simulation Settings ###
    print("------------------------------------------------------------------")
    print("             TSIM SEA ICE AND SNOW THICKNESS MODEL")
    print("------------------------------------------------------------------")
    print(
        "Evolution of the sea ice thickness using numerical Stefan's law.\nintegration range = {} days, T_bo = {:.2f} °C,\nh_i0 = {:.2f} m, ocean_heat_flux = {}, Q_w = {:.2f} W/m²\nsnow = {}, h_s0 = {:.2f} m".format(N_days, T_bo-kelvin, h_i0, ocean_heat, Q_w, snow, h_s0))
    print("------------------------------------------------------------------")

    #### Initialization ###
    # array colecting the values of the sea ice thickness for each day
    h_i = np.zeros(N_days)
    h_i[0] = h_i0  # initial condition for the sea ice thickness for the first day [m]
    # array colecting the values of snow thickness for each day
    h_s = np.zeros(N_days)
    h_s[0] = h_s0  # initial condition for the snow thickness for the first day [m]
    # array colecting the values of surface temperature for each day
    T_su_ar = np.zeros(N_days)
    # initialized surface temperature in regard of the ice thickness, incoming Energy and bottom temp.
    T_su, efm = surface_temp(h_i0, h_s0, day=1)
    T_su_ar[0] = T_su
    # array colecting the values of ocean mix layer temperature for each day
    T_mix_lay_ar = np.zeros(N_days)
    # At the beggining, the mixed layer temperature is equal to the sea ice bottom temperature [K]
    T_w = T_bo
    T_mix_lay_ar[0] = T_w

    time_range = range(0, integration_range)  # integration range in days

    ### Dynamic Model ####
    for day in range(1, integration_range):
        ## Output ##
        print("Day {}/{}".format(day, integration_range))
        print("----------")
        print("Sea ice thickness at begining of Day {} = {:.2f} m".format(
            day, h_i[day-1]))
        if snow == True:
            print("Snow thickness at begining of Day {} = {:.2f} m".format(
                day, h_s[day-1]))

        ## Ice Cover testing condition ##
        # Test if there is some ice cover or not. If they are an ice cover we perform the same computation as before,
        # the temperature of the ocean mixed layer remains at it's initial value of the freezing point temperature.
        # If they are no more ice cover, we compute the energy desequilibrium to warm or cool the mixed layer.
        if h_i[day-1] > 0:
            ice_cover = True
        else:
            ice_cover = False

        if ice_cover == True:
            if snow == True:
                ## Snow thickness instanciation ##
                if h_s[day-1] < 0:
                    h_s[day-1] = 0
                snow_gain = snow_fall(day)
                h_s[day] = h_s[day-1] + snow_fall(day)
            else:
                snow_gain = 0

            ## Surface temperature computation ##
            # Computation of the surface temperature given a particular day and ice and snow thickness
            T_su, efm = surface_temp(h_i[day-1], h_s[day-1], day)
            T_su_ar[day] = T_su

            ## Energy change at the bottom ##
            # Energy lost at the bottom during one day due to flux from water to ice.[J/m^2]
            E_net_bot = E_net_bottom(
                h_i[day-1], ocean_heat, Q_w, snow, h_s[day-1], T_su)
            # Mass of water freezed at the bottom of the ice layer at the end of one day [kg/m^2]
            freezing_water_mass = E_net_bot/L_fus
            # To obtain [m] as needed
            sea_ice_gain = freezing_water_mass / rho_i

            ## Energy change at the surface ##
            # Energy gain at the surface during one day due to non equilibrium.[J/m^2]
            E_net_sur = E_net_surf(efm)  # [J/m²]
            if h_s[day-1] > 0:
                # Mass of ice melted at the surface of the ice layer at the end of one day [kg/m²]
                melt_snow_mass = E_net_sur/L_fus
                # To obtain [m] as needed
                snow_lost = melt_snow_mass/rho_s
            else:
                # Mass of ice melted at the surface of the ice layer at the end of one day [kg/m²]
                melt_ice_mass = E_net_sur/L_fus  # [kg/m²]
                # To obtain [m] as needed
                sea_ice_lost = melt_ice_mass/rho_i
                snow_lost = 0

            ## Mix layer temperature ##
            T_mix_lay_ar[day] = T_bo

            ## Net balance of sea ice thickness ##
            delta_h_i = sea_ice_gain - sea_ice_lost
            h_i[day] = h_i[day-1] + delta_h_i
            ## Net balance of snow thickness ##
            delta_h_s = snow_gain - snow_lost
            h_s[day] = h_s[day-1] + delta_h_s

        if ice_cover == False:
            # set the latest ice thickness to 0 in order to have physical value.
            h_i[day-1] = 0
            # output
            print("No ice cover at Day {}".format(day))
            ## Surface temperature computation ##
            # Computation of the surface temperature given a particular day and ice thickness
            T_su, efm = surface_temp(h_i[day-1], h_s[day-1], day)
            T_su_ar[day] = T_su
            if T_w >= T_bo:
                # In this case the water can warm or cool without producing SI
                # Energy gain by the mixed layer in one day [J/m^2]
                delta_h = 0
                E_gain = E_gain_mixed_layer(T_w, day, Q_w)
                T_w += E_gain/(M_w*c)  # New mixed layer temperature [K]
                T_mix_lay_ar[day] = T_w
            else:
                # In this case the water is cooling below the freezing point so we re-create ice
                delta_T_excess = T_bo - T_w
                # Excess of heat which will be turn into ice [J/m^2]
                E_gain = delta_T_excess * M_w * c
                freezing_water_mass = E_gain/L_fus  # [kg/m^2]
                h_i[day] = freezing_water_mass/rho_i  # [m]
                delta_h = h_i[day]
                T_w = T_bo  # set the bottom temperature [K]
                T_mix_lay_ar[day] = T_w

        ## Output of simulation ##
        print(
            "Energy balance at the bottom during Day {} = {:.2f} MJ/m²".format(day, E_net_bot/1e6))
        print("Variation of sea-ice thickness during Day {} = {:.2f} m".format(
            day, delta_h_i))
        print("Sea ice thickness at the end of Day {} = {:.2f} m".format(
            day, h_i[day]))
        print("------------------------------------------------------------------")
    return h_i, h_s, T_su_ar, T_mix_lay_ar, time_range


########################################### Cases of Simulations ######################################################

##### 3 Addition of Surface Ocean and Snow #####

def first_and_mult_ice():
    ##### Settings for ice-free conditions #####
    ### Instancing ###
    h_ice_free, h_snow_ice_free, T_su_ice_free, T_mix_lay_ice_free, time_range = ice_thick(
        h_i0=0.1, ocean_heat=True, Q_w=5, snow=False, h_s0=0)
    ### Display ###
    ## Ice thickness evolution plot ##
    Q_w = 5
    h_s0 = 0
    plt.plot(time_range, h_ice_free, label="h_ice")
    plt.title('TSIM Ice thickness evolution for {} days\nwith oceanic heat flux Q_w = {:.2f}W/m², a layer of snow h_s0 = {:.2f}m\nalbedo = {}'.format(
        N_days, Q_w, h_s0, alb_sur), size=22)
    plt.xlabel("Days", size=20)
    plt.ylabel("Ice Thickness [m]", size=20)
    plt.legend(fontsize=18)
    plt.grid()
    plt.savefig(save_dir + "first_and_mult_ice.png", dpi=300)
    # plt.show()
    plt.clf()
    ## Temperature evolution plot ##
    plt.plot(time_range, T_su_ice_free - kelvin, label="T_su")
    plt.plot(time_range, T_mix_lay_ice_free - kelvin, label="T_mix")
    plt.title('TSIM Temperature evolution for {} days\nwith oceanic heat flux Q_w = {:.2f}W/m², a layer of snow h_s0 = {:.2f}m\nalbedo = {}'.format(
        N_days, Q_w, h_s0, alb_sur), size=22)
    plt.xlabel("Days", size=20)
    plt.ylabel("Temperature [°C]", size=20)
    plt.legend(fontsize=18)
    plt.grid()
    plt.savefig(save_dir + "first_and_mult_ice_temp.png", dpi=300)
    # plt.show()
    plt.clf()

    ##### Answers to question 3.1 ######################################################################################
    # 3.1.1 Let the model run for 10 years. How thick does your ice get in winter? Are there still year-to-year changes?
    # Answer : The ice tends to have a thickness of 1,7m in winter. After a few years (5-6) it seems that the sea ice
    #          thickness has reached an equilibrium
    # 3.1.2 When does the ocean become ice free?
    # Answer : After a bit less than 200 days so rouglhy speaking near the end of May and the begining of June.
    # 3.1.3 By how much do you have to reduce the non-solar flux to get multi-year ice?
    # Answer : A reduction of 7% of the non-solar flux already produce multi-year ice since the first year.
    # 3.1.4 : By how much do you have to increase the non-solar fluxes to have an ice-free Arctic all year round?
    # Answer : An increase of 9% of the non-solar flux is sufficient to have an ice-free Arctif all year round after
    #          five years.
    ####################################################################################################################


def ctrl_sim_wth_snow():
    ##### Settings for ice-free conditions #####
    ### Instancing ###
    h_ice, h_snow, T_su, T_mix_lay, time_range = ice_thick(
        h_i0=0.1, ocean_heat=True, Q_w=5, snow=False, h_s0=0)
    ### Display ###
    ## Ice thickness evolution plot ##
    h_i0 = 0.1
    Q_w = 5
    h_s0 = 0

    fig, axs = plt.subplots(2, 2)
    fig.suptitle(
        'TSIM Model without snow\n' + r'$\alpha_S$ = {}, $Q_W = {}W/m^2$, $h_i(t=0) = {}m$, $h_s(t=0) = {}m, T = {}$ years'.format(alb_sur, Q_w, h_i0, h_s0, N_years))

    axs[0, 0].plot(time_range, h_ice, label="h_ice", linewidth=0.4)
    axs[0, 0].set_title('Ice thickness')
    axs[0, 0].set_xlabel('Days')
    axs[0, 0].set_ylabel('Thickness [m]')
    # axs[0, 1].set_xticks(np.arange(0, 5, 1))
    axs[0, 0].grid()

    axs[0, 1].plot(time_range, h_snow, label='h_snow',
                   color='c', linewidth=0.4)
    axs[0, 1].set_title('Snow thickness')
    axs[0, 1].sharex(axs[0, 0])
    axs[0, 1].grid()
    # axs[0, 1].set_xlabel('Days')
    # axs[0, 0].set_ylabel('Thickness [m]')

    axs[1, 0].plot(time_range, T_su, label='T_su',
                   color='orange', linewidth=0.4)
    axs[1, 0].set_title('Surface Temperature')
    axs[1, 0].sharex(axs[0, 0])
    axs[1, 0].set_xlabel('Days')
    axs[1, 0].set_ylabel('Temperature [°K]')
    axs[1, 0].grid()

    axs[1, 1].plot(time_range, T_mix_lay, label='T_w',
                   color='green', linewidth=0.4)
    axs[1, 1].set_title('Mix Layered Temperature')
    axs[1, 1].sharex(axs[1, 0])
    axs[1, 1].grid()
    # axs[1, 1].set_xlabel('Days')
    # axs[1, 1].set_ylabel('Temperature [°K]')

    # for ax in axs.flat:
    #    ax.set(xlabel='Days', ylabel='Temperature [°K]')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #    ax.label_outer()

    fig.tight_layout()
    plt.savefig(save_dir + "ctrl_sim_no_snow.png", dpi=300)
    # plt.show()
    plt.clf()


def ctrl_sim():
    ##### Settings for ice-free conditions #####
    ### Instancing ###
    h_ice, h_snow, T_su, T_mix_lay, time_range = ice_thick(
        h_i0=0.1, ocean_heat=True, Q_w=5, snow=True, h_s0=0)
    ### Display ###
    ## Ice thickness evolution plot ##
    Q_w = 5
    h_i0 = 0.1
    h_s0 = 0

    fig, axs = plt.subplots(2, 2)
    fig.suptitle(
        'TSIM Model with snow\n' + r'$\alpha_S$ = {}, $Q_W = {}W/m^2$, $h_i(t=0) = {}m$, $h_s(t=0) = {}m, T = {}$ years'.format(alb_sur, Q_w, h_i0, h_s0, N_years))

    axs[0, 0].plot(time_range, h_ice, label="h_ice", linewidth=0.4)
    axs[0, 0].set_title('Ice thickness')
    axs[0, 0].set_xlabel('Days')
    axs[0, 0].set_ylabel('Thickness [m]')
    axs[0, 0].grid()

    axs[0, 1].plot(time_range, h_snow, label='h_snow',
                   color='c', linewidth=0.4)
    axs[0, 1].set_title('Snow thickness')
    axs[0, 1].sharex(axs[0, 0])
    axs[0, 1].grid()
    # axs[0, 1].set_xlabel('Days')
    # axs[0, 0].set_ylabel('Thickness [m]')

    axs[1, 0].plot(time_range, T_su, label='T_su',
                   color='orange', linewidth=0.4)
    axs[1, 0].set_title('Surface Temperature')
    axs[1, 0].sharex(axs[0, 0])
    axs[1, 0].set_xlabel('Days')
    axs[1, 0].set_ylabel('Temperature [°K]')
    axs[1, 0].grid()

    axs[1, 1].plot(time_range, T_mix_lay, label='T_w',
                   color='green', linewidth=0.4)
    axs[1, 1].set_title('Mix Layered Temperature')
    axs[1, 1].sharex(axs[1, 0])
    axs[1, 1].grid()
    # axs[1, 1].set_xlabel('Days')
    # axs[1, 1].set_ylabel('Temperature [°K]')

    # for ax in axs.flat:
    #    ax.set(xlabel='Days', ylabel='Temperature [°K]')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #    ax.label_outer()

    fig.tight_layout()
    plt.savefig(save_dir + "ctrl_sim_with_snow.png", dpi=300)
    # plt.show()
    plt.clf()


if __name__ == "__main__":
    # first_and_mult_ice()
    ctrl_sim_wth_snow()
    ctrl_sim()
