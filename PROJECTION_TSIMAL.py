############################################################################################################################
# Thermodynamic Sea Ice Model of Amaury Laridon (TSIMAL)
# Author : Amaury Laridon
# Course : LPHYS2265 - Sea ice ocean interactions in polar regions
# Goal : Final version of the TSIM model use for Projections and outputs for Part 2 of the project.
#        Modelisation of the evolution of sea-ice thickness with  a dynamic
#        surface temperature Free Surface Temperature (FST) and an Addition of Ocean and Snow (AOO)
#        More information on the GitHub Page of the project : https://github.com/AmauryLaridon/TSIM.git
# Date : 04/04/23
############################################################################################################################
#################################################### Packages ##############################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import mean_squared_error

####################################################### Parameters #########################################################
################################ Physical Constant #######################################

sigma = 5.67e-8  # Stefan-Boltzman constant [J/°K]
kelvin = 273.15  # Conversion form Celsius to Kelvin [Adim]
ki = 2.2  # sea ice thermal conductivity [W/m/K]
ks = 0.31  # Snow thermal conductivity [W/m/K]
sec_per_day = 86400  # seconds in one day [s/day]
L_fus = 3.35e5  # latent heat of fusion for water [J/kg]
rho_i = 917  # sea ice density [kg/m³]
rho_s = 330  # snow density [kg/m³]
c = 4000  # heat capacity of water [J/(kg.°K)]
rho_w = 1025  # density of sea water [kg/m³]

############################## Simulation Parameters ######################################
N_years = 100  # number of years in the simulation [Adim]
N_days = 365 * N_years  # number of days in the simulation [Adim]
alb_surf = 0.77  # surface albedo [Adim]
alb_wat = 0.1  # albedo of water [Adim]
h_w = 50  # depth of the ocean mix layer [m]
M_w = rho_w * h_w  # masse of water in the mixed layer [kg/m^2]
# temperature at the freezing point of sea water with a salinity of 34g/kg
T_bo = -1.8 + kelvin
Day_0 = 1  # set the first day of the simulation [Adim]


########## Tuning Parameters ##########
init_ctl = True  # condition wheter we start with the initial values of the last day of our CONTROL_TSIMAL.py run or not
# factor which multiplies thermal conductivities of ice and snow for tuning [Adim]
gamma_SM = 1.04
# part of the radiation reflected to space to compensate spurious melting following parametrisation of Semtner (1976) [Adim]
beta_SM = 0.0035
alb_dry_snow = 0.83  # albedo of dry snow [Adim]
alb_bare_ice = 0.64  # albedo of bare ice [Adim]
epsilon = 0.99  # surface emissivity [Adim]
dyn_alb = True  # condition wheter we use a fixed value of the surface albedo or not. Used for output
# critical value of the snow thickness that change the albedo regim. Used for tuning [m]
h_s_crit = 0.1
i_0 = 0.25  # fraction of radiation penetrating below the ice surface [Adim]
# snow fall modulator,=1 represent the standard values [Adim]
snow_fall_mod = 1
temp_lim = True  # temperature limited to 0°C following instruction 2.2.2
snow_ice_form = True  # enable or not the snow-ice formation process cfr instruction 3.2
# maximum longwave perturbation of x W/m² at the end of the century to simulate GHG. [W/m²]
lw_forcing = 3  # long wave radiation forcing [W/m²]

################################ Display Parameters #######################################
plt.rcParams["text.usetex"] = True
save_dir = "/home/amaury/Bureau/LPHYS2265 - Sea ice ocean atmosphere interactions in polar regions/Projet/Figures/"
data_dir = "/home/amaury/Bureau/LPHYS2265 - Sea ice ocean atmosphere interactions in polar regions/Projet/Data/"
figure = plt.figure(figsize=(16, 10))

############################################################################################################################
##################################################### TSIMAL MODEL #########################################################
############################################################################################################################

############################################## Model of Surface Temperature ##############################################

######################### Parameterization of atmospheric fluxes ####################################


def solar_flux(day):
    """Definition of the atmospheric solar heat flux Q_sol for a given day in the year.
    Conversion from Fletcher(1965)"""
    doy = day % 365
    Q_sol = 314 * np.exp((-((doy - 164) ** 2)) / 4608)
    return Q_sol


def non_solar_flux(day):
    """Definition of the atmospheric non solar heat flux Q_nsol for a given day in the year.
    Conversion from Fletcher(1965)"""
    doy = day % 365
    Q_nsol = 118 * np.exp((-0.5 * (doy - 206) ** 2) / (53**2)) + 179
    return Q_nsol


######################### Model of surface temperature evolution ####################################


def surface_temp(h_i, h_s, delta_lw, day, alb_surf, limit_temp=temp_lim):
    """Compute the evolution of the surface temperature with respect to the variation of the atmospheric
    heat fluxes and return a single value for a given day and a given thickness of the ice and snow
    and the energy available for melting at the surface"""

    # finding the surface temperature using roots function from numpy.
    if limit_temp == True:
        # As required in 2.1.2 temperature is not
        # physically sensible for ice in summer so we cap the surface temperature to 273,15°K.
        if h_i > 0:
            k_eff = ((ki * gamma_SM) * (ks * gamma_SM)) / (
                ki * gamma_SM * h_s + ks * gamma_SM * h_i
            )  # [W/m²/K]
            root = min(
                [
                    273.15,
                    np.roots(
                        [
                            -epsilon * sigma,
                            0,
                            0,
                            -k_eff,
                            k_eff * T_bo
                            + solar_flux(day)
                            * (1 - (alb_surf + beta_SM * (1 - alb_surf) * i_0))
                            + delta_lw
                            + non_solar_flux(day),
                        ]
                    ).real[3],
                ]
            )
        else:
            root = min(
                [
                    273.15,
                    np.roots(
                        [
                            -epsilon * sigma,
                            0,
                            0,
                            0,
                            solar_flux(day) * (1 - alb_wat)
                            + delta_lw
                            + non_solar_flux(day),
                        ]
                    ).real[3],
                ]
            )
    else:
        # Case when we dont limitate the surface temperature to remains below or egal to 0°C anymore.
        if h_i > 0:
            k_eff = ((ki * gamma_SM) * (ks * gamma_SM)) / (
                ki * gamma_SM * h_s + ks * gamma_SM * h_i
            )  # [W/m²/K]
            root = np.roots(
                [
                    -epsilon * sigma,
                    0,
                    0,
                    -k_eff,
                    k_eff * T_bo
                    + solar_flux(day)
                    * (1 - (alb_surf + beta_SM * (1 - alb_surf) * i_0))
                    + delta_lw
                    + non_solar_flux(day),
                ]
            ).real[3]
        else:
            root = np.roots(
                [
                    -epsilon * sigma,
                    0,
                    0,
                    0,
                    solar_flux(day) * (1 - alb_wat) + delta_lw + non_solar_flux(day),
                ]
            ).real[3]
    T_su = root

    def net_surf_flux(h_i, h_s, delta_lw, day, T_su, alb_surf):
        """Compute the net solar flux for a given day with a given sea ice and snow thickness"""
        k_eff = ((ki * gamma_SM) * (ks * gamma_SM)) / (
            ki * gamma_SM * h_s + ks * gamma_SM * h_i
        )  # [W/m²/K]
        nsf = (
            solar_flux(day) * (1 - (alb_surf + beta_SM * (1 - alb_surf) * i_0))
            + delta_lw
            + non_solar_flux(day)
            - epsilon * sigma * (T_su**4)
            - k_eff * (T_su - T_bo)
        )
        return nsf

    nsf = net_surf_flux(h_i, h_s, delta_lw, day, T_su, alb_surf)

    if nsf > 0:
        # If the net solar flux is positive, this energy is available for melting and will be stored in a variable efm (energy for melting)
        efm = nsf
    else:
        efm = 0  # If the net solar flux is negative or egal to zero, the efm = 0

    return T_su, efm


########################################### Model of the Ocean Mix Layer ##################################################


def E_gain_mixed_layer(T_w, day, Q_w):
    """Compute the free water energy gain in one day. [J/m^2]"""
    E_gain_mix_lay = (
        solar_flux(day) * (1 - alb_wat)
        + non_solar_flux(day)
        + Q_w
        - epsilon * sigma * (T_w**4)
    ) * sec_per_day
    return E_gain_mix_lay


################################################ Model of snow fall #######################################################


def snow_fall(day):
    """Function that modelise the snowfall in [m]. The values are given in the Exercise_part_1.pdf file available on the GitHub.
    30 cm between 20 August and October, 5 cm between November and april, 5 cm in May. We use an uniform distribution of those snowfall
    during these three different periods. Un snow_fall_mod coefficient is used to linearly multiply the snow fall for other simulations
    settings. Function builded with the help of Augustin Lambotte."""
    doy = day % 365
    if doy >= 232 and doy <= 304:  # between 20 August and October (included)
        return (0.3 / (304 - 232)) * snow_fall_mod
    elif doy >= 305 or doy <= 120:  # between November and April (included)
        return (0.05 / (365 - 305 + 120)) * snow_fall_mod
    elif doy >= 121 and doy <= 151:  # May
        return (0.05 / (151 - 121)) * snow_fall_mod
    else:
        return 0


########################################### Model of Sea ice thickness ####################################################


def fourier_cond_flux(h_i, T_su, snow, h_s):
    """Computation of the conductive heat flux Q_c trough the ice using the Fourier-Fick's law (upward positive)
    [W/m^2]"""
    if snow == False:
        Q_c = ((T_bo - T_su) / (h_i)) * (ki * gamma_SM)
    else:
        k_eff = ((ki * gamma_SM) * (ks * gamma_SM)) / (
            ki * gamma_SM * h_s + ks * gamma_SM * h_i
        )  # [W/m²/K]
        Q_c = (T_bo - T_su) * k_eff  # [W/m²]
    print("Fourier-Thick conductive flux = {:.2f} W/m²".format(Q_c))
    return Q_c


def E_net_bottom(ice_thick, ocean_heat, Q_w, snow, h_s, T_su, T_bo=T_bo):
    """Compute the total energy lost in one day at the bottom of the sea ice layer (upward positive)[J/m^2]"""
    if ocean_heat:
        E_loss_through_ice = fourier_cond_flux(ice_thick, T_su, snow, h_s) * sec_per_day
        E_gain_ocean_flux = Q_w * sec_per_day
        E_net_bot = E_loss_through_ice - E_gain_ocean_flux
    else:
        E_loss_through_ice = fourier_cond_flux(ice_thick, T_su, snow, h_s) * sec_per_day
        E_net_bot = E_loss_through_ice
    return E_net_bot


def E_net_surf(efm):
    """Compute the energy gain at the surface of the sea ice in one day due to a non-equilibrium between the solar and non-solar
    flux incoming, the flux coming from the water through the ice and the ice blackbody radiation. This disequilibrium
    is because the ice can't go over 273.15K without melting. This energy will hence be used to melt the ice during summer.
    Function builded with the help of Augustin Lambotte.
    [J/m^2]
    """
    E_net_surf = efm * sec_per_day
    return E_net_surf


def ice_thick(
    h_i0,
    ocean_heat,
    Q_w,
    snow,
    h_s0,
    integration_range=N_days,
    T_bo=T_bo,
    limit_temp=temp_lim,
    init_ctl=init_ctl,
):
    """Computation of the evolution of the sea ice and snow thickness using Stefan's law.
    An option gives the possibility to add an Oceanic heat flux and a layer of snow.
    This function returns an array with the sea ice thickness, snow thickness, surface temperature, mixed layer temperature,
    an array with the time of integration usefull for plotting, an array with the height of the volume of water displaced
    by the volume of ice and snow and the number of year needed to obtain equilibrium"""

    ##### Output Simulation Settings #####
    print("------------------------------------------------------------------")
    print("            TSIMAL SEA ICE AND SNOW THICKNESS MODEL")
    print("------------------------------------------------------------------")
    print(
        "Evolution of the sea ice thickness using numerical Stefan's law.\nintegration range = {} days, dyn_albedo = {}, T_bo = {:.2f} °C,\nh_i0 = {:.2f} m, ocean_heat_flux = {}, Q_w = {:.2f} W/m²\nsnow = {}, h_s0 = {:.2f} m".format(
            N_days, dyn_alb, T_bo - kelvin, h_i0, ocean_heat, Q_w, snow, h_s0
        )
    )
    print("------------------------------------------------------------------")

    ##### Initialization #####

    if init_ctl == True:
        # We recover for the initial values of h_i, h_s, T_w and T_su the last value
        # of CONTROL_TSIMAL.py at the end of the equilibrium cycle
        ctl_data = np.genfromtxt(
            "/home/amaury/Bureau/LPHYS2265 - Sea ice ocean atmosphere interactions in polar regions/Projet/Data/CTL_TSIMAL.txt"
        )
        h_i0 = ctl_data[-1, 2]
        h_s0 = ctl_data[-1, 3]
        T_su_0 = ctl_data[-1, 1]
        T_mix_lay_0 = ctl_data[-1, 4]
    else:
        T_su_0, efm = surface_temp(
            h_i0, h_s0, delta_lw, alb_surf, day=1, limit_temp=temp_lim
        )
        T_mix_lay_0 = T_bo
        # array colecting the values of the sea ice thickness for each day
    h_i = np.zeros(N_days)
    # initial condition for the sea ice thickness for the first day [m]
    h_i[0] = h_i0
    # array colecting the values of snow thickness for each day
    h_s = np.zeros(N_days)
    # initial condition for the snow thickness for the first day [m]
    h_s[0] = h_s0
    # array colecting the values of surface temperature for each day
    T_su_ar = np.zeros(N_days)
    # initialized surface temperature with CONTROL_TSIMAL.py
    T_su_ar[0] = T_su_0
    # array colecting the values of ocean mix layer temperature for each day
    T_mix_lay_ar = np.zeros(N_days)
    # initialized surface temperature with CONTROL_TSIMAL.py
    T_mix_lay_ar[0] = T_mix_lay_0
    # array that stored the height of the volume of water displaced by the volume of ice and snow [m]
    h_w_ar = np.zeros(N_days)
    T_eq = 0  # time in days needed to obtain equilibrium in the ice thickness
    # array that stored the surface albedo value [Adim]
    alb_sur_ar = np.zeros(N_days)
    alb_sur_ar[
        0
    ] = alb_dry_snow  # for the projection simulations that start from the end of the control run we have an initial snow thickness such that the surface albedo is the one of snow.
    # At the beggining, the mixed layer temperature is equal to the sea ice bottom temperature [K]
    T_w = T_bo
    time_range = range(0, integration_range)  # integration range in days

    ##### Dynamic Model ######
    for day in range(1, integration_range):
        ### Output ###
        print("Day {}/{}   |".format(day, integration_range - 1))
        print("------------------|")
        print(
            "Sea ice thickness at begining of Day {} = {:.2f} m".format(
                day, h_i[day - 1]
            )
        )
        if snow == True:
            print(
                "Snow thickness at begining of Day {} = {:.2f} m".format(
                    day, h_s[day - 1]
                )
            )

        ### Longwave perturbation ###
        delta_lw_max = lw_forcing
        year = np.modf(day / 365)[1]
        # print(year)
        delta_lw = delta_lw_max * year / 100  # [W/m²]
        # delta_lw_J = delta_lw * sec_per_day  # [J/m²]

        ### Definition alb_sur ###
        if dyn_alb == True:
            # If the dynamic surface albedo option is True we consider the differents values that it may takes.
            # If dyn_alb == False we remains with the standard values define in the parameters above.
            if h_s[day - 1] > h_s_crit:
                # If the snow thickness is superior than the critical value, the surface albedo is the one of the dry snow
                alb_sur = alb_dry_snow
                alb_sur_ar[day] = alb_sur
            elif h_s[day - 1] == 0:
                # if there is no snow in the simulation the albedo of the surface is the one of the bare ice
                alb_sur = alb_bare_ice
                alb_sur_ar[day] = alb_sur
            elif h_s[day - 1] < h_s_crit and h_s[day - 1] > 0:
                # slope of the linear relation for the surface albedo between zero thickness of snow and the critical value giving a surface albedo between the bare ice value and the dry snow value
                m = (alb_dry_snow - alb_bare_ice) / h_s_crit
                alb_sur = m * h_s[day - 1] + alb_bare_ice
                alb_sur_ar[day] = alb_sur
        else:
            alb_sur = alb_surf
            alb_sur_ar[0] = alb_sur
            alb_sur_ar[day] = alb_sur

        ### Ice Cover testing condition ###
        # Test if there is some ice cover or not. If they are an ice cover we perform the same computation as before,
        # the temperature of the ocean mixed layer remains at it's initial value of the freezing point temperature.
        # If they are no more ice cover, we compute the energy desequilibrium to warm or cool the mixed layer.
        if h_i[day - 1] > 0:
            ice_cover = True
        else:
            ice_cover = False

        if ice_cover == True:
            # In order to have a layer of snow we need to have a layer of ice.
            if snow == True:
                ## Snow thickness instanciation ##
                # We first add the snow layer corresponding to the snow fall for a given day. We will later compute wheter there is
                # a snow melting or not.
                snow_gain = snow_fall(day)
            else:
                snow_gain = 0  # if we use the option of not having snow there couldn't be snow gain
                # if there is no snow in the simulation the albedo of the surface is the one of the bare ice
                alb_sur = alb_bare_ice

            ### Surface temperature computation ###
            # Computation of the surface temperature given a particular day and ice and snow thickness

            T_su, efm = surface_temp(
                h_i[day - 1],
                h_s[day - 1],
                delta_lw,
                day,
                limit_temp=temp_lim,
                alb_surf=alb_sur,
            )
            T_su_ar[day] = T_su

            ### Energy change at the bottom ###
            # Use to compute wheter there is an sea ice thickness gain from the bottom of sea ice.
            # Energy lost at the bottom during one day due to flux from water to ice.[J/m^2]
            E_net_bot = E_net_bottom(
                h_i[day - 1], ocean_heat, Q_w, snow, h_s[day - 1], T_su
            )
            # Mass of water freezed at the bottom of the ice layer at the end of one day [kg/m²]
            freezing_water_mass = E_net_bot / L_fus
            # To obtain [m] as needed
            sea_ice_gain = freezing_water_mass / rho_i

            ### Energy change at the surface ###
            # Use to compute the energy budget at surface and wheter there is energy available for melting ice or snow or both.
            # Energy gain at the surface during one day due to non equilibrium + the radiative forcing imposed [J/m^2]
            E_net_sur = E_net_surf(efm)  # [J/m²]
            if h_s[day - 1] > 0:
                # Case where there is still a layer of snow above the ice at the end of the previous day
                # We first compute what will be the snow layer loss if the total energy available for melting is used to melt snow.
                # Mass of ice melted at the surface of the ice layer at the end of one day [kg/m²]
                melt_snow_mass = E_net_sur / L_fus  # [kg/m²]
                # To obtain [m] as needed
                snow_lost = melt_snow_mass / rho_s
                if snow_lost > h_s[day - 1]:
                    # If there is more energy in order to melt snow that the energy needed to melt the entire layer of snow
                    # we completely melt the layer of snow and we use the additionnal energy to melt ice.
                    print("Too much energy for melting only the layer of snow.")
                    h_s[day] = 0
                    # conversion of the excessive snow thickness loss in energy
                    excess_snow_lost = np.abs(h_s[day - 1] - snow_lost)  # [m]
                    melt_excess_snow_mass = rho_s * excess_snow_lost  # [kg/m²]
                    E_excess = L_fus * melt_excess_snow_mass  # [J/m²]
                    # we loose the entire old layer of snow from the previous day
                    snow_lost = h_s[day - 1]

                    # conversion of this excessive energy in a ice thickness lost
                    # Mass of ice melted at the surface of the ice layer at the end of one day [kg/m²]
                    melt_ice_mass = E_excess / L_fus  # [kg/m²]
                    # To obtain [m] as needed
                    sea_ice_lost = melt_ice_mass / rho_i
                else:
                    # On this particular day we have still a layer of snow at the beginning and the end of this day. There is only a melting of the snow
                    # and so a loss of snow thickness but not on ice thickness.
                    sea_ice_lost = 0

            else:
                # Case where there is no longer an layer of snow at the end of the previous day. The only thing that can melt is the ice.
                # Mass of ice melted at the surface of the ice layer at the end of one day [kg/m²]
                melt_ice_mass = E_net_sur / L_fus  # [kg/m²]
                # To obtain [m] as needed
                sea_ice_lost = melt_ice_mass / rho_i
                snow_lost = 0

            ## Mix layer temperature ##
            T_mix_lay_ar[day] = T_bo

            ## Net balance of sea ice thickness ##
            delta_h_i = sea_ice_gain - sea_ice_lost
            h_i[day] = h_i[day - 1] + delta_h_i
            ## Net balance of snow thickness ##
            delta_h_s = snow_gain - snow_lost
            h_s[day] = h_s[day - 1] + delta_h_s

            ## Testing for snow ice formation after snow fall and possible melting of snow and ice and ice formation##
            if snow_ice_form == True:
                h_w = (h_i[day] * rho_i + h_s[day] * rho_s) / rho_w
                h_w_ar[day] = h_w
                if h_i[day] > 0 and h_s[day] > 0:
                    # In the case where there is a layer of ice and still a layer of snow above at the end of the day we will add the possibility to have
                    # snow-ice formation. We compute the snow-ice interface and wheter it is above or below sea-level using the fundamental law of static
                    # with the weight force and the Archimede's force.
                    # compute h_w the height of the water volume displaced [m]

                    if h_w > h_i[day]:
                        # If the height of the water volume displaced is superior than the thickness of the ice then all the layer of ice is below sea level
                        # and there is a height of h_w-h_i[day] of snow than can be froozen to ice.
                        h_snow_below_sea = h_w - h_i[day]  # [m]
                        h_s[day] = h_s[day] - h_snow_below_sea
                        if h_s[day] < 0:
                            # if there not enough snow thickness we define de snow thickness as beeing equal to zero to keep physical quantities.
                            h_s[day] = 0
                        h_i[day] = h_i[day] + h_snow_below_sea

        if ice_cover == False:
            # set the latest ice thickness to 0 in order to have physical value.
            h_i[day - 1] = 0
            # output
            print("No ice cover at beginning of Day {}".format(day))
            # if there is no ice there is no layer of snow
            h_s[day - 1] = 0
            ## Surface temperature computation ##
            # Computation of the surface temperature given a particular day and ice thickness
            T_su, efm = surface_temp(
                h_i[day - 1],
                h_s[day - 1],
                delta_lw,
                day,
                limit_temp=temp_lim,
                alb_surf=alb_wat,
            )
            T_su_ar[day] = T_su
            alb_sur_ar[day] = alb_wat
            if T_w >= T_bo:
                # In this case the water can warm without producing sea ice
                # Energy gain by the mixed layer in one day [J/m^2]
                delta_h = 0
                E_gain = E_gain_mixed_layer(T_w, day, Q_w)
                T_w += E_gain / (M_w * c)  # New mixed layer temperature [°K]
                T_mix_lay_ar[day] = T_w
            else:
                # In this case the water is cooling below the freezing point so we re-create ice
                delta_T_excess = np.abs(T_bo - T_w)
                # Excess of heat which will be turn into ice [J/m^2]
                E_gain = delta_T_excess * M_w * c
                freezing_water_mass = E_gain / L_fus  # [kg/m^2]
                h_i[day] = freezing_water_mass / rho_i  # [m]

                # We make this assumption in order to avoid a bug due to a too large time step
                if h_i[day] < 0.1:
                    h_i[day] = 0.1
                    delta_h = 0.1
                delta_h = h_i[day]
                T_w = T_bo  # set the bottom temperature [°K]
                T_mix_lay_ar[day] = T_w

        ## Output of simulation ##
        print(
            "Energy balance at the bottom during Day {} = {:.2f} MJ/m²".format(
                day, E_net_bot / 1e6
            )
        )
        print(
            "Variation of sea-ice thickness during Day {} = {:.2f} m".format(
                day, delta_h_i
            )
        )
        print("Sea ice thickness at the end of Day {} = {:.2f} m".format(day, h_i[day]))
        print("------------------------------------------------------------------")

    data = np.column_stack((time_range, T_su_ar, h_i, h_s, T_mix_lay_ar))
    np.savetxt(
        data_dir + "PR" + str(lw_forcing) + "_TSIMAL_full_data.txt",
        data,
        delimiter=" ",
        fmt="%s ",
    )

    return h_i, h_s, T_su_ar, T_mix_lay_ar, time_range, h_w_ar, alb_sur_ar


########################################################################################################################
############################################## CONTROL SIMULATIONS #####################################################
########################################################################################################################


def first_and_mult_ice():
    ##### Settings for ice-free conditions #####
    ### Instancing ###
    (
        h_ice_free,
        h_snow_ice_free,
        T_su_ice_free,
        T_mix_lay_ice_free,
        time_range,
        h_w_ar,
        alb,
    ) = ice_thick(h_i0=0.1, ocean_heat=True, Q_w=5, snow=False, h_s0=0)

    ### Yearly Diagnostics ###

    # array to store the annual maximum ice thickness
    h_i_max = np.zeros(N_years)
    # array to store the annual minimum ice thickness
    h_i_min = np.zeros(N_years)
    h_i_mean = np.zeros(N_years)  # array to store the annual mean thickness
    # array to store the annual maximum snow thickness
    h_s_max = np.zeros(N_years)
    # array to store the annual minima of surface temperature
    T_su_min = np.zeros(N_years)

    for year in range(0, N_years):
        day00 = year * 365
        day99 = day00 + 364
        h_i_max[year] = max(h_ice_free[day00:day99])
        h_s_max[year] = max(h_snow_ice_free[day00:day99])
        h_i_min[year] = min(h_ice_free[day00:day99])
        h_i_mean[year] = np.mean(h_ice_free[day00:day99])
        T_su_min[year] = min(T_su_ice_free[day00:day99])

    ### Display ###
    ## Ice thickness evolution plot ##
    Q_w = 5
    h_s0 = 0
    h_i0 = 0.1

    plt.plot(time_range, h_ice_free, label="h_ice")
    plt.title(
        "TSIMAL Ice thickness evolution without snow\n"
        + r"dyn_alb = {}, $Q_W = {}W/m^2$, $h_i(t=0) = {}m$, $h_s(t=0) = {}m, T = {}$ years, $\gamma$ = {}".format(
            dyn_alb, Q_w, h_i0, h_s0, N_years, gamma_SM
        ),
        size=22,
    )
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
    plt.title(
        "TSIMAL Temperature evolution without snow\n"
        + r"dyn_alb = {}, $Q_W = {}W/m^2$, $h_i(t=0) = {}m$, $h_s(t=0) = {}m, T = {}$ years, $\gamma$ = {}".format(
            dyn_alb, Q_w, h_i0, h_s0, N_years, gamma_SM
        ),
        size=22,
    )
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


def ctrl_sim_without_snow():
    ##### Settings for ice-free conditions #####
    ### Instancing ###
    h_ice, h_snow, T_su, T_mix_lay, time_range, h_w_ar, alb = ice_thick(
        h_i0=0.1, ocean_heat=True, Q_w=2, snow=False, h_s0=0
    )

    ### Yearly Diagnostics ###

    # array to store the annual maximum ice thickness
    h_i_max = np.zeros(N_years)
    # array to store the annual minimum ice thickness
    h_i_min = np.zeros(N_years)
    h_i_mean = np.zeros(N_years)  # array to store the annual mean thickness
    # array to store the annual maximum snow thickness
    h_s_max = np.zeros(N_years)
    # array to store the annual minima of surface temperature
    T_su_min = np.zeros(N_years)

    for year in range(0, N_years):
        day00 = year * 365
        day99 = day00 + 364
        h_i_max[year] = max(h_ice[day00:day99])
        h_s_max[year] = max(h_snow[day00:day99])
        h_i_min[year] = min(h_ice[day00:day99])
        h_i_mean[year] = np.mean(h_ice[day00:day99])
        T_su_min[year] = min(T_su[day00:day99])

    ### Display ###
    ## Ice thickness evolution plot ##
    h_i0 = 0.1
    Q_w = 2
    h_s0 = 0
    time_range_years = [time_range[i] / 365 for i in range(N_days)]

    fig, axs = plt.subplots(2, 2)
    fig.suptitle(
        "TSIMAL Model without snow\n"
        + r"dyn_alb = {}, $Q_W = {}W/m^2$, $\gamma$ = {}, $\beta$ = {}".format(
            dyn_alb, Q_w, gamma_SM, beta_SM
        )
        + "\n"
        + r"$h_i(t=0) = {}m$, $h_s(t=0) = {}m, T = {}$ years".format(
            h_i0, h_s0, N_years
        )
    )

    axs[0, 0].plot(time_range_years, h_ice, label="h_ice", linewidth=0.4)
    axs[0, 0].set_title("Ice thickness")
    axs[0, 0].set_xlabel("Year")
    axs[0, 0].set_ylabel("Thickness [m]")
    axs[0, 0].set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    # axs[0, 1].set_xticks(np.arange(0, 5, 1))
    axs[0, 0].grid()

    axs[0, 1].plot(time_range_years, h_snow, label="h_snow", color="c", linewidth=0.4)
    axs[0, 1].set_title("Snow thickness")
    axs[0, 1].set_xlabel("Year")
    axs[0, 1].sharex(axs[0, 0])
    axs[0, 1].grid()
    # axs[0, 1].set_xlabel('Days')
    # axs[0, 0].set_ylabel('Thickness [m]')

    axs[1, 0].plot(
        time_range_years, T_su - kelvin, label="T_su", color="orange", linewidth=0.4
    )
    axs[1, 0].set_title("Surface Temperature")
    axs[1, 0].sharex(axs[0, 0])
    axs[1, 0].set_xlabel("Year")
    axs[1, 0].set_ylabel("Temperature [°C]")
    axs[1, 0].grid()

    axs[1, 1].plot(
        time_range_years, T_mix_lay - kelvin, label="T_w", color="green", linewidth=0.4
    )
    axs[1, 1].set_title("Mix Layered Temperature")
    axs[1, 1].sharex(axs[1, 0])
    axs[1, 1].set_xlabel("Year")
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
    h_ice, h_snow, T_su, T_mix_lay, time_range, h_w_ar, alb_sur_ar = ice_thick(
        h_i0=0.1, ocean_heat=True, Q_w=2, snow=True, h_s0=0
    )

    ### Yearly Diagnostics ###

    # array to store the annual maximum ice thickness
    h_i_max = np.zeros(N_years)
    # array to store the annual minimum ice thickness
    h_i_min = np.zeros(N_years)
    h_i_mean = np.zeros(N_years)  # array to store the annual mean thickness
    # array to store the annual maximum snow thickness
    h_s_max = np.zeros(N_years)
    # array to store the annual minima of surface temperature
    T_su_min = np.zeros(N_years)

    for year in range(0, N_years):
        day00 = year * 365
        day99 = day00 + 364
        h_i_max[year] = max(h_ice[day00:day99])
        h_s_max[year] = max(h_snow[day00:day99])
        h_i_min[year] = min(h_ice[day00:day99])
        h_i_mean[year] = np.mean(h_ice[day00:day99])
        T_su_min[year] = min(T_su[day00:day99])

    ## Exporting yearly diagnostics data ##
    yearly_diag = np.column_stack(
        (np.arange(0, N_years), h_i_min, h_i_mean, h_i_max, h_s_max, T_su_min)
    )
    np.savetxt(
        data_dir + "PR" + str(lw_forcing) + "_TSIMAL.txt",
        yearly_diag,
        delimiter=" ",
        fmt="%s ",
    )

    ### Display ###
    Q_w = 2
    h_i0 = 0.1
    h_s0 = 0
    time_range_years = [time_range[i] / 365 for i in range(N_days)]

    ## Submerged height plot ##

    if snow_ice_form == True:
        plt.plot(time_range_years, h_w_ar, label=r"$h_w$")
        plt.plot(time_range_years, h_ice, label=r"$h_{ice}$")
        plt.title(
            "TSIMAL Submerged height and ice thickness evolution\n"
            + r"dyn_alb = {}, $Q_W = {}W/m^2$, $\gamma$ = {}, $\beta$ = {}".format(
                dyn_alb, Q_w, gamma_SM, beta_SM
            )
            + "\n"
            + r"$h_i(t=0) = {}m$, $h_s(t=0) = {}m, T = {}$ years".format(
                h_i0, h_s0, N_years
            ),
            size=11,
        )
        plt.xlabel("Year", size=8)
        plt.ylabel("Height [m]", size=8)
        plt.legend(fontsize=8)
        plt.grid()
        # plt.savefig(
        #    save_dir + "PR" + str(lw_forcing) + "_TSIMAL_" + "water_height" + ".png",
        #    dpi=300,
        # )
        # plt.show()
        plt.clf()

    ## Control Subplot ##

    fig, axs = plt.subplots(2, 2)
    fig.suptitle(
        r"TSIMAL Projection with $\Delta Q = {}W/m^2$".format(lw_forcing)
        + "\n"
        + r"dyn_alb = {}, $Q_W = {}W/m^2$, $\gamma$ = {}, $\beta$ = {}".format(
            dyn_alb, Q_w, gamma_SM, beta_SM
        )
        + "\n"
        + r"$h_i(t=0) = {}m$, $h_s(t=0) = {}m, T = {}$ years".format(
            h_i0, h_s0, N_years
        )
    )

    axs[0, 0].plot(time_range_years, h_ice, label="h_ice", linewidth=0.4)
    axs[0, 0].set_title("Ice thickness")
    axs[0, 0].set_xlabel("Year")
    axs[0, 0].set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    axs[0, 0].set_ylabel("Thickness [m]")
    axs[0, 0].grid()

    axs[0, 1].plot(time_range_years, h_snow, label="h_snow", color="c", linewidth=0.4)
    axs[0, 1].set_title("Snow thickness")
    axs[0, 1].sharex(axs[0, 0])
    axs[0, 1].set_xlabel("Year")
    axs[0, 1].grid()
    # axs[0, 1].set_xlabel('Days')
    # axs[0, 0].set_ylabel('Thickness [m]')

    axs[1, 0].plot(
        time_range_years, T_su - kelvin, label="T_su", color="orange", linewidth=0.4
    )
    axs[1, 0].set_title("Surface Temperature")
    axs[1, 0].sharex(axs[0, 0])
    axs[1, 0].set_xlabel("Year")
    axs[1, 0].set_ylabel("Temperature [°C]")
    axs[1, 0].grid()

    axs[1, 1].plot(
        time_range_years, T_mix_lay - kelvin, label="T_w", color="green", linewidth=0.4
    )
    axs[1, 1].set_title("Mix Layered Temperature")
    axs[1, 1].sharex(axs[1, 0])
    axs[1, 1].grid()
    axs[1, 1].set_xlabel("Year")
    # axs[1, 1].set_xlabel('Days')
    # axs[1, 1].set_ylabel('Temperature [°K]')

    # for ax in axs.flat:
    #    ax.set(xlabel='Days', ylabel='Temperature [°K]')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #    ax.label_outer()

    fig.tight_layout()
    plt.savefig(save_dir + "PR" + str(lw_forcing) + "_TSIMAL.png", dpi=300)
    # plt.show()
    plt.clf()

    ## Control Subplot Yearly diagnostics ##

    fig, axs = plt.subplots(2, 2)
    fig.suptitle(
        r"Yearly diagnostics TSIMAL Projection with $\Delta Q = {}W/m^2$".format(
            lw_forcing
        )
        + "\n"
        + r"dyn_alb = {}, $Q_W = {}W/m^2$, $\gamma$ = {}, $\beta$ = {}".format(
            dyn_alb, Q_w, gamma_SM, beta_SM
        )
        + "\n"
        + r"$h_i(t=0) = {}m$, $h_s(t=0) = {}m, T = {}$ years".format(
            h_i0, h_s0, N_years
        )
    )

    axs[0, 0].plot(np.arange(0, N_years), h_i_max, label=r"$h_i^{max}$", linewidth=1)
    axs[0, 0].plot(np.arange(0, N_years), h_i_min, label=r"$h_i^{min}$", linewidth=1)
    axs[0, 0].plot(np.arange(0, N_years), h_i_mean, label=r"$h_i^{mean}$", linewidth=1)
    axs[0, 0].set_title("Ice thickness")
    axs[0, 0].set_xlabel("Year")
    axs[0, 0].set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    axs[0, 0].set_ylabel("Thickness [m]")
    axs[0, 0].grid()
    axs[0, 0].legend()

    axs[0, 1].plot(
        np.arange(0, N_years), h_s_max, label=r"$h_s^{max}$", color="c", linewidth=1
    )
    axs[0, 1].set_title("Snow thickness")
    axs[0, 1].sharex(axs[0, 0])
    axs[0, 1].set_xlabel("Year")
    axs[0, 1].grid()
    axs[0, 1].legend()
    # axs[0, 1].set_xlabel('Days')
    # axs[0, 0].set_ylabel('Thickness [m]')

    axs[1, 0].plot(
        np.arange(0, N_years),
        T_su_min - kelvin,
        label=r"$T_{su}^{min}$",
        color="orange",
        linewidth=1,
    )
    axs[1, 0].set_title("Surface Temperature")
    axs[1, 0].sharex(axs[0, 0])
    axs[1, 0].set_xlabel("Year")
    axs[1, 0].set_ylabel("Temperature [°C]")
    axs[1, 0].grid()
    axs[1, 0].legend()

    axs[1, 1].plot(
        time_range_years, T_mix_lay - kelvin, label="T_w", color="green", linewidth=1
    )
    axs[1, 1].set_title("Mix Layered Temperature")
    axs[1, 1].sharex(axs[1, 0])
    axs[1, 1].grid()
    axs[1, 1].set_xlabel("Year")
    # axs[1, 1].set_xlabel('Days')
    # axs[1, 1].set_ylabel('Temperature [°K]')

    # for ax in axs.flat:
    #    ax.set(xlabel='Days', ylabel='Temperature [°K]')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #    ax.label_outer()

    fig.tight_layout()
    plt.savefig(save_dir + "PR" + str(lw_forcing) + "_yearly_diag_TSIMAL.png", dpi=300)
    # plt.show()
    plt.clf()

    ## Surface albedo evolution plot ##

    plt.plot(time_range, alb_sur_ar, label="alb_sur")
    plt.plot(time_range, h_snow, label="h_snow")
    plt.title(
        "TSIMAL Surface Albedo Evolution\n"
        + r"dyn_alb = {}, $Q_W = {}W/m^2$, $h_i(t=0) = {}m$, $h_s(t=0) = {}m, T = {}$ years, $\gamma$ = {}".format(
            dyn_alb, Q_w, h_i0, h_s0, N_years, gamma_SM
        ),
        size=10,
    )
    plt.xlabel("Days", size=10)
    plt.ylabel("Surface Albedo [Adim] and Snow Thickness [m]", size=10)
    plt.legend(fontsize=10)
    plt.grid()
    plt.savefig(save_dir + "surf_albedo.png", dpi=300)
    # plt.show()
    plt.clf()


########################################################################################################################
########################################### TUNING AND PROJECTIONS #####################################################
########################################################################################################################

################################################ Analysis tools ########################################################


def month_mean_v1(h):
    """NOT WORKING ! Given the array of the daily thickness of a layer (snow or ice) for one year returns the means for every month
    of that value. A small falsification is made we consider that the 11 firsts month have a length of 31 days
    and the last month of 24 days to get a full year of 365 days. A more accurate computation can be
    done by considering the exact number of days per month in the calendar."""
    nbr_month = 12
    N_days_per_month_1 = 31
    N_days_per_month_2 = 24
    h_mean_month_ar = np.zeros(nbr_month)
    for month in range(nbr_month):
        h_sum = 0
        print(h_mean_month_ar[month])
        if month == nbr_month:
            for day in range(N_days_per_month_2):
                h_sum = 0
                h_sum += h[day]
            h_mean_month = h_sum / N_days_per_month_2
        else:
            for day in range(N_days_per_month_1):
                h_sum += h[day]

            h_mean_month = h_sum / N_days_per_month_1

        h_mean_month_ar[month] = h_mean_month

    return h_mean_month_ar


def month_mean_v2(h):
    """Given the array of the daily thickness of a layer(snow or ice) for one year returns the means for every month
    of that value."""

    h_mean_month_ar = np.zeros(12)

    h_sum = 0
    for day in range(0, 31):
        h_sum += h[day]
    h_mean = h_sum / 31
    h_mean_month_ar[0] = h_mean

    h_sum = 0
    for day in range(31, 59):
        h_sum += h[day]
    h_mean = h_sum / 28
    h_mean_month_ar[1] = h_mean

    h_sum = 0
    for day in range(59, 90):
        h_sum += h[day]
    h_mean = h_sum / 31
    h_mean_month_ar[2] = h_mean

    h_sum = 0
    for day in range(90, 120):
        h_sum += h[day]
    h_mean = h_sum / 30
    h_mean_month_ar[3] = h_mean

    h_sum = 0
    for day in range(120, 151):
        h_sum += h[day]
    h_mean = h_sum / 31
    h_mean_month_ar[4] = h_mean

    h_sum = 0
    for day in range(151, 181):
        h_sum += h[day]
    h_mean = h_sum / 30
    h_mean_month_ar[5] = h_mean

    h_sum = 0
    for day in range(181, 212):
        h_sum += h[day]
    h_mean = h_sum / 31
    h_mean_month_ar[6] = h_mean

    h_sum = 0
    for day in range(212, 243):
        h_sum += h[day]
    h_mean = h_sum / 31
    h_mean_month_ar[7] = h_mean

    h_sum = 0
    for day in range(243, 273):
        h_sum += h[day]
    h_mean = h_sum / 30
    h_mean_month_ar[8] = h_mean

    h_sum = 0
    for day in range(273, 304):
        h_sum += h[day]
    h_mean = h_sum / 31
    h_mean_month_ar[9] = h_mean

    h_sum = 0
    for day in range(304, 334):
        h_sum += h[day]
    h_mean = h_sum / 30
    h_mean_month_ar[10] = h_mean

    h_sum = 0
    for day in range(334, 365):
        h_sum += h[day]
    h_mean = h_sum / 31
    h_mean_month_ar[11] = h_mean

    return h_mean_month_ar


def err_annual_mean_thick(h, mu71):
    """Compute the annual mean thickness for TSIMAL model output and for MU71 serie and compute the absolute error
    and the relative error between the two."""
    model_annual_mean_thick = sum(h) / 12
    mu71_annual_mean_thick = sum(mu71) / 12
    err_abs = model_annual_mean_thick - mu71_annual_mean_thick
    err_rel = (
        (model_annual_mean_thick - mu71_annual_mean_thick) / mu71_annual_mean_thick
    ) * 100
    return model_annual_mean_thick, mu71_annual_mean_thick, err_abs, err_rel


def MSE_annual_mean_thick(h, mu71):
    """Compute the Mean Squared Error for TSIMAL model output regarding the MU71 serie. The MSE will be used as a diagnostic tool
    for the efficienty of TSIMAL to reproduce the MU71 serie. The goal is to minimize MSE with tuning.
    """
    mse = mean_squared_error(h, mu71)
    return mse


def cor_annual_mean_thick(h, mu71):
    """Computes the correlation coefficient for TSIMAL model output regarding the MU71 serie. The value r will be used as a diagnostic
    tool for the efficienty of TSIMAL to reproduce the MU71 serie. The goal is to have the highest coefficient r in absolute value
    with tuning."""
    corr_matrix = np.corrcoef(h, mu71)
    r = corr_matrix[0, 1]
    return r


def std_var_mean_thick(h):
    std = np.std(h)
    return std


##### MU 71 Ice Thickness #####
hi_MU71 = [
    2.82,
    2.89,
    2.97,
    3.04,
    3.10,
    3.14,
    2.96,
    2.78,
    2.73,
    2.71,
    2.72,
    2.75,
]  # Target seasonal cycle of ice thickness of MU71

########################################### Tuning Comparaison ######################################################


def tuning_comp():
    """Simulation and first comparaison between TSIMAL model and MU71 without any tuning.
    alb = 0.77, Q_W = 2W/m² and snow == True"""

    ### Instancing ###
    h_ice, h_snow, T_su, T_mix_lay, time_range, h_w_ar, alb = ice_thick(
        h_i0=0.1, ocean_heat=True, Q_w=2, snow=True, h_s0=0
    )

    ### Display ###
    time_range_years = [time_range[i] / 365 for i in range(N_days)]
    Q_w = 2
    h_s0 = 0
    h_i0 = 0.1

    ## Ice thickness evolution plot ##

    plt.plot(time_range_years, h_ice, label="h_ice")
    plt.title(
        "TSIM Model with snow\n"
        + r"$dyn_alb =$ {}, $Q_W = {}W/m^2$, $h_i(t=0) = {}m$, $h_s(t=0) = {}m, T = {}$ years, $\gamma$ = {}".format(
            dyn_alb, Q_w, h_i0, h_s0, N_years, gamma_SM
        ),
        size=22,
    )
    plt.xlabel("Year", size=20)
    plt.ylabel("Ice Thickness [m]", size=20)
    plt.legend(fontsize=18)
    plt.grid()
    # plt.savefig(save_dir + "no_tuning.png", dpi=300)
    # plt.show()
    plt.clf()

    ## Last year ice thickness evolution subplot comparaison with MU71 ##

    fig, axs = plt.subplots(1, 2)
    fig.suptitle(
        "TSIMAL Last year ice thickness evolution comparaison with MU71\n"
        + r"$dyn_alb =$ {}, $Q_W = {}W/m^2$, $h_i(t=0) = {}m$, $h_s(t=0) = {}m, T = {}$ years, $\gamma$ = {}".format(
            dyn_alb, Q_w, h_i0, h_s0, N_years, gamma_SM
        )
    )
    # Storing only the n (starting from the end of the sim) year value of the sea ice thickness #
    # n = 0 means we keep the last year values, n = T means we keep the first year values
    n = 0
    if n == 0:
        hi_year_n = np.zeros(365)
        hi_year_n = h_ice[-365:]
    if n != 0:
        hi_year_n = np.zeros(365)
        hi_year_n = h_ice[-365 * (n + 1) : -365 * n]

    hi_month_mean_year_n = month_mean_v2(hi_year_n)

    # Plotting #

    axs[0].plot(np.arange(1, 13), hi_month_mean_year_n, label=r"$hi_{TSIMAL}$")
    axs[0].set_title("TSIMAL Model")
    axs[0].set_xlabel("Month")
    axs[0].set_ylabel("Thickness [m]")
    axs[0].set_xticks([2, 4, 6, 8, 10, 12])
    axs[0].grid()

    axs[1].plot(np.arange(1, 13), hi_MU71, label=r"$hi_{M71}$")
    axs[1].set_title("Maykut and Untersteiner (1971)")
    axs[1].set_xlabel("Month")
    axs[1].set_ylabel("Thickness [m]")
    axs[1].set_xticks([2, 4, 6, 8, 10, 12])
    axs[1].grid()

    fig.tight_layout()
    # plt.savefig(save_dir + "no_tuning_comp.png", dpi=300)
    # plt.show()
    plt.clf()

    ## Same plot ##

    plt.plot(np.arange(1, 13), hi_month_mean_year_n, label=r"$hi_{TSIMAL}$")
    plt.plot(np.arange(1, 13), hi_MU71, label=r"$hi_{M71}$")
    plt.title(
        "TSIMAL year {} ice thickness evolution comparaison with MU71\n".format(
            N_years - n
        )
        + r"dyn_alb = {}, $Q_W = {}W/m^2$, $\gamma$ = {}, $\beta$ = {}".format(
            dyn_alb, Q_w, gamma_SM, beta_SM
        )
        + "\n"
        + r"$h_i(t=0) = {}m$, $h_s(t=0) = {}m, T = {}$ years".format(
            h_i0, h_s0, N_years
        ),
        size=11,
    )
    plt.xlabel("Month", size=10)
    plt.ylabel("Ice Thickness [m]", size=10)
    plt.legend(fontsize=8)
    plt.grid()
    plt.savefig(save_dir + "tuning_comp.png", dpi=300)
    fig.tight_layout()
    # plt.show()
    plt.clf()

    ### Computation of the error on annual mean thickness ###
    mean_TSIMAL, mean_mu71, err_abs, err_rel = err_annual_mean_thick(
        hi_month_mean_year_n, hi_MU71
    )
    ### Computation of the standard deviation with regard to annual mean thickness ###
    std_TSIMAL = std_var_mean_thick(hi_month_mean_year_n)
    std_MU71 = std_var_mean_thick(hi_MU71)
    ### Computation of the MSE on annual mean thickness ###
    mse = MSE_annual_mean_thick(hi_month_mean_year_n, hi_MU71)
    ### Computation of the MSE on annual mean thickness ###
    r = cor_annual_mean_thick(hi_month_mean_year_n, hi_MU71)

    ### Output ###
    print("--------------------TSIMAL & MU71 Comparison----------------------")
    print("------------------------------------------------------------------")
    print(
        "dyn_alb = {}, Q_W = {}W/m^2, gamma = {}, beta = {}".format(
            dyn_alb, Q_w, gamma_SM, beta_SM
        )
    )
    print("h_i(t=0) = {}m, h_s(t=0) = {}m, T = {} years".format(h_i0, h_s0, N_years))
    print(
        "alb_s = {}, alb_i = {}, epsilon = {}, i_0 = {}".format(
            alb_dry_snow, alb_bare_ice, epsilon, i_0
        )
    )
    print("------------------------------------------------------------------")

    print(
        "Year {} mean ice thickness TSIMAL = {:.3f}m".format(N_years - n, mean_TSIMAL)
    )
    print("Mean ice thickness MU71 = {:.3f}m".format(mean_mu71))
    print("Absolute Error = {:.4f}m".format(err_abs))
    print("Relative Error = {:.2f}%".format(err_rel))
    print("Standard deviation TSIMAL = {:.3f}".format(std_TSIMAL))
    print("Standard deviation MU71 = {:.3f}".format(std_MU71))
    print("MSE(TSIMAL,MU71) = {:.3f}".format(mse))
    print("r(TSIMAL,MU71) = {:.3f}".format(r))
    print("------------------------------------------------------------------")


if __name__ == "__main__":
    # first_and_mult_ice()
    # ctrl_sim_without_snow()
    ctrl_sim()
    # tuning_comp()
