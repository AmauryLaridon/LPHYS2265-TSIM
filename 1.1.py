import matplotlib.pyplot as plt
import numpy as np

####### - Physical constant - ##########

L_fus = 3.35e5 # Latent heat  of fusion for water [J/kg]
rho_i = 917 #Sea ice density [kg/m^3]
rho_w = 1025 #water density [kg/m^3]
rho_s = 330 #Snow density [kg/m^3]
k_i = 2.2 #Sea ice thermal conductivity [W/m/K]
k_s = 0.31 #Snow thermal conductivity [W/m/K]
sec_per_day = 86400 #Second in one day [s/d]
epsilon = 0.99 #Surface emissivity
sigma = 5.67e-8 #Stefan-Boltzmann constant [W/m^2/K^4]
kelvin = 273.15 #Conversion K <-> C
alb_i = 0.8 #albedo of ice
c_w = 4000 #heat capacity of water [J/kg/K]
alb_w = 0.1 #albedo of water 

########## - Setting simulation parameter - ###########
T_bott = -1.8 + kelvin #Bottom temerature [Celcius]
T_su = -10 + kelvin #Air temperature [Celcius]
h_i0 = 0.1#Thickness of the ice layer in day one [m]
Q_w = 5 #heat flux coming from the ocean (upward positive) [W/m^2]
h_s0 = 0 #Thickness of the snow layer in day one [m]
h_w = 50 #depth of the mixed layer of water [m]
M_w = rho_w*h_w # masse of water in the mixed layer [kg/m^2]

########## - Useful functions - ########

def Q_sol(day):
    return 314 * np.exp((-(day-164)**2)/4608)

def Q_nsol(day):
    return 118 * np.exp((-0.5 * (day - 206)**2)/(53**2)) + 179

def T_surface(ice_thick, snow_thick, T_bott, day):
        """ Compute the surface temperature. This temperature equals the blackbody temperature such as the radiative equilibrium
        is achieve with the incoming solar and non solar flux and the flux coming from water through the ice. However, a surface
        temperature bigger than 273.15K is impossible, in this case the ice will melt (this is why we use the min() function)
        [K]
        """
        #print(np.roots([-epsilon * sigma,0,0,-k_i/ice_thick, k_i/ice_thick * T_bott + Q_sol(day)* (1-alb_i) + Q_nsol(day)]).real)
        k_eff = (k_i*k_s)/(k_i * snow_thick + k_s * ice_thick) # [W/m^2/K]
        return min([273.15, np.roots([-epsilon * sigma,0,0,-k_eff, k_eff * T_bott + Q_sol(day)* (1-alb_i) + Q_nsol(day)]).real[3]])

def Q_c(ice_thick, snow_thick, T_su, T_bott): 
    """Compute the sensible heat flux through the ice and/or snow (upward positive)
    [W/m^2]
    """ 
    k_eff = (k_i*k_s)/(k_i * snow_thick + k_s * ice_thick) # [W/m^2/K]
    return (T_bott - T_su) * k_eff # [W/m^2]

def E_gain_surf(day, T_su, T_bott, ice_thick):
    """ Compute the energy gain at the surface of the sea ice in one day due to a non-equilibrium between the solar and non-solar
    flux incoming, the flux coming from the water through the ice and the ice blackbody radiation. This disequilibrium
    is because the ice can't go over 273.15K without melting. This energy will hence be used to melt the ice during summer.
    [J/m^2]
    """
    return max([0,Q_sol(day%365)*(1-alb_i) + Q_nsol(day%365) - sigma * epsilon * T_su**4 - k_i*(T_su - T_bott)/ice_thick]) * sec_per_day

def E_loss_bottom(ice_thick, T_su,T_bott, snow_thick = 0):
    """ Compute the total energy loss in one day at the bottom of the sea ice layer
        [J/m^2]
    """
    E_loss_through_ice = Q_c(ice_thick, snow_thick, T_su, T_bott) * sec_per_day
    E_gain_by_ocean_flux = Q_w * sec_per_day
    return  E_loss_through_ice - E_gain_by_ocean_flux

def E_gain_mixed_layer(T_w, day):
    """ Compute the free water energy gain in one day.
    Incoming:
    Q_sol: solar flux
    Q_nsol : non solar flux
    Q_w: flux from deep water

    Outgoing: Blackbody radiation with temperature T_w
    [J/m^2]
    """
    return (Q_sol(day) * (1-alb_w) + Q_nsol(day) + Q_w - epsilon * sigma * T_w**4) * sec_per_day

"""def T_surface_free_water(day, T_w):
    print(max(np.roots([-epsilon * sigma,0,0,0,Q_sol(day)*(1-alb_w) + Q_nsol(day)]).real))
    return max(np.roots([-epsilon * sigma,0,0,0,Q_sol(day)*(1-alb_w) + Q_nsol(day)]).real)
"""

def snow_fall(day):
    """ 30 cm between 20 August and October
        5 cm between November and april
        5 cm in May
    """
    day = day%365
    if day >= 232 and day <= 304: #between 20 August and October (included)
        return 0.3/(304-232)
    elif day >= 305 or day <= 120: #between November and April (included)
        return 0.05/(365-305+120)
    elif day >= 121 and day <= 151: #May
        return 0.05/(151-121)
    else:
        return 0
    

######### - Sea ice evolution function - ##########
""" SI_evol_simple is for the exercice until 1.3
    SI_evol_sun_variation is for exercice until 2.3.2
    SI_evol_ice_free_cond for exercice 3.1
    SI_final for exercice 3.2
"""

def SI_evol_simple(integration_range, T_su = T_su, T_bott = T_bott, h_i0 = h_i0):
    #Return two arrays: h_i an array of sea_ice thickness evolution every days and time_range representing the days for poltting.
    #integration range is in days.
    h_i = np.zeros(integration_range) #Array of ice thickness, each column stands for a day.
    h_i[0] = h_i0 #Thickness of the ice layer in day one [m]
    time_range = range(0,integration_range)
    for day in range(1,integration_range):
        E_loss = E_loss_bottom(h_i[day-1], T_su,T_bott, snow_thick=h_s0) # Energy lost at the bottom during one day due to flux from water to ice.[J/m^2]
        Freezing_water_mass = E_loss/L_fus #Mass of water freezed at the bottom of the ice layer at the end of one day [kg/m^2]
        h_i[day] = h_i[day-1] + Freezing_water_mass/rho_i # and we see we obtain [m] as needed
    return h_i, time_range

def SI_evol_sun_variation(integration_range = 365, T_bott = T_bott, h_i0 = h_i0):
    """ Return two arrays: h_i an array of sea_ice thickness evolution every days and time_range representing the days for poltting.
    integration range is in days.
    """
    ######### - Initialization - #########
    h_i = np.zeros(integration_range) #Array of ice thickness, each column stands for a day.
    h_i[0] = h_i0 #Thickness of the ice layer in day one [m]
    time_range = range(0,integration_range)

    ######### - Simulation - ##########
    for day in range(1,integration_range):
        #Computing surface temperature in regard of ice thickness, bottom temperature and day of the year.
        T_su = T_surface(h_i[day-1], 0,T_bott, day %365) #[K]
        
        #Computing Energy change at the bottom
        E_loss = E_loss_bottom(h_i[day-1],T_su,T_bott) #[J/m^2]
        Freezing_water_mass = E_loss/L_fus #Mass of water freezed at the bottom of the ice layer at the end of one day [kg/m^2]

        #Computing energy gain at the top for metling
        E_gain = E_gain_surf(day%365,T_su,T_bott,h_i[day-1]) #[J/m^2]
        melt_ice_mass = E_gain/L_fus #[kg/m^2]

        #Computing of the result SI at the next day
        h_i[day] = h_i[day-1] +Freezing_water_mass/rho_i - melt_ice_mass/rho_i # and we see we obtain [m] as needed

    return h_i, time_range

def SI_evol_ice_free_cond(integration_range = 365, T_bott = T_bott, h_i0 = h_i0):
    """ Return:
        - h_i:              an array with ice thickness for each day
        - time_range:       an array with all the days for plotting
        - T_mixed layer:    an array with the temperature of the mixed water layer for eaxh days
        - T_surface_array:   an array with the surface temperature for each days

    This function allows to reach an ice free condition by considering a mixed water layer which will warm when there is no more ice.
    """
    ######### - Initialization - #########

    h_i = np.zeros(integration_range) #Array of ice thickness, each column stands for a day.
    h_i[0] = h_i0 #Thickness of the ice layer in day one [m]
    time_range = range(0,integration_range)
    T_w = T_bott #At the beggining, the mixed layer temperature is equal to the sea ice bottom temperature [K]
    T_su = T_surface(h_i0, 0,T_bott, day = 0) #initialized surface temperature in regard of the ice thickness, incoming Energy and bottom temp.
    T_mixed_layer = [kelvin - 1.8]
    T_surface_array= [T_su]

    ######### - Simulation - ##########

    for day in range(1,integration_range):
        
        if h_i[day-1] > 0:
            """ In this case the water is covered by sea ice and we perform the same calculation as precedents points.
            """
            T_su = T_surface(h_i[day-1], 0,T_bott, day %365) #[K]
            #Computing Energy change at the bottom
            E_loss = E_loss_bottom(h_i[day-1],T_su,T_bott) #[J/m^2]
            Freezing_water_mass = E_loss/L_fus #Mass of water freezed at the bottom of the ice layer at the end of one day [kg/m^2]

            #Computing energy gain at the top for metling
            E_gain = E_gain_surf(day%365,T_su,T_bott,h_i[day-1]) #[J/m^2]
            melt_ice_mass = E_gain/L_fus #[kg/m^2]

            #Computing of the result SI at the next day
            h_i[day] = h_i[day-1] + Freezing_water_mass/rho_i - melt_ice_mass/rho_i # and we see we obtain [m] as needed
            T_mixed_layer.append(kelvin - 1.8)

            #Computing surface temperature in regard of ice thickness, bottom temperature and day of the year.
            T_surface_array.append(T_su)

        else:
            """ In this case there is no more ice and the energy disequilibrium will warm or cool the mixed layer
            """
            h_i[day-1] = 0 #Set sea_ice to zero (in the case it became negative)

            if T_w >= T_bott:
                """ In this case the water can warm or cool without producing SI
                """
                E_gain = E_gain_mixed_layer(T_w, day%365) # Energy gain by the mixed layer in one day [J/m^2]
                
                T_w += E_gain/(M_w * c_w) #New mixed layer temperature [K]
                T_mixed_layer.append(T_w)
                T_surface_array.append(T_su)
            else:
                """In this case the water is cooling below the freezing point so we re-create ice
                """
                Heat_excess = T_bott - T_w
                E_gain = Heat_excess * M_w * c_w # Excess of heat which will be turn into ice [J/m^2]
                Freezing_water_mass = E_gain/L_fus #[kg/m^2]
                h_i[day] = Freezing_water_mass/rho_i #[m]
                
                T_w = T_bott #We set the bottom temperature [K]
                T_mixed_layer.append(T_w)
                T_surface_array.append(T_su)

    return h_i, time_range, T_mixed_layer, T_surface_array

def SI_evol_final(integration_range, T_bott = T_bott, h_i0 = h_i0, h_s0 = h_s0):
    """ Return:
        - h_i:              an array with ice thickness for each day
        - h_s:              an array with snow thickness for each day
        - time_range:       an array with all the days for plotting
        - T_mixed layer:    an array with the temperature of the mixed water layer for eaxh days
        - T_surface_array:   an array with the surface temperature for each days

    This function allows to reach an ice free condition by considering a mixed water layer which will warm when there is no more ice.
    This function simulate the deposite of snow thanks to the function snow_fall() and, at the top, melt the snow before the ice.
    
    """
    ######### - Initialization - #########

    h_i = np.zeros(integration_range) #Array of ice thickness, each column stands for a day.
    h_s = np.zeros(integration_range) #Array of snow thickness, each column stands for a day.
    h_i[0] = h_i0 #Thickness of the ice layer in day one [m]
    h_s[0] = h_s0 #Thickness of the ice layer in day one [m]
    time_range = range(0,integration_range)

    T_w = T_bott #At the beggining, the mixed layer temperature is equal to the sea ice bottom temperature [K]
    T_su = T_surface(h_i0, h_s0,T_bott, day = 0) #initialized surface temperature in regard of the ice thickness, incoming Energy and bottom temp.
    T_mixed_layer = [kelvin - 1.8]
    T_surface_array= [T_su]

     ######### - Simulation - ##########

    for day in range(1,integration_range):
        
        if h_i[day-1] > 0:
            """ In this case the water is covered by sea ice and snow. During melting we first have to melt snow before ice.
            """
            #snow_fall
            h_s[day] = h_s[day-1] + snow_fall(day)

            #Computing surface temperature
            T_su = T_surface(h_i[day-1], h_s[day-1],T_bott, day %365) #[K]

            
            #Computing Energy change at the bottom
            E_loss = E_loss_bottom(h_i[day-1],T_su,T_bott, snow_thick=h_s[day]) #[J/m^2]
            Freezing_water_mass = E_loss/L_fus #Mass of water freezed at the bottom of the ice layer at the end of one day [kg/m^2]

            #Computing energy gain at the top for metling and what's going to melt between ice or snow (or in which proportion)
            E_gain = E_gain_surf(day%365,T_su,T_bott,h_i[day-1]) #[J/m^2]
            if h_s[day-1] > 0: #If there is snow on ice, snow will melt first.
                snow_mass_on_ice = rho_s*h_s[day-1] #snow available for melting [kg/m^2]
                melt_snow_mass = E_gain/L_fus #[kg/m^2]
                if snow_mass_on_ice > melt_snow_mass:
                    h_s[day] = h_s[day - 1] - melt_snow_mass/rho_s
                    melt_ice_mass = 0
                else: 
                    h_s[day] = 0
                    melt_ice_mass = melt_snow_mass - snow_mass_on_ice
            else:
                melt_ice_mass = E_gain/L_fus
            #Computing the resulting ice and snow thickness at the next day.

            h_i[day] = h_i[day-1] + Freezing_water_mass/rho_i - melt_ice_mass/rho_i # and we see we obtain [m] as needed
            T_mixed_layer.append(kelvin - 1.8)

            #Computing surface temperature in regard of ice thickness, bottom temperature and day of the year.
            T_surface_array.append(T_su)

        else:
            """ In this case there is no more ice and the energy disequilibrium will warm or cool the mixed layer
            """
            h_i[day-1] = 0 #Set sea_ice to zero (in the case it became negative)
            h_s[day-1] = 0

            if T_w >= T_bott:
                """ In this case the water can warm or cool without producing SI
                """
                E_gain = E_gain_mixed_layer(T_w, day%365) # Energy gain by the mixed layer in one day [J/m^2]
                
                T_w += E_gain/(M_w * c_w) #New mixed layer temperature [K]
                T_mixed_layer.append(T_w)
                T_surface_array.append(T_su)
            else:
                """In this case the water is cooling below the freezing point so we re-create ice
                """
                Heat_excess = T_bott - T_w
                E_gain = Heat_excess * M_w * c_w # Excess of heat which will be turn into ice [J/m^2]
                Freezing_water_mass = E_gain/L_fus #[kg/m^2]
                h_i[day] = Freezing_water_mass/rho_i #[m]

                if h_i[day] < 0.1: #To avoid a bug due to a too large time step
                    h_i[day] = 0.1
                T_w = T_bott #We set the bottom temperature [K]
                T_mixed_layer.append(T_w)
                T_surface_array.append(T_su)
    return h_i, h_s, time_range, T_mixed_layer, T_surface_array


########## - Main - ##########

"""h_i, time_range, T_mixed,T_su = SI_evol_ice_free_cond(integration_range=365*10)

plt.plot(time_range,h_i)
plt.show()
"""
h_i, h_s, time_range, T_mixed,T_su = SI_evol_final(integration_range=365*4)
time_range = [day / 365 for day in time_range]
fig, axs = plt.subplots(2,2, figsize = (15,10))
axs[0,0].plot(time_range,h_i)
axs[0,1].plot(time_range,h_s)
axs[1,0].plot(time_range,T_mixed)
axs[1,1].plot(time_range,T_su)

axs[0,0].title.set_text('SIT (m)')
axs[0,1].title.set_text('Snow_thickness (m)')
axs[1,0].title.set_text('Mixed layer temperature (K)')
axs[1,1].title.set_text('T_su (K)')

axs[0,0].grid()
axs[1,0].grid()
axs[0,1].grid()
axs[1,1].grid()

axs[0,0].set_xlabel('year')
axs[1,0].set_xlabel('year')
axs[0,1].set_xlabel('year')
axs[1,1].set_xlabel('year')
fig.suptitle('albedo = {} F_w = {}'. format(alb_i,Q_w))
plt.show()