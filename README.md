# TSIM - Thermodynamic Sea Ice Model
-------

GitHub made for the Project : "**Built your own thermodynamics sea ice model**" of the LPHYS2265 class. 


![PR12_TSIMAL](https://user-images.githubusercontent.com/58213378/232316931-3889bce8-9fc1-44dc-a8fe-e78780ae3d41.png)


## Organization of the files 

### Instructions and Report

The pdf containing the instructions for this project are the *Exercise_part_1.pdf, Exercise_part_2.pdf files*. The *Semtner_JPO76_icemodel.pdf* is the paper of Semtner and al (1976) usefull for this model of sea ice. Thoses files can be found in the *Instructions* folder. **A more detailed presentation of the three models (*IGCT1.py, FST2.py, TSIM.py*) is given in Projet Report/TSIM Presentation**. 

The models and the files are organised following the structure of the project which has three different phases.
1. Building the TSIM model
2. Tuning the TSIM model and running Projections 
3. Ensemble analysis and data analysis

### Numerical models 

1. The *IGCT1.py* script correspond to the **Ice Growth with Constant Temperature** model related to section 1 of *Exercise_part_1.pdf*.
2. The *FST2.py* script correspond to the **Freeing Surface Temperature** model related to section 2 of *Exercise_part_1.pdf*.
3. The *TSIM.py* script correspond to the **Thermodynamic Sea Ice Model** model related to section 3 of *Exercise_part_1.pdf*.

**The *TSIM.py* model is the complete model considering a free surface temperature, an ocean mixed layer and snow.**

**TSIMAL stands for Thermodynamics sea ice model of Amaury Laridon** in order to distinguish more easily all the thermodynamics sea ice models of the LPHYS2265 class.

4. **The *CONTROL_TSIMAL.py* is the TSIM model with additional features as a more elaborate thermal conductivity and albedo** (following *Exercise_part_2.pdf*) that are **used to tune TSIMAL in order to fit the more precisely as possible the *Maykut and Untersteiner-1971 (MU71)* data set** of typical seasonal cycle of ice thickness of Arctic perennial ice before the recent effects of climate change. Some diagnostics tools for comparing the MU71 data set and the TSIMAL output data set are implemented as a mean comparison, a Mean Squared Error (MSE) calculator and a correlation coeficient computation. **The *CONTROL_TSIMAL.py* is then regarded as the control simulation and best tuned values of TSIMAL's parameters.**

5. **The *PROJECTION_TSIMAL.py* is the *CONTROL_TSIMAL* model but used for sea-ice projections** where we progressively increase downwelling longwave radiation over the upcoming 100 years, **to emulate the effect of the increasing greenhouse effect**. Several projections has been made for different values of forcing at the end of the century. 

All the python scripts are written in the most readable way possible with comments. The use of functions has also been maximized in order to be able to call parts of the code more easily.

### Output Data

The most important data set output of TSIMAL are in the *TSIM/Data/ folder.*  
- The full ouptut of *CTL_TSIMAL* run is available at *CTL_TSIMAL.txt*. The columns are organised as follows : (days, Surface temperature [°K], Ice Thickness [m], Snow Thickness [m], Mix layered temperature [°K]). 
- All the *PRX_TSIMAL_full_data.txt* files are the full output (with the same columns structured as defined above) of several projections where "X" stands for the radiative forcing at the end of the century imposed. 
- The *PRX_TSIMAL.txt* files are the output of some yearly diagnostics interesting features of the simulations for the different scenarios. There the columns are organised as (year, Ice thickness minima [m], Ice thickness mean [m], Ice thickness maxima [m], Snow thickness maxima [m], Mix Layered temperature minima [°K]).

### Organization of the figures 

In the */Figures* folder the figures of the differents simulations are store first following the different part of the Project. 
The */Control Simulations Figures* contains all the standards plots made from the complete *TSIM.py* model in order to define first control simulations at the end of Part 1. An additionnal distinction in the folder is made to distinguish simulations with or without snow. Some useful abbreviations for indexing figures have been made. 

- *fig_name.png* are control simulation's figure.
- *fig_name_2sf.png* stands for *Double snowfall*.
- *fig_name_T_x.png* stands for *Period of simulation egals to x years*.
- *fig_name_sif_True.png* stands for *Sea ice formation option activated*.
- *fig_name_2sf.png* stands for *Double snowfall*.
- *fig_name_v1.png* stands for *first version of TSIM* not considering the possibility of having a total melt of the snow layer and a melting of a part of the ice layer.
- *fig_name_v2.png* stands for *second version of TSIM* and takes into account the previous consideration.
- *fig_name_tl_false.png* stands for *Temperatue limit = False*. It was a test of what is going on if we don't limite the surface temperature to be 0°C at max. 
- *fig_name_Q_W_x.png* stands for *Q_w = x W/m²*.
- *fig_name_alb_x.png* stands for *surface_albedo = x*.

The */Tests Figures* folder contains all the intermediate tests made in order to evaluate the model. Thoses figures comes from respectively *IGCT1.py, FST2.py* and *TSIM.py*

The figures of Part 2 have a dedicated folder with first figures associated to the "non tuned case" and then the "tuned case". Subfolders in there are related to the part of the model that has been used for tuning and this can be easily understood following *Exercise_part_2.pdf files*. *ctrl_sim* folder obviously contained the figures of the control model simulation whith the maximazed tuning from the different parameters. Lastly the *Projection* folder has the images related to all the projections that has been done using TSIMAL final and optimzed version. 

## How to run ? 

Simply clone this repository and execute the python script of the model you want to use. All the global parameters of the simulation are placed at the beginning of the script in order to facilitate the user. The execution of some simulations or tests can be done at the instancing part at the end of the script. The user simply have to comment or uncomment what he wants to run. 

It may be possible that the *CONTROL_TSIMAL.py* script requires an installation of the *sklearn* package. To do so simply execute in a console `pip install scikit-learn`. 

