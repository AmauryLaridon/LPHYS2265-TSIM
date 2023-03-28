# TSIM - Thermodynamic Sea Ice Model
-------

GitHub made for the Project : "**Built your own thermodynamics sea ice model**" of the LPHYS2265 class. 

### Organization of the files 

The pdf containing the instructions for this project are the *Exercise_part_1.pdf, Exercise_part_2.pdf files*. The *Semtner_JPO76_icemodel.pdf* is the paper of Semtner and al (1976) usefull for this model of sea ice. Thoses files can be found in the *Instructions* folder. **A more detailed presentation of the three models is given in Projet Report/TSIM Presentation**

The *IGCT1.py* script correspond to the **Ice Growth with Constant Temperature** model related to section 1 of *Exercise_part_1.pdf*.
The *FST2.py* script correspond to the **Freeing Surface Temperature** model related to section 2 of *Exercise_part_1.pdf*.
The *TSIM.py* script correspond to the **Thermodynamic Sea Ice Model** model related to section 3 of *Exercise_part_1.pdf*.

**The *TSIM.py* model is the complete model considering a free surface temperature, an ocean mixed layer and snow.**

All the python scripts are written in the most readable way possible with comments. The use of functions has also been maximized in order to be able to call parts of the code more easily.

### How to run ? 

Simply clone this repository and execute the python script of the model you want to use. All the global parameters of the simulation are placed at the beginning of the script in order to facilitate the user. Some additionnal modifications can be done at the instancing part at the end of the script. 

### Organization of the figures 

In the */Figures* folder the figures of the differents simulations are store. 
The */Control Simulations Figures* contains all the standards plots made from the complete *TSIM.py* model in order to define control simulations. An additionnal distinction in the folder is made to distinguish simulations with or without snow. Some useful abbreviations for indexing figures have been made. 

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
