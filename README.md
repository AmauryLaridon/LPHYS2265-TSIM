# TSIM - Thermodynamic Sea Ice Model
-------

GitHub made for the Project : "**Built your own thermodynamics sea ice model**" of the LPHYS2265 class. 

### Organization of the files 

The pdf containing the instructions for this project are the *Exercise_part_1.pdf, Exercise_part_2.pdf files*. The *Semtner_JPO76_icemodel.pdf* is the paper of Semtner and al (1976) usefull for this model of sea ice. 

The *IGCT1.py* script correspond to the **Ice Growth with Constant Temperature** model related to section 1 of *Exercise_part_1.pdf*
The *FST2.py* script correspond to the **Freeing Surface Temperature** model related to section 2 of *Exercise_part_1.pdf*
The *TSIM.py* script correspond to the **Thermodynamic Sea Ice Model** model related to section 3 of *Exercise_part_1.pdf*

The *TSIM.py* model is the complete model considering a free surface temperature, an ocean mixed layer and snow. 

All the python scripts are written in the most readable way possible with comments. The use of functions has also been maximized in order to be able to call parts of the code more easily.

### Organization of the figures 

In the */Figures* folder the figures of the differents simulations are store. 
The */Control Simulations Figures* contains all the standards plots made from the complete *TSIM.py* model in order to define control simulations. An additionnal distinction in the folder is made to distinguish simulations with or without snow.

The */Tests Figures* folder contains all the intermediate tests made in order to evaluate the model. Thoses figures comes from respectively *IGCT1.py, FST2.py* and *TSIM.py*
