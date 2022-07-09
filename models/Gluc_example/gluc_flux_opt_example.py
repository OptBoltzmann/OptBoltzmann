




import numpy as np
import pandas as pd
import subprocess
import sys
import re
import os
import warnings


import pickle



from scipy.optimize import least_squares
from IPython.core.display import display
from IPython.core.debugger import set_trace
pd.set_option('display.max_columns', None,'display.max_rows', None)

warnings.filterwarnings('ignore')

from importlib import reload



'''

Import Modules for optimizaion

'''

import flux_opt as Fopt




'''

########################################

Load in a model

###########################################

'''


model_file = 'gluconeogenesis_prot4.pkl'

#model_file = 'gluconeogenesis_DF.pkl'


with open(model_file, 'rb') as handle:
    model = pickle.load(handle)





Stoich_matrix = pd.DataFrame(model['S'])
active_reactions = pd.DataFrame(model['active_reactions'])
metabolites = pd.DataFrame(model['metabolites'])







'''

########################################

Specify the inputs for the optimization problem

###########################################

'''

'''

Some general notes on the expected data types and shapes
#########################

Stochiometric matrix: 
    - A matrix with dimension (number of reactions x number of metabolites )
    - Both forward and reverse reactions are captured with just one reaction row 
    - It is assumed metabolite columns are ordered such that all variable metabolites come first then all fixed metabolites are at the end


Metabolite upper bounds:
    - A vector of length equal to the number of variable metabolites that gives the upper bound for each
    - Note zero is the assumed lower bound for each metabolite


Fixed Metabolite log counts:
    - A vector of length equal to the number of fixed metabolites
    - Gives the value at which fixed metabolites will be set for the optimization

K :
    - The equilibrium constants for each reaction
    - A vector with length equal to number of reactions


Objective Coefficients:
    - A vector with length equal to the number of reactions giving the coefficient in the objective
    - assumption is objective is of the form <c,y> where c are the coefficients, y are fluxes, and <_,_> is the standard inner product

'''







'''
The Stochiometric matrix

'''

S = Stoich_matrix.values







'''
The fixed metabolite concentrations
'''


T = 298.15
R = 8.314e-03
RT = R*T
N_avogadro = 6.022140857e+23
VolCell = 1.0e-15
Concentration2Count = N_avogadro * VolCell
concentration_increment = 1/(N_avogadro*VolCell)


#number of variable metabolites
nvar = len(metabolites[ metabolites['Variable']==True].values)


fixed_concs = np.array(metabolites['Conc'].iloc[nvar:].values, dtype=np.float64)
fixed_counts = fixed_concs*Concentration2Count
f_log_counts = np.log(fixed_counts)







'''
The Variable Metabolite Upper bounds
'''

vcount_upper_bound = np.log(np.ones(nvar) *1.0e-03*Concentration2Count)





'''
Equilibrium Constants
'''


Keq = model['Keq']






'''
Objective Coefficients
'''


# Initial choice here as a single reaction we want max flux through
react_for_maxflux = 2


n_REACT = S.shape[0]
obj_coefs = np.zeros(n_REACT)
obj_coefs[react_for_maxflux] = 1






'''

Solve the Optimization problem with general method

'''


y_sol, alpha_sol, n_sol = Fopt.flux_ent_opt(obj_coefs,vcount_upper_bound,f_log_counts,S,Keq)

















