



import numpy as np
import pandas as pd
import random

import numpy.random as nprd
from scipy.linalg import norm
from scipy.optimize import least_squares
import scipy.linalg as spL

import pickle


def derivatives(vcounts,fcounts,mu0,S, R, P, delta, Keq, E_Regulation):
  nvar = vcounts.size
  metabolites = np.append(vcounts,fcounts)
  KQ_f = odds(metabolites,mu0,S, R, P, delta, Keq, 1);
  Keq_inverse = np.power(Keq,-1);
  KQ_r = odds(metabolites,mu0,-S, P, R, delta, Keq_inverse, -1);
  deriv = S.T.dot((E_Regulation *(KQ_f - KQ_r)).T);
  #deriv = S.T.dot(((KQ_f - KQ_r)).T);
  deriv = deriv[0:nvar]
  return(deriv.reshape(deriv.size,))






def odds(log_counts,mu0,S, R, P, delta, K, direction = 1):
  counts = np.exp(log_counts)
  delta_counts = counts+delta;
  log_delta = np.log(delta_counts);
  Q_inv = np.exp(-direction*(R.dot(log_counts) + P.dot(log_delta)))
  KQ = np.multiply(K,Q_inv);
  return(KQ)







def oddsDiff(vcounts,fcounts,mu0,S, R, P, delta,Keq,E_Regulation):
  metabolites = np.append(vcounts,fcounts)
  KQ_f = odds(metabolites,mu0,S, R, P, delta,Keq);
  Keq_inverse = np.power(Keq,-1);
  KQ_r = odds(metabolites,mu0,-S, P, R, delta,Keq_inverse,-1);
  
  #WARNING: Multiply regulation here, not on individual Keq values.
  KQdiff =  E_Regulation * (KQ_f - KQ_r);
  return(KQdiff)





def calc_delta_S(vcounts, fcounts, P):
    #WARNING To avoid negative numbers do not use log concs
    metab = np.append(vcounts, fcounts)
    target_metab = np.ones(vcounts.size) * np.log(0.001*Concentration2Count);
    target_metab = np.append(target_metab, fcounts)
    delta_S =  P.dot(metab) - P.dot(target_metab)
    return(delta_S)





def calc_E_step(E, vcounts, fcounts):
    newE = E -E/2
    return(newE)




def reduced_Stoich(S,f_log_counts,target_log_vcounts):

  # Flip Stoichiometric Matrix
  S_T = S    # S_T is the Stoich matrix with rows as reactions, columns as metabolites
  S = np.transpose(S_T) # this now is the Stoich matrix with rows metabolites, and columns reactions

  #Set the System parameters 
  VarM = target_log_vcounts
  FxdM = np.reshape(f_log_counts,(len(f_log_counts),1) )
  n_M = len(VarM) + len(FxdM)

  #Metabolite parms
  n_M_f = len(f_log_counts)   #total number of fixed metabolites
  n_M_v = len(target_log_vcounts)   #total number of variable metabolites
  n_M = n_M_f + n_M_v   # total number of metabolites

  VarM_idx = np.arange(0,n_M_v) #variable metabolite indices
  FxdM_idx = np.arange(n_M_v,n_M) # fixed metabolite indices

  # Split S into the component corresponding to the variable metabolites S_v 
  # and the component corresponding to the fixed metabolites S_f
  S_v_T = np.delete(S_T,FxdM_idx,1)
  S_v = np.transpose(S_v_T)

  S_f_T = np.delete(S_T,VarM_idx,1)
  S_f = np.transpose(S_f_T)

  #find a basis for the nullspace of S_v
  Sv_N = spL.null_space(S_v)

  return S_v, S_f, Sv_N




def load_model(model_file):

  with open(model_file, 'rb') as handle:
      model = pickle.load(handle)

  
  S_active = pd.DataFrame(model['S'])
  active_reactions = pd.DataFrame(model['active_reactions'])
  metabolites = pd.DataFrame(model['metabolites'])
  Keq = model['Keq']

  return S_active, active_reactions, metabolites, Keq




def numpy_model_preprocessing(model_file):

  with open(model_file, 'rb') as handle:
      model = pickle.load(handle)

  
  S_active = pd.DataFrame(model['S'])
  active_reactions = pd.DataFrame(model['active_reactions'])
  metabolites = pd.DataFrame(model['metabolites'])
  Keq = model['Keq']


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

  S = S_active.values

  return S, Keq, f_log_counts, vcount_upper_bound 




def save_model_solution(save_file,y_sol,alpha_sol,n_sol,S,Keq,f_log_counts,vcount_upper_bound,obj_coefs):
  ### Save a dictionary of the solution for later analysis (note this also saves the associated model file for future reference)

  solution_data = {}

  #Solution values
  solution_data['y'] = y_sol
  solution_data['alpha'] = alpha_sol
  solution_data['n'] = n_sol


  #Optimization parameters
  solution_data['f_log_counts'] = f_log_counts
  solution_data['vcount_upper_bound'] = vcount_upper_bound
  solution_data['obj_coefs'] = obj_coefs

  #Model files
  solution_data['S'] = S
  solution_data['Keq'] = Keq



  with open(save_file, 'wb') as handle:
      pickle.dump(solution_data, handle, protocol=4)




def load_model_solution(save_file):

  with open(save_file, 'rb') as handle:
      solution_data = pickle.load(handle)

  #Solution values
  y_sol = solution_data['y']
  alpha_sol = solution_data['alpha']
  n_sol = solution_data['n']


  #Optimization parameters
  f_log_counts = solution_data['f_log_counts']
  vcount_upper_bound = solution_data['vcount_upper_bound']
  obj_coefs = solution_data['obj_coefs']

  #Model files
  S = solution_data['S']
  Keq = solution_data['Keq']

  return y_sol,alpha_sol,n_sol,S,Keq,f_log_counts,vcount_upper_bound,obj_coefs








def flux_gradient(Sv_N,y_obj_coefs):

  n_react = len(np.ravel(y_obj_coefs))
  y_obj_coefs = np.reshape(y_obj_coefs,(1,n_react))
  beta_grad = np.matmul(y_obj_coefs,Sv_N)

  beta_grad = np.transpose(beta_grad)


  y_grad = np.matmul(Sv_N,beta_grad)

  return y_grad



