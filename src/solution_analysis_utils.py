
import numpy as np
import pandas as pd
import random

from scipy.optimize import least_squares
from scipy.optimize import linprog
import scipy.linalg as spL



'''

Methods for Analyzing Optimization Solutions





General Nomenclature


S: A stochiometric matrix for a metabolic reaction network, where rows correspond to metabolites and columns correspond to reactions
S_T: the transpose

S_v: The stochimoetric submatrix coressponding to the metabolites that can vary

S_f: The stochiometric submatrix coressponding to the fixed metabolites



Sv_N: A basis for the null space of S_v, that gives the steady state representation of the fluxes beta


'''






'''
###############################################

Some General Functions

##############################################

'''






def rxn_flux(v_log_counts, f_log_counts, S, K, E_regulation):
    # Flip Stoichiometric Matrix
    S_T = np.transpose(S)    # S_T is the Stoich matrix with rows as reactions, columns as metabolites
    v_log_counts = np.reshape(v_log_counts,(len(v_log_counts),1))
    f_log_counts = np.reshape(f_log_counts,(len(f_log_counts),1))
    tot_log_counts = np.concatenate((v_log_counts,f_log_counts))
    K = np.reshape(K,(len(K),1))
    E_regulation = np.reshape(E_regulation,(len(E_regulation),1))
    
    forward_odds = K*np.exp(- np.matmul(S_T,tot_log_counts) )
    #forward_odds = K*np.exp(-.25*np.matmul(S_T,tot_log_counts) )**4
    return E_regulation*(forward_odds - np.power(forward_odds,-1) )
    






'''
Function for relating flux values to log metabolite values
'''

def h_func(y):
    y = np.reshape(y,(len(y),1))
    return np.sign(y)*(np.log(2)*np.ones(np.shape(y)) - np.log( np.abs(y) + np.sqrt( np.power(y,2) + 4 )    ) )








'''
partials of h_func when computed with respect to steady state basis representation of the flux beta
'''


def h_beta_partial(B,y):
    Hbp = np.zeros(np.shape(B))
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            Hbp[i,j] = -B[i,j]/np.sqrt(y[j]**2 + 4) 
    return Hbp







def cns_cnstrnt_func(y,n,FxdM,K,S_v,S_f):
    y = np.reshape(y,(len(y),1))
    n = np.reshape(n,(len(n),1))
    FxdM = np.reshape(FxdM,(len(FxdM),1))
    K = np.reshape(K,(len(K),1))

    S_v_T = np.reshape(S_v)
    S_f_T = np.reshape(S_f)

    return np.sign(y)*( np.matmul(S_v_T,n) + np.matmul(S_f_T,FxdM) - np.log(K) - h_func(y) )















'''
###############################################

Identify Equivalent Flux Solutions

##############################################


These functions explore the range of metabolite and activity coefficients that can produce the same flux as a given solution




Given Soultion:

y_sol = flux values for all reactions
beta_sol = the coressponding representation in the steady state basis

n_sol = the log metabolite concentrations







N_Svr: The nullspace of the submatrix corresponding to the reactions for which the consistency constraints are active at the solution


'''




'''
Identify reactions that have neccessairly active consistency constraints at solutions

'''

def obj_restricted_rxns(beta_grad,y_sol,n_sol,FxdM,Sv_N,S_v,S_f,K_v):
    ###check all inputs are appropriate shape
    y_sol = np.reshape(y_sol,(len(y_sol),1))
    n_sol = np.reshape(n_sol,(len(n_sol),1))
    FxdM = np.reshape(FxdM,(len(FxdM),1))
    K_v = np.reshape(K_v,(len(K_v),1))



    #compute how h(y_j) will change when stepping in the gradient direction

    S_v_T = np.transpose(S_v)
    S_f_T = np.transpose(S_f)

    Hbp = h_beta_partial(Sv_N,y_sol)

    delta_hy = np.matmul(Hbp,beta_grad)

    ### compute inequality violations in response to a shift

    delta_beta = .01

    cns_cnstrnt = np.sign(y_sol)*( np.matmul(S_v_T,n_sol) + np.matmul(S_f_T,FxdM) - np.log(K_v) - h_func(y_sol) )

    beta_change = np.sign(y_sol)*(- delta_hy*delta_beta)


    shift_parm = .1

    cns_viol = np.maximum(np.sign(cns_cnstrnt + shift_parm*beta_change),0)

    return cns_viol







def free_metabolite_basis(restricted_rxns,S_v):

    restricted_rxns = np.reshape(restricted_rxns,(len(restricted_rxns),1))

    svu_row_idx = np.where(restricted_rxns[:,0] == 1)
    
    S_v_T = np.transpose(S_v)

    Svr_T = np.zeros(np.shape(S_v_T))

    Svr_T[svu_row_idx] = S_v_T[svu_row_idx]



    ### find the null space of the Svu_block
    N_Svr = spL.null_space(Svr_T)

    return N_Svr





def max_min_total_metabolite_sol(y_sol,n_sol,N_Svr,n_max,FxdM,S,S_v,S_f,K_v,sense = 'max'):


    ###check all inputs are appropriate shape
    y_sol = np.reshape(y_sol,(len(y_sol),1))
    n_sol = np.reshape(n_sol,(len(n_sol),1))
    n_max = np.reshape(n_max,(len(n_max),1))
    FxdM = np.reshape(FxdM,(len(FxdM),1))
    K_v = np.reshape(K_v,(len(K_v),1))

    S_v_T = np.transpose(S_v)
    S_f_T = np.transpose(S_f)

    if sense =='max': obj_sgn = -1
    if sense =='min': obj_sgn = 1
    #Let gamma be our null space representation of the shift to eta we will allow.

    #our objective is then to accomplish the largest possible increase/decrease to the total amount of metabolites so our objective is given by [1]^T N_Svr gamma

    obj_coefs = np.matmul(np.ones((1,N_Svr.shape[0])),N_Svr)
    obj_coefs = obj_coefs[0,:]

    ### subject to the constraint that N_Svr gamma + eta* <= eta_max

    gam_upper_bound = np.ravel(n_max) - np.ravel(n_sol)


    N_Svr_thresh = N_Svr
    N_Svr_thresh[N_Svr_thresh <= 1e-10] = 0

    gub_thresh = gam_upper_bound
    gub_thresh[gub_thresh<=1e-3] = 0



    #### and must satisfy the consistency constraints for the solution flux
    S_v_T_sgn = S_v_T*np.sign(y_sol)
    S_f_T_sgn = S_f_T*np.sign(y_sol)


    #np.sign(y)*( np.matmul(S_v_T,n) + np.matmul(S_f_T,FxdM) - np.log(K_v) - h_func(y) )

    nty_rhs_bound = np.sign(y_sol)*h_func(y_sol) + np.sign(y_sol)*np.log(K_v) - np.matmul(S_f_T_sgn,FxdM ) - np.matmul(S_v_T_sgn,n_sol)

    nty_thresh = nty_rhs_bound
    nty_thresh[nty_thresh<=1e-4] = 0

    nty_lhs_A = np.matmul(S_v_T_sgn,N_Svr_thresh)



    A_ub = np.concatenate( (N_Svr_thresh,nty_lhs_A))

    b_ub = np.concatenate( (np.reshape(gub_thresh,(len(gub_thresh),1)),nty_rhs_bound) )
    b_ub = np.ravel(b_ub)


    #solve the optimization
    lp_sol = linprog(obj_sgn*obj_coefs,A_ub = A_ub, b_ub = b_ub)

    if lp_sol.success != True:
        print('Optimization Failed')

    gam_sol = lp_sol.x
    gam_sol = np.reshape(gam_sol,(len(gam_sol),1))

    free_n_sol = np.ravel(n_sol) + np.ravel(np.matmul(N_Svr_thresh,gam_sol))


    n_vec = free_n_sol
    E_regulation = np.ones(len(y_sol))
    rxn_flux_unregulated = rxn_flux(n_vec, FxdM, S, K_v, E_regulation)

    #print(rxn_flux_unregulated)


    ### compute regulation
    free_alpha_sol = y_sol[:,0]/rxn_flux_unregulated[:,0]

    return [free_n_sol, free_alpha_sol]








def max_min_subset_metabolite_sol(metab_indexes,y_sol,n_sol,N_Svr,n_max,FxdM,S,S_v,S_f,K_v,sense = 'max'):


    ###check all inputs are appropriate shape
    y_sol = np.reshape(y_sol,(len(y_sol),1))
    n_sol = np.reshape(n_sol,(len(n_sol),1))
    n_max = np.reshape(n_max,(len(n_max),1))
    FxdM = np.reshape(FxdM,(len(FxdM),1))
    K_v = np.reshape(K_v,(len(K_v),1))

    S_v_T = np.transpose(S_v)
    S_f_T = np.transpose(S_f)

    if sense =='max': obj_sgn = -1
    if sense =='min': obj_sgn = 1
    #Let gamma be our null space representation of the shift to eta we will allow.

    #our objective is then to accomplish the largest possible increase/decrease to the total amount of metabolites so our objective is given by [1]^T N_Svr gamma

    ### Select only a subset of metabolites
    metab_selection = np.zeros( (1,len(n_sol)))
    metab_selection[0,metab_indexes] = 1

    obj_coefs = np.matmul(metab_selection,N_Svr)
    obj_coefs = obj_coefs[0,:]

    ### subject to the constraint that N_Svr gamma + eta* <= eta_max

    gam_upper_bound = np.ravel(n_max) - np.ravel(n_sol)


    N_Svr_thresh = N_Svr
    N_Svr_thresh[N_Svr_thresh <= 1e-10] = 0

    gub_thresh = gam_upper_bound
    gub_thresh[gub_thresh<=1e-3] = 0



    #### and must satisfy the consistency constraints for the solution flux
    S_v_T_sgn = S_v_T*np.sign(y_sol)
    S_f_T_sgn = S_f_T*np.sign(y_sol)


    #np.sign(y)*( np.matmul(S_v_T,n) + np.matmul(S_f_T,FxdM) - np.log(K_v) - h_func(y) )

    nty_rhs_bound = np.sign(y_sol)*h_func(y_sol) + np.sign(y_sol)*np.log(K_v) - np.matmul(S_f_T_sgn,FxdM ) - np.matmul(S_v_T_sgn,n_sol)

    nty_thresh = nty_rhs_bound
    nty_thresh[nty_thresh<=1e-4] = 0

    nty_lhs_A = np.matmul(S_v_T_sgn,N_Svr_thresh)



    A_ub = np.concatenate( (N_Svr_thresh,nty_lhs_A))

    b_ub = np.concatenate( (np.reshape(gub_thresh,(len(gub_thresh),1)),nty_rhs_bound) )
    b_ub = np.ravel(b_ub)


    #solve the optimization
    lp_sol = linprog(obj_sgn*obj_coefs,A_ub = A_ub, b_ub = b_ub)

    if lp_sol.success != True:
        print('Optimization Failed')

    gam_sol = lp_sol.x
    gam_sol = np.reshape(gam_sol,(len(gam_sol),1))

    free_n_sol = np.ravel(n_sol) + np.ravel(np.matmul(N_Svr_thresh,gam_sol))


    n_vec = free_n_sol
    E_regulation = np.ones(len(y_sol))
    rxn_flux_unregulated = rxn_flux(n_vec, FxdM, S, K_v, E_regulation)

    #print(rxn_flux_unregulated)


    ### compute regulation
    free_alpha_sol = y_sol[:,0]/rxn_flux_unregulated[:,0]

    return [free_n_sol, free_alpha_sol]














def max_min_total_activity_sol(y_sol,n_sol,N_Svr,n_max,FxdM,S,S_v,S_f,K_v,sense = 'max'):


    ###check all inputs are appropriate shape
    y_sol = np.reshape(y_sol,(len(y_sol),1))
    n_sol = np.reshape(n_sol,(len(n_sol),1))
    n_max = np.reshape(n_max,(len(n_max),1))
    FxdM = np.reshape(FxdM,(len(FxdM),1))
    K_v = np.reshape(K_v,(len(K_v),1))

    S_v_T = np.transpose(S_v)
    S_f_T = np.transpose(S_f)

    if sense =='max': obj_sgn = -1
    if sense =='min': obj_sgn = 1




    N_Svr_thresh = N_Svr
    N_Svr_thresh[N_Svr_thresh <= 1e-10] = 0

    S_v_T_sgn = S_v_T*np.sign(y_sol)
    S_v_sgn = np.transpose(S_v_T_sgn)
    S_f_T_sgn = S_f_T*np.sign(y_sol)



    # Objective is to increase or decrease the activity coefficients by as much as possible

    gam_conv = np.matmul(np.transpose(N_Svr), S_v_sgn )

    obj_coefs = np.matmul(gam_conv,np.ones( (len(y_sol),1) ) )
    obj_coefs = obj_coefs[:,0]


    ### subject to the constraints that N_Svr gamma + eta* <= eta_max

    gam_upper_bound = np.ravel(n_max) - np.ravel(n_sol)


    gub_thresh = gam_upper_bound
    gub_thresh[gub_thresh<=1e-3] = 0



    #### and must satisfy the consistency constraints for the solution flux
   

    #np.sign(y)*( np.matmul(S_v_T,n) + np.matmul(S_f_T,FxdM) - np.log(K_v) - h_func(y) )

    nty_rhs_bound = np.sign(y_sol)*h_func(y_sol) + np.sign(y_sol)*np.log(K_v) - np.matmul(S_f_T_sgn,FxdM ) - np.matmul(S_v_T_sgn,n_sol)

    nty_thresh = nty_rhs_bound
    nty_thresh[nty_thresh<=1e-4] = 0

    nty_lhs_A = np.matmul(S_v_T_sgn,N_Svr_thresh)



    A_ub = np.concatenate( (N_Svr_thresh,nty_lhs_A))

    b_ub = np.concatenate( (np.reshape(gub_thresh,(len(gub_thresh),1)),nty_rhs_bound) )
    b_ub = np.ravel(b_ub)


    #solve the optimization
    lp_sol = linprog(obj_sgn*obj_coefs,A_ub = A_ub, b_ub = b_ub)

    if lp_sol.success != True:
        print('Optimization Failed')

    gam_sol = lp_sol.x
    gam_sol = np.reshape(gam_sol,(len(gam_sol),1))

    free_n_sol = np.ravel(n_sol) + np.ravel(np.matmul(N_Svr_thresh,gam_sol))


    n_vec = free_n_sol
    E_regulation = np.ones(len(y_sol))
    rxn_flux_unregulated = rxn_flux(n_vec, FxdM, S, K_v, E_regulation)

    #print(rxn_flux_unregulated)


    ### compute regulation
    free_alpha_sol = y_sol[:,0]/rxn_flux_unregulated[:,0]

    return [free_n_sol, free_alpha_sol]






def max_min_subset_activity_sol(rxn_indexes,y_sol,n_sol,N_Svr,n_max,FxdM,S,S_v,S_f,K_v,sense = 'max'):


    ###check all inputs are appropriate shape
    y_sol = np.reshape(y_sol,(len(y_sol),1))
    n_sol = np.reshape(n_sol,(len(n_sol),1))
    n_max = np.reshape(n_max,(len(n_max),1))
    FxdM = np.reshape(FxdM,(len(FxdM),1))
    K_v = np.reshape(K_v,(len(K_v),1))

    S_v_T = np.transpose(S_v)
    S_f_T = np.transpose(S_f)

    if sense =='max': obj_sgn = -1
    if sense =='min': obj_sgn = 1




    N_Svr_thresh = N_Svr
    N_Svr_thresh[N_Svr_thresh <= 1e-10] = 0

    S_v_T_sgn = S_v_T*np.sign(y_sol)
    S_v_sgn = np.transpose(S_v_T_sgn)
    S_f_T_sgn = S_f_T*np.sign(y_sol)



    # Objective is to increase or decrease the activity coefficients by as much as possible

    gam_conv = np.matmul(np.transpose(N_Svr), S_v_sgn )

    ### Select only a subset of reactions
    reaction_selection = np.zeros( (len(y_sol),1))
    reaction_selection[rxn_indexes,0] = 1

    obj_coefs = np.matmul(gam_conv,reaction_selection )
    obj_coefs = obj_coefs[:,0]


    ### subject to the constraints that N_Svr gamma + eta* <= eta_max

    gam_upper_bound = np.ravel(n_max) - np.ravel(n_sol)


    gub_thresh = gam_upper_bound
    gub_thresh[gub_thresh<=1e-3] = 0



    #### and must satisfy the consistency constraints for the solution flux
   

    #np.sign(y)*( np.matmul(S_v_T,n) + np.matmul(S_f_T,FxdM) - np.log(K_v) - h_func(y) )

    nty_rhs_bound = np.sign(y_sol)*h_func(y_sol) + np.sign(y_sol)*np.log(K_v) - np.matmul(S_f_T_sgn,FxdM ) - np.matmul(S_v_T_sgn,n_sol)

    nty_thresh = nty_rhs_bound
    nty_thresh[nty_thresh<=1e-4] = 0

    nty_lhs_A = np.matmul(S_v_T_sgn,N_Svr_thresh)



    A_ub = np.concatenate( (N_Svr_thresh,nty_lhs_A))

    b_ub = np.concatenate( (np.reshape(gub_thresh,(len(gub_thresh),1)),nty_rhs_bound) )
    b_ub = np.ravel(b_ub)


    #solve the optimization
    lp_sol = linprog(obj_sgn*obj_coefs,A_ub = A_ub, b_ub = b_ub)

    if lp_sol.success != True:
        print('Optimization Failed')

    gam_sol = lp_sol.x
    gam_sol = np.reshape(gam_sol,(len(gam_sol),1))

    free_n_sol = np.ravel(n_sol) + np.ravel(np.matmul(N_Svr_thresh,gam_sol))


    n_vec = free_n_sol
    E_regulation = np.ones(len(y_sol))
    rxn_flux_unregulated = rxn_flux(n_vec, FxdM, S, K_v, E_regulation)

    #print(rxn_flux_unregulated)


    ### compute regulation
    free_alpha_sol = y_sol[:,0]/rxn_flux_unregulated[:,0]

    return [free_n_sol, free_alpha_sol]
