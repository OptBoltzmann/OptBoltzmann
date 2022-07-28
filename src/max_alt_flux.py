



import pyomo
import pyomo.environ as pe
import itertools 
import numpy as np
import numpy.random as nprd
from scipy.linalg import norm
from scipy.optimize import least_squares
import scipy.linalg as spL

import pyutilib.services
from pyomo.opt import TerminationCondition






##### assume that initialized values are at an optimal solution for growth production

### add an additional constraint that restricts the objective value to be within epsilon of the the starting growth objective value

### search to maximize a secondary objective.

### possible to reduce the problem complexity by removing the gradient direction for growth from the steady state basis
### such that all changes would be neutral with respect to the growth objective.




def Max_alt_flux(n_ini,y_ini,target_log_vcounts, f_log_counts, S, K,primary_obj_coefs,primary_obj_tolerance,secondary_obj_coefs):

  # Flip Stoichiometric Matrix
    S_T = S    # S_T is the Stoich matrix with rows as reactions, columns as metabolites
    S = np.transpose(S) # this now is the Stoich matrix with rows metabolites, and columns reactions
    n_react = np.shape(S)[1]

    #Set the System parameters 
    FxdM = np.reshape(f_log_counts,(len(f_log_counts),1) )
    K_eq = np.reshape(K,(n_react,1))

    #Metabolite parms
    n_M_f = len(f_log_counts)   #total number of fixed metabolites
    n_M_v = len(n_ini)   #total number of variable metabolites
    n_M = n_M_f + n_M_v   # total number of metabolites

    ### construct parm indices
    react_idx = np.arange(0,n_react)

    TotM_idx = np.arange(0,n_M)
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
    dSv_N = np.shape(Sv_N)[1] # the dimension of the nullspace

    # Steady State satisfies the following least squares problem
    # || S_T*( SVS* y_hat) - y_hat ||^2 = 0
    # where SVS is the matrix (S_v * S_v_T)^-1 * S_v

    ###precompute SVS
    SvS = np.matmul( np.linalg.inv( np.matmul(S_v,S_v_T) ), S_v) 

    #Get the system parameters to constuct a model

    #Metabolite parms
    n_M = n_M_f + n_M_v   # total number of metabolites

    ### construct parm indices
    react_idx = np.arange(0,n_react)

    beta_idx = np.arange(0,dSv_N)

    
    
    
    #Set the initial condition

    beta_ini = np.ravel(np.matmul(np.transpose(Sv_N),y_ini) )



    b_ini = np.matmul(S_v_T,np.reshape(n_ini,(len(n_ini),1))) + np.matmul(S_f_T,np.reshape(FxdM,(n_M_f,1) )    )
    b_ini = np.reshape(b_ini,len(b_ini))
    
   
    h_ini = np.sign(y_ini)*( np.log(2) - np.log( np.abs(y_ini) + np.sqrt( np.power(y_ini,2) + 4 ) )  )
    

   
    ## Set the optimization parameters
    VarM_lbnd = -300 #lower bound on the log metabolite counts
    

    #Pyomo Model Definition
    ######################
    m =  pe.ConcreteModel()

    #Input the model parameters

    #set the indices
    #####################
    m.react_idx = pe.Set(initialize = react_idx) 

    m.TotM_idx = pe.Set(initialize = TotM_idx)
    m.VarM_idx = pe.Set(initialize = VarM_idx)
    m.FxdM_idx = pe.Set(initialize = FxdM_idx)

    m.beta_idx = pe.Set(initialize = beta_idx)

    #m.obj_rxn_idx = pe.Set(initialize = obj_rxn_idx)

    dm_idx = np.arange(0,1)
    m.dm_idx = pe.Set(initialize = dm_idx)


    # Objective Coefficients
    ##################
    obj_c_dict = dict(list(zip(react_idx,secondary_obj_coefs) ))
    m.obj_coefs=pe.Param(m.react_idx, initialize = obj_c_dict)









    # Stochiometric matrix
    ###########################
    S_idx = list(itertools.product(np.arange(0,n_M),np.arange(0,n_react)))
    S_vals = list(np.reshape(S,[1,n_M*n_react])[0])
    S_dict = dict(list(zip(S_idx,S_vals)))

    m.S = pe.Param(S_idx ,initialize = S_dict,mutable = True)

    # variable metabolite stochiometric psuedo inverse matrix
    ############################
    SvS_idx = list(itertools.product(np.arange(0,n_M_v),np.arange(0,n_react)))
    SvS_vals = list(np.reshape(SvS,[1,n_M_v*n_react])[0])
    SvS_dict = dict(list(zip(SvS_idx,SvS_vals)))

    m.SvS = pe.Param(SvS_idx ,initialize = SvS_dict)

    ## Nullspace basis Matrix
    ####################
    SvN_idx = list(itertools.product(np.arange(0,n_react),np.arange(0,dSv_N)))
    SvN_vals = list(np.reshape(Sv_N,[1,n_react*dSv_N])[0])
    SvN_dict = dict(list(zip(SvN_idx,SvN_vals)))

    m.SvN=pe.Param(SvN_idx, initialize = SvN_dict)

    # Reaction Equilibrium constants
    ##################
    K_dict = dict(list(zip(react_idx,K) ))
    m.K=pe.Param(m.react_idx, initialize = K_dict)

    # Fixed metabolite log counts
    FxdM_dict = dict( list( zip(FxdM_idx,f_log_counts) ) )
    m.FxdM = pe.Param(m.FxdM_idx, initialize = FxdM_dict)

    # Bounds on the log of the metabolites
    #########
    M_ubnd_dict = dict(list(zip(VarM_idx,target_log_vcounts)))
    m.VarM_ubnd = pe.Param(m.VarM_idx, initialize = M_ubnd_dict)


    #SET the Variables
    #############################


    ## Variable metabolites (log)
    ######################

    Mini_dict = dict(list(zip(VarM_idx,n_ini)))
    m.VarM = pe.Var(VarM_idx,initialize = Mini_dict)

    # steady state fluxes
    yini_dict = dict(list(zip(react_idx,y_ini)))
    m.y = pe.Var(react_idx, initialize = yini_dict)

    # flux null space representation
    betaini_dict = dict(list(zip(beta_idx,beta_ini)))
    m.beta = pe.Var(beta_idx, initialize = betaini_dict)

    # Steady state condition RHS
    bini_dict = dict(list(zip(react_idx,b_ini)))
    m.b = pe.Var(react_idx, initialize = bini_dict)

    hini_dict = dict(list(zip(react_idx,h_ini)))
    m.h = pe.Var(react_idx, initialize = hini_dict)

    
    
    # Set the Constraints
    #############################

    #flux null space representation constraint
    def flux_null_rep(m,i):
        return m.y[i]   ==  sum( m.SvN[(i,j)]*m.beta[j]  for j in m.beta_idx )
    m.fxnrep_cns = pe.Constraint(m.react_idx, rule = flux_null_rep)

    

    def steady_state_metab(m,j):
        return m.b[j]  == sum( m.S[(k,j)]*m.VarM[k] for k in m.VarM_idx ) + sum( m.S[(k,j)]*m.FxdM[k] for k in m.FxdM_idx ) 
    m.ssM_cns = pe.Constraint(m.react_idx, rule = steady_state_metab)

    
    
    
    def num_smooth_cns(m,i):
        return m.h[i] ==(m.y[i]*1e50/(abs(m.y[i])*1e50 + 1e-50))*(pe.log(2) -  pe.log(abs(m.y[i]) + pe.sqrt(m.y[i]**2 + 4 ) )  ) 
    m.nms_cns = pe.Constraint(m.react_idx, rule = num_smooth_cns)
    
    
    # y sign variable
    y_sign_ini_dict = dict(list(zip(react_idx,.5+.5*np.sign(y_ini) )))
    #m.u = pe.Var(react_idx,bounds=(0,1),initialize = y_sign_ini_dict)
    m.u = pe.Param(react_idx,initialize = y_sign_ini_dict)


    ysgn_ini_dict = dict(list(zip(react_idx,np.sign(y_ini) )))
    m.ysgn = pe.Param(react_idx,initialize = ysgn_ini_dict)
    
    Mb = 100
    
    def relaxed_reg_cns_upper(m,i):
        return ( m.b[i] - pe.log(m.K[i]) ) >= m.h[i] - Mb*(m.u[i])  
    m.rxr_cns_up = pe.Constraint(m.react_idx,rule = relaxed_reg_cns_upper)
    
    
    def relaxed_reg_cns_lower(m,i):
        return ( m.b[i] - pe.log(m.K[i]) ) <= m.h[i] + Mb*(1 - m.u[i])
    m.rxr_cns_low = pe.Constraint(m.react_idx,rule = relaxed_reg_cns_lower)
    
    
    
    #def sign_constraint(m,i):
    #    return (pe.log(m.K[i]) - m.b[i])*m.y[i] >= 0
    #m.sign_y_cns = pe.Constraint(m.react_idx, rule = sign_constraint)


    def sign_constraint(m,i):
        return m.y[i]*m.ysgn[i] >= 0 
    m.sign_y_cns = pe.Constraint(m.react_idx, rule = sign_constraint)
    
    #def y_sign_relax(m,i):
    #    return 2*m.u[i] - 1 == (m.y[i]/(abs(m.y[i]) + 1e-50) ) 
    #m.y_sign_relax_cns = pe.Constraint(m.react_idx,rule = y_sign_relax)
    
   
    
    # Variable metabolite upper and lower bounds
    def M_upper_cnstrnts(m,i):
        return  m.VarM[i] <= m.VarM_ubnd[i]
    m.VarM_ub_cns = pe.Constraint(m.VarM_idx,rule = M_upper_cnstrnts)


    def M_lower_cnstrnts(m,i):
        return  m.VarM[i] >= VarM_lbnd
    m.VarM_lb_cns = pe.Constraint(m.VarM_idx,rule = M_lower_cnstrnts)

    
    

    # Original objective constraint
    og_obj = np.sum(y_ini*primary_obj_coefs)
    og_ep = primary_obj_tolerance
    obj_bound = np.array([og_obj - og_ep])

    # ## Nullspace basis Matrix
    # ####################
    # SvN_idx = list(itertools.product(np.arange(0,n_react),np.arange(0,dSv_N)))
    # SvN_vals = list(np.reshape(Sv_N,[1,n_react*dSv_N])[0])
    # SvN_dict = dict(list(zip(SvN_idx,SvN_vals)))

    # m.SvN=pe.Param(SvN_idx, initialize = SvN_dict)




    
    #Original Objective Coefficients
    ##################
    obj_og_idx = list(itertools.product(np.arange(0,n_react),np.arange(0,1)))
    obj_og_dict = dict(list(zip(obj_og_idx,list(primary_obj_coefs) ) ))
    m.og_obj_coefs=pe.Param(obj_og_idx, initialize = obj_og_dict)

    rhs_og_idx = np.arange(0,1)
    rhs_og_dict =  dict(list(zip(rhs_og_idx,list(obj_bound) ) ))
    m.rhs_og_coefs = pe.Param(rhs_og_idx, initialize = rhs_og_dict)


    def og_obj_cnstrnt(m,i):
        return m.rhs_og_coefs[i] <= sum( m.og_obj_coefs[(j,i)]*m.y[j]  for j in m.react_idx )
    m.og_obj_cns = pe.Constraint(m.dm_idx, rule = og_obj_cnstrnt)

    
    #def steady_state_metab(m,j):
    #    return m.b[j]  == sum( m.S[(k,j)]*m.VarM[k] for k in m.VarM_idx ) + sum( m.S[(k,j)]*m.FxdM[k] for k in m.FxdM_idx ) 
    #m.ssM_cns = pe.Constraint(m.react_idx, rule = steady_state_metab)

    



    # Set the Objective function

    def _Obj(m):
        return sum( m.y[j]*m.obj_coefs[j]  for j in m.react_idx )
    m.Obj_fn = pe.Objective(rule = _Obj, sense = pe.maximize) 

    
    

    #Find a Solution
    ####################

    max_iter = 10000
    max_cpu_time = 800000



    #Set the solver to use
    opt=pe.SolverFactory('ipopt', solver_io='nl')


    #Set solver otpions
    opts = {'max_iter': max_iter,
          'max_cpu_time': max_cpu_time,
          'tol':1e-7,
          'acceptable_tol':1e-6,
          'linear_solver': 'ma57',
          'hsllib':'/opt/coinhsl/lib/libcoinhsl.dylib'} 
          #'halt_on_ampl_error': 'yes',
          # 'dual_inf_tol':1.0,
          # 'acceptable_dual_inf_tol':1.01,
          # 'OF_print_info_string':'yes'}
          #'print_level': 8}
          #'acceptable_constr_viol_tol':1e-10,
          #'constr_viol_tol': 1e-7,
          #'acceptable_constr_viol_tol':1e-6}
          #'halt_on_ampl_error': 'yes'}



    ## Solve the Model
    status_obj = opt.solve(m, options=opts, tee=True)


    n_sol=np.zeros(n_M_v)
    b_sol = np.zeros(n_react)
    y_sol = np.zeros(n_react)
    beta_sol = np.zeros(dSv_N)
    h_sol = np.zeros(n_react)

    for i in react_idx:
        b_sol[i] = pe.value(m.b[i])
        y_sol[i] = pe.value(m.y[i])
        h_sol[i] = pe.value(m.h[i])

    for i in beta_idx:
        beta_sol[i] = pe.value(m.beta[i])

    for i in VarM_idx:
        n_sol[i] = pe.value(m.VarM[i])
        
    
    
    E_regulation = np.ones(len(y_sol))
    unreg_rxn_flux = np.ravel( rxn_flux(n_sol, f_log_counts,S_T, K, E_regulation) )
    alpha_sol = y_sol/unreg_rxn_flux
    alpha_sol = np.ravel(alpha_sol)

    

    return(y_sol, alpha_sol, n_sol)








def rxn_flux(v_log_counts, f_log_counts, S, K, E_regulation):
    # Flip Stoichiometric Matrix
    S_T = S    # S_T is the Stoich matrix with rows as reactions, columns as metabolites
    S = np.transpose(S) # this now is the Stoich matrix with rows metabolites, and columns reactions
    n_react = np.shape(S)[1]
    v_log_counts = np.reshape(v_log_counts,(len(v_log_counts),1))
    f_log_counts = np.reshape(f_log_counts,(len(f_log_counts),1))
    tot_log_counts = np.concatenate((v_log_counts,f_log_counts))
    K = np.reshape(K,(len(K),1))
    E_regulation = np.reshape(E_regulation,(len(E_regulation),1))
    
    #forward_odds = K*np.exp(- .25*np.matmul(S_T,tot_log_counts) )*np.exp(- .25*np.matmul(S_T,tot_log_counts) )*np.exp(- .25*np.matmul(S_T,tot_log_counts) )*np.exp(- .25*np.matmul(S_T,tot_log_counts) )
    forward_odds = K*np.exp(-.25*np.matmul(S_T,tot_log_counts) )**4
    reverse_odds = np.power(K,-1)*np.exp(.25*np.matmul(S_T,tot_log_counts) )**4
    
    #forward_odds = np.exp(-.25*(np.matmul(S_T,tot_log_counts) + np.log(K) + np.log(E_regulation) ) )**4
    #reverse_odds = np.exp(.25*(np.matmul(S_T,tot_log_counts) - np.log(K) + np.log(E_regulation) )  )**4
    
    return E_regulation*(forward_odds - reverse_odds )
    


    
    
    
