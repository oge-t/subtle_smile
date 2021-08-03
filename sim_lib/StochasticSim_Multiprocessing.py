import multiprocessing
import multiprocessing as mp
import ctypes as c
import os, inspect, sys
import pickle
import StochasticSim_Multiprocessing as smp
import sim_params

###################### Read the G1pp parameters ##########################
import scipy.integrate as integrate
from scipy.integrate import simps
from scipy.integrate import cumtrapz
from numpy.random import multivariate_normal as mvn
from numpy.random import uniform
from numpy.random import normal
import numpy as np
import copy
import inspect
import gc

### simulator api definition #######
#locvol_sim = LocalVolStochasticIRSimulation(domestic_dfcurve, base_dfcurve, spot_FX,
#                                            domestic_ta, domestic_a, domestic_tvol, domestic_vols, domestic_x0,
#                                            base_ta, base_a, base_tvol, base_vols, base_x0,
#                                            rho_domestic_base, rho_domestic_fx, rho_base_fx,
#                                            nr_mcruns, LVOL_NUMMETHOD, SHORTRATE_NUMMETHOD,
#                                            antitheticpaths=ANTITHETIC,
#                                            in_forward_measure=True,
#                                            nrsubsim=NRSUBSIM,
#                                            )

#---------------------------------------------------------------------------
class LocalVolStochasticIRSimulation():

    def __init__(self,  domestic_shifttimes, domestic_shiftvalues, base_shifttimes, base_shiftvalues, spot_FX,
                        domestic_ta, domestic_a, domestic_tvol, domestic_vols, domestic_x0,
                        base_ta, base_a, base_tvol, base_vols, base_x0,
                        rho_domestic_base, rho_domestic_fx, rho_base_fx,
                        nr_mcruns, LVOL_NUMMETHOD, SHORTRATE_NUMMETHOD,
                        antitheticpaths=True,
                        in_forward_measure=False,
                        nrsubsim=2, 
                        observation_names = None,
                        observe_at_given_times=True,
                        domestic_currency_name='USD',
                        base_currency_name='EUR',
                        fx_name='EURUSD',
                        maxtdisargs=None,
                        save_output_flags=None):

        #self.sdedata = sdedata
        #self.g1pp_data = g1pp_data
        
        self.nrsubsim = nrsubsim
        self.nr_mcruns = nr_mcruns
        self.dtype_glob = 'float' #'float32'
        self.nsteps_peryear = 250
        self.in_forward_measure = in_forward_measure
        self.observation_names = observation_names if (observation_names is not None) else []
        self.antitheticpaths = antitheticpaths
        self.in_DRN = True
        self.det_DR = False
        self.det_FR = False
        self.DLV = True
        self.spot_FX = spot_FX
        self.fx_prefix=''
        self.domestic_currency_prefix=''
        self.base_currency_prefix=''
        
        self.g1pp_data = sim_params.g1pp_dataC(spot_FX, base_x0, domestic_x0, domestic_ta, domestic_shifttimes, 
                                                domestic_tvol, domestic_a, domestic_vols, domestic_shiftvalues,
                                                base_ta, base_a, base_shifttimes, base_shiftvalues, base_tvol, base_vols)

        self.sdedata = sim_params.sdedataC(rho_domestic_base, rho_base_fx, rho_domestic_fx)
        
        self.heston_data = None
        
        if (maxtdisargs is None): ### for output shared memory allocation... 
            self.maxT = 10.0
            maxtdisargs = {'maturity':self.maxT, 'time_steps': int(self.nsteps_peryear*self.maxT)}
            
        self.maxT = maxtdisargs['maturity']
        self.maxn_time = maxtdisargs['time_steps']
        maxn_time = self.maxn_time
        maxT = self.maxT
        self.dt = maxT/maxn_time
        self.n_paths_per_process = int(nr_mcruns/nrsubsim)
        
        self.updated_contracts_or_sim = False
        self.lvol_expectations = 0.0
        self.lvol_stderr = 0.0
        
        #mult = 1 if dtype_glob=='float32' else 2 
        
        self.t_stamp_raw = []
        self.Xsamples_raw = []
        self.rd_raw = []
        self.rb_raw = []
        self.volsamples_raw = []
        self.Utsamples_raw = []
        self.dW_raw = []
        self.xd_raw = []
        self.xb_raw = []
        
        #mg = multiprocessing.Manager()
        self.contracts = {}
        self.lvol_data = {}
        
        self.save_output_flags = {'XS':True, 'RD':True, 'RB':True, 'XD':False, 'XB':False, 'DW':False, 'VOLTRAJ':False}
        
        ##### override save output flags ###
        save_keys = self.save_output_flags.keys()
        if save_output_flags is not None:
            for keyval in save_keys:
                if keyval in save_output_flags.keys():
                    self.save_output_flags[keyval] = save_output_flags[keyval]
        
        
        self.mult = 1 if (self.dtype_glob=='float32') else 2

        if type(self) is LocalVolStochasticIRSimulation:
            print('in_forward_measure =%s, Antithetic=%s, in_DRN=%s, det_DR=%s, det_FR=%s, DLV=%s'%(self.in_forward_measure, self.antitheticpaths, self.in_DRN, self.det_DR, self.det_FR, self.DLV))
            self._allocate_shmem()
            print("Initializing pool for ...", type(self))
            self.simpool = multiprocessing.Pool(processes = self.nrsubsim, initializer=init_process, initargs=(self.t_stamp_raw, 
                                                                                                              self.Xsamples_raw, 
                                                                                                              self.rd_raw, 
                                                                                                              self.rb_raw, 
                                                                                                              self.xd_raw, 
                                                                                                              self.xb_raw, 
                                                                                                              self.dW_raw,
                                                                                                              self.volsamples_raw,
                                                                                                              self.Utsamples_raw,
                                                                                                              self.maxn_time, self.maxT))

        return
#---------------------------------------------------------------------------
    def _allocate_shmem(self):
        mult = self.mult
        maxn_time = self.maxn_time
        n_paths_per_process = self.n_paths_per_process
        
        SAVE_XS = self.save_output_flags['XS']
        SAVE_RD = self.save_output_flags['RD']
        SAVE_RB = self.save_output_flags['RB']
        SAVE_XD = self.save_output_flags['XD']
        SAVE_XB = self.save_output_flags['XB']
        SAVE_DW = self.save_output_flags['DW']
        SAVE_VOLTRAJ = self.save_output_flags['VOLTRAJ']
        
        for _ in range(self.nrsubsim):
            print("Allocating shmem ...", end='')
            self.t_stamp_raw.append(mp.Array(c.c_float, mult*maxn_time, lock=False))
            if (SAVE_XS):
                self.Xsamples_raw.append(mp.Array(c.c_float, mult*n_paths_per_process*(maxn_time+1), lock=False))
                print('XS', end=' ')
            if (SAVE_RD):
                self.rd_raw.append(mp.Array(c.c_float, mult*n_paths_per_process*maxn_time, lock=False))
                print('RD', end=' ')
            if (SAVE_RB):
                self.rb_raw.append(mp.Array(c.c_float, mult*n_paths_per_process*maxn_time, lock=False))
                print('RB', end=' ')
            if (SAVE_XD):
                self.xd_raw.append(mp.Array(c.c_float, mult*n_paths_per_process*maxn_time, lock=False))
                print('XD', end=' ')
            if (SAVE_XB):
                self.xb_raw.append(mp.Array(c.c_float, mult*n_paths_per_process*maxn_time, lock=False))
                print('XB', end=' ')
            if (SAVE_DW):
                self.dW_raw.append(mp.Array(c.c_float, mult*n_paths_per_process*(maxn_time+1), lock=False))
                print('DW', end=' ')
            if (SAVE_VOLTRAJ):
                self.volsamples_raw.append(mp.Array(c.c_float, mult*n_paths_per_process*maxn_time, lock=False))
                print('VOLTRAJ', end=' ')
            print()
        return
#---------------------------------------------------------------------------
    def delete_all(self):
#---------------------------------------------------------------------------
        self.simpool.close()

        SAVE_XS = self.save_output_flags['XS']
        SAVE_XD = self.save_output_flags['XD']
        SAVE_XB = self.save_output_flags['XB']
        SAVE_RD = self.save_output_flags['RD']
        SAVE_RB = self.save_output_flags['RB']
        SAVE_DW = self.save_output_flags['DW']
        SAVE_VOLTRAJ = self.save_output_flags['VOLTRAJ']
        SAVE_HESTON = self.save_output_flags['HESTON'] if (self.DLV==False) else False
        
        for ts in self.t_stamp_raw: del(ts) 
        if (SAVE_XS):
            for Xs in self.Xsamples_raw: del(Xs)
            print('Deleting Xs...')
        if (SAVE_RB):
            for rb in self.rb_raw: del(rb)
            print('Deleting RB...')
        if (SAVE_RD):
            for rd in self.rd_raw: del(rd) 
            print('Deleting RD...')
        if (SAVE_XB):
            for xb in self.xb_raw: del(xb) 
            print('Deleting XB...')
        if (SAVE_XD):
            for xd in self.xd_raw: del(xd)
            print('Deleting XD...')
        if (SAVE_DW):
            for dW in self.dW_raw: del(dW) 
            print('Deleting DW...')
        if (SAVE_VOLTRAJ):
            for vol in self.volsamples_raw: del(vol) 
            print('Deleting VOLTRAJ...')
        if (SAVE_HESTON):
            for ut in self.Utsamples_raw: del(ut) 
            print('Deleting HESTON UT...')

        del(self.t_stamp_raw, self.Xsamples_raw, self.rd_raw, self.rb_raw, self.xb_raw, self.xd_raw, self.dW_raw, self.volsamples_raw, self.Utsamples_raw)
        mp.heap.BufferWrapper._heap = mp.heap.Heap ()
        gc.collect ()

#----------------------------------------------------------------------------------------------------------
    def run(self, det_DR = None, det_FR = None, DLV = None):
#----------------------------------------------------------------------------------------------------------
        pinputs = []
        tdisargs = self.tdisargs
        sdedata = self.sdedata
        g1pp_data = self.g1pp_data
        heston_data = self.heston_data
        lvol_data = self.lvol_data
        self.updated_contracts_or_sim = True

        if (det_DR is None): det_DR = self.det_DR
        if (det_FR is None): det_FR = self.det_FR
        if (DLV is None): DLV = self.DLV

        n_trials = self.n_paths_per_process
        for pid in range(self.nrsubsim):
            pinputs.append((pid, n_trials, tdisargs, sdedata, g1pp_data, heston_data, lvol_data, det_DR, det_FR, DLV, self.in_DRN, self.in_forward_measure, self.antitheticpaths, self.dtype_glob, self.save_output_flags))
            print('ProcessID:%d, n_trials:%d, '%(pinputs[-1][0], pinputs[-1][1])+'maturity:{maturity:.3f}, timesteps:{time_steps:d}'.format(**pinputs[-1][2]))
    
        self.simpool.starmap(smp.simulate_paths, pinputs)

        self._collect_simulation_results()

        return 

#---------------------------------------------------------------------------        
    def _get_X_and_U(self):
        dtype_glob= self.dtype_glob
        n_timesteps = self.tdisargs['time_steps']
        n_trials = self.n_paths_per_process
        observation_idxs = -1
        spots = np.concatenate([np.frombuffer(xs, dtype=dtype_glob).reshape((n_trials, -1))[:,:n_timesteps+1][:, observation_idxs] for xs in self.Xsamples_raw], axis=0)
        variances = np.concatenate([np.frombuffer(ut, dtype=dtype_glob).reshape((n_trials, -1))[:,:n_timesteps][:, observation_idxs] for ut in self.Utsamples_raw], axis=0)
        return spots, variances

#---------------------------------------------------------------------------        
    def _collect_simulation_results(self):
        self.outputMC = {}
        self.outputMC['MCResults'] = {}
        self.outputMC['MCResults']['SimulationObservations'] = {}
        self.outputMC['MCResults']['SimulationObservations']['Samples'] = []
        dtype_glob= self.dtype_glob
        n_timesteps = self.tdisargs['time_steps']
        n_trials = self.n_paths_per_process
        observation_idxs = [-1]
        observation_names = self.observation_names #['FXrate', 'FXStochvar']
        print('Collecting results for ', observation_names)
        self.MCobservations = []
        
        for obsname in observation_names:
            if (obsname.lower() == 'fxrate'):
                resout = np.concatenate([np.frombuffer(xs, dtype=dtype_glob).reshape((n_trials, -1))[:,:n_timesteps+1][:, observation_idxs] for xs in self.Xsamples_raw], axis=0)
            if (obsname.lower() == 'fxstochvariance'):
                resout = np.concatenate([np.frombuffer(ut, dtype=dtype_glob).reshape((n_trials, -1))[:,:n_timesteps][:, observation_idxs] for ut in self.Utsamples_raw], axis=0)
            if (obsname.lower() == 'rd'):
                resout = np.concatenate([np.frombuffer(rd, dtype=dtype_glob).reshape((n_trials, -1))[:,:n_timesteps][:, observation_idxs] for rd in self.rd_raw], axis=0)
            if (obsname.lower() == 'rb'):
                resout = np.concatenate([np.frombuffer(rb, dtype=dtype_glob).reshape((n_trials, -1))[:,:n_timesteps][:, observation_idxs] for rb in self.rb_raw], axis=0)
            if (obsname.lower() == 'domestic_xfactor'):
                resout = np.concatenate([np.frombuffer(xd, dtype=dtype_glob).reshape((n_trials, -1))[:,:n_timesteps][:, observation_idxs] for xd in self.xd_raw], axis=0)
            if (obsname.lower() == 'base_xfactor'):
                resout = np.concatenate([np.frombuffer(xb, dtype=dtype_glob).reshape((n_trials, -1))[:,:n_timesteps][:, observation_idxs] for xb in self.xb_raw], axis=0)
            #if (obsname.lower() == 'tstamp'):
            #    #resout = np.concatenate([np.frombuffer(ts, dtype=dtype_glob).reshape((1, -1))[:,:n_timesteps] for ts in self.t_stamp_raw], axis=0)
            #    resout = np.frombuffer(self.t_stamp_raw[0], dtype=dtype_glob).reshape(-1)[:n_timesteps][observation_idxs]

            self.MCobservations.append(list(resout))

        for t in zip(*self.MCobservations):
            self.outputMC['MCResults']['SimulationObservations']['Samples'].append({obsname: res for obsname,res in zip(observation_names, t)})

        return
        
#---------------------------------------------------------------------------        
    def update_base_bfactors(self, finaltime):
        
        print(finaltime, self.nsteps_peryear)
        self.tdisargs = {'maturity':finaltime, 'time_steps':int(self.nsteps_peryear*finaltime)}
        return 
    
#---------------------------------------------------------------------------        
    def update_domestic_bfactors(self, finaltime):
    
        self.tdisargs = {'maturity':finaltime, 'time_steps':int(self.nsteps_peryear*finaltime)}
        return 

#---------------------------------------------------------------------------            

    def set_simulation_times(self, finaltimes):
    
        self.tdisargs = {'maturity':finaltimes[-1], 'time_steps':int(self.nsteps_peryear*finaltimes[-1])}
        return 
        
    def set_observation_times(self, obstimes):
    
        self.tdisargs = {'maturity':obstimes[-1], 'time_steps':int(self.nsteps_peryear*obstimes[-1])}
        return 
        
    def set_vanilla_contract(self, contracttype, maturity, strike):
    
        return
#---------------------------------------------------------------------------        
    def update_localvol(self, strikes, ts, locvols):
    
        self.lvol_data['spots'] = strikes
        self.lvol_data['surfacevalues'] = locvols
        self.lvol_data['times'] = ts
        return
    
#---------------------------------------------------------------------------    
    def _compute_means_from_output(self):
        
        print('Recomputing localvol means and stderr from output...')
        n_trials_per_process = self.n_paths_per_process
        n_timesteps = self.tdisargs['time_steps']
        strikes = self.contracts['strikes']
        dtype_glob = self.dtype_glob
        Xsamples = np.concatenate([np.frombuffer(xs, dtype=dtype_glob).reshape((n_trials_per_process, -1))[:,:n_timesteps+1] for xs in self.Xsamples_raw], axis=0)
        rd = np.concatenate([np.frombuffer(rds, dtype=dtype_glob).reshape((n_trials_per_process, -1))[:,:n_timesteps] for rds in self.rd_raw], axis=0)
        rb = np.concatenate([np.frombuffer(rbs, dtype=dtype_glob).reshape((n_trials_per_process, -1))[:,:n_timesteps] for rbs in self.rb_raw], axis=0)
        
        self.lvol_expectations = np.array([np.mean((rd[:, n_timesteps-1]*K - rb[:, n_timesteps-1]*Xsamples[:, n_timesteps])*(Xsamples[:,n_timesteps]>K), axis=0) for K in strikes])
        self.lvol_stderr = [0.0]*len(strikes)
        self.updated_contracts_or_sim = False
        
        return
        
#---------------------------------------------------------------------------
    def get_means_from_output(self):
    
        if (self.updated_contracts_or_sim==True):
            self._compute_means_from_output()
            
        return self.lvol_expectations
    
#---------------------------------------------------------------------------
    def get_stderrs_from_output(self):
    
        if (self.updated_contracts_or_sim==True):
            self._compute_means_from_output()
            
        return self.lvol_stderr

#---------------------------------------------------------------------------
    def set_contracts_for_calibration(self, Ks, expiry):
    
        self.contracts['strikes'] = Ks
        self.contracts['expiry'] = expiry
        self.updated_contracts_or_sim = True
        return

#---------------------------------------------------------------------------    
class StochasticLocalVolSimulation(LocalVolStochasticIRSimulation):
#---------------------------------------------------------------------------
    def __init__(self,  domestic_shifttimes, domestic_shiftvalues, base_shifttimes, base_shiftvalues, spot_FX,
                        initialvar, volofvar_dict, kappa_dict, theta_dict, 
                        domestic_ta, domestic_a, domestic_tvol, domestic_vols, domestic_x0,
                        base_ta, base_a, base_tvol, base_vols, base_x0,
                        rho_domestic_base, rho_domestic_fx, rho_base_fx,
                        rho_fx_v, rho_domestic_v, rho_base_v,
                        nr_mcruns, LVOL_NUMMETHOD, SHORTRATE_NUMMETHOD,
                        antitheticpaths=True,
                        in_forward_measure=False,
                        nrsubsim=2, 
                        observation_names=['|FXrate', '|FXStochVariance'],
                        observe_at_given_times=True,
                        domestic_currency_name='USD',
                        base_currency_name='EUR',
                        fx_name='EURUSD',
                        maxtdisargs=None,
                        save_output_flags=None,
                        ):
                        
        super(StochasticLocalVolSimulation, self).__init__(domestic_shifttimes, domestic_shiftvalues, base_shifttimes, base_shiftvalues, spot_FX,
                                                                        domestic_ta, domestic_a, domestic_tvol, domestic_vols, domestic_x0,
                                                                        base_ta, base_a, base_tvol, base_vols, base_x0,
                                                                        rho_domestic_base, rho_domestic_fx, rho_base_fx,
                                                                        nr_mcruns, LVOL_NUMMETHOD, SHORTRATE_NUMMETHOD,
                                                                        antitheticpaths=antitheticpaths,
                                                                        in_forward_measure=in_forward_measure,
                                                                        nrsubsim=nrsubsim, 
                                                                        observation_names = observation_names,
                                                                        observe_at_given_times=observe_at_given_times,
                                                                        domestic_currency_name='USD',
                                                                        base_currency_name='EUR',
                                                                        fx_name='EURUSD',
                                                                        maxtdisargs=maxtdisargs,
                                                                        save_output_flags=save_output_flags)
                        
        
        self.nrmcruns = nr_mcruns
        self.DLV = False
        self.in_DRN = True
        self.det_DR = True
        self.det_FR = True
        
        save_dict = {'HESTON': True, 'XD': False, 'XB': False}
        
        ###### override default save_output_flags  ###
        for key in save_dict.keys():
            self.save_output_flags[key] = save_output_flags[key] if (save_output_flags is not None) and (key in save_output_flags.keys()) else save_dict[key]
        
        self.cir_v0 = initialvar
        SAVE_HESTON = self.save_output_flags['HESTON']
        
        self.heston_data = sim_params.heston_dataC(kappa_dict, theta_dict, volofvar_dict, initialvar, rho_fx_v, rho_domestic_v, rho_base_v)
        
        for _ in range(self.nrsubsim):
            if (SAVE_HESTON):
                print("Allocating shmem ...", end='')
                self.Utsamples_raw.append(mp.Array(c.c_float, self.mult*self.n_paths_per_process*self.maxn_time, lock=False))
                print('HESTON UT', end=' ')
            print()

        if type(self) is StochasticLocalVolSimulation:
            self._allocate_shmem()
            print("Initializing pool for ...", type(self))
            self.simpool = multiprocessing.Pool(processes = self.nrsubsim, initializer=init_process, initargs=(self.t_stamp_raw, 
                                                                                                      self.Xsamples_raw, 
                                                                                                      self.rd_raw, 
                                                                                                      self.rb_raw, 
                                                                                                      self.xd_raw, 
                                                                                                      self.xb_raw, 
                                                                                                      self.dW_raw,
                                                                                                      self.volsamples_raw,
                                                                                                      self.Utsamples_raw,
                                                                                                      self.maxn_time, self.maxT))


#---------------------------------------------------------------------------
def init_process(t_stamp, Xsamples, rd, rb, xd, xb, dW, volsamples, Utsamples, maxn_time, maxT):
    
    smp.t_stamp_raw = t_stamp
    smp.Xsamples_raw = Xsamples
    smp.rd_raw = rd
    smp.rb_raw = rb
    smp.xd_raw = xd
    smp.xb_raw = xb
    smp.dW_raw = dW
    smp.volsamples_raw = volsamples
    smp.Utsamples_raw = Utsamples
    smp.maxn_time = maxn_time
    smp.maxT = maxT

#---------------------------------------------------------------------------
def simulate_paths(pid, n_trials, tdisargs, sdedata, g1pp_data, heston_data, lvol_data, det_DR = False, det_FR = True, DLV=True, in_DRN=True, 
                    in_forward_measure=False, antitheticpaths=True, dtype_glob='float32', save_output_flags=None):

    print("using passed and extracted g1pp_data for pid=%d"%pid)

    dim = 1
    T = tdisargs['maturity']
    n_time = tdisargs['time_steps']
    maxn_time = smp.maxn_time 
    dt = T/n_time #smp.maxT/smp.maxn_time

    in_forward_measure = ((det_DR==False) and in_forward_measure)
    
    print('in_forward_measure =%s, Antithetic=%s, in_DRN=%s, det_DR=%s, det_FR=%s, DLV=%s'%(in_forward_measure, antitheticpaths, in_DRN, det_DR, det_FR, DLV))

    MILSTEIN_SDE = False #True #self.MILSTEIN_SDE
    EULER_SDE = True #False

    if (save_output_flags is not None):
        SAVE_XS = save_output_flags['XS']
        SAVE_XD = save_output_flags['XD']
        SAVE_XB = save_output_flags['XB']
        SAVE_RD = save_output_flags['RD']
        SAVE_RB = save_output_flags['RB']
        SAVE_DW = save_output_flags['DW'] 
        SAVE_VOLTRAJ = save_output_flags['VOLTRAJ']
        if (DLV==False): SAVE_HESTON = save_output_flags['HESTON']
    else:
        SAVE_DW, SAVE_XD, SAVE_XB, SAVE_XS, SAVE_RD, SAVE_RB, SAVE_VOLTRAJ, SAVE_HESTON = False, False, False, False, False, False, False, False

    x0_range = [g1pp_data.spot_FX]*2
    
    #######################################################################################################
    domestic_shifttimes, domestic_ta, domestic_tvol = g1pp_data.domestic_shifttimes, g1pp_data.domestic_ta, g1pp_data.domestic_tvol
    base_shifttimes, base_ta, base_tvol = g1pp_data.base_shifttimes, g1pp_data.base_ta, g1pp_data.base_tvol

    domestic_shiftvalues, domestic_a, domestic_vol = g1pp_data.domestic_shiftvalues, g1pp_data.domestic_a, g1pp_data.domestic_vol
    base_shiftvalues, base_a, base_vol = g1pp_data.base_shiftvalues, g1pp_data.base_a, g1pp_data.base_vol
    #######################################################################################################

    bdtT = lambda t:(1-np.exp(-domestic_a[0]*(T-t)))/domestic_a[0]
    #######################################################################################################

    t_stamp = np.frombuffer(smp.t_stamp_raw[pid], dtype=dtype_glob)
    t_stamp[:n_time] = dt*np.arange(n_time)
    
    if (SAVE_VOLTRAJ):
        volsamples = np.frombuffer(smp.volsamples_raw[pid], dtype=dtype_glob).reshape((n_trials, maxn_time))
    
    if (SAVE_DW):
        dWs = np.frombuffer(smp.dW_raw[pid], dtype=dtype_glob).reshape((n_trials, maxn_time+1))
        dWs[:,0] = 0
    
    xds = np.ones(n_trials)*g1pp_data.domestic_x0
    xbs = np.ones(n_trials)*g1pp_data.base_x0
    if (SAVE_XD):
        xd = np.frombuffer(smp.xd_raw[pid], dtype=dtype_glob).reshape((n_trials, maxn_time))
        xd[:,0]= xds
    if (SAVE_XB):
        xb = np.frombuffer(smp.xb_raw[pid], dtype=dtype_glob).reshape((n_trials, maxn_time))
        xb[:,0]= xbs

    rds = xds + domestic_shiftvalues[0]
    rbs = xbs + base_shiftvalues[0]
    if (SAVE_RD):
        rd = np.frombuffer(smp.rd_raw[pid], dtype=dtype_glob).reshape((n_trials, maxn_time))
        rd[:,0] = rds
    if (SAVE_RB):
        rb = np.frombuffer(smp.rb_raw[pid], dtype=dtype_glob).reshape((n_trials, maxn_time))
        rb[:,0] = rbs

    ###################Asset Initialization###################
    Xs = np.random.uniform(low = x0_range[0], high=x0_range[1], size=n_trials) #spot_FX
    if (SAVE_XS):
        Xsamples = np.frombuffer(smp.Xsamples_raw[pid], dtype=dtype_glob).reshape((n_trials, maxn_time+1))
        Xsamples[:,0] = Xs
    lvol_spots, lvol_vols = np.array(lvol_data['spots']), np.array(lvol_data['surfacevalues'])
    tslices = np.array(lvol_data['times'])

    #####################################################
    
    if (DLV==False):
        Ut = np.ones(n_trials)*heston_data.initialvar
        kappa_times = np.array(heston_data.kappa_times)
        kappa_vals = heston_data.kappa_vals
        theta_times = np.array(heston_data.theta_times)
        theta_vals = heston_data.theta_vals
        volofvar_times = np.array(heston_data.volofvar_times)
        volofvar_vals = heston_data.volofvar_vals
        rho_fx_v = heston_data.rho_fx_v
        rho_domestic_v = heston_data.rho_domestic_v
        rho_base_v = heston_data.rho_base_v
        if (SAVE_HESTON):
            Utsamples = np.frombuffer(smp.Utsamples_raw[pid], dtype=dtype_glob).reshape((n_trials, maxn_time))
            Utsamples[:,0] = Ut
    else:
        rho_fx_v, rho_domestic_v, rho_base_v = 1, 1, 1
    
    covariances_mat = np.array([[1.0, sdedata.rho_domestic_fx, sdedata.rho_base_fx, rho_fx_v], 
                             [sdedata.rho_domestic_fx, 1.0, sdedata.rho_domestic_base, rho_domestic_v], 
                             [sdedata.rho_base_fx, sdedata.rho_domestic_base, 1.0, rho_base_v],
                             [rho_fx_v, rho_domestic_v, rho_base_v, 1.0]])

    #aoidx, doidx, foidx, uoidx = 0, 1, 2, 3
    #xpos, dpos, fpos, upos = 0, 1, 2, 3

    det_X = False
    
    stochastic_components = np.logical_not([det_X, det_DR, det_FR, DLV])
    cov_mat = covariances_mat[stochastic_components, :][:,stochastic_components]

    xpos, dpos, fpos, upos = tuple(np.cumsum(stochastic_components) - 1)
    meanval = [0.0]*np.sum(stochastic_components)

    print('Using covariance matrix, ', cov_mat)
    print('xpos, dpos, fpos, upos:', xpos, dpos, fpos, upos)
    ################################### Start the simulation ####################################
    search_index = lambda array_vals, val: min(len(array_vals)-1, np.searchsorted(array_vals, val))
    search_index_closest = lambda array_vals, val: (np.abs(array_vals - val)).argmin()
    
    volvals = np.zeros(Xsamples.shape[0])
    sig_d = domestic_vol[0]

    print('PID, Time, idx, TsliceIdx, TsliceTime, Volvals[0], Shapes: ', end='')
    if (DLV==False): print('Ut ', end='')
    print('Incr, Xdrift, Xs, idxvals, volvals, rds, rbs, xds, xbs')
    for idx, (t, dt) in enumerate(zip(t_stamp[1:n_time], np.diff(t_stamp[:n_time])), 1):
    
        if (idx%20==0):
            print(('{:2d}'+'{:8.3f}'+'{:5d}'*2+'{:8.3f}'*2).format(pid, t,idx, tidx, tslices[tidx], volvals[0]), end='')
            
        sqrtdt = np.sqrt(dt)
        if (antitheticpaths==True):
            samplevals = mvn(mean = meanval, cov=cov_mat, size=[int(0.5*n_trials),1])
            dW_sample = np.concatenate([samplevals, -1*samplevals], axis=0)
        else:
            dW_sample = mvn(mean = meanval, cov=cov_mat, size=[n_trials,1]) ###### generates random samples of size (n_trials x 1 x len(meanval))
        ####### save dWs to output
        if (SAVE_DW):
            dWs[:,idx] = dW_sample[:,0,xpos]

        
        ################# Local Volatility ################################
        dW = dW_sample[:,0, xpos]
        tidx = search_index(tslices, t)
        indexvals = np.argmin(np.abs(lvol_spots[tidx:tidx+1,:] - Xs.reshape(-1,1)), axis=1)
        volvals = lvol_vols[tidx, indexvals]
        if (DLV==False): ### SLV
            volvals = volvals*np.sqrt(Ut)
        ######### save localvol trajectory ###############################
        if (SAVE_VOLTRAJ):
            volsamples[:,idx-1] = volvals
        ###################################################################
            
        sig2o2 = volvals**2/2
        incr = volvals*dW*sqrtdt
        ####### save xsamples to output #####################
        Xdrift = rds - rbs
        if (in_forward_measure==True): Xdrift += -(sdedata.rho_domestic_fx*bdtT(t)*sig_d*volvals)
        if (MILSTEIN_SDE==True):
            Xs = Xs*(1.0 + (Xdrift + sig2o2*(dW**2-1))*dt + incr)
        elif (EULER_SDE==True):
            Xs = Xs*(1.0 + Xdrift*dt + incr)

        if (SAVE_XS):
            Xsamples[:,idx] = Xs
        
        
        ################# SLV ###########################################
        if (DLV==False):
            uidx = search_index(kappa_times, t)
            kappa_u = 0.0 if (uidx==0) else kappa_vals[uidx-1] #kappa_vals[uidx] #
            uidx = search_index(theta_times, t)
            theta_u = 0.0 if (uidx==0) else theta_vals[uidx-1] #theta_vals[uidx] #
            uidx = search_index(volofvar_times, t)
            volofvar_u = 0.0 if (uidx==0) else volofvar_vals[uidx-1] #volofvar_vals[uidx] #
            Udrift = kappa_u*(theta_u - Ut)*dt
            Uincr = volofvar_u*np.sqrt(Ut)*dW_sample[:,0, upos]*sqrtdt
            if (in_forward_measure==True): 
                didx = search_index(domestic_tvol, t)
                sig_d = 0.0 if (didx==0) else domestic_vol[didx-1]
                Udrift += -(rho_domestic_v*bdtT(t)*sig_d*volofvar_u*np.sqrt(Ut)*dt)
            Ut = Ut + Udrift + Uincr
            Ut = np.abs(Ut) #Ut[Ut<0] = 0 #-1*Ut[idxmask1]
            if (idx%20==0):
                print(Ut.shape, end='')
            if (SAVE_HESTON):
                Utsamples[:,idx] = Ut
        #################################################################
        
        ################## Domestic G1pp ####################################
        if (det_DR==False):
            didx = search_index(domestic_tvol, t)
            sig_d = 0.0 if (didx==0) else domestic_vol[didx-1]
            sigd2_2 = sig_d**2/2
            didx = search_index(domestic_ta, t)
            a_d = 0.0 if (didx==0) else domestic_a[didx-1]
            incrd = sig_d*dW_sample[:,0, dpos]*sqrtdt
            xddrift = (-a_d*xds*dt)
            ############# in T-Forward Measure ########################################
            if (in_forward_measure==True): xddrift += -(sig_d**2)*bdtT(t)*dt 
            ###########################################################################
            xds = xds + xddrift + incrd
            if (SAVE_XD):
                xd[:, idx] = xds
        
        ################## Foreign G1pp ####################################
        if (det_FR==False):
            bidx = search_index(base_tvol, t)
            sig_b = 0.0 if (bidx==0) else base_vol[bidx-1]
            sigb2_2 = sig_b**2/2
            bidx = search_index(base_ta, t)
            a_b = 0.0 if (bidx==0) else base_a[bidx-1]
            
            incrb = sig_b*dW_sample[:,0, fpos]*sqrtdt 
            xbdrift = (-a_b*xbs*dt)

            ############ in Domestic Risk-Neutral measure ###################################
            if (in_DRN==True): xbdrift += -sdedata.rho_base_fx*sig_b*volvals*dt 
            ###############################################################################
            if (in_forward_measure==True): xbdrift += -sdedata.rho_domestic_base*bdtT(t)*sig_b*sig_d*dt
            xbs = xbs + xbdrift + incrb
            if (SAVE_XB):
                xb[:, idx] = xbs
        #################################################

        didx = search_index(domestic_shifttimes, t)
        phi_d = 0.0 if (didx==0) else domestic_shiftvalues[didx-1]
        
        bidx = search_index(base_shifttimes, t)
        phi_b = 0.0 if (bidx==0) else base_shiftvalues[bidx-1]
        
        rds = xds + phi_d
        rbs = xbs + phi_b
        if (SAVE_RD):
            rd[:,idx] = rds
        if (SAVE_RB):
            rb[:,idx] = rbs

        if (idx%20==0):
            print(incr.shape, Xdrift.shape, Xs.shape, indexvals.shape, volvals.shape, rds.shape, rbs.shape, xds.shape, xbs.shape)

    ################################## last iteration outside the loop ##############################################
    idx+=1
    dW_sample = mvn(mean=meanval, cov=cov_mat, size=[n_trials,1]) ###### generates random samples of size (n_trials x 1 x 3)
    ####### save dWs to output
    if (SAVE_DW):
        dWs[:,idx] = dW_sample[:, 0, xpos]
    ##################*
    dW = dW_sample[:,0, xpos]
    tidx = search_index(tslices, t)
    indexvals = np.argmin(np.abs(lvol_spots[tidx:tidx+1,:] - Xs.reshape(-1,1)), axis=1)
    volvals = lvol_vols[tidx, indexvals]
    ####### save localvol to output
    if (SAVE_VOLTRAJ):
        volsamples[:,idx-1] = volvals
    ##################
    sig2o2 = volvals**2/2
    incr = volvals*dW*sqrtdt
    Xdrift = rds - rbs
    if (in_forward_measure==True): Xdrift += -(sdedata.rho_domestic_fx*bdtT(t)*sig_d*volvals)
    if (MILSTEIN_SDE==True):
        Xs = Xs*(1.0 + (Xdrift + sig2o2*(dW**2-1))*dt + incr)
    elif (EULER_SDE==True):
        Xs = Xs*(1.0 + Xdrift*dt + incr)
    ########################################################################################

    if (SAVE_XS):
        Xsamples[:,idx] = Xs

    print('PiD:%d returned to caller'%pid)
    return
