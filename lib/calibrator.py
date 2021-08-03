"""Local volatility calibration utilities under stochastic interest rates and/or
stochastic volatility"""
import inspect
import math
import os
import pickle
import statistics

import numpy as np
import interpolator

def split_seq(seq, size):
    """
    Splits a given list evenly to sublists of roughly the same size

    Parameters
    ----------
    seq : list
        Original sequence
    size : int
        Number of sublists

    Returns
    -------
    newseq : list of lists

    """
    newseq = []
    splitsize = 1.0/size*len(seq)
    for i in range(size):
        newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
    return newseq

class LocalVolIRCalibration:

    def __init__(self, surface, simulator, filename=None):
        self.surface = surface
        self.simulator = simulator
        self.filename = filename

    def save_lv_surface(self, all_strikes, times, surfacevalues, surface_errs=None):
        """
        Save the local vol surface data

        Parameters
        ----------
        all_strikes : list of floats
            List of strikes
        times : list of floats
            List of times
        surfacevalues : list of lists of floats
            Vol surface data
        surface_errs : list of lists of floats
            Vol surface errors

        """
        if self.filename:
            dirname = os.path.dirname(self.filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            lvol_output = { 'spots': all_strikes,
                            'times': times,
                            'surfacevalues': surfacevalues,
                           }
            if surface_errs:
                lvol_output['errors'] = surface_errs
            with open(self.filename, 'wb') as f:
                pickle.dump(lvol_output, f)
            print("Written: %s" % (self.filename))

class DeterministicLocalVolDeterministicIRCalibration(LocalVolIRCalibration):


    def __init__(self, surface, simulator, filename=None):
        super(DeterministicLocalVolDeterministicIRCalibration, self).__init__(surface, simulator,
                                                                              filename=filename,
                                                                              )

    def calibrate_localvol(self, Ks, ts):
        """
        Do the calibration

        Parameters
        ----------
        Ks : list of lists of floats
            2D strike grid
        ts : list of floats
            Time grid

        Returns
        -------
        locvols : list of lists of floats
            The local volatility surface

        """
        if len(Ks) != len(ts):
            raise Exception("Strike grid length does not match times grid length")
        locvols = []
        all_strikes = Ks
        for idx, t, in enumerate(ts):
            print("Processing time: %f" % t)
            strikes = all_strikes[idx]
            locvol_slice = self.surface.evaluate_localvol_slice_det_ir(strikes, t)
            locvols.append(locvol_slice)
            self.save_lv_surface(all_strikes[:idx+1], ts[:idx+1], locvols)

        return locvols


class DeterministicLocalVolStochasticIRCalibration(LocalVolIRCalibration):


    def __init__(self, surface, simulator, filename=None):
        super(DeterministicLocalVolStochasticIRCalibration, self).__init__(surface, simulator,
                                                                           filename=filename,
                                                                           )
        self.deterministic_ir_calibration = DeterministicLocalVolDeterministicIRCalibration(surface, simulator,
                                                                                            filename=filename,
                                                                                            )

    def calibrate_localvol(self, Ks, ts):
        """
        Do the calibration

        Parameters
        ----------
        Ks : list of lists of floats
            2D strike grid
        ts : list of floats
            Time grid

        Returns
        -------
        locvols : list of lists of floats
            The local volatility surface

        """
        if len(Ks) != len(ts):
            raise excp.InvalidLengthException("Strike grid length does not match times grid length")
        locvols = []
        locvol_errs = []
        all_strikes = Ks
        for idx, t, in enumerate(ts):
            print("Processing time: %f" % t)
            strikes = all_strikes[idx]
            if idx == 0:
                locvol_slice = self.deterministic_ir_calibration.surface.evaluate_localvol_slice_det_ir(strikes, t)
                locvol_err_slice = [0.0] * len(locvol_slice)
            else:
                self.simulator.set_contracts_for_calibration(strikes, t)
                self.simulator.update_localvol(all_strikes[:idx], ts[:idx], locvols)
                self.simulator.update_domestic_bfactors(t)
                self.simulator.update_base_bfactors(t)
                self.simulator.run()
                expectations = self.simulator.get_means_from_output()
                stderrs = self.simulator.get_stderrs_from_output()
                locvol_slice, locvol_err_slice = self.surface.evaluate_localvol_slice_stoc_ir(strikes, t, expectations, stderrs)
            locvols.append(locvol_slice)
            locvol_errs.append(locvol_err_slice)
            self.save_lv_surface(all_strikes[:idx+1], ts[:idx+1], locvols, locvol_errs)

        return locvols

class StochasticLocalVolDeterministicIRCalibration(LocalVolIRCalibration):


    def __init__(self, surface, simulator, filename=None, var_method='curve_fit'):
        super(StochasticLocalVolDeterministicIRCalibration, self).__init__(surface, simulator,
                                                                           filename=filename,
                                                                           )
        self.var_method = var_method # curve_fit binning

    def calibrate_localvol(self, Ks, ts):
        """
        Do the calibration

        Parameters
        ----------
        Ks : list of lists of floats
            2D strike grid
        ts : list of floats
            Time grid

        Returns
        -------
        all_Ls : list of lists of floats
            The leverage surface

        """
        if len(Ks) != len(ts):
            raise excp.InvalidLengthException("Strike grid length does not match times grid length")
        all_strikes = Ks
        #all_locvols = []
        all_Ls = []
        for idx, t, in enumerate(ts):
            print("Processing time: %f" % t)
            strikes = all_strikes[idx]
            locvol_slice = self.surface.evaluate_localvol_slice_det_ir(strikes, t)
            if idx == 0:
                locvol_S0 = interpolator.getLinearInterpEx(self.simulator.spot_FX, strikes, locvol_slice)
                L0 = locvol_S0 / math.sqrt(self.simulator.cir_v0)
                Ls = [L0] * len(strikes)
                #all_locvols.append(locvol_slice)
                all_Ls.append(Ls)
            if self.var_method in ('curve_fit', 'binning'):
                self.simulator.update_localvol([all_strikes[0]] + all_strikes[:idx], [0.0] + ts[:idx], all_Ls)
                self.simulator.set_simulation_times(ts[:idx + 1])
                self.simulator.set_observation_times([t])
                self.simulator.set_vanilla_contract('spot', t, None)
                self.simulator.run()
    
                cspots, cvars = self.simulator._get_X_and_U()
    
                if self.var_method == 'curve_fit':
                    Ls = self.compute_L_curve_fit(cspots, cvars, strikes, t)
                else: # elif self.var_method == 'binning':
                    Ls = self.compute_L_simple_bin_average(cspots, cvars, strikes, t)
            else: 
                raise Exception("Method "+self.var_method+" is not implemented!")

            all_Ls.append(Ls)

            self.save_lv_surface([all_strikes[0]] + all_strikes[:idx+1],
                                 [0.0] + ts[:idx+1],
                                 all_Ls)

        return all_Ls
    
    def compute_L_curve_fit(self, spots, variances, strikes, time):
        """
        Compute the leverage function by estimating the conditional expectation
        on variance by least squares regression

        Parameters
        ----------
        spots : list of floats
            List of underlier values
        variances : list of floats
            List of variance process values. Must be of the same length
            as *spots*
        strikes : list of floats
            Strikes grid to compute the leverage grid at
        time : float
            Time for the calibration slice

        Returns
        -------
        Ls : list of floats
            The leverage surface slice for the given time

        """
        import numpy as np
        import scipy.optimize as scopt
        # Monomial
        #ls_eqn = lambda x, a, b, c: a * x**2 + b * x + c
        # Laguerre
        ls_eqn = lambda x, a, b, c: a * 0.5 * (x**2 - 4 * x + 2) + b * (1 - x) + c
        popt, pcov = scopt.curve_fit(ls_eqn, spots, variances)
        cstrikes = np.linspace(min(spots), max(spots), len(strikes))
        
        vars = np.array([max(ls_eqn(strike, popt[0], popt[1], popt[2]), 1e-6) for strike in cstrikes])
        locvol = np.array(self.surface.evaluate_localvol_slice_det_ir(cstrikes, time))
        cLs = locvol/np.sqrt(vars)
        
        Ls = [interpolator.getLinearInterpEx(strike, cstrikes, cLs) for strike in strikes]
        return Ls

    def compute_L_simple_bin_average(self, spots, variances, strikes, time):
        """
        Compute the leverage function by estimating the conditional expectation
        on variance by binning

        Parameters
        ----------
        spots : list of floats
            List of underlier values
        variances : list of floats
            List of variance process values. Must be of the same length
            as *spots*
        strikes : list of floats
            Strikes grid to compute the leverage grid at
        time : float
            Time for the calibration slice

        Returns
        -------
        Ls : list of floats
            The leverage surface slice for the given time

        """
        fpairs = sorted(zip(spots, variances))
        chunked_fpairs = split_seq(fpairs, len(strikes))
        
        
        cstrikes_and_vars = [[np.mean(x) for x in zip(*C)] for C in chunked_fpairs ]
        cstrikes, cvars = zip(*cstrikes_and_vars)
        locvol = np.array(self.surface.evaluate_localvol_slice_det_ir(cstrikes, time))
        cLs = locvol/np.sqrt(cvars)

        Ls = [interpolator.getLinearInterpEx(strike, cstrikes, cLs) for strike in strikes]
        return Ls


class StochasticLocalVolStochasticIRCalibration(StochasticLocalVolDeterministicIRCalibration):


    def __init__(self, surface, simulator, filename=None, var_method='curve_fit'):
        super(StochasticLocalVolStochasticIRCalibration, self).__init__(surface, simulator,
                                                                        filename=filename,
                                                                        var_method=var_method,
                                                                        )

    def calibrate_localvol(self, Ks, ts, localvols=None):
        """
        Do the calibration

        Parameters
        ----------
        Ks : list of lists of floats
            2D strike grid
        ts : list of floats
            Time grid
        localvols : list of lists of floats
            Given precalibrated deterministic local volatilities, e.g. with
            :class:`DeterministicLocalVolStochasticIRCalibration`.

        Returns
        -------
        all_Ls : list of lists of floats
            The leverage surface

        """
        if len(Ks) != len(ts):
            raise excp.InvalidLengthException("Strike grid length does not match times grid length")
        all_strikes = Ks
        all_Ls = []
        for idx, t, in enumerate(ts):
            print("Processing time: %f" % t)
            strikes = all_strikes[idx]
            if localvols:
                locvol_slice = localvols[idx]
            else:
                locvol_slice = self.surface.evaluate_localvol_slice_det_ir(strikes, t)
            if idx == 0:
                locvol_S0 = interpolator.getLinearInterpEx(self.simulator.spot_FX, strikes, locvol_slice)
                L0 = locvol_S0 / math.sqrt(self.simulator.cir_v0)
                Ls = [L0] * len(strikes)
                all_Ls.append(Ls)
            self.simulator.update_localvol([all_strikes[0]] + all_strikes[:idx], [0.0] + ts[:idx], all_Ls) # nonuniform
            self.simulator.update_domestic_bfactors(t)
            self.simulator.update_base_bfactors(t)
            self.simulator.set_simulation_times(ts[:idx + 1])
            self.simulator.set_observation_times([t])
            self.simulator.set_vanilla_contract('spot', t, None)
            self.simulator.run()

            obs_result = self.simulator.outputMC['MCResults']['SimulationObservations']
            cspots = [path[self.simulator.fx_prefix + 'FXrate'][0] for path in obs_result["Samples"]]
            cvars = [path[self.simulator.fx_prefix + 'FXStochVariance'][0] for path in obs_result["Samples"]]

            if self.var_method == 'curve_fit':
                Ls = self.compute_L_curve_fit(cspots, cvars, strikes, t)
            else: # binning
                Ls = self.compute_L_simple_bin_average(cspots, cvars, strikes, t)
            all_Ls.append(Ls)

            self.save_lv_surface([all_strikes[0]] + all_strikes[:idx+1],
                                 [0.0] + ts[:idx+1],
                                 all_Ls)

        return all_Ls


class StochasticLocalVolStochasticIRCalibrationMultiRegression(StochasticLocalVolDeterministicIRCalibration):


    def __init__(self, surface, simulator, filename=None, var_method='curve_fit'):
        super(StochasticLocalVolStochasticIRCalibrationMultiRegression, self).__init__(surface, simulator,
                                                                                       filename=filename,
                                                                                       var_method=var_method,
                                                                                       )

    def calibrate_localvol(self):
        """
        Do the calibration. Uses the same grid as the given localvol surface.

        Returns
        -------
        regcoeffs, rsq : list of lists of floats, list of floats
            Regression coefficients and R-squares from regression for each time
            slice

        """
        all_popt = []
        all_rsq = []
        Ks = self.surface.surface_interpolator.xs
        ts = self.surface.surface_interpolator.ts
        locvols = self.surface.surface_interpolator.vols
        self.simulator.update_localvol(Ks, ts, locvols)
        for idx, t, in enumerate(self.surface.surface_interpolator.ts):
            print("Processing time: %f" % t)
            strikes = Ks[idx]
            if idx == 0:
                locvol_slice = self.surface.surface_interpolator.vols[idx]
                locvol_S0 = interpolator.getLinearInterpEx(self.simulator.spot_FX, strikes, locvol_slice)
                L0 = self.simulator.cir_v0
                all_popt.append([L0] + [0.0] * (len(inspect.signature(self.simulator.regression_equation).parameters) - 2)) # the length must match the total number of basis functions
                all_rsq.append(1.0)
            self.simulator.update_regression_coefficients(ts[:idx + 1], all_popt, all_rsq, self.simulator.nrmcruns)
            self.simulator.update_domestic_bfactors(t)
            self.simulator.update_base_bfactors(t)
            self.simulator.set_simulation_times(ts[:idx + 1])
            self.simulator.set_observation_times([t])
            self.simulator.set_vanilla_contract('spot', t, None)
            self.simulator.run()

            obs_result = self.simulator.outputMC['MCResults']['SimulationObservations']
            cspots = [path[self.simulator.fx_prefix + 'FXrate'][0] for path in obs_result["Samples"]]
            cvars = [path[self.simulator.fx_prefix + 'FXStochVariance'][0] for path in obs_result["Samples"]]
            cdomxs = [path[self.simulator.domestic_currency_prefix + 'Domestic_XFactor'][0] for path in obs_result["Samples"]]
            cbasxs = [path[self.simulator.base_currency_prefix + 'Base_XFactor'][0] for path in obs_result["Samples"]]

            if self.var_method == 'curve_fit':
                popt, rsq = self.compute_L_curve_fit(cspots, cdomxs, cbasxs, cvars, len(strikes), t)
            else:
                raise excp.NotImplementedException("Method %s not implemented" % (self.var_method))
            all_popt.append(popt)
            all_rsq.append(rsq)

            self.save_regression_coefficients([0.0] + ts[:idx+1], all_popt, all_rsq)

        return all_popt, all_rsq

    def save_regression_coefficients(self, times, regcoeffs, rsq):
        """
        Save regression coefficients in a pickle file. The file name is given in
        the constructor.

        Parameters
        ----------
        times : list of floats
            Time slices
        regcoeffs : list of lists of floats
            Regression coefficients for each time slice
        rsq : list of floats
            R-squares from regression results for each time slice

        """
        if self.filename:
            dirname = os.path.dirname(self.filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            l_output = {'times': times,
                        'regcoeffs': regcoeffs,
                        'R2': rsq,
                        'nrdatapoints': self.simulator.nrmcruns,
                        }

            with open(self.filename, 'wb') as f:
                pickle.dump(l_output, f)
            print("Written: %s" % (self.filename))

    def compute_L_curve_fit(self, spots, domxs, basxs, variances, nr_chunks, time):
        """
        Compute the leverage function by estimating the conditional expectation
        on variance by least squares regression. Returns the regression results.

        Parameters
        ----------
        spots : list of floats
            List of underlier values
        variances : list of floats
            List of variance process values. Must be of the same length
            as *spots*
        nr_chunks : int
            Number of bins. Corresponds to the size of the output strikes grid
        time : float
            Time for the calibration slice

        Returns
        -------
        reg, rsq : List of floats, float
            Regression coefficients and R-square from regression

        """
        import numpy as np
        import scipy.optimize as scopt

        ls_eqn = self.simulator.regression_equation
        popt, pcov = scopt.curve_fit(ls_eqn, (spots, domxs, basxs), variances)
        residuals = np.asarray(variances) - ls_eqn((np.asarray(spots), np.asarray(domxs), np.asarray(basxs)), *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((np.asarray(variances) - np.mean(variances))**2)
        rsq = 1.0 - (ss_res / ss_tot)
        return popt.tolist(), rsq
