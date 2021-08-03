"Surface classes used in local volatility calibration"
import scipy.stats
import math
import bsanalytic as bsan
import interpolator

import numpy as np

class Surface:
    """
    Parent level surface class

    """
    def __init__(self, surface_interpolator, domestic_dfcurve, base_dfcurve, spot_FX, impvolobj=None, localvol_cap=1e20):
        self.surface_interpolator = surface_interpolator
        self.domestic_dfcurve = domestic_dfcurve
        self.base_dfcurve = base_dfcurve
        self.spot_FX = spot_FX
        self.localvol_cap = localvol_cap
        self.delta_t = 5e-4

    def get_fwd(self, t):
        """
        Returns the forward value of the underlier at the given time

        Parameters
        ----------
        t : float
            time

        Returns
        -------
        fwd : float
            Forward

        """
        domestic_df = self.domestic_dfcurve(t)
        base_df = self.base_dfcurve(t)
        rd = - math.log(domestic_df) / t
        rf = - math.log(base_df) / t
        erdt = base_df / domestic_df
        return self.spot_FX * erdt

    def get_atmf_vol(self, t):
        """
        Returns at the money forward implied volatility at the given time

        """
        raise Exception

    def create_strike_grid(self, t, nrstrikes, gridtype='uniform_in_y', width_nr_stdev=3.0):
        """
        Compute 1D strike grid for a given time using the ATMF vol

        Parameters
        ----------
        t : float
            Time
        nrstrikes : int
            Number of strikes in the grid
        gridtype : {'uniform_in_y'}
            Grid type
        width_nr_stdev : float
            Width of the generated grid in terms of standard deviations from ATMF.

        Returns
        -------
        strikes : list of floats
            1D strike grid

        """
        if gridtype == 'uniform_in_y':
            fwd = self.get_fwd(t)
            ref_impvol = self.get_atmf_vol(t)
            ymin = -width_nr_stdev * ref_impvol * math.sqrt(t)
            ymax = +width_nr_stdev * ref_impvol * math.sqrt(t)
            ys = list(np.arange(ymin, ymax, (ymax - ymin) / nrstrikes))[:nrstrikes]
            strikes = [fwd * math.exp(y) for y in ys]
        else:
            raise Exception('Gridtype %s not implemented' % gridtype)
        return strikes


class CallSurface(Surface):
    """
    Call price surface class

    """
    def __init__(self, surface_interpolator, domestic_dfcurve, base_dfcurve, spot_FX, localvol_cap=1e20):
        super(CallSurface, self).__init__(surface_interpolator, domestic_dfcurve, base_dfcurve, spot_FX, localvol_cap)    
        
    def evaluate_localvol_slice_det_ir(self, strikes, t):
        return [self.evaluate_localvol_deterministic_ir(K, t) for K in strikes]
    
    def evaluate_localvol_deterministic_ir(self, K, t):
        """
        Evaluate local volatility using standard Dupire's formula

        Parameters
        ----------
        K : float
            strike
        t : float
            time

        Returns
        -------
        localvol : float
            Local volatility

        """
        domestic_df = self.domestic_dfcurve(t)
        base_df = self.base_dfcurve(t)
        rd = -math.log(domestic_df) / t
        rf = -math.log(base_df) / t
        c = self.surface_interpolator.eval(K, t)
        d_c_d_t = self.surface_interpolator.eval(K, t, dt=1)
        d_c_d_K = self.surface_interpolator.eval(K, t, dx=1)
        d2_c_d_K2 = self.surface_interpolator.eval(K, t, dx=2)
        nom = d_c_d_t + (rd - rf) * K * d_c_d_K + rf * c
        denom = 0.5 * K * K * d2_c_d_K2
        return math.sqrt(min(self.localvol_cap, max(0.0, nom / denom)))

    def evaluate_localvol_slice_stoc_ir(self, strikes, t, expectations, stderrs):
        """
        Evaluate local volatility using Dupire's formula for stochastic rates
        for various strikes on a given time slice

        Parameters
        ----------
        strikes : list of floats
            strikes
        t : float
            time
        expectations : list of floats
            Values of the expectations in Dupire's formula
        stderrs : list of floats
            Standard error for the expectations

        Returns
        -------
        localvols, localvol_errs : list of floats, list of floats
            Local volatilities, Errors in local volatility

        """
        locvols = []
        locvol_errs = []
        for K,expect,stderr in zip(strikes, expectations, stderrs):
            locvol, locvol_err = self.evaluate_localvol_stochastic_ir(K, t, expect, stderr)
            locvols.append(locvol)
            locvol_errs.append(locvol_err)
        return locvols, locvol_errs
        #return [self.evaluate_localvol_stochastic_ir(K, t, expect) for K,expect in zip(strikes, expectations)]
    
    def evaluate_localvol_stochastic_ir(self, K, t, expectation, stderr):
        """
        Evaluate local volatility using Dupire's formula for stochastic rates

        Parameters
        ----------
        K : float
            strike
        t : float
            time
        expectation : float
            Value of the expectation in Dupire's formula
        stderr : float
            Standard error for the expectation

        Returns
        -------
        localvol, localvol_err : float, float
            Local volatility, Error in local volatility

        """
        domestic_df = self.domestic_dfcurve(t)
        base_df = self.base_dfcurve(t)
        rd = -math.log(domestic_df) / t
        rf = -math.log(base_df) / t
        c = self.surface_interpolator.eval(K, t)
        d_c_d_t = self.surface_interpolator.eval(K, t, dt=1)
        d_c_d_K = self.surface_interpolator.eval(K, t, dx=1)
        d2_c_d_K2 = self.surface_interpolator.eval(K, t, dx=2)
        nom = d_c_d_t - domestic_df * expectation
        nom_err = domestic_df * stderr
        denom = 0.5 * K * K * d2_c_d_K2
        locvol = math.sqrt(min(self.localvol_cap, max(0.0, nom / denom)))
        locvol_err = 0.5 * (nom_err / denom) / math.sqrt(locvol) if locvol > 0.0 else 0.0
        return locvol, locvol_err

    def get_atmf_vol(self, t):
        """
        Returns at the money forward implied volatility at the given time

        Parameters
        ----------
        t : float

        Returns
        -------
        atmfvol : float

        """
        domestic_df = self.domestic_dfcurve(t)
        base_df = self.base_dfcurve(t)
        rd = - math.log(domestic_df) / t
        rf = - math.log(base_df) / t
        erdt = base_df / domestic_df
        fwd = self.spot_FX * erdt
        ref_call = self.surface_interpolator.eval(fwd, t)
        return bsan.impliedvol_call(self.spot_FX, fwd, t, rd, rf, ref_call, 0.1)

    
class TIVSurface(Surface):
    """
    Total implied variance surface class

    """
    def __init__(self, surface_interpolator, impvolobj, domestic_dfcurve, base_dfcurve, spot_FX, localvol_cap=1e20):
        super(TIVSurface, self).__init__(surface_interpolator, domestic_dfcurve, base_dfcurve, spot_FX, localvol_cap)
        self.fxvolsurf = impvolobj

    def evaluate_localvol_slice_det_ir(self, strikes, t):
        return [self.evaluate_localvol_deterministic_ir(K, t) for K in strikes]
    
    def evaluate_localvol_deterministic_ir(self, K, t):
        """
        Evaluate local volatility using standard Dupire's formula

        Parameters
        ----------
        K : float
            strike
        t : float
            time

        Returns
        -------
        localvol : float
            Local volatility

        """
        domestic_df = self.domestic_dfcurve(t)
        base_df = self.base_dfcurve(t)
        erdt = base_df / domestic_df
        fwd = self.spot_FX * erdt
        y = math.log(K / fwd)
        w = self.surface_interpolator.eval(y, t)
        d_w_d_t = self.surface_interpolator.eval(y, t, dt=1)
        d_w_d_y = self.surface_interpolator.eval(y, t, dx=1)
        d2_w_d_y2 = self.surface_interpolator.eval(y, t, dx=2)
        nom = d_w_d_t
        denom = 1.0 - d_w_d_y * y / w + 0.5 * d2_w_d_y2 + 0.25 * (-0.25 - 1.0 / w + y ** 2 / w ** 2) * d_w_d_y ** 2
        return math.sqrt(min(self.localvol_cap, max(0.0, nom / denom)))
    

    def evaluate_localvol_slice_stoc_ir(self, strikes, t, expectations, stderrs):
        """
        Evaluate local volatility using Dupire's formula for stochastic rates
        for various strikes on a given time slice

        Parameters
        ----------
        strikes : list of floats
            strikes
        t : float
            time
        expectations : list of floats
            Values of the expectations in Dupire's formula
        stderrs : list of floats
            Standard error for the expectations

        Returns
        -------
        localvols, localvol_errs : list of floats, list of floats
            Local volatilities, Errors in local volatility

        """
        locvols = []
        locvol_errs = []
        for K,expect,stderr in zip(strikes, expectations, stderrs):
            locvol, locvol_err = self.evaluate_localvol_stochastic_ir(K, t, expect, stderr)
            locvols.append(locvol)
            locvol_errs.append(locvol_err)
        return locvols, locvol_errs

    
    def evaluate_localvol_stochastic_ir(self, K, t, expectation, stderr):
        """
        Evaluate local volatility using Dupire's formula for stochastic rates

        Parameters
        ----------
        K : float
            strike
        t : float
            time
        expectation : float
            Value of the expectation in Dupire's formula
        stderr : float
            Standard error for the expectation

        Returns
        -------
        localvol, localvol_err : float, float
            Local volatility, Error in local volatility

        """
        domestic_df = self.domestic_dfcurve(t)
        base_df = self.base_dfcurve(t)
        domestic_shortrate = interpolator.constructShortRateCurve(self.domestic_dfcurve)
        rd = domestic_shortrate(t)
        base_shortrate = interpolator.constructShortRateCurve(self.base_dfcurve)
        rf = base_shortrate(t)
        mu_T = rd - rf
        erdt = base_df / domestic_df
        fwd = self.spot_FX * erdt
        y = math.log(K / fwd)
        e_y = math.exp(y)
        w = self.surface_interpolator.eval(y, t)
        d_w_d_t = self.surface_interpolator.eval(y, t, dt=1)
        d_w_d_y = self.surface_interpolator.eval(y, t, dx=1)
        d2_w_d_y2 = self.surface_interpolator.eval(y, t, dx=2)
        d_1 = -y / math.sqrt(w) + 0.5 * math.sqrt(w)
        d_2 = d_1 - math.sqrt(w)
        call = fwd * (scipy.stats.norm.cdf(d_1) - e_y * scipy.stats.norm.cdf(d_2))
        d_call_d_w = 0.5 * fwd * e_y * scipy.stats.norm.pdf(d_2) / math.sqrt(w)
        d_call_d_y = -fwd * e_y * scipy.stats.norm.cdf(d_2)

        nom = -mu_T * (d_call_d_y + d_call_d_w * d_w_d_y) - rf * call + d_call_d_w * d_w_d_t - expectation
        nom_err = stderr
        denom = d_call_d_w * (1.0 - d_w_d_y * y / w + 0.5 * d2_w_d_y2 + 0.25 * (-0.25 - 1.0 / w + y ** 2 / w ** 2) * d_w_d_y ** 2)

        locvol = math.sqrt(min(self.localvol_cap, max(0.0, nom / denom)))
        locvol_err = 0.5 * (nom_err / denom) / math.sqrt(locvol) if locvol > 0.0 else 0.0
        return locvol, locvol_err

    def get_atmf_vol(self, t):
        """
        Returns at the money forward implied volatility at the given time

        Parameters
        ----------
        t : float

        Returns
        -------
        atmfvol : float

        """
        ref_totimpvar = self.surface_interpolator.eval(0.0, t)
        return math.sqrt(ref_totimpvar / t)


class LVSurface(Surface):
    """
    Wrapper to evaluate a given local volatility surface through interpolation

    """
    def __init__(self, surface_interpolator, domestic_dfcurve, base_dfcurve, spot_FX, localvol_cap=1e20):
        super(LVSurface, self).__init__(surface_interpolator, domestic_dfcurve, base_dfcurve, spot_FX, localvol_cap)

    def evaluate_localvol_deterministic_ir(self, K, t):
        """
        Evaluate local volatility at the given strike and time

        Parameters
        ----------
        K : float
            strike
        t : float
            time

        Returns
        -------
        localvol : float
            Local volatility

        """
        return max(0.0, self.surface_interpolator.eval(K, t)) # localvol_cap is ignored for now

    def evaluate_localvol_slice_det_ir(self, strikes, t):
        return [self.evaluate_localvol_deterministic_ir(K, t) for K in strikes]

    def evaluate_localvol_stochastic_ir(self, K, t, expectation, stderr):
        """
        Evaluate local volatility at the given strike and time

        Parameters
        ----------
        K : float
            strike
        t : float
            time
        expectation : ignored
        stderr : ignored

        Returns
        -------
        localvol, 0.0 : float, float
            Local volatility, 0.0

        """
        return max(0.0, self.surface_interpolator.eval(K, t)), 0.0 # localvol_cap is ignored for now

    def get_atmf_vol(self, t): # not really the ATMF implied vol but should do for our purposes
        """
        Returns at the money forward local volatility at the given time

        Parameters
        ----------
        t : float

        Returns
        -------
        vol : float

        """
        fwd = self.get_fwd(t)
        return self.evaluate_localvol_deterministic_ir(fwd, t)
        

def generate_call_price_surface(impvolobj, strikes, times, domestic_dfcurve, base_dfcurve, spot, localvol_cap=1e20):
    """
    Generates a uniform call price surface for local volatility evaluation

    Parameters
    ----------
    impvolobj : 
        Implied vol surface
    strikes : list of floats
        Grid of strikes
    times : list of floats
        Grid of times
    domestic_dfcurve : 
        (domestic) discount curve
    base_dfcurve : 
        base discount curve or dividend yield curve
    spot : float
        Time zero value of the underlier
    localvol_cap : float
        Cap for local volatility

    Returns
    -------
    calls : :class:`CallSurface`
        Surface for local volatility evaluation

    """
    interp = interpolator.generate_call_price_interpolator(impvolobj, strikes, times, domestic_dfcurve, base_dfcurve, spot)
    return CallSurface(interp, domestic_dfcurve, base_dfcurve, spot, localvol_cap)

def generate_call_price_surface_nonuniform(impvolobj, nrstrikes, times, domestic_dfcurve,
                                           base_dfcurve, spot, localvol_cap=1e20,
                                           width_nr_stdev=3.5):
    """
    Generates a nonuniform call price surface for local volatility evaluation

    Parameters
    ----------
    impvolobj : 
        Implied vol surface
    nrstrikes : list of floats
        Number of strikes
    times : list of floats
        Grid of times
    domestic_dfcurve : 
        (domestic) discount curve
    base_dfcurve : 
        base discount curve or dividend yield curve
    spot : float
        Time zero value of the underlier
    localvol_cap : float
        Cap for local volatility
    width_nr_stdev : float
        Width of the generated grid in terms of standard deviations from ATMF

    Returns
    -------
    calls : :class:`CallSurface`
        Surface for local volatility evaluation

    """
    interp = interpolator.generate_call_price_interpolator_nonuniform(impvolobj, nrstrikes, times, domestic_dfcurve, base_dfcurve, spot, width_nr_stdev)
    return CallSurface(interp, domestic_dfcurve, base_dfcurve, spot, localvol_cap)

def generate_tiv_surface(impvolobj, ys, times, domestic_dfcurve, base_dfcurve, spot, localvol_cap=1e20):
    """
    Generates a uniform total implied variance surface for local volatility
    evaluation

    Parameters
    ----------
    impvolobj : 
        Implied vol surface
    ys : list of floats
        Grid of log-moneynesses
    times : list of floats
        Grid of times
    domestic_dfcurve : 
        (domestic) discount curve
    base_dfcurve : 
        base discount curve or dividend yield curve
    spot : float
        Time zero value of the underlier
    localvol_cap : float
        Cap for local volatility

    Returns
    -------
    calls : :class:`TIVSurface`
        Surface for local volatility evaluation

    """
    interp = interpolator.generate_implied_totalvariance_interpolator(impvolobj, ys, times)
    return TIVSurface(interp, domestic_dfcurve, base_dfcurve, spot, localvol_cap)

def generate_tiv_surface_nonuniform(impvolobj, nrys, times, domestic_dfcurve, base_dfcurve, spot, localvol_cap=1e20,
                                    width_nr_stdev=3.5):
    """
    Generates a nonuniform total implied volatility surface for local volatility evaluation

    Parameters
    ----------
    impvolobj : 
        Implied vol surface
    nrstrikes : list of floats
        Number of log-moneynesses
    times : list of floats
        Grid of times
    domestic_dfcurve : 
        (domestic) discount curve
    base_dfcurve : 
        base discount curve or dividend yield curve
    spot : float
        Time zero value of the underlier
    localvol_cap : float
        Cap for local volatility
    width_nr_stdev : float
        Width of the generated grid in terms of standard deviations from ATMF

    Returns
    -------
    calls : :class:`TIVSurface`
        Surface for local volatility evaluation

    """
    interp = interpolator.generate_implied_totalvariance_interpolator_nonuniform(impvolobj, nrys, times, width_nr_stdev)
    return TIVSurface(interp, impvolobj, domestic_dfcurve, base_dfcurve, spot, localvol_cap)

def generate_lv_surface_from_values_nonuniform(strikes, times, localvols, domestic_dfcurve, base_dfcurve, spot, localvol_cap=1e20):
    """
    Generates a nonuniform local volatility surface for direct evaluation

    Parameters
    ----------
    strikes : list of lists of floats
        Grid of strikes
    times : list of floats
        Grid of times
    localvols : list of lists of floats
        Grid of local volatilities. Must be of the same shape with *strikes*
    domestic_dfcurve : 
        (domestic) discount curve
    base_dfcurve : 
        base discount curve or dividend yield curve
    spot : float
        Time zero value of the underlier
    localvol_cap : float
        Cap for local volatility

    Returns
    -------
    calls : :class:`LVSurface`
        Surface for local volatility evaluation

    """
    interp = interpolator.SurfaceInterpolator(strikes, times, localvols)
    return LVSurface(interp, domestic_dfcurve, base_dfcurve, spot, localvol_cap)

def generate_2d_strike_grid(surface, nrstrikes, times, rectangular, width_nr_stdev=3.5):
    """
    Compute 2D strike grid for given times and number of strikes

    Parameters
    ----------
    surface : :class:`Surface`
        Surface to query for the 1D subgrids
    nrstrikes : int
        Number of strikes in the 1D subgrids
    times : list of floats
        Times
    rectangular : bool
        The shape of the output grid
    width_nr_stdev : float
        Width of the generated grid in terms of standard deviations from ATMF.

    Returns
    -------
    strikes : list of lists of floats
        2D strike grid

    """
    strike_matrix = []
    if rectangular:
        finaltime = times[-1]
        strike_grid = surface.create_strike_grid(finaltime, nrstrikes, width_nr_stdev=width_nr_stdev)
        for t in times:
            strike_matrix.append(strike_grid[:])
    else:
        for t in times:
            strike_grid = surface.create_strike_grid(t, nrstrikes, width_nr_stdev=width_nr_stdev)
            strike_matrix.append(strike_grid)
    return strike_matrix
