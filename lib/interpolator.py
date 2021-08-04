"Uniform and nonuniform surface interpolators"
import scipy.interpolate
import math

import numpy as np
import bsanalytic as bsan
import scipy
import bisect
import datetime


class InterpolatedCurve:
    def __init__(self, times, values, interpolation_fnc=None):
        self.times = times
        self.values = values
        ASOFDATE=datetime.datetime(2020, 4, 30)
        self.dates = [ASOFDATE+datetime.timedelta(days=timeval * 365) for timeval in times]
        if (interpolation_fnc is None):
            self.interpolation_fnc = getLinearInterpEx
        elif interpolation_fnc.lower() == 'pwc_left_cts':
            self.interpolation_fnc = getLeftConstantInterpEx
        elif interpolation_fnc.lower() == 'loglinear':
            self.interpolation_fnc = getLogLinearInterpEx
        return

    def __call__(self, dateortimes):
        if isinstance(dateortimes, list) or isinstance(dateortimes, np.ndarray):
            return [self.interpolation_fnc(tval, self.times, self.values) for tval in dateortimes]
        else:
            return self.interpolation_fnc(dateortimes, self.times, self.values)


class SurfaceInterpolator:
    """
    Nonuniform surface interpolator based on scipy implementation.
    x values are interpolated with the splines of the given degree.
    t values are interpolated linearly.

    Parameters
    ----------
    xs : list of lists of floats
        Nonuniform grid of x values (e.g. strikes)
    ts : list of floats
        Grid of t values (e.g. times).
    vols : list of lists of floats
        Nonuniform grid of values to be interpolated (e.g. volatilities).
        Must be of the same shape with *xs*
    k : int
        Degree of spline to interpolate the x values

    """
    def __init__(self, xs, ts, vols, k=3):
        self.xs = xs
        self.ts = ts
        self.vols = vols
        if len(ts) != len(xs):
            raise Exception("ts length must match xs length")
        if len(ts) != len(vols):
            raise Exception("ts length must match vols length")
        self.interp = []
        for idx, t in enumerate(ts):
            slice_interp = scipy.interpolate.CubicSpline(xs[idx], vols[idx], bc_type='natural') ## 'clamped'
            self.interp.append(slice_interp)

    def eval(self, x, t, dx=0, dt=0):
        """
        Interpolate the surface at the given point

        Parameters
        ----------
        x : float
        t : float
        dx : int
            Degree of derivative in the x direction
        dt : int
            Degree of derivative in the y direction

        Returns
        -------
        out : float
            Interpolated value

        """
        if t < self.ts[0]:
            return self.interp[0](x, nu=dx) if dt == 0 else 0.0
        elif self.ts[-1] < t:
            return self.interp[-1](x, nu=dx) if dt == 0 else 0.0
        else:
            interp_vols = [self.interp[idx](x, nu=dx) for idx in range(len(self.ts))]
            times_interp = scipy.interpolate.InterpolatedUnivariateSpline(self.ts, interp_vols, k=1)
            return times_interp(t, nu=dt).tolist() # tolist needs to be used to convert it to float for some reason

class SurfaceInterpolatorInterp2D:
    """
    Uniform surface interpolator based on scipy implementation.

    Parameters
    ----------
    xs : list of floats
        Grid of x values (e.g. strikes)
    ts : list of floats
        Grid of t values (e.g. times).
    vols : list of lists of floats
        Uniform grid of values to be interpolated (e.g. volatilities)
    kind : str
        Interpolation type. See :func:`scipy.interpolate.interp2d` documentation

    """
    def __init__(self, xs, ts, vols, kind='cubic'):
        self.xs = xs
        self.ts = ts
        self.vols = vols
        self.interp = scipy.interpolate.interp2d(xs, ts, vols, kind=kind, copy=False)

    def eval(self, x, t, dx=0, dt=0):
        """
        Interpolate the surface at the given point

        Parameters
        ----------
        x : float
        t : float
        dx : int
            Degree of derivative in the x direction
        dt : int
            Degree of derivative in the y direction

        Returns
        -------
        out : float
            Interpolated value

        """
        return self.interp(x, t, dx, dt)[0]

def generate_call_price_interpolator(impvolobj, strikes,
                                     times, domestic_dfcurve, base_dfcurve, spot):
    """
    Generate a uniform call price surface

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

    Returns
    -------
    interp : :class:`SurfaceInterpolatorInterp2D`
        Interpolator object

    """
    print("Creating call price interpolator")
    allcalls = []
    for T in times:
        callslice = []
        domestic_df = domestic_dfcurve(T)
        base_df = base_dfcurve(T)
        rd = - math.log(domestic_df) / T
        rf = - math.log(base_df) / T
        for strike in strikes:
            impvol = impvolobj.impliedvol_K(T, strike)
            call = max(0.0, bsan.Call(spot, strike, T, rd, rf, impvol))
            callslice.append(call)
        allcalls.append(callslice)
    call_interp = SurfaceInterpolatorInterp2D(strikes, times, allcalls)
    print("Created call price interpolator")
    return call_interp

def generate_call_price_interpolator_nonuniform(impvolobj, nrstrikes,
                                                times, domestic_dfcurve, base_dfcurve, spot,
                                                width_nr_stdev=3.5):
    """
    Generate a nonuniform call price surface

    Parameters
    ----------
    impvolobj :
        Implied vol surface
    nrstrikes : int
        Number of strikes
    times : list of floats
        Grid of times
    domestic_dfcurve :
        (domestic) discount curve
    base_dfcurve :
        base discount curve or dividend yield curve
    spot : float
        Time zero value of the underlier
    width_nr_stdev : float
        Width of the generated grid in terms of standard deviations from ATMF

    Returns
    -------
    interp : :class:`SurfaceInterpolator`
        Interpolator object

    """
    print("Creating call price interpolator")
    allcalls = []
    allstrikes = []
    for T in times:
        domestic_df = domestic_dfcurve(T)
        base_df = base_dfcurve(T)
        rd = - math.log(domestic_df) / T
        rf = - math.log(base_df) / T
        fwd = spot * base_df / domestic_df
        ref_impvol = impvolobj.impliedvol_K(T, fwd)
        ymin = -width_nr_stdev * ref_impvol * math.sqrt(T)
        ymax = +width_nr_stdev * ref_impvol * math.sqrt(T)
        Kmin = math.exp(ymin) * fwd
        Kmax = math.exp(ymax) * fwd
        Ks = list(np.arange(Kmin, Kmax, (Kmax - Kmin) / nrstrikes))[:nrstrikes]
        allstrikes.append(Ks)
        callslice = []
        for K in Ks:
            impvol = impvolobj.impliedvol_K(T, K)
            call = max(0.0, bsan.Call(spot, K, T, rd, rf, impvol))
            callslice.append(call)
        allcalls.append(callslice)
    all_times = times
    impvars = SurfaceInterpolator(allstrikes, all_times, allcalls)
    print("Created Call surface interpolator")
    return impvars

def generate_implied_totalvariance_interpolator(impvolobj, ys, times):
    """
    Generate a uniform total implied variance surface

    Parameters
    ----------
    impvolobj :
        Implied vol surface
    ys : list of floats
        Grid of log-moneynesses
    times : list of floats
        Grid of times

    Returns
    -------
    interp :
        Interpolator object

    """
    print("Creating implied total variance interpolator")
    totalvariance = []
    for T in times:
        w_slice = []
        for y in ys:
            impvol = impvolobj.impliedvol_lkf(T, y)
            w_slice.append(impvol * impvol * T)
        totalvariance.append(w_slice)
    impvars = SurfaceInterpolatorInterp2D(ys, times, totalvariance)
    print("Created implied total variance interpolator")
    return impvars

def generate_implied_totalvariance_interpolator_nonuniform(impvolobj, nrys, times,
                                                           width_nr_stdev=3.5):
    """
    Generate a nonuniform total implied variance surface

    Parameters
    ----------
    impvolobj :
        Implied vol surface
    nrys : int
        Number of log-moneynesses
    times : list of floats
        Grid of times
    width_nr_stdev : float
        Width of the generated grid in terms of standard deviations from ATMF

    Returns
    -------
    interp : :class:`SurfaceInterpolator`
        Interpolator object

    """
    print("Creating implied total variance interpolator")
    totalvariance = []
    all_ys = []
    for T in times:
        ref_impvol = impvolobj.impliedvol_lkf(T, 0.0)
        ymin = -width_nr_stdev * ref_impvol * math.sqrt(T)
        ymax = +width_nr_stdev * ref_impvol * math.sqrt(T)
        ys = list(np.arange(ymin, ymax, (ymax - ymin) / nrys))[:nrys]
        all_ys.append(ys)
        w_slice = []
        for y in ys:
            impvol = impvolobj.impliedvol_lkf(T, y)
            w_slice.append(impvol * impvol * T)
        totalvariance.append(w_slice)
    all_times = times
    impvars = SurfaceInterpolator(all_ys, all_times, totalvariance)
    print("Created implied total variance interpolator")
    return impvars


def constructShortRateCurve(dfcurve, name=None):
    """
    Construct short rate curve from given discount factors

    Parameters
    ----------
    dfcurve :
        Discount factor curve
    name : str
        Name of the curve

    Returns
    -------
    ratecurve :
        Short rate curve

    """
    if name is None:
        name = 'shortRate'
    logdfs = [np.log(df) for df in dfcurve.values]

    srcurve = {'values': [(logdf1-logdf2)/(t2-t1) for logdf1,logdf2,t1,t2 in zip(logdfs,logdfs[1:],
                                                                         dfcurve.times,dfcurve.times[1:])], 'times':dfcurve.times[:-1]}

    shortratecurve = InterpolatedCurve(srcurve['times'], srcurve['values'], interpolation_fnc='pwc_left_cts')

    return shortratecurve


def constructDiscountcurve(times, vals):
    crv = InterpolatedCurve(times, vals, interpolation_fnc='loglinear')
    return crv

def getLinearInterpEx(t,tlist,xlist,diffun=lambda x,y:x-y):
    """
    getLinearInterpEx(t,tlist,xlist,diffun=lambda x,y:x-y)
    Linear interpolation. Constant extrapolation.

    Parameters
    ----------
    t : float
        point to be interpolated
    tlist : list
        independent variables
    xlist : list of floats
        dependent variables
    diffun : function
        difference function for two *tlist* values

    Returns
    -------
    out : float
        Linear interpolation at point *t*

    """
    ts,xs=getBracketingPoints(t,tlist,xlist)
    if len(ts)==1:
        # extrapolating outside range of tlist by constant
        return xs[0]
    else:
        w=diffun(t,ts[0])/diffun(ts[1],ts[0])
        return w*xs[1]+(1.0-w)*xs[0]

def getLeftConstantInterpEx(t,tlist,xlist,diffun=None):
    """
    Left constant interpolation. Constant extrapolation.

    Parameters
    ----------
    t : float
        point to be interpolated
    tlist : list of floats
        independent variables
    xlist : list of floats
        dependent variables

    Returns
    -------
    out : float
        Left constant interpolation at point *t*

    """
    ts,xs=getBracketingPoints(t,tlist,xlist)
    return xs[0]


def getLogLinearInterpEx(t,tlist,xlist,diffun=lambda x,y:x-y):
    """
    getLogLinearInterpEx(t,tlist,xlist,diffun=lambda x,y:x-y)
    Log-linear interpolation. Constant extrapolation.

    Parameters
    ----------
    t : float
        point to be interpolated
    tlist : list
        independent variables
    xlist : list of floats
        dependent variables
    diffun : function
        difference function for two *tlist* values

    Returns
    -------
    out : float
        Log-linear interpolation at point *t*

    """
    ts,xs=getBracketingPoints(t,tlist,xlist)
    if len(ts) == 1:
        return xs[0]
    lxlist=[math.log(x) for x in xs]
    w=diffun(t,ts[0])/diffun(ts[1],ts[0])
    lxi = w*lxlist[1]+(1.0-w)*lxlist[0]
    return math.exp(lxi)


def getBracketingPoints(t,tlist,xlist,left_continuous=True):
    """
    Points bracketing *t*

    Parameters
    ----------
    t : float
        point to be bracketed
    tlist : list of floats
        independent variables
    xlist : list of floats
        dependent variables

    Returns
    -------
    out : list of lists
        Bracketing points

    """
    #print(t,tlist,xlist)
    if left_continuous:
        if (t>=tlist[-1]):
            return [[tlist[-1]],[xlist[-1]]]
        elif (t<tlist[0]):
            return [[tlist[0]],[xlist[0]]]
        else:
            idx = bisect.bisect_right(tlist, t) - 1
            return [[tlist[idx], tlist[idx + 1]], [xlist[idx], xlist[idx + 1]]]
    else:
        if (t>tlist[-1]):
            return [[tlist[-1]],[xlist[-1]]]
        elif (t<=tlist[0]):
            return [[tlist[0]],[xlist[0]]]
        else:
            idx = bisect.bisect_left(tlist, t)
            return [[tlist[idx - 1], tlist[idx]], [xlist[idx - 1], xlist[idx]]]
