import numpy as np
import scipy

class FXIVolInterpolator:
    
    def __init__(self, ivoljsondata, spot_FX, dfcurve, base_dfcurve):
    
        self.strikes = [ivol_at_t['strikes'] for ivol_at_t in ivoljsondata]
        self.times = [ivol_at_t['time'] for ivol_at_t in ivoljsondata]
        self.ivolvals = [ivol_at_t['vols'] for ivol_at_t in ivoljsondata]
        self.dfcurve = dfcurve
        self.base_dfcurve = base_dfcurve
        self.spot_FX = spot_FX
        self.interps  = [scipy.interpolate.CubicSpline(Ks, ivols, bc_type='natural') for Ks, ivols in zip(self.strikes, self.ivolvals)]
        self.fwds = self.spot_FX * np.array(self.base_dfcurve(self.times))/np.array(self.dfcurve(self.times))
        ref_impvolvals = [interp(fwd) for interp, fwd in zip(self.interps, self.fwds)]
        
        self.ref_impvols = scipy.interpolate.CubicSpline(self.times, ref_impvolvals, bc_type='clamped')
         
        
    def impliedvol_lkf(self, tval, lkf):
        
        ref_impvol = self.ref_impvols(tval) ## reference implied vols
        nr_stddev = lkf/(ref_impvol*np.sqrt(tval)) ## number of standard deviations for current lkf
        Ks = [np.exp(nr_stddev * self.ref_impvols(t1) * np.sqrt(t1))*fwd for t1,fwd in zip(self.times, self.fwds)]
        
        tinterp = scipy.interpolate.InterpolatedUnivariateSpline(self.times, [interp(K) for interp, K in zip(self.interps, Ks)])
        
        self.tinterp = tinterp
        self.lastKs = Ks
        return tinterp(tval) 
        
    def impliedvol_K(self, tval, strike):
        
        forward = self.spot_FX * self.base_dfcurve(tval)/self.dfcurve(tval)
        lkf = np.log(strike/forward)
        
        ivol = self.impliedvol_lkf(tval, lkf)
        
        return ivol
