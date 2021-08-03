
##################################################################################
class sdedataC():
    def __init__(self, rho_domestic_base, rho_base_fx, rho_domestic_fx):

        self.rho_domestic_base = rho_domestic_base
        self.rho_base_fx = rho_base_fx
        self.rho_domestic_fx = rho_domestic_fx
        return
##################################################################################

##################################################################################
class g1pp_dataC():
    def __init__(self, spot_FX, base_x0, domestic_x0, g1ppTA, domestic_shifttimes, g1ppTVOL, g1ppA, g1ppVOL, domestic_shiftvalues,
                g1ppdivTA, g1ppdivMEANREV, base_shifttimes, base_shiftvalues, g1ppdivTVOL, g1ppdivVOL):
        
        self.spot_FX=spot_FX
        self.base_x0 = base_x0
        self.domestic_x0 = domestic_x0
        
        self.domestic_ta = g1ppTA
        self.domestic_shifttimes = domestic_shifttimes
        self.domestic_tvol = g1ppTVOL

        self.domestic_a = g1ppA
        self.domestic_vol = g1ppVOL
        self.domestic_shiftvalues = domestic_shiftvalues

        self.base_ta = g1ppdivTA
        self.base_a = g1ppdivMEANREV
        self.base_shifttimes = base_shifttimes
        self.base_shiftvalues = base_shiftvalues
        self.base_tvol = g1ppdivTVOL
        self.base_vol = g1ppdivVOL
        return
##################################################################################


##################################################################################
class heston_dataC():
    def __init__(self, kappa_dict, theta_dict, volofvar_dict, initialvar, rho_fx_v, rho_domestic_v, rho_base_v):
        self.kappa_times = kappa_dict['times']
        self.kappa_vals = kappa_dict['values']
        self.theta_times = theta_dict['times']
        self.theta_vals = theta_dict['values']
        self.volofvar_times = volofvar_dict['times']
        self.volofvar_vals = volofvar_dict['values']
        self.initialvar = initialvar
        self.rho_fx_v = rho_fx_v
        self.rho_domestic_v = rho_domestic_v
        self.rho_base_v = rho_base_v
        return
##################################################################################