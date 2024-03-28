import numpy as np
from scipy import optimize
from scipy import stats
from functools import partial
from itertools import compress

class implied_vol_surface:
    def __init__(self, snapshot):
        '''
        This class contains:
            1. Different methods to calculate implied volatility
            2. Plotting functions
        '''
        self.data = snapshot
        self.imp_vol_call_bids = []
        self.imp_vol_call_asks = []
        self.imp_vol_put_bids = []
        self.imp_vol_put_asks = []
        self.days_to_exp = []
        self.strikes = []
        
    def normal(self, x):
        '''
        Helper function to return the normal cdf
        '''
        return stats.norm.cdf(x, 0.0, 1.0)
        
    def bs_call(self, S, K, r, q, T, sigma):
        '''
        Helper funtion to calculate Black-Scholes European call
        '''
        d1 = (np.log(S / K) + (r - q + (sigma**2) / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * np.exp(-q * T) * self.normal(d1) - K * np.exp(-r * T) * self.normal(d2)
    
    def bs_put(self, S, K, r, q, T, sigma):
        '''
        Helper funtion to calculate Black-Scholes European put
        '''
        d1 = (np.log(S / K) + (r - q + (sigma**2) / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * self.normal(-d2) - S * np.exp(-q * T) * self.normal(-d1)
    
    
    def root_finding(self, method):
        '''
        This function uses root-finding methods to get the implied vol given all other parameters
        This assumes the options have the same expiry with futures
        The function is seperated into two steps:
            1. Calculate the implied dividend yield + repo rate (the implied q) from future contracts for each expiry
            2. Use a root-finding method to find the implied volatility that minimises squared error between market price and BS option
        '''
        def interpolate_r(data):
            r_bids = []
            r_asks = []
            for T in self.data.futures.days_to_exp:
                
                # use exact match if exist
                if T in self.data.fx_rates.days_to_exp:
                    r_bid = self.data.fx_rates.implied_yield_bids[self.data.fx_rates.days_to_exp.index(T)]
                    r_ask = self.data.fx_rates.implied_yield_asks[self.data.fx_rates.days_to_exp.index(T)]
                
                # linearly interplote between two rates
                else:
                    for idx, days in enumerate(self.data.fx_rates.days_to_exp):
                        if T > days:
                            pass
                        else:
                            break
                    r_bid = ((self.data.fx_rates.implied_yield_bids[idx] - self.data.fx_rates.implied_yield_bids[idx - 1]) / \
                    (self.data.fx_rates.days_to_exp[idx] - self.data.fx_rates.days_to_exp[idx - 1])) \
                    * (T - self.data.fx_rates.days_to_exp[idx - 1]) + self.data.fx_rates.implied_yield_bids[idx - 1]
                    r_ask = ((self.data.fx_rates.implied_yield_asks[idx] - self.data.fx_rates.implied_yield_asks[idx - 1]) / \
                    (self.data.fx_rates.days_to_exp[idx] - self.data.fx_rates.days_to_exp[idx - 1])) \
                    * (T - self.data.fx_rates.days_to_exp[idx - 1]) + self.data.fx_rates.implied_yield_asks[idx - 1]
                
                r_bids.append(r_bid/100)
                r_asks.append(r_ask/100)

            return r_bids, r_asks
        
        
        def calc_implied_q(data, r_bids, r_asks):
            '''
            This function calculates the implied q from future contracts for each maturities
            '''
            q_bids = []
            q_asks = []
            
            for idx, r_ask in enumerate(r_bids): 
                q_bids.append(r_ask - (1 / self.data.futures.days_to_exp[idx] * 360) * np.log(self.data.futures.asks[idx] / self.data.spot.mid))
            
            for idx, r_bid in enumerate(r_asks):
                q_asks.append(r_bid - (1 / self.data.futures.days_to_exp[idx] * 360) * np.log(self.data.futures.bids[idx] / self.data.spot.mid))

            return q_bids, q_asks
                              
                              
        def calc_single_implied_vol(option_price, S, K, r, q, T, sigma, option_valuation, method):
            '''
            This function calculates and returns the implied volatility of a European option using Newton-Raphson
            '''
            if np.isnan(option_price):
                return np.nan
            
            def mse(option_price, S, K, r, q, T, sigma):
                return (option_price - option_valuation(S, K, r, q, T, sigma))**2
            
            f = partial(mse, option_price, S, K, r, q, T/360)
            
            if method == 'secant':
                imp_vol, res = optimize.newton(f, sigma, full_output = True, disp=False)
                if res.converged == False:
                    return np.nan
            elif method == 'newton raphson':
                imp_vol = optimize.newton(f, sigma, disp=False)
            elif method == 'halley':
                imp_vol = optimize.newton(f, sigma)
                
            return imp_vol
            
        
        # Step one: find the implied q
        r_bids, r_asks = interpolate_r(self.data)
        imp_q_bids, imp_q_asks = calc_implied_q(self.data, r_bids, r_asks)
        
        # Step two: Use Newton-Raphson to find the implied vol
        imp_vol_call_bids = []
        imp_vol_call_asks = [] 
        imp_vol_put_bids = []                      
        imp_vol_put_asks = []                 
        spot = self.data.spot.mid
        i = 0
        for idx, maturity_days in enumerate(self.data.calls.days_to_exp):
            # initialise a random number to initialise optimisation
            for strike in self.data.calls.strikes:
                imp_vol_call_bid = imp_vol_call_ask = imp_vol_put_bid = imp_vol_put_ask = 0.21
                
                imp_vol_call_bid = calc_single_implied_vol(self.data.calls.bids[i], spot, strike, r_bids[idx], imp_q_asks[idx], maturity_days, imp_vol_call_bid, self.bs_call, method)
                imp_vol_call_ask = calc_single_implied_vol(self.data.calls.asks[i], spot, strike, r_asks[idx], imp_q_bids[idx], maturity_days, imp_vol_call_ask, self.bs_call, method)
                imp_vol_put_bid = calc_single_implied_vol(self.data.puts.bids[i], spot, strike, r_asks[idx], imp_q_bids[idx], maturity_days, imp_vol_put_bid, self.bs_put, method)
                imp_vol_put_ask = calc_single_implied_vol(self.data.puts.asks[i], spot, strike, r_bids[idx], imp_q_asks[idx], maturity_days, imp_vol_put_ask, self.bs_put, method)
                self.imp_vol_call_bids.append(imp_vol_call_bid)
                self.imp_vol_call_asks.append(imp_vol_call_ask)
                self.imp_vol_put_bids.append(imp_vol_put_bid)
                self.imp_vol_put_asks.append(imp_vol_put_ask)
                self.days_to_exp.append(maturity_days)
                self.strikes.append(strike)
                
                i += 1