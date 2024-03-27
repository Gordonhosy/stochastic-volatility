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
        self.imp_vol = None
        
    def normal(x):
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
    
    
    def newton_raphson(self):
        '''
        This function uses Newton-Raphson method to get the implied vol given all other parameters
        This assumes the options have the same expiry with futures
        The function is seperated into two steps:
            1. Calculate the implied dividend yield + repo rate (the implied q) from future contracts for each expiry
            2. 
        
        '''
        def interpolate_r(data):
            r_bids = []
            r_asks = []
            for T in self.data.futures.days_to_exp:
                
                # use exact match if exist
                if T in self.data.fx_rates.days_to_exp:
                    r_bid = self.data.fx_rates.bids[self.data.fx_rates.days_to_exp.index(T)]
                    r_ask = self.data.fx_rates.asks[self.data.fx_rates.days_to_exp.index(T)]
                
                # linearly interplote between two rates
                else:
                    for idx, days in enumerate(self.data.fx_rates.days_to_exp):
                        if T > days:
                            pass
                        else:
                            break
                    r_bid = ((self.data.fx_rates.bids[idx] - self.data.fx_rates.bids[idx - 1]) / \
                    (self.data.fx_rates.days_to_exp[idx] - self.data.fx_rates.days_to_exp[idx - 1])) \
                    * (T - self.data.fx_rates.days_to_exp[idx - 1]) + self.data.fx_rates.bids[idx - 1]
                    r_ask = ((self.data.fx_rates.asks[idx] - self.data.fx_rates.asks[idx - 1]) / \
                    (self.data.fx_rates.days_to_exp[idx] - self.data.fx_rates.days_to_exp[idx - 1])) \
                    * (T - self.data.fx_rates.days_to_exp[idx - 1]) + self.data.fx_rates.asks[idx - 1]
                
                r_bids.append(r_bid)
                r_asks.append(r_ask)
            
            return r_bids, r_asks
        
        
        def calc_implied_q(data, r_bids, r_asks):
            '''
            This function calculates the implied q from future contracts for each maturities
            '''
            q_bids = []
            q_asks = []
            
            for idx, r_bid in enumerate(r_bids): 
                q_bids.append(r_bid - (1 / T)*np.log(self.data.futures.bids[idx] / self.data.spot.mid))
            
            for idx, r_ask in enumerate(r_asks):
                q_asks.append(r_ask - (1 / T)*np.log(self.data.futures.asks[idx] / self.data.spot.mid))
            
            return q_bids, q_asks
                              
                              
        def calc_single_implied_vol(option_price, S, K, r, q, T, sigma, option_valuation):
            '''
            This function calculates and returns the implied volatility of a European option using Newton-Raphson
            '''
            def mse(option_price, S, K, r, q, T, sigma):
                return (option_price - option_valuation(S, K, r, q, T, sigma))**2
            
            f = partial(mse, option_price, S, K, r, q, T)
            imp_vol = optimize.newton(mse, sigma)
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
                              
        for idx, maturity_days in enumerate(self.data.calls.days_to_exp):
            # initialise a random number to initialise optimisation
            imp_vol_call_bid = 0.1
            imp_vol_call_ask = 0.1
            imp_vol_put_bid = 0.1                    
            imp_vol_put_ask = 0.1 
            strikes = list(compress(self.data.calls.strikes, [maturity_days == x for x in self.data.calls.days_to_exp]))
            call_price_bids = list(compress(self.data.calls.bids, [maturity_days == x for x in self.data.calls.days_to_exp]))
            call_price_asks = list(compress(self.data.calls.asks, [maturity_days == x for x in self.data.calls.days_to_exp]))
            put_price_bids = list(compress(self.data.puts.bids, [maturity_days == x for x in self.data.calls.days_to_exp]))
            put_price_asks = list(compress(self.data.puts.asks, [maturity_days == x for x in self.data.calls.days_to_exp]))
            for strike, call_bid, call_ask, put_bid, put_ask in zip(strikes, call_price_bids, call_price_asks, put_price_bids, put_price_asks):
                imp_vol_call_bid = calc_single_implied_vol(spot, strike, r_bids[idx], imp_q_bids[idx], maturity_days, imp_vol_call_bid, self.bs_call)
                imp_vol_call_ask = calc_single_implied_vol(spot, strike, r_asks[idx], imp_q_asks[idx], maturity_days, imp_vol_call_ask, self.bs_call)
                imp_vol_put_bid = calc_single_implied_vol(spot, strike, r_bids[idx], imp_q_bids[idx], maturity_days, imp_vol_put_bid, self.bs_put)
                imp_vol_put_ask = calc_single_implied_vol(spot, strike, r_asks[idx], imp_q_asks[idx], maturity_days, imp_vol_put_ask, self.bs_put)
                imp_vol_call_bids.append(imp_vol_call_bid)
                imp_vol_call_asks.append(imp_vol_call_ask)
                imp_vol_put_bids.append(imp_vol_put_bid)
                imp_vol_put_asks.append(imp_vol_put_ask)
        
        self.imp_vol_call_bids = imp_vol_call_bids
        self.imp_vol_call_asks = imp_vol_call_asks
        self.imp_vol_put_bids = imp_vol_put_bids
        self.imp_vol_put_asks = imp_vol_put_asks