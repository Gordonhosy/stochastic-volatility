import numpy as np
import pandas as pd
from scipy import stats, optimize
from functools import partial
from itertools import compress
import plotly.graph_objects as go



class black_scholes:
    def __init__(self, snapshot):
        '''
        This class contains:
            1. Different methods to calculate implied volatility
            2. Plotting functions
        '''
        self.name = 'bs'
        self.data = snapshot
        self.imp_vol_call_bids = []
        self.imp_vol_call_asks = []
        self.imp_vol_put_bids = []
        self.imp_vol_put_asks = []
        self.imp_vol_call_mids = []
        self.imp_vol_put_mids = []
        self.days_to_exp = []
        self.strikes = []
        self.r = []
        self.q = []
        self.adjust_rates()
        self.calc_static()
        self.calc_r_q()
        
        self.obs_exps = []
        self.obs_strikes = []
        self.obs_prices = []
        
    def normal(self, x):
        '''
        Helper function to return the normal cdf
        '''
        return stats.norm.cdf(x, 0.0, 1.0)
    
    def normal_prime(self, x):
        '''
        Helper function to approximate the normal inverse
        '''
        return (np.exp(-(x**2)/2)) / (np.sqrt(2 * np.pi))
    
    def d1(self, S, K, r, q, sigma, T):
        '''
        Helper function to calculate d1
        '''
        return (np.log(S / K) + (r - q + (sigma**2) / 2) * T) / (sigma * np.sqrt(T))
        
    def bs_call(self, S, K, r, q, T, sigma):
        '''
        Helper function to calculate Black-Scholes European call
        '''
        d1 = (np.log(S / K) + (r - q + (sigma**2) / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * np.exp(-q * T) * self.normal(d1) - K * np.exp(-r * T) * self.normal(d2)
    
    def bs_put(self, S, K, r, q, T, sigma):
        '''
        Helper function to calculate Black-Scholes European put
        '''
        d1 = (np.log(S / K) + (r - q + (sigma**2) / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * self.normal(-d2) - S * np.exp(-q * T) * self.normal(-d1)
    
    def bs_vega(self, S, K, r, q, T, sigma):
        '''
        Helper funtion to calculate option vega
        '''
        d1 = (np.log(S / K) + (r - q + (sigma**2) / 2) * T) / (sigma * np.sqrt(T))
        return S * np.sqrt(T) * self.normal_prime(d1)
    
    def bs_volga(self, S, K, r, q, T, sigma):
        '''
        Helper function to calculate option volga
        '''
        d1 = (np.log(S / K) + (r - q + (sigma**2) / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return np.exp(-q * T) * np.sqrt(T) * self.normal_prime(d1) * (d1 * d2 / sigma)
    
    def calc_r_q(self):
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
                    
                    if T < min(self.data.fx_rates.days_to_exp): # linear extrapolate for T smaller than all available rates
                        idx += 1

                    r_bid = ((self.data.fx_rates.implied_yield_bids[idx] - self.data.fx_rates.implied_yield_bids[idx - 1]) / \
                    (self.data.fx_rates.days_to_exp[idx] - self.data.fx_rates.days_to_exp[idx - 1])) \
                    * (T - self.data.fx_rates.days_to_exp[idx - 1]) + self.data.fx_rates.implied_yield_bids[idx - 1]
                    r_ask = ((self.data.fx_rates.implied_yield_asks[idx] - self.data.fx_rates.implied_yield_asks[idx - 1]) / \
                    (self.data.fx_rates.days_to_exp[idx] - self.data.fx_rates.days_to_exp[idx - 1])) \
                    * (T - self.data.fx_rates.days_to_exp[idx - 1]) + self.data.fx_rates.implied_yield_asks[idx - 1]
                
                # transform annual rate to continuous rate
                r_bids.append(np.log(1 + r_bid / 100))
                r_asks.append(np.log(1 + r_ask / 100))

            return r_bids, r_asks
        
        
        def calc_implied_q(data, r_bids, r_asks):
            '''
            This function calculates the continuous implied q from future contracts for each maturities
            '''
            q_bids = []
            q_asks = []
            
            for idx, r_ask in enumerate(r_asks):
                q_bids.append(r_ask - (1 / self.data.futures.days_to_exp[idx] * 360) * np.log(self.data.futures.asks[idx] / self.data.spot.mid))
                
            for idx, r_bid in enumerate(r_bids): 
                q_asks.append(r_bid - (1 / self.data.futures.days_to_exp[idx] * 360) * np.log(self.data.futures.bids[idx] / self.data.spot.mid))

            return q_bids, q_asks
                              
                              
        
               
        r_bids, r_asks = interpolate_r(self.data)
        imp_q_bids, imp_q_asks = calc_implied_q(self.data, r_bids, r_asks)
        self.r = [(x + y)/2 for x, y in zip(r_bids, r_asks)]
        self.q = [(x + y)/2 for x, y in zip(imp_q_bids, imp_q_asks)]
        
    
    def root_finding(self, method):
        '''
        This function uses root-finding methods to get the implied vol given all other parameters
        This assumes the options have the same expiry with futures
        The function is seperated into two steps:
            1. Calculate the implied dividend yield + repo rate (the implied q) from future contracts for each expiry
            2. Use a root-finding method to find the implied volatility that minimises squared error between market price and BS option
        '''
        # Step one: find the implied q
        r_mids = self.r
        imp_q_mids = self.q
        
        # Step two: Use root-finding methdods to calculate the implied vol
        spot = self.data.spot.mid
        i = 0
        for idx, maturity_days in enumerate(self.data.calls.days_to_exp):
            for strike in self.data.calls.strikes:
                # initialise a random number to initialise optimisation
                imp_vol_call_bid = imp_vol_call_ask = imp_vol_put_bid = imp_vol_put_ask = imp_vol_call_mid = imp_vol_put_mid = 0.2
                
                imp_vol_call_bid = self.calc_single_implied_vol(self.data.calls.bids[i], spot, strike, r_mids[idx], imp_q_mids[idx], maturity_days, imp_vol_call_bid, self.bs_call, method)
                imp_vol_call_ask = self.calc_single_implied_vol(self.data.calls.asks[i], spot, strike, r_mids[idx], imp_q_mids[idx], maturity_days, imp_vol_call_ask, self.bs_call, method)
                imp_vol_put_bid = self.calc_single_implied_vol(self.data.puts.bids[i], spot, strike, r_mids[idx], imp_q_mids[idx], maturity_days, imp_vol_put_bid, self.bs_put, method)
                imp_vol_put_ask = self.calc_single_implied_vol(self.data.puts.asks[i], spot, strike, r_mids[idx], imp_q_mids[idx], maturity_days, imp_vol_put_ask, self.bs_put, method)
                imp_vol_call_mid = self.calc_single_implied_vol((self.data.calls.bids[i] + self.data.calls.asks[i])/2, spot, strike, r_mids[idx], imp_q_mids[idx], maturity_days, imp_vol_call_mid, self.bs_call, method)
                imp_vol_put_mid = self.calc_single_implied_vol((self.data.puts.bids[i] + self.data.puts.asks[i])/2, spot, strike, r_mids[idx], imp_q_mids[idx], maturity_days, imp_vol_put_mid, self.bs_put, method)
                
                self.imp_vol_call_bids.append(imp_vol_call_bid)
                self.imp_vol_call_asks.append(imp_vol_call_ask)
                self.imp_vol_put_bids.append(imp_vol_put_bid)
                self.imp_vol_put_asks.append(imp_vol_put_ask)
                self.imp_vol_call_mids.append(imp_vol_call_mid)
                self.imp_vol_put_mids.append(imp_vol_put_mid)
                
                i += 1      
                
                
    def calc_single_implied_vol(self, option_price, S, K, r, q, T, sigma, option_valuation, method):
        '''
        This function calculates and returns the implied volatility of a European option using Newton-Raphson
        '''
        if np.isnan(option_price):
            return np.nan

        def mse(option_price, S, K, r, q, T, sigma):
            return (option_valuation(S, K, r, q, T, sigma) - option_price)**2

        f = partial(mse, option_price, S, K, r, q, T/360)

        if method == 'secant':
            imp_vol, res = optimize.newton(f, sigma, full_output = True, disp = False, tol=1.48e-15, maxiter=500)
            if res.converged == False:
                return np.nan

        elif method == 'newton':
            def f_prime(option_price, S, K, r, q, T, sigma):
                return 2 * (option_valuation(S, K, r, q, T, sigma) - option_price) * self.bs_vega(S, K, r, q, T, sigma)
            fp = partial(f_prime, option_price, S, K, r, q, T/360)
            imp_vol, res = optimize.newton(f, sigma, fprime = fp, full_output = True, disp = False)
            if res.converged == False:
                return np.nan

        elif method == 'halley':
            def f_prime(option_price, S, K, r, q, T, sigma):
                return 2 * (option_valuation(S, K, r, q, T, sigma) - option_price) * self.bs_vega(S, K, r, q, T, sigma)
            def f_prime2(option_price, S, K, r, q, T, sigma):
                return 2 * ((option_valuation(S, K, r, q, T, sigma) - option_price) * self.bs_volga(S, K, r, q, T, sigma)\
                           + self.bs_vega(S, K, r, q, T, sigma) ** 2)
            fp = partial(f_prime, option_price, S, K, r, q, T/360)
            fp2 = partial(f_prime2, option_price, S, K, r, q, T/360)
            imp_vol, res = optimize.newton(f, sigma, fprime = fp, fprime2 = fp2, full_output = True, disp = False)
            if res.converged == False:
                return np.nan

        return imp_vol

    def calc_convergence(self):
        '''
        Function to calculate the number of convergence
        '''
        return len([x for x in self.imp_vol_call_asks + self.imp_vol_call_bids + self.imp_vol_put_asks + self.imp_vol_put_bids if not np.isnan(x)])
    
    
    def calc_static(self):
        '''
        Helper function to created ordered strikes and expiries
        '''
        for T in self.data.calls.days_to_exp:
            for K in self.data.calls.strikes:
                self.days_to_exp.append(T)
                self.strikes.append(K)
    
    def adjust_rates(self):
        '''
        Some currencies need adjustments since data is not available for all tenors
        '''
        if self.data.ticker in ['HSI Index', 'HSCEI Index']:
            self.data.fx_rates.days_to_exp = self.data.us_rates.days_to_exp \
            = [1, 2, 3, 7, 14, 21, 30, 61, 91, 122, 153, 183, 214, 244, 275, 306, 334, 365, 456, 548, 640, 730]
            self.data.fx_rates.implied_yield_bids = []
            self.data.fx_rates.implied_yield_asks = []
            self.data.fx_rates.calc_implied_rate(self.data.us_rates)
        # CNH no 15m, 21m
        elif self.data.ticker in ['SSE50 Index', 'SHSN300 Index', 'CSI1000 Index']:
            self.data.fx_rates.days_to_exp = self.data.us_rates.days_to_exp \
            = [1, 2, 3, 7, 14, 21, 30, 61, 91, 122, 153, 183, 214, 244, 275, 306, 334, 365, 548, 730]
            self.data.fx_rates.implied_yield_bids = []
            self.data.fx_rates.implied_yield_asks = []
            self.data.fx_rates.calc_implied_rate(self.data.us_rates)
        # NTN no ON, TN, SN, 11m, 15m, 18m, 21m
        elif self.data.ticker in ['TWSE Index']:
            self.data.fx_rates.days_to_exp = self.data.us_rates.days_to_exp \
            = [7, 14, 21, 30, 61, 91, 122, 153, 183, 214, 244, 275, 306, 365, 730]
            self.data.fx_rates.implied_yield_bids = []
            self.data.fx_rates.implied_yield_asks = []
            self.data.fx_rates.calc_implied_rate(self.data.us_rates)
        # IRN no ON, TN, SN, 2w, 3w, 4m, 5m, 7m, 8m, 10m, 11m, 15m, 18m, 21m
        elif self.data.ticker in ['NIFTY Index']:
            self.data.fx_rates.days_to_exp = self.data.us_rates.days_to_exp \
            = [7, 30, 61, 91, 183, 275, 365, 730]
            self.data.fx_rates.implied_yield_bids = []
            self.data.fx_rates.implied_yield_asks = []
            self.data.fx_rates.calc_implied_rate(self.data.us_rates)
    
# Visualisations
# --------------------------------------------------------------------------------------------------------------------
    
    
    def vol_smile(self, ith_contract):
        '''
        Return the volatility smile curve given the i-th contract
        '''
        days_to_exp = sorted(list(set(self.days_to_exp)))[ith_contract - 1]
        
        return list(compress(self.strikes, [days_to_exp == x for x in self.days_to_exp])), \
                list(compress(self.imp_vol_call_bids, [days_to_exp == x for x in self.days_to_exp])), \
                list(compress(self.imp_vol_call_asks, [days_to_exp == x for x in self.days_to_exp])), \
                list(compress(self.imp_vol_put_bids, [days_to_exp == x for x in self.days_to_exp])), \
                list(compress(self.imp_vol_put_asks, [days_to_exp == x for x in self.days_to_exp]))
                                 
                                                                 
    def vol_smile_plot(self, ith_contract):
        '''
        Return the volatility smile plot given the i-th contract
        '''
        strikes, iv_call_bids, iv_call_asks, iv_put_bids, iv_put_asks = self.vol_smile(ith_contract)
        
        fig = go.Figure()

        fig.add_trace(
                go.Scatter(x = strikes,
                           y = iv_call_bids,
                           mode='lines+markers',
                           #line_color = 'indigo', 
                           name = 'Call Bid')
        )
        
        fig.add_trace(
                go.Scatter(x = strikes,
                           y = iv_call_asks,
                           mode='lines+markers',
                           name = 'Call Ask')
        )
        fig.add_trace(
                go.Scatter(x = strikes,
                           y = iv_put_bids,
                           mode='lines+markers',
                           name = 'Put Bid')
        )
        fig.add_trace(
                go.Scatter(x = strikes,
                           y = iv_put_asks,
                           mode='lines+markers',
                           name = 'Put Ask')
        )

        fig.update_layout(
                title = dict(
                    text = r'Implied Volatility Smile',
                    x = 0.5,
                    y = 0.9
                ),
                font=dict(size=14),
                xaxis_title = r'Strikes',
                yaxis_title = r'Implied Vol',
                legend=dict(
                x=1,
                y=0.5,
                traceorder="normal",
                font=dict(
                    family="Arial",
                    size=14,
                    color="black")
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=75, b=20),
                autosize=False,
                width=980,
                height=500
        )
        
        return fig
    
    
    def vol_surface_plot(self):
        '''
        Return the volatility surface plot
        '''
        
        x = np.array(self.strikes)
        y = np.array(self.days_to_exp)
        z1 = np.array(self.imp_vol_call_mids)
        z2 = np.array(self.imp_vol_put_mids)
            
        
        fig = go.Figure()

        fig.add_trace(
            go.Mesh3d(x=x, y=y, z=z1, opacity=0.8, intensity=z1, colorscale='Tealgrn', coloraxis='coloraxis1')
        )
        fig.add_trace(
            go.Mesh3d(x=x, y=y, z=z2, opacity=0.8, intensity=z2, colorscale='Peach', coloraxis='coloraxis2')
        )

        fig.update_layout(
                title = dict(
                    text = r'Implied Volatility Surface',
                    x = 0.5,
                    y = 0.9
                ),
                font=dict(size=12),
                scene=dict(
                    xaxis_title='Strike',
                    yaxis_title='Days to Maturity',
                    zaxis_title='Implied Vol',
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=75, b=20),
                autosize=False,
                width=980,
                height=500,
                coloraxis=dict(colorscale='Tealgrn', colorbar_x=0.9, colorbar_thickness=23, colorbar_title = 'Call IV'),
                coloraxis2=dict(colorscale='Peach', colorbar_x=1, colorbar_thickness=23, colorbar_title = 'Put IV'),
        )
        
        return fig
    
    
    def call_price_df(self):
        '''
        Return a DataFrame for call options
        '''
        call_prices = [str((x,y)) for x,y in zip(self.data.calls.bids, self.data.calls.asks)]
        call_prices_arr = np.array(call_prices).reshape(len(set(self.days_to_exp)), len(set(self.strikes)))
        return pd.DataFrame(call_prices_arr,
                            index = sorted(list(set(self.days_to_exp))),
                            columns = sorted(list(set(self.strikes)))
                           )
    
    def call_imp_vol_df(self):
        '''
        Return a DataFrame for call options implied vol
        '''
        call_vols = [str((x,y)) for x,y in zip(self.imp_vol_call_bids, self.imp_vol_call_asks)]
        call_vols_arr = np.array(call_vols).reshape(len(set(self.days_to_exp)), len(set(self.strikes)))
        return pd.DataFrame(call_vols_arr,
                            index = sorted(list(set(self.days_to_exp))),
                            columns = sorted(list(set(self.strikes)))
                           )
    
    def put_price_df(self):
        '''
        Return a DataFrame for put options
        '''
        put_prices = [str((x,y)) for x,y in zip(self.data.puts.bids, self.data.puts.asks)]
        put_prices_arr = np.array(put_prices).reshape(len(set(self.days_to_exp)), len(set(self.strikes)))
        return pd.DataFrame(put_prices_arr,
                            index = sorted(list(set(self.days_to_exp))),
                            columns = sorted(list(set(self.strikes)))
                           )
    
    def put_imp_vol_df(self):
        '''
        Return a DataFrame for call options implied vol
        '''
        put_vols = [str((x,y)) for x,y in zip(self.imp_vol_put_bids, self.imp_vol_put_asks)]
        put_vols_arr = np.array(put_vols).reshape(len(set(self.days_to_exp)), len(set(self.strikes)))
        return pd.DataFrame(put_vols_arr,
                            index = sorted(list(set(self.days_to_exp))),
                            columns = sorted(list(set(self.strikes)))
                           )
    
