from constant_vol import black_scholes

import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.integrate import quad
import plotly.graph_objects as go
import math
from scipy.optimize import minimize
import pickle
import os


class stochastic_volatility(black_scholes):
    def __init__(self, snapshot):
        super().__init__(snapshot)
        '''
        This class inherits the black_scholes class.
        The additional functions include:
            1. Computing stochastic volatility
        '''
        self.name = 'sv'
        self.root_finding('halley')
        
        # use only observable prices
        obs_idx = np.argwhere(~np.isnan(self.imp_vol_call_bids)).flatten()
        self.obs_exps = np.array(self.days_to_exp)[[obs_idx]].flatten()
        self.obs_strikes = np.array(self.strikes)[[obs_idx]].flatten()
        self.obs_prices = np.array(self.data.calls.bids)[[obs_idx]].flatten()
        self.obs_imp_vol = np.array(self.imp_vol_call_bids)[[obs_idx]].flatten()
        self.Ts = sorted(list(set(self.obs_exps)))
        self.v = None
        self.kappa = None
        self.theta = None
        self.sigma = None
        self.lam = None
        self.rho = None

        
    def rank_T(self, T):
            '''
            Helper function to determine which interval T lies in
            '''
            for idx, x in enumerate(self.Ts):
                if T < x:
                    return idx
            return len(self.Ts)
        
    def heston_mse(self):
        '''
        Calculate estimated Heston parameters by specific methods
        '''
        
        # define call price functions
        def C_1(u, r, q, tau, kappa, theta, sigma, lam, rho):
            d = np.sqrt((rho * sigma * 1j * u + rho*sigma - kappa - sigma)**2 - sigma**2 * (1j*u - u**2))
            g = (kappa + lam - rho*sigma - rho*sigma * 1j * u + d)/(kappa + lam - rho*sigma - rho*sigma * 1j * u - d)
            return (r - q) * 1j * u *tau + kappa * theta/(sigma**2) * (((kappa + lam - rho*sigma) - rho*sigma*1j*u + d)*tau - 2*np.log((1 - g * np.exp(d * tau))/(1 - g)))

        def C_2(u, r, q, tau, kappa, theta, sigma, lam, rho):
            d = np.sqrt((rho * sigma * 1j * u - kappa - sigma)**2 - sigma**2 * (-1j*u - u**2))
            g = (kappa + lam - rho*sigma * 1j * u + d)/(kappa + lam - rho*sigma * 1j * u - d)
            return (r - q) * 1j * u *tau + kappa * theta/(sigma**2) * (((kappa + lam) - rho*sigma*1j*u + d)*tau - 2*np.log((1 - g * np.exp(d * tau))/(1 - g)))

        def D_1(u, tau, kappa, sigma, lam, rho):
            d = np.sqrt((rho * sigma * 1j * u + rho*sigma - kappa - sigma)**2 - sigma**2 * (1j*u - u**2))
            g = (kappa + lam - rho*sigma - rho*sigma * 1j * u + d)/(kappa + lam - rho*sigma - rho*sigma * 1j * u - d)
            return (((kappa + lam - rho*sigma) - rho * sigma * 1j * u + d)/(sigma**2)) * ((1 - np.exp(d*tau))/(1 - g*np.exp(d*tau)))

        def D_2(u, tau, kappa, sigma, lam, rho):
            d = np.sqrt((rho * sigma * 1j * u - kappa - sigma)**2 - sigma**2 * (-1j*u - u**2))
            g = (kappa + lam - rho*sigma * 1j * u + d)/(kappa + lam - rho*sigma * 1j * u - d)
            return (((kappa + lam) - rho * sigma * 1j * u + d)/(sigma**2)) * ((1 - np.exp(d*tau))/(1 - g*np.exp(d*tau)))

        def phi_1(u, v, x, r, q, tau, kappa, theta, sigma, lam, rho):
            C = C_1(u, r, q, tau, kappa, theta, sigma, lam, rho)
            D = D_1(u, tau, kappa, sigma, lam, rho)
            return np.exp(C + D * v + 1j * u * x)

        def phi_2(u, v, x, r, q, tau, kappa, theta, sigma, lam, rho):
            C = C_2(u, r, q, tau, kappa, theta, sigma, lam, rho)
            D = D_2(u, tau, kappa, sigma, lam, rho)
            return np.exp(C + D * v + 1j * u * x)

        def intergrand_1(u, K, v, x, r, q, tau, kappa, theta, sigma, lam, rho):
            phi = phi_1(u, v, x, r, q, tau, kappa, theta, sigma, lam, rho)
            return np.exp(-1j *np.log(K)) * phi / (1j * u)

        def intergrand_2(u, K, v, x, r, q, tau, kappa, theta, sigma, lam, rho):
            phi = phi_2(u, v, x, r, q, tau, kappa, theta, sigma, lam, rho)
            return np.exp(-1j *np.log(K)) * phi / (1j * u)

        def intergrate_rec(func, args, lower_bound, upper_bound):
            sum_int = 0
            steps = 10000
            du = (upper_bound - lower_bound) / steps
            for i in range(0, steps):
                mid_pt = du * (i + 0.5)
                sum_int += func(mid_pt, *args)*du
            return np.real(sum_int)
        
        def P_1(K, v, x, r, q, tau, kappa, theta, sigma, lam, rho):
            args = (K, v, x, r, q, tau, kappa, theta, sigma, lam, rho)
            return 0.5 + intergrate_rec(intergrand_1, args, 0, 100) / np.pi
        
        def P_2(K, v, x, r, q, tau, kappa, theta, sigma, lam, rho):
            args = (K, v, x, r, q, tau, kappa, theta, sigma, lam, rho)
            return 0.5 + intergrate_rec(intergrand_2, args, 0, 100) / np.pi
        
        def heston_call(S, K, v, r, q, tau, kappa, theta, sigma, lam, rho):
            x = np.log(S)
            args = (K, v, x, r, q, tau, kappa, theta, sigma, lam, rho)
            return np.exp(x) * P_1(*args) - K * np.exp(-(r - q)*tau) * P_2(*args)
    
        
        # calibrate to market data by minimising mse
        init_est = [0.1, # v
                    3, # kappa
                    0.05, # theta
                    0.3, # sigma
                    -0.8, # rho
                    0.03 # lam
                   ]
        boundaries = [[0.001, 0.1], # v
                      [0.001, 5], # kappa
                      [0.001, 0.1], # theta
                      [0.01, 1], # sigma
                      [-1, 0], # rho
                      [-1, 1] # lam
                     ]
        
        # interpolate r and q
        obs_r = []
        obs_q = []
        for T in self.obs_exps:
            i = self.rank_T(T)
            if i == 0:    
                i += 1
            elif i == len(self.Ts):
                i -= 1
            obs_r.append((self.r[i] - self.r[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.r[i - 1])
            obs_q.append((self.q[i] - self.q[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.q[i - 1])
        obs_r = np.array(obs_r)
        obs_q = np.array(obs_q)
            
        
        def mse(parameters):
            '''
            Mean square error function between market option prices and Heston prices
            '''
            v, kappa, theta, sigma, rho, lam = [x for x in parameters]
            n = len(self.obs_prices)
            return (1 / n) * np.sum((self.obs_prices - heston_call(self.data.spot.mid, self.obs_strikes, v, obs_r, obs_q, self.obs_exps / 360, kappa, theta, sigma, lam, rho))**2)
        
        min_res = optimize.minimize(mse, init_est, method = "SLSQP", bounds = boundaries, options = {"maxiter": 1000, 'disp': True}, tol = 1e-3)
        
        self.v, self.kappa, self.theta, self.sigma, self.rho, self.lam = [x for x in min_res.x]
        
        
        
    def heston_mse2(self):
        
        def C_1(u, r, q, tau, kappa, theta, sigma, lam, rho):
            d = np.sqrt((rho * sigma * 1j * u + rho*sigma - kappa - sigma)**2 - sigma**2 * (1j*u - u**2))
            g = (kappa + lam - rho*sigma - rho*sigma * 1j * u + d)/(kappa + lam - rho*sigma - rho*sigma * 1j * u - d)
            return (r - q) * 1j * u *tau + kappa * theta/(sigma**2) * (((kappa + lam - rho*sigma) - rho*sigma*1j*u + d)*tau - 2*np.log((1 - g * np.exp(d * tau))/(1 - g)))

        def C_2(u, r, q, tau, kappa, theta, sigma, lam, rho):
            d = np.sqrt((rho * sigma * 1j * u - kappa - sigma)**2 - sigma**2 * (-1j*u - u**2))
            g = (kappa + lam - rho*sigma * 1j * u + d)/(kappa + lam - rho*sigma * 1j * u - d)
            return (r - q) * 1j * u *tau + kappa * theta/(sigma**2) * (((kappa + lam) - rho*sigma*1j*u + d)*tau - 2*np.log((1 - g * np.exp(d * tau))/(1 - g)))

        def D_1(u, tau, kappa, sigma, lam, rho):
            d = np.sqrt((rho * sigma * 1j * u + rho*sigma - kappa - sigma)**2 - sigma**2 * (1j*u - u**2))
            g = (kappa + lam - rho*sigma - rho*sigma * 1j * u + d)/(kappa + lam - rho*sigma - rho*sigma * 1j * u - d)
            return (((kappa + lam - rho*sigma) - rho * sigma * 1j * u + d)/(sigma**2)) * ((1 - np.exp(d*tau))/(1 - g*np.exp(d*tau)))

        def D_2(u, tau, kappa, sigma, lam, rho):
            d = np.sqrt((rho * sigma * 1j * u - kappa - sigma)**2 - sigma**2 * (-1j*u - u**2))
            g = (kappa + lam - rho*sigma * 1j * u + d)/(kappa + lam - rho*sigma * 1j * u - d)
            return (((kappa + lam) - rho * sigma * 1j * u + d)/(sigma**2)) * ((1 - np.exp(d*tau))/(1 - g*np.exp(d*tau)))

        def phi_1(u, v, x, r, q, tau, kappa, theta, sigma, lam, rho):
            C = C_1(u, r, q, tau, kappa, theta, sigma, lam, rho)
            D = D_1(u, tau, kappa, sigma, lam, rho)
            return np.exp(C + D * v + 1j * u * x)

        def phi_2(u, v, x, r, q, tau, kappa, theta, sigma, lam, rho):
            C = C_2(u, r, q, tau, kappa, theta, sigma, lam, rho)
            D = D_2(u, tau, kappa, sigma, lam, rho)
            return np.exp(C + D * v + 1j * u * x)        
        
        
        def heston_price(S, K, v, kappa, theta, sigma, rho, lam, tau, r, q):            
            sum_int = 0
            steps = 100
            upper_bound = 100
            du = upper_bound / steps
            x = np.log(S)
            for i in range(1, steps):
                u = du * (2*i + 1)/2
                sum_int += du * np.exp(-1j*u*np.log(K))/(1j*u)*(S*np.exp(-q*tau)\
                                            *phi_1(u, v, x, r, q, tau, kappa, theta, sigma, lam, rho) \
                                            - K*np.exp(-r*tau)*phi_2(u, v, x, r, q, tau, kappa, theta, sigma, lam, rho))

            return np.real(sum_int)/np.pi + 0.5*(S*np.exp(-q*tau) - K*np.exp(-r*tau))
        
        def mse(x):
            v0, kappa, theta, sigma, rho, lambd = [param for param in x]

            return np.sum( (self.obs_prices-heston_price(self.data.spot.mid, self.obs_strikes, \
                                v0, kappa, theta, sigma, rho, lambd, self.obs_exps / 365, obs_r, obs_q))**2 /len(self.obs_prices) )
        
        obs_r = []
        obs_q = []
        for T in self.obs_exps:
            i = self.rank_T(T)
            if i == 0:    
                i += 1
            elif i == len(self.Ts):
                i -= 1
            obs_r.append((self.r[i] - self.r[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.r[i - 1])
            obs_q.append((self.q[i] - self.q[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.q[i - 1])
        obs_r = np.array(obs_r)
        obs_q = np.array(obs_q)
        
        init_est = [0.1, # v
                    3, # kappa
                    0.05, # theta
                    0.3, # sigma
                    -0.8, # rho
                    0.03 # lam
                   ]
        boundaries = [[0.001, 0.1], # v
                      [0.001, 5], # kappa
                      [0.001, 0.1], # theta
                      [0.01, 1], # sigma
                      [-1, 0], # rho
                      [-1, 1] # lam
                     ]
        
        result = minimize(mse, init_est, method='SLSQP', bounds=boundaries)
        
        self.v, self.kappa, self.theta, self.sigma, self.rho, self.lam = [x for x in result.x]
        
        
    def heston_mle(self, lam):
        '''
        Estimate Heston parameters by Maximium Likelihood
        '''
        self.lam = lam
        # extract the index spot time series
        ticker = self.data.ticker.split(' ')[0].lower()
        date = self.data.date.replace('-', '')
        if ticker == 'shsn300':
            ticker = 'csi300'
        
        spot_ts = []
        r_ts = []
        q_ts = []
        for filename in os.listdir(f'data_{ticker}'):
            f = os.path.join(f'data_{ticker}', filename)
            d = f.split('_')[2].split('.')[0]
            if int(d) <= int(date):
                try:
                    pkl_file = open(f, 'rb')
                    raw = pickle.load(pkl_file)
                    bs = black_scholes(raw)
                    spot_ts.append(raw.spot.mid)
                    r_ts.append(bs.r[0])
                    q_ts.append(bs.q[0])
                except:
                    pass
                
        spot_ts = np.array(spot_ts)
        # R doesnt have t = 1
        R_ts = spot_ts[1:]/spot_ts[:-1]
        
        # v doesnt have t = 1 and t = n
        v_ts = []
        for i in range(len(R_ts)):
            if i == 0:
                pass
            else:
                v_ts.append(np.var(R_ts[:i+1]))
        v_ts = np.array(v_ts)
        
        # matching the time series
        v_t = np.array(v_ts[0:-1])
        v_t1 = np.array(v_ts[1:])
        R_t1 = np.array(R_ts[1:-1])
        r_t = np.array(r_ts[1:-2])
        q_t = np.array(q_ts[1:-2])
        
        def nmle(x):
            kappa, theta, sigma, rho = [param for param in x]

            log_lik = - np.sum(-np.log(sigma) - np.log(v_t) - 0.5*np.log(1-rho**2)\
                              - (R_t1 - 1 - (r_t - q_t))**2/(2*v_t*(1-rho**2))\
                              + (rho * (R_t1 - 1 - (r_t - q_t)) * (v_t1 - v_t - kappa * (theta - v_t)))/(v_t * sigma * (1-rho**2))\
                              - (v_t1 - v_t - kappa * (theta - v_t))**2/(2*sigma**2*v_t*(1-rho**2)))
            return log_lik
        

        init_est = [3, # kappa
                    0.05, # theta
                    0.3, # sigma
                    -0.8, # rho
                   ]
        boundaries = [[0.001, 5], # kappa
                      [0.001, 0.1], # theta
                      [0.01, 1], # sigma
                      [-1, 0], # rho
                     ]
        
        result = minimize(nmle, init_est, method='Powell', options={'maxiter': 1e4}, bounds=boundaries)
        
        self.kappa, self.theta, self.sigma, self.rho = [x for x in result.x]
        
        def nmle1(v):
            return - (- np.log(v_t[0]) - 0.5*np.log(1-self.rho**2)\
                              - (R_t1[0] - 1 - (r_t[0] - q_t[0]))**2/(2*v_t[0]*(1-self.rho**2))\
                              + (self.rho * (R_t1[0] - 1 - (r_t[0] - q_t[0])) * (v_t1[0] - v_t[0] - self.kappa * (self.theta - v_t[0])))\
                                  /(v_t[0] * self.sigma * (1-self.rho**2))\
                              - (v_t1[0] - v_t[0] - self.kappa * (self.theta - v_t[0]))**2/(2*self.sigma**2*v_t[0]*(1-self.rho**2)))
        
        init_est = [0.05]
        boundaries = [[1e-3,0.1]]
        
        result = minimize(nmle1, init_est, method='Powell', bounds=boundaries)
        
        self.v = result.x[0]
        
        
    def mse_performance(self):
        '''
        Calculate the mse against Heston price
        '''
        def C_1(u, r, q, tau, kappa, theta, sigma, lam, rho):
            d = np.sqrt((rho * sigma * 1j * u + rho*sigma - kappa - sigma)**2 - sigma**2 * (1j*u - u**2))
            g = (kappa + lam - rho*sigma - rho*sigma * 1j * u + d)/(kappa + lam - rho*sigma - rho*sigma * 1j * u - d)
            return (r - q) * 1j * u *tau + kappa * theta/(sigma**2) * (((kappa + lam - rho*sigma) - rho*sigma*1j*u + d)*tau - 2*np.log((1 - g * np.exp(d * tau))/(1 - g)))

        def C_2(u, r, q, tau, kappa, theta, sigma, lam, rho):
            d = np.sqrt((rho * sigma * 1j * u - kappa - sigma)**2 - sigma**2 * (-1j*u - u**2))
            g = (kappa + lam - rho*sigma * 1j * u + d)/(kappa + lam - rho*sigma * 1j * u - d)
            return (r - q) * 1j * u *tau + kappa * theta/(sigma**2) * (((kappa + lam) - rho*sigma*1j*u + d)*tau - 2*np.log((1 - g * np.exp(d * tau))/(1 - g)))

        def D_1(u, tau, kappa, sigma, lam, rho):
            d = np.sqrt((rho * sigma * 1j * u + rho*sigma - kappa - sigma)**2 - sigma**2 * (1j*u - u**2))
            g = (kappa + lam - rho*sigma - rho*sigma * 1j * u + d)/(kappa + lam - rho*sigma - rho*sigma * 1j * u - d)
            return (((kappa + lam - rho*sigma) - rho * sigma * 1j * u + d)/(sigma**2)) * ((1 - np.exp(d*tau))/(1 - g*np.exp(d*tau)))

        def D_2(u, tau, kappa, sigma, lam, rho):
            d = np.sqrt((rho * sigma * 1j * u - kappa - sigma)**2 - sigma**2 * (-1j*u - u**2))
            g = (kappa + lam - rho*sigma * 1j * u + d)/(kappa + lam - rho*sigma * 1j * u - d)
            return (((kappa + lam) - rho * sigma * 1j * u + d)/(sigma**2)) * ((1 - np.exp(d*tau))/(1 - g*np.exp(d*tau)))

        def phi_1(u, v, x, r, q, tau, kappa, theta, sigma, lam, rho):
            C = C_1(u, r, q, tau, kappa, theta, sigma, lam, rho)
            D = D_1(u, tau, kappa, sigma, lam, rho)
            return np.exp(C + D * v + 1j * u * x)

        def phi_2(u, v, x, r, q, tau, kappa, theta, sigma, lam, rho):
            C = C_2(u, r, q, tau, kappa, theta, sigma, lam, rho)
            D = D_2(u, tau, kappa, sigma, lam, rho)
            return np.exp(C + D * v + 1j * u * x)        
        
        
        def heston_price(S, K, v, kappa, theta, sigma, rho, lam, tau, r, q):            
            sum_int = 0
            steps = 10000
            upper_bound = 100
            du = upper_bound / steps
            x = np.log(S)
            for i in range(1, steps):
                u = du * (2*i + 1)/2
                sum_int += du * np.exp(-1j*u*np.log(K))/(1j*u)*(S*np.exp(-q*tau)\
                                            *phi_1(u, v, x, r, q, tau, kappa, theta, sigma, lam, rho) \
                                            - K*np.exp(-r*tau)*phi_2(u, v, x, r, q, tau, kappa, theta, sigma, lam, rho))

            return np.real(sum_int)/np.pi + 0.5*(S*np.exp(-q*tau) - K*np.exp(-r*tau))
        
        
        obs_r = []
        obs_q = []
        for T in self.obs_exps:
            i = self.rank_T(T)
            if i == 0:    
                i += 1
            elif i == len(self.Ts):
                i -= 1
            obs_r.append((self.r[i] - self.r[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.r[i - 1])
            obs_q.append((self.q[i] - self.q[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.q[i - 1])
        obs_r = np.array(obs_r)
        obs_q = np.array(obs_q)
        
        mse = 0
        for exp, strike, mkt_price, r, q in zip(self.obs_exps, self.obs_strikes, self.obs_prices, obs_r, obs_q):
        
            mse += (mkt_price - heston_price(self.data.spot.mid, strike, \
                                self.v, self.kappa, self.theta, self.sigma, self.rho, self.lam, exp / 365, r, q))**2
            
        return mse/len(self.obs_prices)