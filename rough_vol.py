from constant_vol import black_scholes
import stochastic_vol

import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.integrate import quad
import plotly.graph_objects as go
import math
from scipy.optimize import minimize
from functools import partial
from scipy.special import gamma


class rough_volatility(black_scholes):
    def __init__(self, snapshot):
        super().__init__(snapshot)
        '''
        This class inherits the black_scholes class.
        The additional functions include:
            1. Computing rough volatility
        '''
        self.name = 'rv'
        self.root_finding('halley')
        
        # use only observable prices
        obs_idx = np.argwhere(~np.isnan(self.imp_vol_call_bids)).flatten()
        self.obs_exps = np.array(self.days_to_exp)[[obs_idx]].flatten()
        self.obs_strikes = np.array(self.strikes)[[obs_idx]].flatten()
        self.obs_prices = np.array(self.data.calls.bids)[[obs_idx]].flatten()
        self.obs_imp_vol = np.array(self.imp_vol_call_bids)[[obs_idx]].flatten()
        self.Ts = sorted(list(set(self.obs_exps)))
        self.n = 10
        self.v = None
        self.kappa = None
        self.theta = None
        self.sigma = None
        self.lam = None
        self.rho = None
        self.H = None

        
    def rank_T(self, T):
            '''
            Helper function to determine which interval T lies in
            '''
            for idx, x in enumerate(self.Ts):
                if T < x:
                    return idx
            return len(self.Ts)
        
        
    def rheston_mse(self):
        
        def rHeston_call(S, K, v, r, q, tau, kappa, theta, sigma, lam, rho, H):
            phi_p = partial(phi, v, tau, kappa, theta, sigma, lam, rho, H)
            return S*np.exp(-q*tau) - np.sqrt(S*K)*np.exp(-(r+q)*tau/2)/np.pi*real_integral(S, K, phi_p)
        
        def real_integral(S, K, phi_f):

            int_sum, umax, N = 0, 100, 10
            dh=umax/N #dphi is width

            for i in range(1,N):
                # rectangular integration
                h = dh * (2*i + 1)/2 # midpoint to calculate height
                numerator = np.exp(-1j*h*np.log(K/S)) * phi_f(h - 1j/2)
                denominator = (h**2 + 1/4)
                int_sum += dh * numerator/denominator

            return np.real(int_sum)
        
        def phi(v, tau, kappa, theta, sigma, lam, rho, H, x):
            integral = 0
            self.dt = tau/self.n
            for k in range(1, self.n):
                integral += u_hat(x, k, H, lam, rho, sigma)
            
            frac_sum = 0
            for k in range(1, self.n):
                t = self.dt * (2*k + 1)/2
                t1 = self.dt * (2*(k + 1) + 1)/2
                frac_sum += ((tau - t)**(1/2 - H) - (tau - t1)**(1/2 - H))*u_hat(x, k, H, lam, rho, sigma)
            frac_integral = 1 / (gamma(1/2 - H) ** 2) * frac_sum
            return np.exp(theta*lam*integral + v*frac_integral)
        
        def G(eta, uh, lam, rho, sigma):
            return 1/2*(-eta**2 - 1j*eta) + lam*(1j*eta*rho*sigma - 1)*uh + ((lam**2 * sigma**2)*(uh)**2)/2
            
        def z(j, i, H):
            if j == 0:
                return self.dt**(H + 1/2) / gamma(H + 5/2) * (i**(H + 3/2) - (i - H - 1/2)*(i + 1)**(H + 1/2))
            else:
                return self.dt**(H + 1/2) / gamma(H + 5/2) * ((i - j + 2)**(H + 3/2) + (i - j)**(H + 3/2) - 2*(i - j + 1)**(H + 3/2))
            
        def up(eta, i, H, lam, rho, sigma):
            up_sum = 0
            for j in range(i):
                up_sum += (self.dt**(H + 1/2) / gamma(H + 3/2)*((i - j + 1)**(H + 1/2) - (i - j)**(H + 1/2))) * G(eta, u_hat(eta, j, H, lam, rho, sigma), lam, rho, sigma)
            return up_sum

        def u_hat(eta, i, H, lam, rho, sigma):
            sum_term = 0
            for j in range(i):
                sum_term += z(j, i + 1, H) * G(eta, u_hat(eta, j, H, lam, rho, sigma), lam, rho, sigma)
            return self.dt**(H + 1/2) / gamma(H + 5/2) * G(eta, up(eta, i, H, lam, rho, sigma), lam, rho, sigma) + sum_term
            
        
        def mse(x):
            v, kappa, theta, sigma, rho, lam, H = [param for param in x]

            err = np.sum((P - rHeston_call(S0, K, v, r, q, tau, kappa, theta, sigma, lam, rho, H))**2 / len(P))

            return err
        
        S0 = self.data.spot.mid
        K = self.obs_strikes
        tau = self.obs_exps / 365
        P = self.obs_prices
        
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
        r = np.array(obs_r)
        q = np.array(obs_q)
        
        init_est = [0.1, # v
                    3, # kappa
                    0.05, # theta
                    0.3, # sigma
                    -0.8, # rho
                    0.03, # lam
                    0.25
                   ]
        boundaries = [[0.001, 0.1], # v
                      [0.001, 5], # kappa
                      [0.001, 0.1], # theta
                      [0.01, 1], # sigma
                      [-1, 0], # rho
                      [-1, 1], # lam
                      [0.001, 0.5] # H
                     ]
        
        parameters = {"v0": {"init_est": 0.05, "limits": [1e-3,0.1]},
                  "kappa": {"init_est": 2, "limits": [1e-3,5]},
                  "theta": {"init_est": 0.04, "limits": [1e-3,0.1]},
                  "sigma": {"init_est": 0.5, "limits": [1e-2,1]},
                  "rho": {"init_est": -0.5, "limits": [-1,0]},
                  "lam": {"init_est": 0.05, "limits": [-1,1]},
                  "H": {"init_est": 0.25, "limits": [1e-3,0.5]},
                  }

        init_est = [param["init_est"] for key, param in parameters.items()]
        boundaries = [param["limits"] for key, param in parameters.items()]
        
        result = minimize(mse, init_est, method='CG', options={'maxiter': 10}, bounds=boundaries)
        
        self.v, self.kappa, self.theta, self.sigma, self.rho, self.lam, self.H = [x for x in result.x]
        
        
        
    def rheston_heston_H(self):
        
        def rHeston_call(S, K, v, r, q, tau, kappa, theta, sigma, lam, rho, H):
            phi_p = partial(phi, v, tau, kappa, theta, sigma, lam, rho, H)
            return S*np.exp(-q*tau) - np.sqrt(S*K)*np.exp(-(r+q)*tau/2)/np.pi*real_integral(S, K, phi_p)
        
        def real_integral(S, K, phi_f):

            int_sum, umax, N = 0, 5, 10
            dh=umax/N #dphi is width

            for i in range(1,N):
                # rectangular integration
                h = dh * (2*i + 1)/2 # midpoint to calculate height
                numerator = np.exp(-1j*h*np.log(K/S)) * phi_f(h - 1j/2)
                denominator = (h**2 + 1/4)
                int_sum += dh * numerator/denominator

            return np.real(int_sum)
        
        def phi(v, tau, kappa, theta, sigma, lam, rho, H, x):
            integral = 0
            self.dt = tau/self.n
            for k in range(1, self.n):
                integral += u_hat(x, k, H, lam, rho, sigma)
            
            frac_sum = 0
            for k in range(1, self.n):
                t = self.dt * (2*k + 1)/2
                t1 = self.dt * (2*(k + 1) + 1)/2
                frac_sum += ((tau - t)**(1/2 - H) - (tau - t1)**(1/2 - H))*u_hat(x, k, H, lam, rho, sigma)
            frac_integral = 1 / (gamma(1/2 - H) ** 2) * frac_sum
            return np.exp(theta*lam*integral + v*frac_integral)
        
        def G(eta, uh, lam, rho, sigma):
            return 1/2*(-eta**2 - 1j*eta) + lam*(1j*eta*rho*sigma - 1)*uh + ((lam**2 * sigma**2)*(uh)**2)/2
            
        def z(j, i, H):
            if j == 0:
                return self.dt**(H + 1/2) / gamma(H + 5/2) * (i**(H + 3/2) - (i - H - 1/2)*(i + 1)**(H + 1/2))
            else:
                return self.dt**(H + 1/2) / gamma(H + 5/2) * ((i - j + 2)**(H + 3/2) + (i - j)**(H + 3/2) - 2*(i - j + 1)**(H + 3/2))
            
        def up(eta, i, H, lam, rho, sigma):
            up_sum = 0
            for j in range(i):
                up_sum += (self.dt**(H + 1/2) / gamma(H + 3/2)*((i - j + 1)**(H + 1/2) - (i - j)**(H + 1/2))) * G(eta, u_hat(eta, j, H, lam, rho, sigma), lam, rho, sigma)
            return up_sum

        def u_hat(eta, i, H, lam, rho, sigma):
            sum_term = 0
            for j in range(i):
                sum_term += z(j, i + 1, H) * G(eta, u_hat(eta, j, H, lam, rho, sigma), lam, rho, sigma)
            return self.dt**(H + 1/2) / gamma(H + 5/2) * G(eta, up(eta, i, H, lam, rho, sigma), lam, rho, sigma) + sum_term
            
        
        def mse(H):

            err = np.sum((P - rHeston_call(S0, K, self.v, r, q, tau, self.kappa, self.theta, self.sigma, self.lam, self.rho, H))**2 / len(P))

            return err
        
        S0 = self.data.spot.mid
        K = self.obs_strikes
        tau = self.obs_exps / 365
        P = self.obs_prices
        
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
        r = np.array(obs_r)
        q = np.array(obs_q)
        
        sv_model = stochastic_vol.stochastic_volatility(self.data)
        sv_model.heston_mse2()
        self.v = sv_model.v
        self.kappa = sv_model.kappa
        self.theta = sv_model.theta
        self.sigma = sv_model.sigma
        self.lam = sv_model.lam
        self.rho = sv_model.rho
        
        init_est = [0.25]
        boundaries = [[1e-3,0.5]]
        
        result = minimize(mse, init_est, method='Powell', options={'maxiter': 10}, bounds=boundaries)
        
        self.H = result.x[0]
        
        
    def mse_performance(self):
        '''
        Calculate the mse against rough Heston price with pre-set H
        '''
        sv_model = stochastic_vol.stochastic_volatility(self.data)
        sv_model.heston_mse2()
        self.v = sv_model.v
        self.kappa = sv_model.kappa
        self.theta = sv_model.theta
        self.sigma = sv_model.sigma
        self.lam = sv_model.lam
        self.rho = sv_model.rho
        self.H = 0.1
        
        def rHeston_call(S, K, v, r, q, tau, kappa, theta, sigma, lam, rho, H):
            phi_p = partial(phi, v, tau, kappa, theta, sigma, lam, rho, H)
            return S*np.exp(-q*tau) - np.sqrt(S*K)*np.exp(-(r+q)*tau/2)/np.pi*real_integral(S, K, phi_p)
        
        def real_integral(S, K, phi_f):

            int_sum, umax, N = 0, 5, 10
            dh=umax/N #dphi is width

            for i in range(1,N):
                # rectangular integration
                h = dh * (2*i + 1)/2 # midpoint to calculate height
                numerator = np.exp(-1j*h*np.log(K/S)) * phi_f(h - 1j/2)
                denominator = (h**2 + 1/4)
                int_sum += dh * numerator/denominator

            return np.real(int_sum)
        
        def phi(v, tau, kappa, theta, sigma, lam, rho, H, x):
            integral = 0
            self.dt = tau/self.n
            for k in range(1, self.n):
                integral += u_hat(x, k, H, lam, rho, sigma)
            
            frac_sum = 0
            for k in range(1, self.n):
                t = self.dt * (2*k + 1)/2
                t1 = self.dt * (2*(k + 1) + 1)/2
                frac_sum += ((tau - t)**(1/2 - H) - (tau - t1)**(1/2 - H))*u_hat(x, k, H, lam, rho, sigma)
            frac_integral = 1 / (gamma(1/2 - H) ** 2) * frac_sum
            return np.exp(theta*lam*integral + v*frac_integral)
        
        def G(eta, uh, lam, rho, sigma):
            return 1/2*(-eta**2 - 1j*eta) + lam*(1j*eta*rho*sigma - 1)*uh + ((lam**2 * sigma**2)*(uh)**2)/2
            
        def z(j, i, H):
            if j == 0:
                return self.dt**(H + 1/2) / gamma(H + 5/2) * (i**(H + 3/2) - (i - H - 1/2)*(i + 1)**(H + 1/2))
            else:
                return self.dt**(H + 1/2) / gamma(H + 5/2) * ((i - j + 2)**(H + 3/2) + (i - j)**(H + 3/2) - 2*(i - j + 1)**(H + 3/2))
            
        def up(eta, i, H, lam, rho, sigma):
            up_sum = 0
            for j in range(i):
                up_sum += (self.dt**(H + 1/2) / gamma(H + 3/2)*((i - j + 1)**(H + 1/2) - (i - j)**(H + 1/2))) * G(eta, u_hat(eta, j, H, lam, rho, sigma), lam, rho, sigma)
            return up_sum

        def u_hat(eta, i, H, lam, rho, sigma):
            sum_term = 0
            for j in range(i):
                sum_term += z(j, i + 1, H) * G(eta, u_hat(eta, j, H, lam, rho, sigma), lam, rho, sigma)
            return self.dt**(H + 1/2) / gamma(H + 5/2) * G(eta, up(eta, i, H, lam, rho, sigma), lam, rho, sigma) + sum_term
        
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
            rhc = rHeston_call(self.data.spot.mid, strike, self.v, r, q, exp/360, self.kappa, self.theta, self.sigma, self.lam, self.rho, self.H)
            mse += (mkt_price - rhc)**2
            
        return mse/len(self.obs_prices)