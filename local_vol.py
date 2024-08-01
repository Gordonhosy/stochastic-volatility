from constant_vol import black_scholes

import numpy as np
import pandas as pd
from scipy import stats, optimize
from functools import partial
from itertools import compress
import plotly.graph_objects as go
import cvxopt as opt
from cvxopt import matrix, spmatrix, sparse
from cvxopt.solvers import qp, options
from cvxopt import blas
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import Rbf
import math


class local_volatility(black_scholes):
    def __init__(self, snapshot):
        super().__init__(snapshot)
        '''
        This class inherits the black_scholes class.
        Local volatility model is calibrated according to Fengler (2009). Functions include:
        1. Smoothing the implied volatility surface
        2. Calibrating cubic splines
        3. FDM for pricing
        4. Ploting functions
        '''
        self.name = 'lv'
        self.root_finding('halley')
        self.fdm_diff = 1
        
        # use only observable prices
        obs_idx = np.argwhere(~np.isnan(self.imp_vol_call_bids)).flatten()
        self.obs_exps = np.array(self.days_to_exp)[[obs_idx]].flatten()
        self.obs_strikes = np.array(self.strikes)[[obs_idx]].flatten()
        self.obs_prices = np.array(self.data.calls.bids)[[obs_idx]].flatten()
        self.obs_imp_vol = np.array(self.imp_vol_call_bids)[[obs_idx]].flatten()
        self.Ts = sorted(list(set(self.obs_exps)))
        self.Ms = dict()
        self.Ks = dict()
        self.obs_moneyness = self.calc_moneyness()
        self.splines = dict()
        
        _, _, self.init_estimate = self.call_price_smoothed2()
        
        self.m_space = None
        self.T_space = None
        self.call_surface = None
        self.iv_surface = None
        self.lv_surface = None
    
    def find_K(self, m, T):
        '''
        Return strike given moneyness and maturity
        '''
        i = self.rank_T(T)
        if i == 0:    
            i += 1
        elif i == len(self.Ts):
            i -= 1
        r = (self.r[i] - self.r[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.r[i - 1]
        q = (self.q[i] - self.q[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.q[i - 1]
        return m * self.data.spot.mid * np.exp((r - q) * T / 360)
    
    def find_m(self, K, T):
        '''
        Return moneyness given strike and maturity
        '''
        i = self.rank_T(T)
        if i == 0:    
            i += 1
        elif i == len(self.Ts):
            i -= 1
        r = (self.r[i] - self.r[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.r[i - 1]
        q = (self.q[i] - self.q[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.q[i - 1]
        return (np.exp((q - r) * T / 360) * K) / self.data.spot.mid
        
        
    def fengler(self):
        '''
        Calibrate local volatility according to Fengler (2009)
        '''
        # group the call prices by maturity
        calls = dict() 
        i = 0
        for T in self.Ts:
            calls[T] = []
            for m in self.Ms[T]:
                calls[T].append(self.init_estimate[i])
                i += 1
                
        # for each maturity, fit cubic spline
        for T in reversed(self.Ts):
            n = len(self.Ks[T])
            h = np.diff(self.Ks[T])
        
            Q = np.zeros((n, n))
            for i in range(1, n + 1):
                for j in range(2, n):
                    if i + 1 == j:
                        Q[i - 1][j - 1] = 1/h[j - 1 - 1]
                    elif i == j:
                        Q[i - 1][j - 1] = -1/h[j - 1 - 1] - 1/h[j - 1]
                    elif i - 1 == j:
                        Q[i - 1][j - 1] = 1/h[j - 1]
            Q = Q[:,1:-1]
            
            R = np.zeros((n, n))
            for i in range(2, n):
                for j in range(2, n):
                    if i == j:
                        R[i - 1][j - 1] = (h[i - 1 - 1] + h[i - 1])/3
                    elif i - j == 1:
                        R[i - 1][j - 1] = h[j - 1] / 6
                    elif j - i == 1:
                        R[i - 1][j - 1] = h[i - 1] / 6
            R = R[1:-1, 1:-1]
            
            y = np.concatenate((np.array(calls[T]), np.zeros(n - 2)), axis = None)
            y = np.reshape(y, (len(y), 1))
            
            W = np.identity(n)
            
            B = np.concatenate((np.concatenate((W, np.zeros((n, n - 2))), axis = 1), \
                                np.concatenate((np.zeros((n - 2, n)), 1e-9*R), axis = 1)), axis = 0)
            
            A = np.concatenate((Q.T, -R), axis = 1).T
            
            cond1_l = -np.concatenate((np.zeros((n - 2, n)), np.identity(n - 2)), axis = 1)
            cond1_r = np.zeros((n - 2, 1))
            
            cond2_l = -np.concatenate((np.array([-1/h[0], 1/h[0]]), np.zeros(n - 2), np.array([-h[0] / 6]), np.zeros(n - 2 - 1)),
                                   axis = None).reshape((1, 2 * n - 2))
            cond2_r = np.array([-np.exp(-self.fwd_r(T, max(self.data.calls.days_to_exp)))]).reshape((1,1))
            
            cond3_l = np.zeros((n - 2, 2 * n - 2))
            for i in range(n - 2):
                for j in range(2 * n - 2):
                    if j - i == 1:
                        cond3_l[i][j] = -1 / h[i + 1]
                    elif j - i == 2:
                        cond3_l[i][j] = 1 / h[i + 1]
                    elif j - i == n:
                        cond3_l[i][j] = h[i + 1] / 6
            cond3_r = np.zeros((n - 2, 1))
            
            if T == max(self.Ts):
                cond4_l = np.concatenate((np.array([1]), np.zeros(2 * n - 3)), axis = None).reshape((1, 2 * n - 2))
                cond4_r = np.array([self.data.spot.mid * np.exp(-self.fwd_q(T, max(self.data.calls.days_to_exp)))]).reshape((1,1))
            else:
                cond4_l = np.concatenate((np.identity(n), np.zeros((n, n - 2))), axis = 1)
                fwd_q = self.fwd_q(T, next_T)
                cond4_r = np.array([np.exp(fwd_q) * next_g_i for next_g_i in self.splines[next_T].g]).reshape((n, 1))
                
            cond5_l = -np.concatenate((np.array([1]), np.zeros(2 * n - 3)), axis = None).reshape((1, 2 * n - 2))
            cond5_r = np.array([np.exp(-self.fwd_q(T, max(self.data.calls.days_to_exp))) * self.data.spot.mid - np.exp(-self.fwd_r(T, max(self.data.calls.days_to_exp))) * self.Ks[T][0]]).reshape((1,1))
            
            cond6_l = -np.concatenate((np.identity(n), np.zeros((n, n - 2))), axis = 1)
            cond6_r = np.zeros((n, 1))
            
            G = np.concatenate((cond1_l, cond3_l, cond4_l, cond5_l, cond6_l), axis = 0)
            H = np.concatenate((cond1_r, cond3_r, cond4_r, cond5_r, cond6_r), axis = 0)
            
            sol = qp(matrix(B), matrix(-y), matrix(G), matrix(H), matrix(A.T), matrix(np.zeros((n - 2, 1))))
            
            self.splines[T] = self.cubic_spline(list(np.array(sol['x']).flatten()), self.Ks[T], h)
            
            next_T = T
            
    def fwd_r(self, T1, T2):
        '''
        Calculate the continuous forward rate
        '''
        if T1 == T2:
            return 0
        else:
            i = self.data.calls.days_to_exp.index(T1)
            j = self.data.calls.days_to_exp.index(T2)
            return (self.r[j] * T2 - self.r[i] * T1) / 360
        
    def fwd_q(self, T1, T2):
        '''
        Calculate the continuous forward implied q
        '''
        if T1 == T2:
            return 0
        else:
            i = self.data.calls.days_to_exp.index(T1)
            j = self.data.calls.days_to_exp.index(T2)
            return (self.q[j] * T2 - self.q[i] * T1) / 360
            
    def rank_T(self, T):
            '''
            Helper function to determine which interval T lies in
            '''
            for idx, x in enumerate(self.Ts):
                if T < x:
                    return idx
            return len(self.Ts)
        
    def dcdT(self, K, T):
        '''
        Calculate the first partial derivative w.r.t. T by finite difference
        '''
        if T in self.Ts:
            i = self.Ts.index(T)
            if i == 0:
                return (self.splines[self.Ts[i + 1]].predict(K) - self.splines[self.Ts[i]].predict(K))\
            /(self.Ts[i + 1] - self.Ts[i]) * 360
            elif i == len(self.Ts) - 1:
                return (self.splines[self.Ts[i]].predict(K) - self.splines[self.Ts[i - 1]].predict(K))\
            /(self.Ts[i] - self.Ts[i - 1]) * 360
            else:
                return (self.splines[self.Ts[i + 1]].predict(K) - self.splines[self.Ts[i - 1]].predict(K))\
            /(self.Ts[i + 1] - self.Ts[i - 1]) * 360
        else:
            i = self.rank_T(T)
            # assume linear trend for extrapolation
            if i == 0:
                i += 1
            elif i == len(self.Ts):
                i -= 1
            return (self.splines[self.Ts[i]].predict(K) - self.splines[self.Ts[i - 1]].predict(K))\
            /(self.Ts[i] - self.Ts[i - 1]) * 360
    
    def c(self, K, T):
        '''
        Return interpolated call price at any point
        '''
        if self.call_surface is not None:
            i = np.absolute(self.m_space - self.find_m(K, T)).argmin()
            j = list(self.T_space).index(T)
            return self.call_surface[i][j]
        else:
            i = self.rank_T(T)
            if i == 0:    
                i += 1
            elif i == len(self.Ts):
                i -= 1
            c = self.dcdT(K, T)/360 * (T - self.Ts[i - 1]) + self.splines[self.Ts[i - 1]].predict(K)
            if c > 0:
                return c
            else:
                return 0
        
    def dcdK(self, K, T):
        '''
        Calculate the first partial derivative w.r.t. K by hybrid methods
        '''
        if T in self.Ts:
            return self.splines[T].dcdK(K)
        else:
            h = self.fdm_diff
            c1 = self.c(K + h, T)
            c2 = self.c(K - h, T)
            return (c1 - c2) / (2 * h)   
    
    def d2cdK2(self, K, T):
        '''
        Calculate the second partial derivative w.r.t. K by hybrid methods
        '''
        if T in self.Ts:
            return self.splines[T].d2cdK2(K)
        else:
            i = self.rank_T(T)
            h = self.fdm_diff
            c1 = self.c(K + h, T)
            c2 = self.c(K - h, T)
            c = self.c(K, T)
            return (c1 + c2 - 2 * c) / (h ** 2)
    
    def lv_call(self, K, T):
        '''
        Return the local volatility with Dupire's formula
        '''
        i = self.rank_T(T)
        if i == 0:    
            i += 1
        elif i == len(self.Ts):
            i -= 1
        r = (self.r[i] - self.r[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.r[i - 1]
        q = (self.q[i] - self.q[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.q[i - 1]
        return np.sqrt((self.dcdT(K, T) + (r - q) * K * self.dcdK(K, T) + q * self.c(K, T)) / (0.5 * K**2 * self.d2cdK2(K, T)))
        
    def lv_iv(self, K, T):
        '''
        Return the local volatility with Dupire's formula written in implied vol
        '''
        i = self.rank_T(T)
        if i == 0:    
            i += 1
        elif i == len(self.Ts):
            i -= 1
        r = (self.r[i] - self.r[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.r[i - 1]
        q = (self.q[i] - self.q[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.q[i - 1]

        d1 = self.d1(self.data.spot.mid, K, r, q, self.iv(K, T), T/360)
        return np.sqrt((2 * self.divdT(K, T) + (self.iv(K, T) / T * 360) + 2 * K * (r - q) * self.divdK(K, T)) \
                / (K**2 * (self.d2ivdK2(K, T) - d1 * np.sqrt(T / 360) * self.divdK(K, T)**2 + (1/(K * np.sqrt(T / 360)) \
                + d1 * self.divdK(K, T))**2 / self.iv(K, T))))
    
    def iv(self, K, T):
        '''
        Return implied volatility of any point
        '''
        if self.iv_surface is not None:
            i = np.absolute(self.m_space - self.find_m(K, T)).argmin()
            j = list(self.T_space).index(T)
            return self.iv_surface[i][j]
        else:
            i = self.rank_T(T)
            if i == 0:    
                i += 1
            elif i == len(self.Ts):
                i -= 1
            r = (self.r[i] - self.r[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.r[i - 1]
            q = (self.q[i] - self.q[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.q[i - 1]
            return self.calc_single_implied_vol(call_price[j], self.data.spot.mid, K, r, q, T, 0.25, self.bs_call, 'halley')
        
    def divdK(self, K, T):
        '''
        Calculate the first partial derivative of imp. vol. w.r.t. K by central FDM
        '''
        i = np.absolute(self.m_space - self.find_m(K, T)).argmin()
        j = list(self.T_space).index(T)
        if i == 0:
            i += 1
        elif i == len(self.m_space) - 1:
            i -= 1
        return (self.iv_surface[i + 1][j] - self.iv_surface[i - 1][j]) / (self.find_K(self.m_space[i + 1], T) - self.m_space[i - 1])
    
    def d2ivdK2(self, K, T):
        '''
        Calculate the second partial derivative of imp. vol. w.r.t. K by central FDM
        '''
        i = np.absolute(self.m_space - self.find_m(K, T)).argmin()
        j = list(self.T_space).index(T)
        if i == 0:
            i += 1
        elif i == len(self.m_space) - 1:
            i -= 1
        return (self.iv_surface[i + 1][j] + self.iv_surface[i - 1][j] - 2 * self.iv_surface[i][j]) / ((self.find_K(self.m_space[i + 1], T) - self.m_space[i - 1])**2)
    
    def divdT(self, K, T):
        '''
        Calculate the first partial derivative of imp. vol. w.r.t. T by central FDM
        '''
        i = np.absolute(self.m_space - self.find_m(K, T)).argmin()
        j = list(self.T_space).index(T)
        if j == 0:
            j += 1
        elif j == len(self.T_space) - 1:
            j -= 1
        return (self.iv_surface[i][j + 1] - self.iv_surface[i][j - 1]) / (self.T_space[j + 1] - self.T_space[j - 1]) * 360
    
    def lv_tvar(self, K, T):
        '''
        Return the local volatility with Gatheral's formula
        '''
        w = self.tvar(K, T)
        y = self.y(K, T)
        
        return np.sqrt(self.dwdT(K, T) / (1 - y / w * self.dwdy(K, T) + self.d2wdy2(K, T) / 2 \
                + (-1 / 4 - 1 / w + y**2 / w) * self.dwdy(K, T)**2 / 4))
    
    def y(self, K, T):
        '''
        Return y given K and T
        '''
        i = self.rank_T(T)
        if i == 0:    
            i += 1
        elif i == len(self.Ts):
            i -= 1
        r = (self.r[i] - self.r[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.r[i - 1]
        q = (self.q[i] - self.q[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.q[i - 1]
        return np.log((np.exp((q - r) * T / 360) * K) / self.data.spot.mid)
    
    def tvar(self, K, T):
        '''
        Calculate the total implied variance given a point
        '''
        if self.tvar_surface is not None:
            i = np.absolute(self.m_space - self.find_m(K, T)).argmin()
            j = list(self.T_space).index(T)
            return self.tvar_surface[i][j]
        else:
            i = self.rank_T(T)
            if i == 0:    
                i += 1
            elif i == len(self.Ts):
                i -= 1
            r = (self.r[i] - self.r[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.r[i - 1]
            q = (self.q[i] - self.q[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.q[i - 1]
            return self.calc_single_implied_vol(call_price[j], self.data.spot.mid, K, r, q, T, 0.25, self.bs_call, 'halley')**2 * T / 360
    
    def dwdT(self, K, T):
        '''
        Calculate the first partial derivative of total var w.r.t. T by central FDM
        '''
        i = np.absolute(self.m_space - self.find_m(K, T)).argmin()
        j = list(self.T_space).index(T)
        if j == 0:
            j += 1
        elif j == len(self.T_space) - 1:
            j -= 1
        return (self.tvar_surface[i][j + 1] - self.tvar_surface[i][j - 1]) / (self.T_space[j + 1] - self.T_space[j - 1]) * 360

    def dwdy(self, K, T):
        '''
        Calculate the first partial derivative of total var w.r.t. y by central FDM
        '''
        i = np.absolute(self.m_space - self.find_m(K, T)).argmin()
        j = list(self.T_space).index(T)
        if i == 0:
            i += 1
        elif i == len(self.m_space) - 1:
            i -= 1
        return (self.tvar_surface[i + 1][j] - self.tvar_surface[i - 1][j]) / (self.y(self.find_K(self.m_space[i + 1], T), T) - self.y(self.m_space[i - 1], T))
    
    def d2wdy2(self, K, T):
        '''
        Calculate the second partial derivative of total var w.r.t. y by central FDM
        '''
        i = np.absolute(self.m_space - self.find_m(K, T)).argmin()
        j = list(self.T_space).index(T)
        if i == 0:
            i += 1
        elif i == len(self.m_space) - 1:
            i -= 1
        return (self.tvar_surface[i + 1][j] + self.tvar_surface[i - 1][j] - 2 * self.tvar_surface[i][j]) / ((self.y(self.find_K(self.m_space[i + 1], T), T) - self.y(self.m_space[i - 1], T))**2)    
    
    
    def grid_call(self, num_strike, num_maturity):
        '''
        Return a grid of interpolated call prices
        '''
        m_space = np.linspace(max([min(self.Ms[T]) for T in self.Ms]), min([max(self.Ms[T]) for T in self.Ms]), num_strike)
        T_space = np.linspace(0, max(self.Ts), num_maturity)
        
        call_prices = []
        m_array = []
        T_array = []
        for T in T_space:
            for m in m_space:
                K = self.find_K(m, T)
                call_prices.append(self.c(K,T))
                m_array.append(m)
                T_array.append(T)
                
        return m_array, T_array, call_prices
    
    def grid_imp_vol(self, num_strike, num_maturity):
        '''
        Return a grid of interpolated call prices
        '''
        m_array, T_array, call_price = self.grid_call(num_strike, num_maturity)
        m_space = sorted(list(set(m_array)))
        T_space = sorted(list(set(T_array)))
        j = 0
        imp_vol = []
        for T in T_space:
            i = self.rank_T(T)
            if i == 0:    
                i += 1
            elif i == len(self.Ts):
                i -= 1
            r = (self.r[i] - self.r[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.r[i - 1]
            q = (self.q[i] - self.q[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.q[i - 1]
            for m in m_space:
                K = self.find_K(m, T)
                imp_vol.append(self.calc_single_implied_vol(call_price[j], self.data.spot.mid, K, r, q, T, 0.25, self.bs_call, 'halley'))
                j += 1
        
        return m_array, T_array, imp_vol
    
    def grid_total_var(self, num_strike, num_maturity):
        '''
        Return a grid of implied total variance
        '''
        m_array, T_array, imp_vol = self.grid_imp_vol(num_strike, num_maturity)
        m_space = sorted(list(set(m_array)))
        T_space = sorted(list(set(T_array)))
        
        j = 0
        total_var = []
        y_array = []
        m_array = []
        for T in T_space:
            i = self.rank_T(T)
            if i == 0:    
                i += 1
            elif i == len(self.Ts):
                i -= 1
            r = (self.r[i] - self.r[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.r[i - 1]
            q = (self.q[i] - self.q[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.q[i - 1]
            for m in m_space:
                K = self.find_K(m, T)
                total_var.append(imp_vol[j]**2 * T / 360)
                y_array.append(np.log((np.exp((q - r) * T / 360) * K) / self.data.spot.mid))
                j += 1
                m_array.append(m)
                
        return m_array, T_array, total_var
        
    
    def grid_lv(self, num_strike, num_maturity, method):
        '''
        Return a grid of interpolated call prices
        '''
        m_space = np.linspace(max([min(self.Ms[T]) for T in self.Ms]), min([max(self.Ms[T]) for T in self.Ms]), num_strike)
        T_space = np.linspace(0, max(self.Ts), num_maturity)
        
        if method == 'var':
            _, _, tvars = self.grid_total_var(num_strike, num_maturity)
            self.tvar_surface = np.array(tvars).reshape((num_maturity, num_strike)).T
            self.m_space = m_space
            self.T_space = T_space
            lv = self.lv_tvar
        
            lvs = []
            y_array = []
            T_array = []
            m_array = []
            for T in T_space:
                for m in m_space:
                    K = self.find_K(m, T)
                    lv_tmp = lv(K,T)
                    if lv_tmp > 1:
                        lv_tmp = np.nan
                    lvs.append(lv_tmp)
                    y_array.append(self.y(K, T))
                    T_array.append(T)
                    m_array.append(m)
                    
            self.lv_surface = np.array(lvs).reshape((num_maturity, num_strike)).T

            return m_array, T_array, lvs
        
        else:
            if method == 'iv':
                _, _, ivs = self.grid_imp_vol(num_strike, num_maturity)
                self.iv_surface = np.array(ivs).reshape((num_maturity, num_strike)).T
                self.m_space = m_space
                self.T_space = T_space
                lv = self.lv_iv
            
            elif method == 'call':
                self.m_space = m_space
                self.T_space = T_space
                lv = self.lv_call
        
            lvs = []
            m_array = []
            T_array = []
            for T in T_space:
                for m in m_space:
                    K = self.find_K(m, T)
                    lvs.append(lv(K,T))
                    m_array.append(m)
                    T_array.append(T)
                    
            self.lv_surface = np.array(lvs).reshape((num_maturity, num_strike)).T
                
            return m_array, T_array, lvs
    
    def fengler_call_surface_plot(self, num_strike, num_maturity):
        '''
        Return the smoothed call surface plot
        '''
        
        x, y, z1 = self.grid_call(num_strike, num_maturity)
        x = np.array(sorted(list(set(x))))
        y = np.array(sorted(list(set(y))))
        z1 = np.array(z1).reshape((len(y), len(x)))
        
        fig = go.Figure()

        fig.add_trace(
            go.Surface(x=x, y=y, z=z1, opacity=0.8, colorscale='Sunset', colorbar_title = 'Call Prices')
        )
        fig.update_layout(
                title = dict(
                    text = r'Smoothed Call Surface',
                    x = 0.5,
                    y = 0.9
                ),
                font=dict(size=12),
                scene=dict(
                    xaxis_title='Moneyness',
                    yaxis_title='Days to Maturity',
                    zaxis_title='Call Price',
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=75, b=20),
                autosize=False,
                width=980,
                height=500,
                coloraxis=dict(colorbar_x=0.9, colorbar_thickness=23),
        )
        
        return fig
    
    def fengler_imp_vol_surface_plot(self, num_strike, num_maturity):
        '''
        Return the smoothed imp vol surface plot
        '''
        
        x, y, z1 = self.grid_imp_vol(num_strike, num_maturity)
        x = np.array(sorted(list(set(x))))
        y = np.array(sorted(list(set(y))))
        z1 = np.array(z1).reshape((len(y), len(x)))
        
        fig = go.Figure()

        fig.add_trace(
            go.Surface(x=x, y=y, z=z1, opacity=0.8, colorscale='Sunset', colorbar_title = 'IV')
        )
        fig.update_layout(
                title = dict(
                    text = r'Smoothed IV Surface',
                    x = 0.5,
                    y = 0.9
                ),
                font=dict(size=12),
                scene=dict(
                    xaxis_title='Moneyness',
                    yaxis_title='Days to Maturity',
                    zaxis_title='IV',
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=75, b=20),
                autosize=False,
                width=980,
                height=500,
                coloraxis=dict(colorbar_x=0.9, colorbar_thickness=23),
        )
        
        return fig
    
    def fengler_total_var_surface_plot(self, num_strike, num_maturity):
        '''
        Return the smoothed imp vol surface plot
        '''
        
        x, y, z1 = self.grid_total_var(num_strike, num_maturity)
        x = np.array(sorted(list(set(x))))
        y = np.array(sorted(list(set(y))))
        z1 = np.array(z1).reshape((len(y), len(x)))
        
        fig = go.Figure()

        fig.add_trace(
            go.Surface(x=x, y=y, z=z1, opacity=0.8, colorscale='Sunset', colorbar_title = 'Total Var')
        )
        fig.update_layout(
                title = dict(
                    text = r'Smoothed Total Var Surface',
                    x = 0.5,
                    y = 0.9
                ),
                font=dict(size=12),
                scene=dict(
                    xaxis_title='Moneyness',
                    yaxis_title='Days to Maturity',
                    zaxis_title='Total Var',
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=75, b=20),
                autosize=False,
                width=980,
                height=500,
                coloraxis=dict(colorscale='Tealgrn', colorbar_x=0.9, colorbar_thickness=23),
        )
        
        return fig
    
    def fengler_lv_surface_plot(self, num_strike, num_maturity, method):
        '''
        Return the local vol surface plot
        '''
        
        x, y, z1 = self.grid_lv(num_strike, num_maturity, method)
        x = np.array(sorted(list(set(x))))
        y = np.array(sorted(list(set(y))))
        z1 = np.array(z1).reshape((len(y), len(x)))
        
        fig = go.Figure()

        fig.add_trace(
            go.Surface(x=x, y=y, z=z1, opacity=0.8, colorscale='Sunset', colorbar_title = 'LV')
        )
        fig.update_layout(
                title = dict(
                    text = r'Local Vol Surface',
                    x = 0.5,
                    y = 0.9
                ),
                font=dict(size=12),
                scene=dict(
                    xaxis_title='Moneyness',
                    yaxis_title='Days to Maturity',
                    zaxis_title='LV',
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=75, b=20),
                autosize=False,
                width=980,
                height=500,
                coloraxis=dict(colorbar_x=0.9, colorbar_thickness=23),
        )
        
        return fig
    
    
    
# Create an object for a cubic spline
    class cubic_spline():
        '''
        Cubic spline object declared with value-second derivative representation Green & Silverman (1993)
        '''
        def __init__(self, params, knots, h):
            self.knots = knots
            self.n = len(knots)
            self.a = np.empty(self.n + 1)
            self.b = np.empty(self.n + 1)
            self.c = np.empty(self.n + 1)
            self.d = np.empty(self.n + 1)
            self.g = params[:self.n]
            self.gamma = params[self.n:]
            self.gamma.insert(0,0)
            self.gamma.append(0)
            self.transform(h)
        
        def transform(self, h):
            '''
            Transformation from value-second derivative representation to piecewise polynomial representation
            '''
            g = self.g
            gamma = self.gamma

            for i in range(self.n + 1):
                if i == 0:
                    self.a[i] = g[i]
                    self.b[i] = (g[i + 1] - g[i]) / h[i] - h[i] * (2 * gamma[i] + gamma[i + 1]) / 6
                    self.c[i] = 0
                    self.d[i] = 0
                elif i == self.n:
                    self.a[i] = g[i - 2]
                    self.b[i] = (g[i - 1] - g[i - 2]) / h[i - 2] + h[i - 2] * (gamma[i - 1] + 2 * gamma[i - 1]) / 6
                    self.c[i] = 0
                    self.d[i] = 0
                else:
                    self.a[i] = g[i - 1]
                    self.b[i] = (g[i] - g[i - 1]) / h[i - 1] - h[i - 1] * (2 * gamma[i - 1] + gamma[i]) / 6
                    self.c[i] = gamma[i - 1] / 2
                    self.d[i] = (gamma[i] - gamma[i - 1]) / (6 * h[i - 1])      
                    
        def rank(self, u):
            '''
            Helper function to determine which interval u lies in
            '''
            for idx, x in enumerate(self.knots):
                if u < x:
                    return idx
            return len(self.knots)
        
        def predict(self, u):
            '''
            Return smoothed call price given an arbitrary strike
            '''
            i = self.rank(u)
            if i == 0:
                return self.a[i] + self.b[i] * (u - self.knots[i])
            else:
                return self.a[i] + self.b[i] * (u - self.knots[i - 1]) + self.c[i] * (u - self.knots[i - 1]) ** 2 + self.d[i] * (u - self.knots[i - 1]) ** 3
            
        def dcdK(self, K):
            '''
            Return first derivative w.r.t. K on the spline
            '''
            i = self.rank(K)
            if i == 0:
                return self.b[i]
            elif i == len(self.knots):
                i -= 1
            
            return self.b[i] + 2 * self.c[i] * (K - self.knots[i]) + self.d[i] * (3 * K**2 - 6 * K * self.knots[i] + 3 * self.knots[i]**2)
            
        def d2cdK2(self, K):
            '''
            Return second derivative w.r.t. K on the spline
            '''
            i = self.rank(K)
            if i == 0:
                return self.c[i]
            elif i == len(self.knots):
                i -= 1
            return 2 * self.c[i] + 6 * self.d[i] * (K - self.knots[i])
            
            
            
            
# Smoothing Functions
# --------------------------------------------------------------------------------------------------------------------
    
    def calc_moneyness(self):
        '''
        Calculate the forward moneyless for observable values
        '''
        moneyness = []
        for T, K in zip(self.obs_exps, self.obs_strikes):
            i = self.rank_T(T)
            if i == 0:    
                i += 1
            elif i == len(self.Ts):
                i -= 1
            r = (self.r[i] - self.r[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.r[i - 1]
            q = (self.q[i] - self.q[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.q[i - 1]
            moneyness.append((np.exp((q - r) * T / 360) * K) / self.data.spot.mid)
        return np.array(moneyness).flatten()
    
    def imp_vol_smoothed(self):
        '''
        Return values of interpolated/extrapolated implied vol surface
        '''
        full_exps = []
        full_moneyness = []
        for T in sorted(list(set(self.obs_exps))):
            self.Ms[T] = []
            self.Ks[T] = []
            for K in sorted(list(set(self.obs_strikes))):
                i = self.rank_T(T)
                if i == 0:    
                    i += 1
                elif i == len(self.Ts):
                    i -= 1
                r = (self.r[i] - self.r[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.r[i - 1]
                q = (self.q[i] - self.q[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.q[i - 1]
                m = (np.exp((q - r) * T / 360) * K) / self.data.spot.mid
                full_exps.append(T)
                full_moneyness.append(m)
                self.Ms[T].append(m)
                self.Ks[T].append(K)
        
        # smoothing function
        tps = Rbf(self.obs_moneyness, self.obs_exps, self.obs_imp_vol, function="thin_plate", smooth=1e-5)
        tps_prices = tps(full_moneyness, full_exps)
        return full_moneyness, full_exps, tps_prices
    
    
    def presmooth_vol_surface_plot(self):
        '''
        Return the smoothed imp vol surface plot
        '''
        
        x, y, z1 = self.imp_vol_smoothed()
        
        x = np.array(x)
        y = np.array(y)
        z1 = np.array(z1)
            
        
        fig = go.Figure()

        fig.add_trace(
            go.Mesh3d(x=x, y=y, z=z1, opacity=0.8, intensity=z1, colorscale='Tealgrn', coloraxis='coloraxis1')
        )
        fig.update_layout(
                title = dict(
                    text = r'Smoothed Call Surface',
                    x = 0.5,
                    y = 0.9
                ),
                font=dict(size=12),
                scene=dict(
                    xaxis_title='Moneyness',
                    yaxis_title='Days to Maturity',
                    zaxis_title='IV',
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=75, b=20),
                autosize=False,
                width=980,
                height=500,
                coloraxis=dict(colorscale='Tealgrn', colorbar_x=0.9, colorbar_thickness=23, colorbar_title = 'IV'),
        )
        
        return fig
    
    def call_price_smoothed(self):
        '''
        Smooth call prices directly
        '''
        full_exps = []
        full_moneyness = []
        for T in sorted(list(set(self.obs_exps))):
            self.Ms[T] = []
            self.Ks[T] = []
            for K in sorted(list(set(self.obs_strikes))):
                i = self.rank_T(T)
                if i == 0:    
                    i += 1
                elif i == len(self.Ts):
                    i -= 1
                r = (self.r[i] - self.r[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.r[i - 1]
                q = (self.q[i] - self.q[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.q[i - 1]
                m = (np.exp((q - r) * T / 360) * K) / self.data.spot.mid
                full_exps.append(T)
                full_moneyness.append(m)
                self.Ms[T].append(m)
                self.Ks[T].append(K)
        
        tps = Rbf(self.obs_moneyness, self.obs_exps, self.obs_prices, function="thin_plate", smooth=1)
        tps_calls = tps(full_moneyness, full_exps)
        return full_moneyness, full_exps, tps_calls
    
    def call_price_smoothed2(self):
        '''
        Smooth call prices from smoothed imp vol
        '''
        full_moneyness, full_exps, tps_vols = self.imp_vol_smoothed()
        
        
        tps_calls = []
        
        j = 0
        for T, m in zip(full_exps, full_moneyness):
            i = self.rank_T(T)
            if i == 0:    
                i += 1
            elif i == len(self.Ts):
                i -= 1
            r = (self.r[i] - self.r[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.r[i - 1]
            q = (self.q[i] - self.q[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.q[i - 1]
            K = m * self.data.spot.mid * np.exp((r - q) * T / 360) 
            tps_calls.append(self.bs_call(self.data.spot.mid, K, r, q, T / 360, tps_vols[j]))
            j += 1
            
        return full_moneyness, full_exps, tps_calls
    
    
    def presmooth_call_surface_plot(self):
        '''
        Return the smoothed call surface plot
        '''
        
        x, y, z1 = self.call_price_smoothed()
        
        x = np.array(x)
        y = np.array(y)
        z1 = np.array(z1)
            
        
        fig = go.Figure()

        fig.add_trace(
            go.Mesh3d(x=x, y=y, z=z1, opacity=0.8, intensity=z1, colorscale='Tealgrn', coloraxis='coloraxis1')
        )
        fig.update_layout(
                title = dict(
                    text = r'Smoothed Call Surface',
                    x = 0.5,
                    y = 0.9
                ),
                font=dict(size=12),
                scene=dict(
                    xaxis_title='Moneyness',
                    yaxis_title='Days to Maturity',
                    zaxis_title='Call Price',
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=75, b=20),
                autosize=False,
                width=980,
                height=500,
                coloraxis=dict(colorscale='Tealgrn', colorbar_x=0.9, colorbar_thickness=23, colorbar_title = 'Call Prices'),
        )
        
        return fig
    
    
    def fdm_call(self, K, T):
        '''
        Return the call price grid by fdm methods
        '''
        
        # define the grid sizes
        t_space = np.linspace(0, T, 100)
        dt = t_space[1] - t_space[0]
        
        # define the upper bound for S
        max_S = self.find_K(self.m_space[-1], t_space[0])
        I = int(np.sqrt(1/(0.05**2 * dt)))
        
        S_space = np.linspace(0, max_S, I)   
        dS = S_space[1] - S_space[0]
        
        V = dict()
        
        for k, t in enumerate(reversed(t_space)):
            V[k] = []
            for idx, S in enumerate(S_space):
                
                if k == 0:
                    V[k].append(max(S - K, 0))
                else:
                    # boundary condition at lower bound of S
                    if idx == 0:
                        V[k].append(0)
                    # boundary condition at upper bound of S
                    elif idx == I - 1:
                        i = self.rank_T(t)
                        if i == 0:    
                            i += 1
                        elif i == len(self.Ts):
                            i -= 1
                        r = (self.r[i] - self.r[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.r[i - 1]
                        q = (self.q[i] - self.q[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.q[i - 1]
                        V[k].append(I * dS - K * np.exp(-r*k*dt/360))
                    else:
                        i = np.absolute(self.m_space - self.find_m(S, t)).argmin()
                        j = np.absolute(self.T_space - t).argmin()
                        sigma_lv = self.lv_surface[i][j]
                        if np.isnan(sigma_lv):
                            sigma_lv = 0

                        i = self.rank_T(t)
                        if i == 0:
                            i += 1
                        elif i == len(self.Ts):
                            i -= 1
                        r = (self.r[i] - self.r[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.r[i - 1]
                        q = (self.q[i] - self.q[i - 1])/(self.Ts[i] - self.Ts[i - 1]) * (T - self.Ts[i - 1]) + self.q[i - 1]

                        a = 0.5 * ((idx)**2 * sigma_lv**2 - (idx) * (r - q)) * dt / 360
                        b = 1 - (r + (idx)**2 * sigma_lv**2) * dt / 360
                        c = 0.5 * ((idx) ** 2 * sigma_lv**2 + (idx) * (r - q)) * dt / 360


                        V[k].append(a * V[k - 1][idx - 1] + b * V[k - 1][idx] + c * V[k - 1][idx + 1])
        
        i = None
        for idx, x in enumerate(S_space):
            if self.data.spot.mid < x:
                i = idx
                break
        if i == None:
            i = len(S_space)

        if i == 0:    
            i += 1
        elif i == len(S_space):
            i -= 1

        return (V[max(V.keys())][i] - V[max(V.keys())][i - 1])/(S_space[i] - S_space[i - 1]) * (self.data.spot.mid - S_space[i - 1]) + V[max(V.keys())][i - 1]    

    
    def fdm_obs_calls(self):
        '''
        Calculate call prices for all observable points
        '''
        calls = []
        for K, T in zip(self.obs_strikes, self.obs_exps):
            calls.append(self.fdm_call(K, T))
        return calls
    
    def call_scatter_plot(self, num_strike, num_maturity, method):
        '''
        Return the smoothed call surface plot
        '''        
        _, _, _ = self.grid_lv(num_strike, num_maturity, method)
        
        x = np.array(self.obs_strikes)
        y = np.array(self.obs_exps)
        z1 = np.array(self.obs_prices)
        z2 = np.array(self.fdm_obs_calls())
        
        fig = go.Figure()

        fig.add_trace(
            go.Scatter3d(x=x, y=y, z=z1, opacity=0.8, mode='markers', name='Observed Prices', marker=dict(color='LightSkyBlue',size=120, symbol='diamond'))
        )
        fig.add_trace(
            go.Scatter3d(x=x, y=y, z=z2, opacity=0.8, mode='markers', name='FDM Prices', marker=dict(color='Red',size=120, symbol='cross'))
        )
        fig.update_layout(
                title = dict(
                    text = r'Calls Scatter Plot',
                    x = 0.5,
                    y = 0.9
                ),
                font=dict(size=12),
                scene=dict(
                    xaxis_title='Moneyness',
                    yaxis_title='Days to Maturity',
                    zaxis_title='Call Price',
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=75, b=20),
                autosize=False,
                width=980,
                height=500,
                coloraxis=dict(colorscale='Tealgrn', colorbar_x=0.9, colorbar_thickness=23, colorbar_title = 'Call Prices'),
        )
        
        return fig
    

        
