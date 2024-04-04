from xbbg import blp
import numpy as np
from datetime import datetime


class raw_snapshot:
    '''
    Raw Snapshot:
    The class is for storing all information, given an index and a date
    It returns all necessary information about the vanilla options, from a perspective of that date

    Inputs:
        Ticker - Bloomberg Tickers
        Date - 'YYYY-MM-DD'
    '''
    def __init__(self, ticker, date):

        self.ticker = ticker
        self.date = date
        self.spot = self.Spot(ticker, date)
        self.us_rates = self.US_rates(ticker, date) # note that rates are in percentages (i.e. 5.0 for 5%), need to divided by 100 for calculations
        self.fx_rates = self.FX_rates(ticker, date, self.us_rates) # note that implied rates are in percentages
        self.futures = self.Futures(ticker, date, self.calc_maturities(ticker, date))
        self.maturity_dates = self.futures.maturity_dates
        self.calls = self.Calls(ticker, date, self.maturity_dates, self.spot)
        self.puts = self.Puts(ticker, date, self.maturity_dates, self.spot)
        
    def calc_maturities(self, ticker, date):
        '''
        This function returns the visible maturities for index options/futures given a date
        Bloomberg have a system for index derivative maturities
        Data only available at the front month, back month, next quarter and next next quarter
        '''
        # Bloomberg month to ticker mapping
        bbg_mapping = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']

        year = date.split('-')[0][-1]
        front_month = int(date.split('-')[1])
        day = int(date.split('-')[2])
        
        # check if the closest future is in the same month
        active_future = self.map_future_ticker(ticker) + bbg_mapping[front_month - 1] + year + ' Index'
        df = blp.bdp(tickers = active_future, flds = ['FUTURES_VALUATION_DATE'])
        if df.values.flatten()[0].day <= day: # if not the same, we have passed the expiry date of this month already (i.e. near end of month)
            front_month += 1

        if front_month == 1:
            return ['F' + year, 'G' + year, 'H' + year, 'M' + year]
        elif front_month == 2:
            return ['G' + year, 'H' + year, 'M' + year, 'U' + year]
        elif front_month == 3:
            return ['H' + year, 'J' + year, 'M' + year, 'U' + year]
        elif front_month == 4:
            return ['J' + year, 'K' + year, 'M' + year, 'U' + year]
        elif front_month == 5:
            return ['K' + year, 'M' + year, 'U' + year, 'Z' + year]
        elif front_month == 6:
            return ['M' + year, 'N' + year, 'U' + year, 'Z' + year]
        elif front_month == 7:
            return ['N' + year, 'Q' + year, 'U' + year, 'Z' + year]
        elif front_month == 8:
            return ['Q' + year, 'U' + year, 'Z' + year, 'H' + str(int(year) + 1)]
        elif front_month == 9:
            return ['U' + year, 'V' + year, 'Z' + year, 'H' + str(int(year) + 1)]
        elif front_month == 10:
            return ['V' + year, 'X' + year, 'Z' + year, 'H' + str(int(year) + 1)]
        elif front_month == 11:
            return ['X' + year, 'Z' + year, 'H' + str(int(year) + 1), 'M' + str(int(year) + 1)]
        elif front_month == 12:
            return ['Z' + year, 'F' + str(int(year) + 1), 'H' + str(int(year) + 1), 'M' + str(int(year) + 1)]
        
    def map_future_ticker(self, ticker):
        '''
        Function to map index tickers to future tickers
        '''
        index_future_map = {'HSI Index':'HI',
                        'HSCEI Index':'HC',
                        'SSE50 Index': 'FFB',
                        'SHSN300 Index': 'IFB',
                        'CSI1000 Index': 'IFD',
                        'TWSE Index': 'FT',
                        'KOSPI2 Index': 'KM',
                        'NIFTY Index': 'JGS',
                        'SPX Index': 'ES'
                       }
        return index_future_map[ticker]

#----------------------------------------------------------------------------------------------
    class Spot:
        def __init__(self, ticker, date):
            self.tickers = ticker
            self.maturities = []
            self.mid = None
            self.date = date
            self.extract_spot(ticker, date)

        def extract_spot(self, ticker, date):
            df = blp.bdh(tickers = ticker, flds = ['PX_LAST'], start_date = date, end_date = date)
            self.mid = df.values[0][0]                

#----------------------------------------------------------------------------------------------
    class Futures:
        def __init__(self, ticker, date, maturities):
            self.date = date
            self.tickers = []
            self.maturity_dates = []
            self.bids = []
            self.asks = []
            self.extract_futures(self.map_future_ticker(ticker), maturities)
            self.days_to_exp = self.calc_days_to_exp()

        def map_future_ticker(self, ticker):
            '''
            Function to map index tickers to future tickers
            '''
            index_future_map = {'HSI Index':'HI',
                                'HSCEI Index':'HC',
                                'SSE50 Index': 'FFB',
                                'SHSN300 Index': 'IFB',
                                'CSI1000 Index': 'IFD',
                                'TWSE Index': 'FT',
                                'KOSPI2 Index': 'KM',
                                'NIFTY Index': 'JGS',
                                'SPX Index': 'ES'
                               }
            return index_future_map[ticker]

        def extract_futures(self, future_ticker, maturities):
            fut_tickers = [future_ticker + maturity + ' Index' for maturity in maturities]
            for fut_ticker in fut_tickers:
                self.tickers.append(fut_ticker)
            df = blp.bdh(tickers = fut_tickers, flds = ['PX_BID', 'PX_ASK'], start_date = self.date, end_date = self.date)
            for idx, v in enumerate(df.values[0]):
                if idx % 2 == 0:
                    self.bids.append(v)
                else:
                    self.asks.append(v)

            df = blp.bdp(tickers = fut_tickers, flds = ['FUTURES_VALUATION_DATE'])
            for maturity_date in df.values.flatten():
                self.maturity_dates.append(maturity_date)
                
        def calc_days_to_exp(self):
            today = datetime.strptime(self.date, '%Y-%m-%d').date()
            return [(maturity - today).days for maturity in self.maturity_dates]
            

#----------------------------------------------------------------------------------------------
    class US_rates:
        def __init__(self, ticker, date):
            self.date = date
            self.tickers = ['USOSFR2T Curncy', 'USOSFR3T Curncy', 'USOSFR1Z Curncy', 'USOSFR2Z Curncy', 'USOSFR3Z Curncy', 'USOSFRA Curncy', 'USOSFRB Curncy', 'USOSFRC Curncy', 'USOSFRD Curncy', 'USOSFRE Curncy', 'USOSFRF Curncy', 'USOSFRG Curncy', 'USOSFRH Curncy', 'USOSFRI Curncy', 'USOSFRJ Curncy', 'USOSFRK Curncy', 'USOSFR1 Curncy', 'USOSFR1C Curncy', 'USOSFR1F Curncy', 'USOSFR1I Curncy', 'USOSFR2 Curncy']
            self.days_to_exp = [2, 3, 7, 14, 21, 30, 61, 91, 122, 153, 183, 214, 244, 275, 306, 334, 365, 456, 548, 640, 730] # just an estimate
            self.bids = []
            self.asks = []
            self.extract_us_rates()
            
        def extract_us_rates(self):
            df = blp.bdh(tickers = self.tickers, flds = ['PX_BID', 'PX_ASK'], start_date = self.date, end_date = self.date)
            for idx, v in enumerate(df.values[0]):
                if idx % 2 == 0:
                    self.bids.append(v)
                else:
                    self.asks.append(v)
                    
            # for SOFRRATE
            df = blp.bdh(tickers = 'SOFRRATE Index', flds = ['PX_LAST'], start_date = self.date, end_date = self.date)
            sofr = df.values.flatten()[0]
            self.tickers.insert(0, 'SOFRRATE Index')
            self.days_to_exp.insert(0, 1)
            self.bids.insert(0, sofr)
            self.asks.insert(0, sofr)
            
#----------------------------------------------------------------------------------------------
    class FX_rates:
        def __init__(self, ticker, date, us_rates):
            if ticker != 'SPX Index':
                self.date = date
                self.currency = None
                self.tickers = self.map_local_rates(ticker)
                self.spot_bid, self.spot_ask = self.extract_spot_rate(ticker)
                self.days_to_exp = [2, 3, 7, 14, 21, 30, 61, 91, 122, 153, 183, 214, 244, 275, 306, 334, 365, 456, 548, 640, 730]
                self.bids = []
                self.asks = []
                self.implied_yield_bids = []
                self.implied_yield_asks = []
                self.extract_fx_rates()
                self.calc_implied_rate(us_rates)
            else:
                self.date = date
                self.currency = 'USD'
                self.tickers = self.map_local_rates(ticker)
                self.spot_bid = self.spot_ask = 1
                self.days_to_exp = [2, 3, 7, 14, 21, 30, 61, 91, 122, 153, 183, 214, 244, 275, 306, 334, 365, 456, 548, 640, 730]
                self.bids = us_rates.bids
                self.asks = us_rates.asks
                self.implied_yield_bids = us_rates.bids
                self.implied_yield_asks = us_rates.asks
                
        def map_local_rates(self, ticker):
            index_fx_map = {'HSI Index':'HKD',
                            'HSCEI Index':'HKD',
                            'SSE50 Index': 'CNH',
                            'SHSN300 Index': 'CNH',
                            'CSI1000 Index': 'CNH',
                            'TWSE Index': 'NTN',
                            'KOSPI2 Index': 'KWN',
                            'NIFTY Index': 'IRN',
                            'SPX Index': 'USD'
                           }
            self.currency = index_fx_map[ticker]
            tenors = ['ON', 'TN', 'SN', '1W', '2W', '3W', '1M', '2M', '3M', '4M', '5M', '6M', '7M', '8M', '9M', '10M', '11M', '12M', '15M', '18M', '21M', '2Y']
            return [self.currency + tenor + ' Curncy' for tenor in tenors]

        def extract_fx_rates(self):
            df = blp.bdh(tickers = self.tickers, flds = ['PX_BID', 'PX_ASK'], start_date = self.date, end_date = self.date)
            for idx, v in enumerate(df.values[0]):
                if idx % 2 == 0:
                    self.bids.append(v)
                else:
                    self.asks.append(v)
                    
        def extract_spot_rate(self, ticker):
            index_spot_map = {'HSI Index':'HKD',
                            'HSCEI Index':'HKD',
                            'SSE50 Index': 'CNH',
                            'SHSN300 Index': 'CNH',
                            'CSI1000 Index': 'CNH',
                            'TWSE Index': 'TWD',
                            'KOSPI2 Index': 'KRW',
                            'NIFTY Index': 'INR',
                           }
            spot_ticker = index_spot_map[ticker]
            df = blp.bdh(tickers = 'USD' + spot_ticker + ' BGN Curncy', flds = ['PX_BID', 'PX_ASK'], start_date = self.date, end_date = self.date)
            return df.values.flatten()[0], df.values.flatten()[1]
            
        def calc_implied_rate(self, us_rates):
            index_fx_map = {'HKD': 10000,
                            'CNH': 10000,
                            'NTN': 1,
                            'KWN': 1,
                            'IRN': 100
                           }
            
            pts_adj = index_fx_map[self.currency]
            
            for i_us, fwd_pts in zip(us_rates.bids, self.bids):
                self.implied_yield_bids.append((1 + (fwd_pts/pts_adj)/self.spot_ask) * (1 + i_us) - 1)
            for i_us, fwd_pts in zip(us_rates.asks, self.asks):
                self.implied_yield_asks.append((1 + (fwd_pts/pts_adj)/self.spot_bid) * (1 + i_us) - 1)
                
            
#----------------------------------------------------------------------------------------------
    class Calls:
        def __init__(self, ticker, date, maturity_dates, spot):
            self.date = date
            self.strike_range = 0.25
            self.strikes = self.calc_strikes(ticker, spot)
            self.maturity_dates = maturity_dates
            self.tickers = []
            self.bids = []
            self.asks = []         
            self.form_tickers(ticker)
            self.extract_calls()
            self.days_to_exp = self.calc_days_to_exp()
            
        def calc_strikes(self, ticker, spot):
            index_strike_tick_map = {'HSI Index': 100,
                                'HSCEI Index': 100,
                                'SSE50 Index': 25,
                                'SHSN300 Index': 50,
                                'CSI1000 Index': 100,
                                'TWSE Index': 100,
                                'KOSPI2 Index': 2.5,
                                'NIFTY Index': 50,
                                'SPX Index': 5
                               }
            strike_tick = index_strike_tick_map[ticker]
            lower_strike = float(round(spot.mid * (1 - self.strike_range) / strike_tick) * strike_tick)
            upper_strike = float(round(spot.mid * (1 + self.strike_range) / strike_tick) * strike_tick)
            if lower_strike.is_integer():
                lower_strike = int(lower_strike)
                upper_strike = int(upper_strike)
            return np.arange(lower_strike, upper_strike, strike_tick)
        
        def form_tickers(self, ticker):
            for maturity in self.maturity_dates:
                for strike in self.strikes:
                    self.tickers.append(ticker.split(' ')[0] + ' %d/%d '%(maturity.month, maturity.year - 2000)  + 'C' + str(strike) + ' Index')
                    
        def extract_calls(self):
            for ticker in self.tickers:
                df = blp.bdh(tickers = ticker, flds = ['PX_BID', 'PX_ASK'], start_date = self.date, end_date = self.date)
                if df.shape[1] == 2:
                    self.bids.append(df.values.flatten()[0])
                    self.asks.append(df.values.flatten()[1])
                elif df.shape[1] == 1:
                    if df.columns[0][1] == 'PX_BID':
                        self.bids.append(df.values.flatten()[0])
                        self.asks.append(np.nan)
                    elif df.columns[0][1] == 'PX_ASK':
                        self.bids.append(np.nan)
                        self.asks.append(df.values.flatten()[0])
                else:
                    self.bids.append(np.nan)
                    self.asks.append(np.nan)
                    
        def calc_days_to_exp(self):
            today = datetime.strptime(self.date, '%Y-%m-%d').date()
            return [(maturity - today).days for maturity in self.maturity_dates]
            
#----------------------------------------------------------------------------------------------
    class Puts:
        def __init__(self, ticker, date, maturity_dates, spot):
            self.date = date
            self.strike_range = 0.25
            self.strikes = self.calc_strikes(ticker, spot)
            self.maturity_dates = maturity_dates
            self.tickers = []
            self.bids = []
            self.asks = []         
            self.form_tickers(ticker)
            self.extract_puts()
            self.days_to_exp = self.calc_days_to_exp()
            
        def calc_strikes(self, ticker, spot):
            index_strike_tick_map = {'HSI Index': 100,
                                'HSCEI Index': 100,
                                'SSE50 Index': 25,
                                'SHSN300 Index': 50,
                                'CSI1000 Index': 100,
                                'TWSE Index': 100,
                                'KOSPI2 Index': 2.5,
                                'NIFTY Index': 50,
                                'SPX Index': 5
                               }
            strike_tick = index_strike_tick_map[ticker]
            lower_strike = float(round(spot.mid * (1 - self.strike_range) / strike_tick) * strike_tick)
            upper_strike = float(round(spot.mid * (1 + self.strike_range) / strike_tick) * strike_tick)
            if lower_strike.is_integer():
                lower_strike = int(lower_strike)
                upper_strike = int(upper_strike)
            return np.arange(lower_strike, upper_strike, strike_tick)
        
        def form_tickers(self, ticker):
            for maturity in self.maturity_dates:
                for strike in self.strikes:
                    self.tickers.append(ticker.split(' ')[0] + ' %d/%d '%(maturity.month, maturity.year - 2000)  + 'P' + str(strike) + ' Index')
                    
        def extract_puts(self):
            for ticker in self.tickers:
                df = blp.bdh(tickers = ticker, flds = ['PX_BID', 'PX_ASK'], start_date = self.date, end_date = self.date)
                if df.shape[1] == 2:
                    self.bids.append(df.values.flatten()[0])
                    self.asks.append(df.values.flatten()[1])
                elif df.shape[1] == 1:
                    if df.columns[0][1] == 'PX_BID':
                        self.bids.append(df.values.flatten()[0])
                        self.asks.append(np.nan)
                    elif df.columns[0][1] == 'PX_ASK':
                        self.bids.append(np.nan)
                        self.asks.append(df.values.flatten()[0])
                else:
                    self.bids.append(np.nan)
                    self.asks.append(np.nan)
                    
        def calc_days_to_exp(self):
            today = datetime.strptime(self.date, '%Y-%m-%d').date()
            return [(maturity - today).days for maturity in self.maturity_dates]