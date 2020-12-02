from collections import OrderedDict
from time import time

#quantopian imports
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.factors import Latest
from quantopian.pipeline.data import morningstar, Fundamentals
from quantopian.pipeline.factors import CustomFactor, SimpleMovingAverage, AverageDollarVolume,SimpleBeta, Returns, RSI,EWMA
from quantopian.pipeline.data.zacks import EarningsSurprises
from quantopian.pipeline.data import factset
from quantopian.pipeline.data.psychsignal import stocktwits
#Algo imports
import quantopian.optimize as opt
import quantopian.algorithm as algo
from quantopian.pipeline.experimental import risk_loading_pipeline  
from quantopian.pipeline.classifiers.fundamentals import Sector as _Sector

#Python imports
import math
import talib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import linear_model, decomposition, ensemble, preprocessing, isotonic, metrics
from scipy.stats.mstats import winsorize
from zipline.utils.numpy_utils import (
    repeat_first_axis,
    repeat_last_axis,
)
from scipy.stats.mstats import gmean
from sklearn.cluster import SpectralClustering
 
from collections import Counter

bs = morningstar.balance_sheet
cfs = morningstar.cash_flow_statement
is_ = morningstar.income_statement
or_ = morningstar.operation_ratios
er = morningstar.earnings_report
v = morningstar.valuation
vr = morningstar.valuation_ratios

from quantopian.algorithm import (
    attach_pipeline,
    date_rules,
    order_optimal_portfolio,
    pipeline_output,
    record,
    schedule_function,
    set_commission,
    set_slippage,
    time_rules,
)

# If you have eventvestor, it's a good idea to screen out aquisition targets
# Comment out & ~IsAnnouncedAcqTarget() as well. You can also run this over
# the free period.
# from quantopian.pipeline.filters.eventvestor import IsAnnouncedAcqTarget

# Will be split 50% long and 50% short
N_STOCKS_TO_TRADE = 500

# Number of days to train the classifier on, easy to run out of memory here
ML_TRAINING_WINDOW = 63

# train on returns over N days into the future
PRED_N_FORWARD_DAYS = 21

# How often to trade, for daily, set to date_rules.every_day()
TRADE_FREQ = date_rules.week_start(days_offset=1) #date_rules.every_day()

"""
HELPER FUNCTIONS
"""

def plus_dm_helper(high, low):
    """
    Returns positive directional movement. Abstracted for use with more complex factors

    https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/DMI

    Parameters
    ----------
    high : np.array
        matrix of high prices
    low : np.array
        matrix of low prices

    Returns
    -------
    np.array : matrix of positive directional movement

    """
    # get daily differences between high prices
    high_diff = (high - np.roll(high, 1, axis=0))[1:]

    # get daily differences between low prices
    low_diff = (np.roll(low, 1, axis=0) - low)[1:]

    # matrix of positive directional movement
    return np.where(((high_diff > 0) | (low_diff > 0)) & (high_diff > low_diff), high_diff, 0.)

def minus_dm_helper(high, low):
    """
    Returns negative directional movement. Abstracted for use with more complex factors

    https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/DMI

    Parameters
    ----------
    high : np.array
        matrix of high prices
    low : np.array
        matrix of low prices

    Returns
    -------
    np.array : matrix of negative directional movement

    """
    # get daily differences between high prices
    high_diff = (high - np.roll(high, 1, axis=0))[1:]

    # get daily differences between low prices
    low_diff = (np.roll(low, 1, axis=0) - low)[1:]

    # matrix of megative directional movement
    return np.where(((high_diff > 0) | (low_diff > 0)) & (high_diff < low_diff), low_diff, 0.)

def trange_helper(high, low, close):
    """
    Returns true range

    http://www.macroption.com/true-range/

    Parameters
    ----------
    high : np.array
        matrix of high prices
    low : np.array
        matrix of low prices
    close: np.array
        matrix of close prices

    Returns
    -------
    np.array : matrix of true range

    """
    # define matrices to be compared
    close = close[:-1]
    high = high[1:]
    low = low[1:]

    # matrices for comparison
    high_less_close = high - close
    close_less_low = close - low
    high_less_low = high - low

    # return maximum value for each cel
    return np.maximum(high_less_close, close_less_low, high_less_low)

def preprocess(a):
    a = a.astype(np.float64)
    a[np.isinf(a)] = np.nan
    a = np.nan_to_num(a - np.nanmean(a))
    a = winsorize(a, limits=[0.02,0.98])  
    return a

"""
FEATURES
"""
class Sector(_Sector):
    window_safe = True

#Momentum Indicators
class Momentum(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 252

    def compute(self, today, assets, out, close):       
        out[:] = close[-20] / close[0]

class aa_momentum(CustomFactor):
    """ Alpha Architect - Momentum factor """
    inputs = [USEquityPricing.close,
    Returns(window_length=126)]
    window_length = 252
    def compute(self, today, assets, out, prices, returns):  
            out[:] = ((prices[-21] - prices[-252])/prices[-252] -  
                      (prices[-1] - prices[-21])/prices[-21]) / np.nanstd(returns, axis=0)

class ATR(CustomFactor):
        """
        Average True Range

        Momentum indicator

        **Default Inputs:** USEquityPricing.high, USEquityPricing.low, USEquityPricing.close

        **Default Window Length:** 15 (14+1)

        https://en.wikipedia.org/wiki/Average_true_range
        """
        inputs=[USEquityPricing.high, USEquityPricing.low, USEquityPricing.close]
        window_length = 21

        def compute(self, today, assets, out, high, low, close):

            tr_frame = trange_helper(high, low, close)
            decay_rate= 2./(len(tr_frame) + 1.)
            weights = np.full(len(tr_frame), decay_rate, float) ** np.arange(len(tr_frame) + 1, 1, -1)
            out[:] = np.average(tr_frame, axis=0, weights=weights)

class MINUS_DM(CustomFactor):
    """
    Negative directional movement

    Momentum indicator

    **Default Inputs:** USEquityPricing.high, USEquityPricing.low

    **Default Window Length:** 15

    https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/DMI
    """    

    inputs = [USEquityPricing.high, USEquityPricing.low]
    window_length = 22

    def compute(self, today, assets, out, high, low):
            out[:] = np.sum(minus_dm_helper(high, low), axis=0)

class PLUS_DM(CustomFactor):
    """
    Positive directional movement

    Momentum indicator

    **Default Inputs:** USEquityPricing.high, USEquityPricing.low

    **Default Window Length:** 15

    https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/DMI
    """    
    inputs = [USEquityPricing.high, USEquityPricing.low]
    window_length = 22

    def compute(self, today, assets, out, high, low):
            out[:] = np.sum(plus_dm_helper(high, low), axis=0)

def Price_Momentum_12M():
    """
    12-Month Price Momentum:
    12-month closing price rate of change.
    https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf # NOQA
    Notes:
    High value suggests momentum (long term)
    Equivalent to analysis of returns (12-month window)
    """
    return Returns(window_length=252)

#Trend Indicators
class Trendline(CustomFactor):
        inputs = [USEquityPricing.close]
        window_length = 252
        window_safe = True

        _x = np.arange(window_length)
        _x_var = np.var(_x)

        def compute(self, today, assets, out, close):

            x_matrix = repeat_last_axis(
            (self.window_length - 1) / 2 - self._x,
            len(assets),
            )

            y_bar = np.nanmean(close, axis=0)
            y_bars = repeat_first_axis(y_bar, self.window_length)
            y_matrix = close - y_bars

            out[:] = preprocess(-np.divide(
            (x_matrix * y_matrix).sum(axis=0) / self._x_var,
            self.window_length
            ))

class LINEARREG_INTERCEPT(CustomFactor):
    """
    Intercept of Trendline

    **Default Inputs:**  USEquitypricing.close

    **Default Window Length:** 14

    http://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/06/lecture-06.pdf
    """
    inputs=[USEquityPricing.close]
    window_length=21

    # using MLE
    def compute(self, today, assets, out, close):

        # prepare X matrix (x_is - x_bar)
        X = range(self.window_length)
        X_bar = np.nanmean(X)
        X_vector = X - X_bar
        X_matrix = np.tile(X_vector, (len(close.T), 1)).T

        # prepare Y vectors (y_is - y_bar)
        Y_bar = np.nanmean(close, axis=0)
        Y_bars = np.tile(Y_bar, (self.window_length, 1))
        Y_matrix = close - Y_bars

        # prepare variance of X
        X_var = np.nanvar(X)

        # multiply X matrix an Y matrix and sum (dot product)
        # then divide by variance of X
        # this gives the MLE of Beta
        betas = (np.sum((X_matrix * Y_matrix), axis=0) / X_var) / (self.window_length)

        # now use to get to MLE of alpha
        out[:] = Y_bar - (betas * X_bar)

class MaxGap(CustomFactor): 
    # the biggest absolute overnight gap in the previous 90 sessions
    inputs = [USEquityPricing.close] ; window_length = 90
    window_safe = True
    def compute(self, today, assets, out, close):
        abs_log_rets = np.abs(np.diff(np.log(close),axis=0))
        max_gap = np.max(abs_log_rets, axis=0)
        out[:] = preprocess(max_gap)

class TEM(CustomFactor):
        """
        TEM = standard deviation of past 6 quarters' reports
        """
        inputs=[factset.Fundamentals.capex_qf_asof_date,
            factset.Fundamentals.capex_qf,
            factset.Fundamentals.assets]
        window_length = 390
        window_safe = True
        def compute(self, today, assets, out, asof_date, capex, total_assets):
            values = capex/total_assets
            values[np.isinf(values)] = np.nan
            out_temp = np.zeros_like(values[-1,:])
            for column_ix in range(asof_date.shape[1]):
                _, unique_indices = np.unique(asof_date[:, column_ix], return_index=True)
                quarterly_values = values[unique_indices, column_ix]
                if len(quarterly_values) < 6:
                    quarterly_values = np.hstack([
                    np.repeat([np.nan], 6 - len(quarterly_values)),
                    quarterly_values,
                    ])

                out_temp[column_ix] = np.std(quarterly_values[-6:])

            out[:] = preprocess(-out_temp) 

class PriceOscillator(CustomFactor):
    inputs = (USEquityPricing.close,)
    window_length = 252

    def compute(self, today, assets, out, close):
        four_week_period = close[-20:]
        np.divide(
            np.nanmean(four_week_period, axis=0),
            np.nanmean(close, axis=0),
            out=out,
        )
        out -= 1

class MeanReversion1M(CustomFactor):
    inputs = (Returns(window_length=21),)
    window_length = 252

    def compute(self, today, assets, out, monthly_rets):
        np.divide(
            monthly_rets[-1] - np.nanmean(monthly_rets, axis=0),
            np.nanstd(monthly_rets, axis=0),
            out=out,
        )

class ADX(CustomFactor):
    """
    Average Directional Movement Index

    Momentum indicator. Smoothed DX

    **Default Inputs:** USEquityPricing.high, USEquityPricing.low, USEquitypricing.close

    **Default Window Length:** 29

    https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/DMI
    """        
    inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close]
    window_length = 29

    def compute(self, today, assets, out, high, low, close):

        # positive directional index
        plus_di = 100 * np.cumsum(plus_dm_helper(high, low) / trange_helper(high, low, close), axis=0)

        # negative directional index
        minus_di = 100 * np.cumsum(minus_dm_helper(high, low) / trange_helper(high, low, close), axis=0)

        # full dx with 15 day burn-in period
        dx_frame = (np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100.)[14:]

        # 14-day EMA
        span = 14.
        decay_rate = 2. / (span + 1.)
        weights = weights_long = np.full(span, decay_rate, float) ** np.arange(span + 1, 1, -1)

        # return EMA
        out[:] = np.average(dx_frame, axis=0, weights=weights)

class DX(CustomFactor):
    """
    Directional Movement Index

    Momentum indicator

    **Default Inputs:** USEquityPricing.high, USEquityPricing.low, USEquitypricing.close

    **Default Window Length:** 15

    https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/DMI
    """        
    inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close]
    window_length = 22

    def compute(self, today, assets, out, high, low, close):

        # positive directional index
        plus_di = 100 * np.sum(plus_dm_helper(high, low) / (trange_helper(high, low, close)), axis=0)

        # negative directional index
        minus_di = 100 * np.sum(minus_dm_helper(high, low) / (trange_helper(high, low, close)), axis=0)

        # DX
        out[:] = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100.

class MEDPRICE(CustomFactor):
    """
    Mean of a day's high and low prices

    **Default Inputs:**  USEquityPricing.high, USEquityPricing.low

    **Default Window Length:** 1

    http://www.fmlabs.com/reference/default.htm?url=MedianPrices.htm
    """        
    inputs = [USEquityPricing.high, USEquityPricing.low]
    window_length = 1

    def compute(self, today, assets, out, high, low):
        out[:] = (high + low) / 2.

class TYPPRICE(CustomFactor):
    """
    Typical Price 

    **Default Inputs:**  USEquityPricing.high, USEquityPricing.low, USEquityPricing.close

    **Default Window Length:** 1

    https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/typical-price
    """    
    inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close]
    window_length = 1

    def compute(self, today, assets, out, high, low, close):
        out[:] = (high + low + close) / 3.

class TRANGE(CustomFactor):
    """
    True Range 

    **Default Inputs:**  USEquityPricing.high, USEquityPricing.low, USEquityPricing.close

    **Default Window Length:** 2

    https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/atr
    """    
    inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close]
    window_length = 2

    def compute(self, today, assets, out, high, low, close):
        out[:] = np.nanmax([(high[-1] - close[0]), (close[0] - low[-1]), (high[-1] - low[-1])], axis=0)

class MACD_Signal_10d(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 60

    def compute(self, today, assets, out, close):

        sig_lines = []

        for col in close.T:
            # get signal line only
            try:
                _, signal_line, _ = talib.MACD(col, fastperiod=12,
                                               slowperiod=26, signalperiod=10)
                sig_lines.append(signal_line[-1])
            # if error calculating, return NaN
            except:
                sig_lines.append(np.nan)
        out[:] = sig_lines

#TRADITIONAL VALUE
def Dividend_Yield():
    """
    Dividend Yield:
    Dividends per share divided by closing price.
    https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf # NOQA
    Notes:
    High Dividend Yield Ratio suggests that an equity
    is attractive to an investor as the dividends
    paid out will be a larger proportion of
    the price they paid for it.
    """
    return vr.dividend_yield.latest

def Price_To_Free_Cashflows():
    """
    Price to Free Cash Flows:
    Closing price divided by free cash flow.
    https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf # NOQA
    Notes:
    Low P/ Free Cash Flows suggests that equity is cheap
    Differs substantially between sectors
    """
    return USEquityPricing.close.latest / \
        vr.fcf_per_share.latest

def Price_To_Operating_Cashflows():
    """
    Price to Operating Cash Flows:
    Closing price divided by operating cash flow.
    https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf # NOQA
    Notes:
    Low P/ Operating Cash Flows suggests that equity is cheap
    Differs substantially between sectors
    """
    return USEquityPricing.close.latest / \
        vr.cfo_per_share.latest

def Price_To_Book():
    """
    Price to Book Value:
    Closing price divided by book value.
    https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf # NOQA
    Notes:
    Low P/B Ratio suggests that equity is cheap
    Differs substantially between sectors
    """
    return USEquityPricing.close.latest / \
        vr.book_value_per_share.latest

def EV_To_Cashflows():
    """
    Enterprise Value to Cash Flows:
    Enterprise Value divided by Free Cash Flows.
    https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf # NOQA
    Notes:
    Low EV/FCF suggests that a company has a good amount of
    money relative to its size readily available
    """
    return v.enterprise_value.latest / \
        cfs.free_cash_flow.latest

#Efficiency
def Capex_To_Sales():
    """
    Capital Expnditure to Sales:
    Capital Expenditure divided by Total Revenue.
    https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf # NOQA
    Notes:
    High value suggests good efficiency, as expenditure is
    being used to generate greater sales figures
    """
    return (cfs.capital_expenditure.latest * 4.) / \
        (is_.total_revenue.latest * 4.)

def EBIT_To_Assets():
    """
    Earnings Before Interest and Taxes (EBIT) to Total Assets:
    EBIT divided by Total Assets.
    https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf # NOQA
    Notes:
    High value suggests good efficiency, as earnings are
    being used to generate more assets
    """
    return (is_.ebit.latest * 4.) / \
        bs.total_assets.latest

#Quality
class Piotroski(CustomFactor):
    inputs = [
        morningstar.operation_ratios.roa,
        morningstar.cash_flow_statement.operating_cash_flow,
        morningstar.cash_flow_statement.cash_flow_from_continuing_operating_activities,

        morningstar.operation_ratios.long_term_debt_equity_ratio,
        morningstar.operation_ratios.current_ratio,
        morningstar.valuation.shares_outstanding,

        morningstar.operation_ratios.gross_margin,
        morningstar.operation_ratios.assets_turnover,
    ]
    window_length = 22

    def compute(self, today, assets, out,
                roa, cash_flow, cash_flow_from_ops,
                long_term_debt_ratio, current_ratio, shares_outstanding,
                gross_margin, assets_turnover):
        profit = (
            (roa[-1] > 0).astype(int) +
            (cash_flow[-1] > 0).astype(int) +
            (roa[-1] > roa[0]).astype(int) +
            (cash_flow_from_ops[-1] > roa[-1]).astype(int)
        )

        leverage = (
            (long_term_debt_ratio[-1] < long_term_debt_ratio[0]).astype(int) +
            (current_ratio[-1] > current_ratio[0]).astype(int) + 
            (shares_outstanding[-1] <= shares_outstanding[0]).astype(int)
        )

        operating = (
            (gross_margin[-1] > gross_margin[0]).astype(int) +
            (assets_turnover[-1] > assets_turnover[0]).astype(int)
        )

        out[:] = profit + leverage + operating

class ROA(CustomFactor):
    window_length = 1
    inputs = [morningstar.operation_ratios.roa]

    def compute(self, today, assets, out, roa):
        out[:] = (roa[-1] > 0).astype(int)

class ROAChange(CustomFactor):
    window_length = 22
    inputs = [morningstar.operation_ratios.roa]

    def compute(self, today, assets, out, roa):
        out[:] = (roa[-1] > roa[0]).astype(int)

class CashFlow(CustomFactor):
    window_length = 1
    inputs = [morningstar.cash_flow_statement.operating_cash_flow]

    def compute(self, today, assets, out, cash_flow):
        out[:] = (cash_flow[-1] > 0).astype(int)

class CashFlowFromOps(CustomFactor):
    window_length = 1
    inputs = [morningstar.cash_flow_statement.cash_flow_from_continuing_operating_activities, morningstar.operation_ratios.roa]

    def compute(self, today, assets, out, cash_flow_from_ops, roa):
        out[:] = (cash_flow_from_ops[-1] > roa[-1]).astype(int)

class LongTermDebtRatioChange(CustomFactor):
    window_length = 22
    inputs = [morningstar.operation_ratios.long_term_debt_equity_ratio]

    def compute(self, today, assets, out, long_term_debt_ratio):
        out[:] = (long_term_debt_ratio[-1] < long_term_debt_ratio[0]).astype(int)

class CurrentDebtRatioChange(CustomFactor):
    window_length = 22
    inputs = [morningstar.operation_ratios.current_ratio]

    def compute(self, today, assets, out, current_ratio):
        out[:] = (current_ratio[-1] > current_ratio[0]).astype(int)

class SharesOutstandingChange(CustomFactor):
    window_length = 22
    inputs = [morningstar.valuation.shares_outstanding]

    def compute(self, today, assets, out, shares_outstanding):
        out[:] = (shares_outstanding[-1] <= shares_outstanding[0]).astype(int)

class GrossMarginChange(CustomFactor):
    window_length = 22
    inputs = [morningstar.operation_ratios.gross_margin]

    def compute(self, today, assets, out, gross_margin):
        out[:] = (gross_margin[-1] > gross_margin[0]).astype(int)

class AssetsTurnoverChange(CustomFactor):
    window_length = 22
    inputs = [morningstar.operation_ratios.assets_turnover]

    def compute(self, today, assets, out, assets_turnover):
        out[:] = (assets_turnover[-1] > assets_turnover[0]).astype(int)

def Asset_Growth_3M():
    """
    3-month Asset Growth:
    Increase in total assets over 3 months
    https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf # NOQA
    Notes:
    High value represents good financial health as quantity of
    assets is increasing
    """
    return Returns(inputs=[bs.total_assets], window_length=63)

def Asset_To_Equity_Ratio():
    """
    Asset / Equity Ratio
    Total current assets divided by common equity
    https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf # NOQA
    Notes:
    High value suggests that company has taken on substantial debt
    Vaires substantially with industry
    """
    return bs.total_assets.latest / bs.common_stock_equity.latest

def Debt_To_Asset_Ratio():
    """
    Debt / Asset Ratio:
    Total Debts divided by Total Assets
    https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf # NOQA
    Notes:
    High value suggests that company has taken on substantial debt
    Low value suggests good financial health as assets greater than debt
    Long Term Debt
    """
    return bs.total_debt.latest / bs.total_assets.latest

def Dividend_Growth():
    """
    Dividend Growth:
    Growth in dividends observed over a 1-year lookback window
    https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf # NOQA
    Notes:
    High value suggests that rate at which the quantity of dividends
    paid out is increasing Morningstar built-in fundamental
    better as adjusts inf values
    """
    return morningstar.earnings_ratios.dps_growth.latest

def FCF_EV():
    return factset.Fundamentals.free_cf_fcfe_qf.latest / \
               factset.Fundamentals.entrpr_val_qf.latest

def DEBT_TOTAL_ASSETS():
    return -factset.Fundamentals.debt.latest / \
                factset.Fundamentals.assets.latest

class ROIC_GROWTH_2YR(CustomFactor):  
        inputs = [Fundamentals.roic]  
        window_length = 504
        window_safe = True
        def compute(self, today, assets, out, roic):  
            out[:] = (gmean([roic[-1]+1, roic[-252]+1,roic[-504]+1])-1)

class GM_STABILITY_8YR(CustomFactor):  
        inputs = [Fundamentals.gross_margin]  
        window_length = 9
        window_safe = True
        def compute(self, today, assets, out, gm):  
            out[:] = (gm[-8]) 

class GM_STABILITY_2YR(CustomFactor):  
        inputs = [Fundamentals.gross_margin]  
        window_length = 504
        window_safe = True
        def compute(self, today, assets, out, gm):  
            out[:] = preprocess(np.std([gm[-1]-gm[-252],gm[-252]-gm[-504]],axis=0)) 

#Profitability
def Net_Income_Margin():
    """
    Gross Income Margin:
    Gross Profit divided by Net Sales
    https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf # NOQA
    Notes:
    High value suggests that the company is generating large profits
    Builtin used as cleans inf values
    """
    return or_.net_margin.latest

def Return_On_Total_Equity():
    """
    Return on Total Equity:
    Net income divided by average of total shareholder equity
    https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf # NOQA
    Notes:
    High value suggests that the company is generating large profits
    Builtin used as cleans inf values
    """
    return or_.roe.latest

def Return_On_Total_Assets():
    """
    Return on Total Assets:
    Net income divided by average total assets
    https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf # NOQA
    Notes:
    High value suggests that the company is generating large profits
    Builtin used as cleans inf values
    """
    return or_.roa.latest

def Return_On_Total_Invest_Capital():
    """
    Return on Total Invest Capital:
    Net income divided by average total invested capital
    https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf # NOQA
    Notes:
    High value suggests that the company is generating large profits
    Builtin used as cleans inf values
    """
    return or_.roic.latest

#Sentiment
class MessageSum(CustomFactor):
        inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close, stocktwits.bull_scored_messages, stocktwits.bear_scored_messages, stocktwits.total_scanned_messages]
        window_length = 21
        window_safe = True
        def compute(self, today, assets, out, high, low, close, bull, bear, total):
            v = np.nansum((high-low)/close, axis=0)
            out[:] = preprocess(v*np.nansum(total*(bear-bull), axis=0))

#Risk
class Volatility(CustomFactor):

    inputs = [USEquityPricing.close]
    window_length = 252

    def compute(self, today, assets, out, close):  
        close = pd.DataFrame(data=close, columns=assets) 
        # Since we are going to rank largest is best we need to invert the sdev.
        out[:] = 1 / np.log(close).diff().std()

class Volatility3M(CustomFactor):
    inputs = [Returns(window_length=2)]
    window_length = 63

    def compute(self, today, assets, out, rets):
        np.nanstd(rets, axis=0, out=out) 


features = {
    #Momentum
    'Momentum': Momentum,
    'aa_momentum': aa_momentum,
    'ATR': ATR,
    'MINUS_DM': MINUS_DM,
    'PLUS_DM': PLUS_DM,
    'Price Momentum 12M':Price_Momentum_12M,
    #Trend/Direction
    'Trendline': Trendline,
    'LINEARREG_INTERCEPT': LINEARREG_INTERCEPT,
    'MaxGap': MaxGap,
    'TEM': TEM,
    'PriceOscillator': PriceOscillator,
    'MeanReversion1M': MeanReversion1M,
    'ADX': ADX,
    'DX': DX,
    'MEDPRICE': MEDPRICE,
    'TYPPRICE': TYPPRICE,
    'TRANGE': TRANGE,
    'MACD_Signal_10d': MACD_Signal_10d,
    #TRADITIONAL VALUE
    'Dividend Yield':Dividend_Yield,
    'Price to Free Cashflows': Price_To_Free_Cashflows,
    'Price to Operating Cashflows':Price_To_Operating_Cashflows,
    'Price to Book': Price_To_Book,
    'EV to Cashflows': EV_To_Cashflows,
    #EFFICIENCY
    'Capex to Sales': Capex_To_Sales,
    'EBIT to Assets': EBIT_To_Assets,
    #Quality
    'Piotroski': Piotroski,
    'Asset Growth 3M': Asset_Growth_3M,
    'Asset to Equity Ratio': Asset_To_Equity_Ratio,
    'Debt to Asset Ratio': Debt_To_Asset_Ratio,
    'Dividend Growth': Dividend_Growth,
    'ROIC_GROWTH_2YR': ROIC_GROWTH_2YR,
    'GM_STABILITY_8YR': GM_STABILITY_8YR,
    'GM_STABILITY_2YR': GM_STABILITY_2YR,
    'FCF_EV': FCF_EV,
    'DEBT_TOTAL_ASSETS': DEBT_TOTAL_ASSETS,
    #PROFITABILITY
    'Net Income Margin': Net_Income_Margin,
    'Return on Total Assets': Return_On_Total_Assets,
    'Return on Total Equity': Return_On_Total_Equity,
    'Return on Invest Capital': Return_On_Total_Invest_Capital,
    #Sentiment
    'MessageSum': MessageSum,
    #Risk
    'Volatility': Volatility,
    'Volatility3M': Volatility3M,
}          
 

def shift_mask_data(features,
                    labels,
                    n_forward_days,
                    lower_percentile,
                    upper_percentile):
    """Align features to the labels ``n_forward_days`` into the future and
    return the discrete, flattened features and masked labels.

    Parameters
    ----------
    features : np.ndarray
        A 3d array of (days, assets, feature).
    labels : np.ndarray
        The labels to predict.
    n_forward_days : int
        How many days into the future are we predicting?
    lower_percentile : float
        The lower percentile in the range [0, 100].
    upper_percentile : float
        The upper percentile in the range [0, 100].

    Returns
    -------
    selected_features : np.ndarray
        The flattened features that are not masked out.
    selected_labels : np.ndarray
        The labels that are not masked out.
    """

    # Slice off rolled elements
    shift_by = n_forward_days + 1
    aligned_features = features[:-shift_by]
    aligned_labels = labels[shift_by:]

    cutoffs = np.nanpercentile(
        aligned_labels,
        [lower_percentile, upper_percentile],
        axis=1,
    )
    discrete_labels = np.select(
        [
            aligned_labels <= cutoffs[0, :, np.newaxis],
            aligned_labels >= cutoffs[1, :, np.newaxis],
        ],
        [-1, 1],
    )

    # flatten the features per day
    flattened_features = aligned_features.reshape(
        -1,
        aligned_features.shape[-1],
    )

    # Drop stocks that did not move much, meaning they are in between
    # ``lower_percentile`` and ``upper_percentile``.
    mask = discrete_labels != 0

    selected_features = flattened_features[mask.ravel()]
    selected_labels = discrete_labels[mask]

    return selected_features, selected_labels


class ML(CustomFactor):
    """
    """
    train_on_weekday = 1

    def __init__(self, *args, **kwargs):
        CustomFactor.__init__(self, *args, **kwargs)

        self._imputer = preprocessing.Imputer()
        self._scaler = preprocessing.MinMaxScaler()
        self._classifier = linear_model.SGDClassifier(penalty='elasticnet')
        self._trained = False
        #ensemble.AdaBoostClassifier(
        #    random_state=1337,
        #    n_estimators=50,
        #)

    def _compute(self, *args, **kwargs):
        ret = CustomFactor._compute(self, *args, **kwargs)
        return ret

    def _train_model(self, today, returns, inputs):
        log.info('training model for window starting on: {}'.format(today))

        imputer = self._imputer
        scaler = self._scaler
        classifier = self._classifier

        features, labels = shift_mask_data(
            np.dstack(inputs),
            returns,
            n_forward_days=PRED_N_FORWARD_DAYS,
            lower_percentile=30,
            upper_percentile=70,
        )
        features = scaler.fit_transform(imputer.fit_transform(features))

        start = time()
        classifier.fit(features, labels)
        log.info('training took {} secs'.format(time() - start))
        self._trained = True

    def _maybe_train_model(self, today, returns, inputs):
        if (today.weekday() == self.train_on_weekday) or not self._trained:
            self._train_model(today, returns, inputs)

    def compute(self, today, assets, out, returns, *inputs):
        # inputs is a list of factors, for example, assume we have 2 alpha
        # signals, 3 stocks, and a lookback of 2 days. Each element in the
        # inputs list will be data of one signal, so len(inputs) == 2. Then
        # each element will contain a 2-D array of shape [time x stocks]. For
        # example:
        # inputs[0]:
        # [[1, 3, 2], # factor 1 rankings of day t-1 for 3 stocks
        #  [3, 2, 1]] # factor 1 rankings of day t for 3 stocks
        # inputs[1]:
        # [[2, 3, 1], # factor 2 rankings of day t-1 for 3 stocks
        #  [1, 2, 3]] # factor 2 rankings of day t for 3 stocks
        self._maybe_train_model(today, returns, inputs)

        # Predict
        # Get most recent factor values (inputs always has the full history)
        last_factor_values = np.vstack([input_[-1] for input_ in inputs]).T
        last_factor_values = self._imputer.transform(last_factor_values)
        last_factor_values = self._scaler.transform(last_factor_values)

        # Predict the probability for each stock going up
        # (column 2 of the output of .predict_proba()) and
        # return it via assignment to out.
        #out[:] = self._classifier.predict_proba(last_factor_values)[:, 1]
        out[:] = self._classifier.predict(last_factor_values)
        
def make_ml_pipeline(universe, window_length=63, n_forward_days=21):
    pipeline_columns = OrderedDict()

    # ensure that returns is the first input
    pipeline_columns['Returns'] = Returns(
        inputs=(USEquityPricing.open,),
        mask=universe, window_length=n_forward_days + 1,
    )

    # rank all the factors and put them after returns
    pipeline_columns.update({
        k: v().zscore(mask=universe) for k, v in features.items()
    })

    # Create our ML pipeline factor. The window_length will control how much
    # lookback the passed in data will have.
    pipeline_columns['ML'] = ML(
        inputs=pipeline_columns.values(),
        window_length=window_length + 1,
        mask=universe,
    )

    pipeline_columns['Sector'] = Sector()

    return Pipeline(screen=universe, columns=pipeline_columns)


def initialize(context):
    """
    Called once at the start of the algorithm.
    """
    set_slippage(slippage.FixedSlippage(spread=0.00))
    set_commission(commission.PerShare(cost=0, min_trade_cost=0))

    schedule_function(
        rebalance,
        TRADE_FREQ,
        time_rules.market_open(minutes=1),
    )

    # Record tracking variables at the end of each day.
    schedule_function(
        record_vars,
        date_rules.every_day(),
        time_rules.market_close(),
    )

    # Set up universe, alphas and ML pipline
    context.universe = QTradableStocksUS()

    ml_pipeline = make_ml_pipeline(
        context.universe,
        n_forward_days=PRED_N_FORWARD_DAYS,
        window_length=ML_TRAINING_WINDOW,
    )
    # Create our dynamic stock selector.
    attach_pipeline(ml_pipeline, 'alpha_model')
    # Attach the risk loading pipeline to our algorithm.
    algo.attach_pipeline(risk_loading_pipeline(), 'risk_loading_pipeline')

    context.past_predictions = {}
    context.hold_out_accuracy = 0
    context.hold_out_log_loss = 0
    context.hold_out_returns_spread_bps = 0


def evaluate_and_shift_hold_out(output, context):
    # Look at past predictions to evaluate classifier accuracy on hold-out data
    # A day has passed, shift days and drop old ones
    context.past_predictions = {
        k - 1: v
        for k, v in context.past_predictions.iteritems()
        if k > 0
    }

    if 0 in context.past_predictions:
        # Past predictions for the current day exist, so we can use todays'
        # n-back returns to evaluate them
        raw_returns = output['Returns']
        raw_predictions = context.past_predictions[0]

        # Join to match up equities
        returns, predictions = raw_returns.align(raw_predictions, join='inner')

        # Binarize returns
        returns_binary = returns > returns.median()
        predictions_binary = predictions > 0.5

        # Compute performance metrics
        context.hold_out_accuracy = metrics.accuracy_score(
            returns_binary.values,
            predictions_binary.values,
        )
        context.hold_out_log_loss = metrics.log_loss(
            returns_binary.values,
            predictions.values,
        )
        long_rets = returns[predictions_binary == 1].mean()
        short_rets = returns[predictions_binary == 0].mean()
        context.hold_out_returns_spread_bps = (long_rets - short_rets) * 10000

    # Store current predictions
    context.past_predictions[PRED_N_FORWARD_DAYS] = context.predicted_probs


def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    # Get the risk loading data every day.
    context.risk_loading_pipeline = pipeline_output('risk_loading_pipeline')
    
    output = pipeline_output('alpha_model')
    context.predicted_probs = output['ML']
    context.predicted_probs.index.rename(['date', 'equity'], inplace=True)

    context.risk_factors = pipeline_output('alpha_model')[['Sector']]
    context.risk_factors.index.rename(['date', 'equity'], inplace=True)
    context.risk_factors.Sector = context.risk_factors.Sector.map(
        Sector.SECTOR_NAMES,
    )

    evaluate_and_shift_hold_out(output, context)

    # These are the securities that we are interested in trading each day.
    context.security_list = context.predicted_probs.index


def rebalance(context, data):
    """
    Execute orders according to our schedule_function() timing.
    """
      # Constrain our risk exposures. We're using version 0 of the default bounds
    # which constrain our portfolio to 18% exposure to each sector and 36% to
    # each style factor.
    
    predictions = context.predicted_probs

    # Filter out stocks that can not be traded
    predictions = predictions.loc[data.can_trade(predictions.index)]
    # Select top and bottom N stocks
    n_long_short = min(N_STOCKS_TO_TRADE // 2, len(predictions) // 2)
    predictions_top_bottom = pd.concat([
        predictions.nlargest(n_long_short),
        predictions.nsmallest(n_long_short),
    ])

    # If classifier predicts many identical values, the top might contain
    # duplicate stocks
    predictions_top_bottom = predictions_top_bottom.iloc[
        ~predictions_top_bottom.index.duplicated()
    ]

    # predictions are probabilities ranging from 0 to 1
    predictions_top_bottom = (predictions_top_bottom - 0.5) * 2

    # Setup Optimization Objective
    objective = opt.MaximizeAlpha(predictions_top_bottom)

    # Setup Optimization Constraints
    constrain_gross_leverage = opt.MaxGrossExposure(1.0)
    constrain_pos_size = opt.PositionConcentration.with_equal_bounds(
        -0.02,
        +0.02,
    )
    market_neutral = opt.DollarNeutral()

    if predictions_top_bottom.index.duplicated().any():
        log.debug(predictions_top_bottom.head())

    sector_neutral = opt.NetGroupExposure.with_equal_bounds(
        labels=context.risk_factors.Sector.dropna(),
        min=-0.0001,
        max=0.0001,
    )
    
    constrain_sector_style_risk = opt.experimental.RiskModelExposure(  
        risk_model_loadings=context.risk_loading_pipeline,  
        version=0,
    )
    # Run the optimization. This will calculate new portfolio weights and
    # manage moving our portfolio toward the target.
    algo.order_optimal_portfolio(
        objective=objective,
        constraints=[
            constrain_gross_leverage,
            constrain_pos_size,
            market_neutral,
            sector_neutral,
            constrain_sector_style_risk,
        ],
    )


def record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
    record(
        leverage=context.account.leverage,
        hold_out_accuracy=context.hold_out_accuracy,
        hold_out_log_loss=context.hold_out_log_loss,
        hold_out_returns_spread_bps=context.hold_out_returns_spread_bps,
    )


def handle_data(context, data):
    pass