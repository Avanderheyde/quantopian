# References:
# https://www.quantopian.com/posts/quantcon-nyc-2017-advanced-workshop
# https://blog.quantopian.com/a-professional-quant-equity-workflow/
# https://www.lib.uwo.ca/business/betasbydatabasebloombergdefinitionofbeta.html
from collections import OrderedDict
from time import time

#quantopian imports
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.data import morningstar, Fundamentals
from quantopian.pipeline.factors import CustomFactor, SimpleMovingAverage, AverageDollarVolume,SimpleBeta, Returns, RSI,EWMA,RollingLinearRegressionOfReturns,Latest
from quantopian.pipeline.data.zacks import EarningsSurprises
from quantopian.pipeline.data import factset
from quantopian.pipeline.data.psychsignal import stocktwits
#Algo imports
import quantopian.optimize as opt
from quantopian.algorithm import attach_pipeline, pipeline_output, order_optimal_portfolio
from quantopian.pipeline.classifiers.fundamentals import Sector

#Python imports
import math
import talib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import linear_model, decomposition, ensemble, preprocessing, isotonic, metrics
from scipy.stats.mstats import winsorize
from scipy.stats.mstats import zscore
from scipy.stats import rankdata
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

# Constraint Parameters
MAX_GROSS_EXPOSURE = 1.0
NUM_LONG_POSITIONS = 50
NUM_SHORT_POSITIONS = 50

MAX_SHORT_POSITION_SIZE = 10*1.0/(NUM_LONG_POSITIONS + NUM_SHORT_POSITIONS)
MAX_LONG_POSITION_SIZE = 10*1.0/(NUM_LONG_POSITIONS + NUM_SHORT_POSITIONS)

# Risk Exposures
MAX_SECTOR_EXPOSURE = 0.10
MAX_BETA_EXPOSURE = 0.20

EPS = 1.005 # optimization parameter
        
def make_factors():
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
                
    class GrossMarginChange_long(CustomFactor):
        window_length = 2*252
        window_safe = True
        inputs = [factset.Fundamentals.ebit_oper_mgn_qf]
        def compute(self, today, assets, out, ebit_oper_mgn):
            ebit_oper_mgn = np.nan_to_num(ebit_oper_mgn)
            ebit_oper_mgn = preprocessing.scale(ebit_oper_mgn,axis=0)
            out[:] = preprocess(ebit_oper_mgn[-1])

    class ROA_GROWTH_2YR(CustomFactor):  
        inputs = [Fundamentals.roa]  
        window_length = 504
        window_safe = True
        def compute(self, today, assets, out, roa):  
            out[:] = (gmean([roa[-1]+1, roa[-252]+1,roa[-504]+1])-1)

    class ROIC_GROWTH_8YR(CustomFactor):  
        inputs = [Fundamentals.roic]  
        window_length = 9
        window_safe = True
        def compute(self, today, assets, out, roic):  
            out[:] = (gmean([roic[-1]/100+1, roic[-2]/100+1,roic[-3]/100+1,roic[-4]/100+1,roic[-5]/100+1,roic[-6]/100+1,roic[-7]/100+1,roic[-8]/100+1])-1)

    class fcf(CustomFactor):
        inputs = [Fundamentals.fcf_yield]
        window_length = 1
        window_safe = True
        def compute(self, today, assets, out, fcf_yield):
            out[:] = preprocess(np.nan_to_num(fcf_yield[-1,:]))
    
         
    return {
            'Piotroski':                      Piotroski,
            'ROIC_GROWTH_2YR':                ROIC_GROWTH_2YR,
            'GM_STABILITY_8YR':               GM_STABILITY_8YR,
            'GrossMarginChange_long':         GrossMarginChange_long,
            'ROA_GROWTH_2YR':                 ROA_GROWTH_2YR,
            'ROIC_GROWTH_8YR':                ROIC_GROWTH_8YR,
            'fcf':                            fcf,
        }


def make_pipeline():
    
   # Define universe
   # ===============================================    
    
    value = morningstar.valuation_ratios.ev_to_ebitda.latest
    market_cap = morningstar.valuation.market_cap.latest > 2e9   
    universe = value.bottom(2*(NUM_LONG_POSITIONS + NUM_SHORT_POSITIONS), mask = (QTradableStocksUS() & market_cap)) 

    sector = Sector(mask=universe)  # sector needed to construct portfolio
    # ===============================================
    
    factors = make_factors()
    
    combined_alpha = None
    for name, f in factors.iteritems():
        if combined_alpha == None:
            combined_alpha = f(mask=universe)
        else:
            combined_alpha = combined_alpha + f(mask=universe)

    longs = combined_alpha.top(NUM_LONG_POSITIONS)
    shorts = combined_alpha.bottom(NUM_SHORT_POSITIONS)

    long_short_screen = (longs | shorts)
    
    beta = 0.66*RollingLinearRegressionOfReturns(
                    target=sid(8554),
                    returns_length=5,
                    regression_length=260,
                    mask=long_short_screen
                    ).beta + 0.33*1.0

# Create pipeline
    pipe = Pipeline(columns = {
        'combined_alpha':combined_alpha,
        'sector':sector,
        'market_beta':beta
    },
    screen = long_short_screen)
    return pipe

def initialize(context):

    context.spy = sid(8554)
    
    attach_pipeline(make_pipeline(), 'long_short_equity_template')

    # Schedule my rebalance function
    schedule_function(func=rebalance,
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_open(hours=0,minutes=30),
                      half_days=True)
    # record my portfolio variables at the end of day
    schedule_function(func=recording_statements,
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_close(),
                      half_days=True)
    
def before_trading_start(context, data):

    context.pipeline_data = pipeline_output('long_short_equity_template')

def recording_statements(context, data):

    record(num_positions=len(context.portfolio.positions))

def rebalance(context, data):
    
    pipeline_data = context.pipeline_data
    todays_universe = pipeline_data.index

    risk_factor_exposures = pd.DataFrame({
            'market_beta':pipeline_data.market_beta.fillna(1.0)
        })
 
    objective = opt.MaximizeAlpha(pipeline_data.combined_alpha)
    
    constraints = []

    constraints.append(opt.MaxGrossExposure(MAX_GROSS_EXPOSURE))
    constraints.append(opt.DollarNeutral())
    constraints.append(
        opt.NetGroupExposure.with_equal_bounds(
            labels=pipeline_data.sector,
            min=-MAX_SECTOR_EXPOSURE,
            max=MAX_SECTOR_EXPOSURE,
        ))
    neutralize_risk_factors = opt.FactorExposure(
        loadings=risk_factor_exposures,
        min_exposures={'market_beta':-MAX_BETA_EXPOSURE},
        max_exposures={'market_beta':MAX_BETA_EXPOSURE}
        )
    constraints.append(neutralize_risk_factors)
    constraints.append(
        opt.PositionConcentration.with_equal_bounds(
            min=-MAX_SHORT_POSITION_SIZE,
            max=MAX_LONG_POSITION_SIZE
        ))

    try:
        order_optimal_portfolio(
        objective=objective,
        constraints=constraints,
        universe=todays_universe
    )
    except:
        return
    
def simplex_projection(v, b=1):
#     """Projection vectors to the simplex domain

# Implemented according to the paper: Efficient projections onto the
# l1-ball for learning in high dimensions, John Duchi, et al. ICML 2008.
# Implementation Time: 2011 June 17 by Bin@libin AT pmail.ntu.edu.sg
# Optimization Problem: min_{w}\| w - v \|_{2}^{2}
# s.t. sum_{i=1}^{m}=z, w_{i}\geq 0

# Input: A vector v \in R^{m}, and a scalar z > 0 (default=1)
# Output: Projection vector w

# :Example:
# >>> proj = simplex_projection([.4 ,.3, -.4, .5])
# >>> print proj
# array([ 0.33333333, 0.23333333, 0. , 0.43333333])
# >>> print proj.sum()
# 1.0

# Original matlab implementation: John Duchi (jduchi@cs.berkeley.edu)
# Python-port: Copyright 2012 by Thomas Wiecki (thomas.wiecki@gmail.com).
# """

    v = np.asarray(v)
    p = len(v)

    # Sort v into u in descending order
    v = (v > 0) * v
    u = np.sort(v)[::-1]
    sv = np.cumsum(u)

    rho = np.where(u > (sv - b) / np.arange(1, p+1))[0][-1]
    theta = np.max([0, (sv[rho] - b) / (rho+1)])
    w = (v - theta)
    w[w<0] = 0
    return w

def preprocess(a):
    
    a = a.astype(np.float64)
    a[np.isinf(a)] = np.nan
    a = np.nan_to_num(a - np.nanmean(a))
    a = winsorize(a, limits=[0.02,0.98])
    
    return a

def get_weights(p,c):
    
        # EPS = 1.0 # optimization parameter

        x_tilde = np.nan_to_num(np.mean(p,axis=0)/c)
        x_tilde[x_tilde==0] = 1
        y_tilde = np.nan_to_num(1.0/x_tilde)
        y_tilde[y_tilde==0] = 1
        
        m = len(x_tilde)
        d = np.ones(m)
        d[x_tilde < 1] = -1
    
        x_tilde[x_tilde < 1] = 0
        y_tilde[x_tilde != 0] = 0
    
        x_tilde = x_tilde + y_tilde      
        
        b_t = 1.0*np.ones(m)/m
        
        ###########################
        # Inside of OLMAR (algo 2)

        x_bar = x_tilde.mean()

        # Calculate terms for lambda (lam)
        dot_prod = np.dot(b_t, x_tilde)
        num = EPS - dot_prod
        denom = (np.linalg.norm((x_tilde-x_bar)))**2

        # test for divide-by-zero case
        if denom == 0.0:
            lam = 0 # no portolio update
        else:     
            lam = max(0, num/denom)
                
        b = b_t + lam*(x_tilde-x_bar)

        a = simplex_projection(b)

        w = np.dot(x_tilde,a)
        
        return (d*a,w)