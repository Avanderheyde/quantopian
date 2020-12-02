# References:
# https://www.quantopian.com/posts/quantcon-nyc-2017-advanced-workshop
# https://blog.quantopian.com/a-professional-quant-equity-workflow/
# https://www.lib.uwo.ca/business/betasbydatabasebloombergdefinitionofbeta.html

from quantopian.algorithm import attach_pipeline, pipeline_output, order_optimal_portfolio
from quantopian.pipeline import Pipeline
from quantopian.pipeline.factors import CustomFactor, RollingLinearRegressionOfReturns
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.classifiers.morningstar import Sector
from quantopian.pipeline.factors import Latest, Returns
import quantopian.experimental.optimize as opt
from quantopian.pipeline.data.psychsignal import stocktwits
from scipy.stats.mstats import zscore
from scipy.stats import rankdata

import numpy as np
import pandas as pd

from quantopian.pipeline.filters import Q1500US

# Constraint Parameters
MAX_GROSS_LEVERAGE = 1.0
NUM_LONG_POSITIONS = 300
NUM_SHORT_POSITIONS = 300

MAX_SHORT_POSITION_SIZE = 10*1.0/(NUM_LONG_POSITIONS + NUM_SHORT_POSITIONS)
MAX_LONG_POSITION_SIZE = 10*1.0/(NUM_LONG_POSITIONS + NUM_SHORT_POSITIONS)

# Risk Exposures
MAX_SECTOR_EXPOSURE = 0.10
MAX_BETA_EXPOSURE = 0.20

EPS = 1.005 # optimization parameter
        
def make_factors():
   
    class OptRev5d(CustomFactor):   
        inputs = [USEquityPricing.open,USEquityPricing.high,USEquityPricing.low,USEquityPricing.close]
        window_length = 5
        def compute(self, today, assets, out, open, high, low, close):

            p = (open+high+low+close)/4

            m = len(p)
            a = np.zeros(m)
            w = np.zeros(m)

            for k in range(1,m+1):
                (a,w) = get_weights(p[-k:,:],close[-1,:])
                a += w*a
                w += w

            out[:] = preprocess(a/w)
            
    class OptRev30d(CustomFactor):   
        inputs = [USEquityPricing.open,USEquityPricing.high,USEquityPricing.low,USEquityPricing.close]
        window_length = 30
        def compute(self, today, assets, out, open, high, low, close):

            p = (open+high+low+close)/4

            m = len(p)
            a = np.zeros(m)
            w = np.zeros(m)

            for k in range(3,m+1):
                (a,w) = get_weights(p[-k:,:],close[-1,:])
                a += w*a
                w += w

            out[:] = preprocess(a/w) 
        
    class MessageSum(CustomFactor):
        inputs = [stocktwits.bull_scored_messages, stocktwits.bear_scored_messages, stocktwits.total_scanned_messages]
        window_length = 21
        def compute(self, today, assets, out, bull, bear, total):
            out[:] = preprocess(-(np.nansum(bull, axis=0)+np.nansum(bear, axis=0)))

    class Volatility(CustomFactor):    
        inputs = [USEquityPricing.open,USEquityPricing.high,USEquityPricing.low,USEquityPricing.close]
        window_length = 3*252 
        def compute(self, today, assets, out, open, high, low, close):
            p = (open+high+low+close)/4
            price = pd.DataFrame(data=p, columns=assets) 
            # Since we are going to rank largest is best we need to invert the sdev.
            out[:] = preprocess(1 / np.log(price).diff().std())

    class Yield(CustomFactor):  
        inputs = [morningstar.valuation_ratios.total_yield]  
        window_length = 1  
        def compute(self, today, assets, out, syield):  
            out[:] =  preprocess(syield[-1])

    class Momentum(CustomFactor):
        inputs = [USEquityPricing.open, USEquityPricing.high, USEquityPricing.low, USEquityPricing.close]
        window_length = 252

        def compute(self, today, assets, out, open, high, low, close):

            p = (open + high + low + close)/4

            out[:] = preprocess(((p[-21] - p[-252])/p[-252] -
                      (p[-1] - p[-21])/p[-21]))

    class Quality(CustomFactor):     
        inputs = [morningstar.income_statement.gross_profit, morningstar.balance_sheet.total_assets]
        window_length = 3*252

        def compute(self, today, assets, out, gross_profit, total_assets):
            norm = gross_profit / total_assets
            out[:] = preprocess((norm[-1] - np.mean(norm, axis=0)) / np.std(norm, axis=0))
         
    return {
            'OptRev5d':              OptRev5d,
            'OptRev30d':             OptRev30d,
            'MessageSum':            MessageSum,
            'Volatility':            Volatility,
            'Yield':                 Yield,
            'Momentum':              Momentum,
            'Quality':               Quality,
        }


def make_pipeline():
    
   # Define universe
   # ===============================================   
    pricing = USEquityPricing.close.latest    
    base_universe = (Q1500US() & (pricing > 5))  
    ev = Latest(inputs=[morningstar.valuation.enterprise_value], mask=base_universe)
    ev_positive = ev > 0   
    ebitda = Latest(inputs=[morningstar.income_statement.ebitda], mask=ev_positive)
    ebitda_positive = ebitda > 0         
    market_cap = Latest(inputs=[morningstar.valuation.market_cap], mask = ebitda_positive)    
    universe = market_cap.top(2*(NUM_LONG_POSITIONS + NUM_SHORT_POSITIONS))
    
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

    constraints.append(opt.MaxGrossLeverage(MAX_GROSS_LEVERAGE))
    constraints.append(opt.DollarNeutral())
    constraints.append(
        opt.NetPartitionExposure.with_equal_bounds(
            labels=pipeline_data.sector,
            min=-MAX_SECTOR_EXPOSURE,
            max=MAX_SECTOR_EXPOSURE,
        ))
    neutralize_risk_factors = opt.WeightedExposure(
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
    
    a = np.nan_to_num(a - np.nanmean(a))

    return zscore(a)

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