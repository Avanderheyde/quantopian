import quantopian.algorithm as algo
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.data.sentdex import sentiment
from quantopian.pipeline.factors import CustomFactor, Returns
from quantopian.pipeline.factors import Latest
from quantopian.pipeline.data import morningstar
import numpy as np


def initialize(context):
    """
    Called once at the start of the algorithm.
    """
    # Rebalance every day, 1 hour after market open.
    algo.schedule_function(
        rebalance,
        algo.date_rules.every_day(),
        algo.time_rules.market_open(hours=1),
    )

    # Record tracking variables at the end of each day.
    algo.schedule_function(
        record_vars,
        algo.date_rules.every_day(),
        algo.time_rules.market_close(),
    )

    # Create our dynamic stock selector.
    algo.attach_pipeline(make_pipeline(), 'pipeline')
    #set_commission(commission.PerTrade(cost=0.001))


def make_pipeline():
    or_ = morningstar.operation_ratios
    class LowVol(CustomFactor):
        inputs = [Returns(window_length=2)]
        window_length = 25
    
        def compute(self, today, assets, out, close):
            out[:] = -np.nanstd(close, axis=0)
    #LowVol, Revenue Growth, Sentiment
    universe = QTradableStocksUS()
    
    #use factors[] then loop through
    
    testing_factor1 = LowVol(mask=universe)
    testing_factor2 = or_.revenue_growth.latest
    testing_factor3 = sentiment.sentiment_signal.latest

    universe = (QTradableStocksUS() 
                & testing_factor1.notnull()
                & testing_factor2.notnull()
                & testing_factor3.notnull())
    
    testing_factor1 = testing_factor1.rank(mask=universe, method='average')
    testing_factor2 = testing_factor2.rank(mask=universe, method='average')
    testing_factor3 = testing_factor3.rank(mask=universe, method='average')
    
    testing_factor = testing_factor1 + testing_factor2 + testing_factor3
    
    testing_quantiles = testing_factor.quantiles(2)
    
    pipe = Pipeline(columns={
        'testing_factor': testing_factor,
        'shorts':testing_quantiles.eq(0),
        'longs':testing_quantiles.eq(1),
    },screen= universe)
    
    return pipe

def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    context.output = algo.pipeline_output('pipeline')

    # These are the securities that we are interested in trading each day.
    context.security_list = context.output.index


def rebalance(context, data):
    "half money go long and half money go short"
    long_secs = context.output[context.output['longs']].index
    long_weight = 0.5/len(long_secs)
    
    short_secs = context.output[context.output['shorts']].index
    short_weight = -0.5/len(short_secs)
    
    for security in long_secs:
        if data.can_trade(security):
            order_target_percent(security, long_weight)
    
    for security in short_secs:
        if data.can_trade(security):
            order_target_percent(security, short_weight)
    
    for security in context.portfolio.positions:
        if data.can_trade(security) and security not in long_secs and security not in short_secs:
            order_target_percent(security,0)
                             


def record_vars(context, data):
    long_count = 0
    short_count = 0
    
    for position in context.portfolio.positions.itervalues():
        if position.amount >0:
            long_count +=1
        elif position.amount <0:
            short_count += 1
     
    record(num_longs = long_count, num_shorts = short_count, leverage = context.account.leverage)