from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import CustomFactor, SimpleMovingAverage
from quantopian.pipeline.data import morningstar

import pandas as pd
import numpy as np


class AvgDailyDollarVolumeTraded(CustomFactor):
    
    inputs = [USEquityPricing.close, USEquityPricing.volume]
    window_length = 20
    
    def compute(self, today, assets, out, close_price, volume):
        out[:] = np.mean(close_price * volume, axis=0)
        

class Value(CustomFactor):
    
    inputs = [morningstar.valuation_ratios.book_value_yield,
              morningstar.valuation_ratios.sales_yield,
              morningstar.valuation_ratios.fcf_yield] 
    
    window_length = 1
    
    def compute(self, today, assets, out, book_value, sales, fcf):
        value_table = pd.DataFrame(index=assets)
        value_table["book_value"] = book_value[-1]
        value_table["sales"] = sales[-1]
        value_table["fcf"] = fcf[-1]
        out[:] = value_table.rank().mean(axis=1)
    
    
class Momentum(CustomFactor):
    
    inputs = [USEquityPricing.close]
    window_length = 252
    
    def compute(self, today, assets, out, close):
        out[:] = close[-20] / close[0]

            
class Quality(CustomFactor):
    
    inputs = [morningstar.operation_ratios.roe,
              morningstar.operation_ratios.roa,
              morningstar.operation_ratios.gross_margin]
    window_length = 1
    
    def compute(self, today, assets, out, roe, roa, gross_margin):  
        quality_table = pd.DataFrame(index=assets)
        quality_table["roe"] = roe[-1]
        quality_table["roa"] = roa[-1]
        quality_table["gross_margin"] = gross_margin[-1]
        out[:] = quality_table.rank().mean(axis=1)
        
        
class Volatility(CustomFactor):
    
    inputs = [USEquityPricing.close]
    window_length = 252
    
    def compute(self, today, assets, out, close):  
        close = pd.DataFrame(data=close, columns=assets) 
        # Since we are going to rank largest is best we need to invert the sdev.
        out[:] = 1 / np.log(close).diff().std()
    

# Compute final rank and assign long and short baskets.
def before_trading_start(context, data):
    results = pipeline_output('factors').dropna()
    ranks = results.rank().mean(axis=1).order()
    
    context.shorts = 1 / ranks[results["momentum"] < 1].head(200)    
    context.longs = ranks[results["momentum"] > 1].tail(200) 
    
    context.shorts /= context.shorts.sum()
    context.longs /= context.longs.sum()
    
    update_universe(context.longs.index | context.shorts.index)
    
    
# Put any initialization logic here. The context object will be passed to
# the other methods in your algorithm.
def initialize(context):
    pipe = Pipeline()
    pipe = attach_pipeline(pipe, name='factors')
        
    value = Value()
    momentum = Momentum()
    quality = Quality()
    volatility = Volatility()
    
    pipe.add(value, "value")
    pipe.add(momentum, "momentum")
    pipe.add(quality, "quality")
    pipe.add(volatility, "volatility")

    
    sma_200 = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=200)
    dollar_volume = AvgDailyDollarVolumeTraded()
    
    # Screen out penny stocks and low liquidity securities.
    pipe.set_screen((sma_200 > 5) & (dollar_volume > 10**7))
    
    context.spy = sid(8554)
    context.shorts = None
    context.longs = None
    
    schedule_function(rebalance, date_rules.month_start())
    schedule_function(cancel_open_orders, date_rules.every_day(),
                      time_rules.market_close())

    
# Will be called on every trade event for the securities you specify. 
def handle_data(context, data):
    record(lever=context.account.leverage,
           exposure=context.account.net_leverage,
           num_pos=len(context.portfolio.positions),
           oo=len(get_open_orders()))

    
def cancel_open_orders(context, data):
    for security in get_open_orders():
        for order in get_open_orders(security):
            cancel_order(order)
        
    
def rebalance(context, data):
    for security in context.shorts.index:
        if get_open_orders(security):
            continue
        if security in data:
            order_target_percent(security, -context.shorts[security])
            
    for security in context.longs.index:
        if get_open_orders(security):
            continue
        if security in data:
            order_target_percent(security, context.longs[security])
            
    for security in context.portfolio.positions:
        if get_open_orders(security):
            continue
        if security in data:
            if security not in (context.longs.index | context.shorts.index):
                order_target_percent(security, 0)