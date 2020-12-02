"""

High Yield Low Vol system

Attempts to replicate http://imarketsignals.com/2016/trading-the-high-yield-low-volatility-stocks-of-the-sp500-with-the-im-hid-lov-7-system/

"""

from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline import CustomFactor
from quantopian.pipeline.filters import Q500US
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data import morningstar 
import numpy as np
import pandas as pd

# Volatility factor
class Volatility(CustomFactor):
    
    inputs = [USEquityPricing.close]
 
    def compute(self, today, assets, out, close):  
        close = pd.DataFrame(data=close, columns=assets) 
        # Since we are going to rank largest is best we need to invert the sdev.
        out[:] = 1 / np.log(close).diff().std()

# Yield Factor
class Yield(CustomFactor):  
    inputs = [morningstar.valuation_ratios.total_yield]  
    window_length = 1  
    def compute(self, today, assets, out, syield):  
        out[:] =  syield[-1]
        
        
def initialize(context):
    
    # how many days to look back volatility and returns
    context.lookback=3*252  # 3 years
    context.long_leverage = 1.0
   
    
    #set_benchmark(sid(41382)) #SPLV
    
    
    pipe = Pipeline()
    attach_pipeline(pipe, 'lvhy')
    
   
    # This is an approximation of the S&P 500
    top_500=Q500US()  # Q1500US()
    
    volatility=Volatility(window_length=context.lookback)
    pipe.add(volatility, 'volatility')
   
    # Rank factor 1 and add the rank to our pipeline
    volatility_rank = volatility.rank(mask=top_500)
    pipe.add(volatility_rank, 'volatility_rank')
    
    
    syield = Yield()  
    pipe.add(syield, 'yield')  
    
    # Rank factor 2 and add the rank to our pipeline
    yield_rank = syield.rank(mask=top_500)
    pipe.add(yield_rank, 'yield_rank')
   
    # Take the average of the two factor rankings, add this to the pipeline
    combo_raw = (volatility_rank + yield_rank)/2
    pipe.add(combo_raw, 'combo_raw') 
    
    # Rank the combo_raw and add that to the pipeline
    pipe.add(combo_raw.rank(mask=top_500), 'combo_rank')
    
    # Set a screen to capture max top 100 best stocks
    pipe.set_screen(top_500 )
            
    # Scedule my rebalance function
    schedule_function(func=rebalance, 
                      date_rule=date_rules.month_start(days_offset=0), 
                      time_rule=time_rules.market_open(hours=0,minutes=30), 
                      half_days=True)
    
    # Schedule my plotting function
    schedule_function(func=record_vars,
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_close(),
                      half_days=True) 
   
    
            
def before_trading_start(context, data):
    # Call pipelive_output to get the output
    context.output = pipeline_output('lvhy')
      
    # Load the long list       
    context.long_list = context.output.sort_values(by='combo_rank', ascending=False).iloc[:7]
  
def record_vars(context, data):  
       
    record(leverage = context.account.leverage, long_count=len(context.portfolio.positions))

    
# This rebalancing is called according to our schedule_function settings.     
def rebalance(context,data):
    
    if len(context.long_list) :
        long_weight = context.long_leverage / float(len(context.long_list))

        log.info("\n" + str(context.long_list.sort_values(by='combo_rank', ascending=False)))
        for long_stock in context.long_list.index:
            order_target_percent(long_stock, long_weight)
      
    for stock in context.portfolio.positions.iterkeys():
        if stock not in context.long_list.index :
            order_target(stock, 0)
