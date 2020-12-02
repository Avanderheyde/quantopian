import quantopian.algorithm as algo
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.filters import Q1500US
from quantopian.pipeline.data.sentdex import sentiment


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
    
    set_commission(commission.PerTrade(cost=0.001))


def make_pipeline():
    sentiment_factor = sentiment.sentiment_signal.latest
    universe = (Q1500US() & sentiment_factor.notnull())
    
    sentiment_quantiles = sentiment_factor.rank(mask=universe, method='average').quantiles(2)
    
    pipe = Pipeline(columns = {'sentiment':sentiment_factor,
                              'longs':(sentiment_factor >=4),
                              'shorts':(sentiment_factor<=-2)},
                   screen = universe)
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
     
