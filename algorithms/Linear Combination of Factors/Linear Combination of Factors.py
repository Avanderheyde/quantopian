import pandas as pd

import quantopian.algorithm as algo
import quantopian.optimize as opt

from quantopian.pipeline import Pipeline
from quantopian.pipeline.data import builtin, Fundamentals, psychsignal
from quantopian.pipeline.factors import AverageDollarVolume, RollingLinearRegressionOfReturns
from quantopian.pipeline.factors.fundamentals import MarketCap
from quantopian.pipeline.classifiers.fundamentals import Sector
from quantopian.pipeline.experimental import QTradableStocksUS, risk_loading_pipeline

# Algorithm Parameters
# --------------------
UNIVERSE_SIZE = 1000
LIQUIDITY_LOOKBACK_LENGTH = 100

MINUTES_AFTER_OPEN_TO_TRADE = 5

MAX_GROSS_LEVERAGE = 1.0
MAX_SHORT_POSITION_SIZE = 0.01  # 1%
MAX_LONG_POSITION_SIZE = 0.01   # 1%


def initialize(context):
    # Universe Selection
    # ------------------
    base_universe = QTradableStocksUS()

    # From what remains, each month, take the top UNIVERSE_SIZE stocks by average dollar
    # volume traded.
    monthly_top_volume = (
        AverageDollarVolume(window_length=LIQUIDITY_LOOKBACK_LENGTH)
        .top(UNIVERSE_SIZE, mask=base_universe)
        .downsample('week_start')
    )
    # The final universe is the monthly top volume &-ed with the original base universe.
    # &-ing these is necessary because the top volume universe is calculated at the start 
    # of each month, and an asset might fall out of the base universe during that month.
    universe = monthly_top_volume & base_universe

    # Alpha Generation
    # ----------------
    # Compute Z-scores of free cash flow yield and earnings yield. 
    # Both of these are fundamental value measures.
    fcf_zscore = Fundamentals.fcf_yield.latest.zscore(mask=universe)
    yield_zscore = Fundamentals.earning_yield.latest.zscore(mask=universe)
    sentiment_zscore = psychsignal.stocktwits.bull_minus_bear.latest.zscore(mask=universe)
    
    # Alpha Combination
    # -----------------
    # Assign every asset a combined rank and center the values at 0.
    # For UNIVERSE_SIZE=500, the range of values should be roughly -250 to 250.
    combined_alpha = (fcf_zscore + yield_zscore + sentiment_zscore).rank().demean()
    
    beta = 0.66*RollingLinearRegressionOfReturns(
                    target=sid(8554),
                    returns_length=5,
                    regression_length=260,
                    mask=combined_alpha.notnull() & Sector().notnull()
                    ).beta + 0.33*1.0

    # Schedule Tasks
    # --------------
    # Create and register a pipeline computing our combined alpha and a sector
    # code for every stock in our universe. We'll use these values in our 
    # optimization below.
    pipe = Pipeline(
        columns={
            'alpha': combined_alpha,
            'sector': Sector(),
            'sentiment': sentiment_zscore,
            'beta': beta,
        },
        # combined_alpha will be NaN for all stocks not in our universe,
        # but we also want to make sure that we have a sector code for everything
        # we trade.
        screen=combined_alpha.notnull() & Sector().notnull() & beta.notnull(),
    )
    
    # Multiple pipelines can be used in a single algorithm.
    algo.attach_pipeline(pipe, 'pipe')
    algo.attach_pipeline(risk_loading_pipeline(), 'risk_loading_pipeline')
    

    # Schedule a function, 'do_portfolio_construction', to run twice a week
    # ten minutes after market open.
    algo.schedule_function(
        do_portfolio_construction,
        date_rule=algo.date_rules.week_start(),
        time_rule=algo.time_rules.market_open(minutes=MINUTES_AFTER_OPEN_TO_TRADE),
        half_days=False,
    )


def before_trading_start(context, data):
    # Call pipeline_output in before_trading_start so that pipeline
    # computations happen in the 5 minute timeout of BTS instead of the 1
    # minute timeout of handle_data/scheduled functions.
    context.pipeline_data = algo.pipeline_output('pipe')
    context.risk_loading_pipeline = algo.pipeline_output('risk_loading_pipeline')


# Portfolio Construction
# ----------------------
def do_portfolio_construction(context, data):
    pipeline_data = context.pipeline_data

    # Objective
    # ---------
    # For our objective, we simply use our naive ranks as an alpha coefficient
    # and try to maximize that alpha.
    # 
    # This is a **very** naive model. Since our alphas are so widely spread out,
    # we should expect to always allocate the maximum amount of long/short
    # capital to assets with high/low ranks.
    #
    # A more sophisticated model would apply some re-scaling here to try to generate
    # more meaningful predictions of future returns.
    objective = opt.MaximizeAlpha(pipeline_data.alpha)

    # Constraints
    # -----------
    # Constrain our gross leverage to 1.0 or less. This means that the absolute
    # value of our long and short positions should not exceed the value of our
    # portfolio.
    constrain_gross_leverage = opt.MaxGrossExposure(MAX_GROSS_LEVERAGE)
    
    # Constrain individual position size to no more than a fixed percentage 
    # of our portfolio. Because our alphas are so widely distributed, we 
    # should expect to end up hitting this max for every stock in our universe.
    constrain_pos_size = opt.PositionConcentration.with_equal_bounds(
        -MAX_SHORT_POSITION_SIZE,
        MAX_LONG_POSITION_SIZE,
    )

    # Constrain ourselves to allocate the same amount of capital to 
    # long and short positions.
    market_neutral = opt.DollarNeutral()
    
    # Constrain beta-to-SPY to remain under the contest criteria.
    beta_neutral = opt.FactorExposure(
        pipeline_data[['beta']],
        min_exposures={'beta': -0.05},
        max_exposures={'beta': 0.05},
    )

    # Constrain exposure to common sector and style risk 
    # factors, using the latest default values. At the time
    # of writing, those are +-0.18 for sector and +-0.36 for 
    # style.
    constrain_sector_style_risk = opt.experimental.RiskModelExposure(
        context.risk_loading_pipeline,
        version=opt.Newest,
    )

    # Run the optimization. This will calculate new portfolio weights and
    # manage moving our portfolio toward the target.
    algo.order_optimal_portfolio(
        objective=objective,
        constraints=[
            constrain_gross_leverage,
            constrain_pos_size,
            market_neutral,
            constrain_sector_style_risk,
            beta_neutral,
        ],
    )