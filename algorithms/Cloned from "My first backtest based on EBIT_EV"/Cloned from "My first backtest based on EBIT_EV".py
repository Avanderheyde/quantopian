import datetime as dt
import numpy as np

from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import CustomFactor, Pipeline
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import Returns, SimpleMovingAverage

class AvgDailyDollarVolumeTraded(CustomFactor):
    inputs = [USEquityPricing.close, USEquityPricing.volume]
    def compute(self, today, assets, out, close_price, volume):
        out[:] = np.nanmedian(close_price * volume, axis=0)

class MarketCap(CustomFactor):

    # Pre-declare inputs and window_length
    inputs = [USEquityPricing.close, morningstar.valuation.shares_outstanding]

    # Compute market cap value
    def compute(self, today, assets, out, close, shares):
        out[:] = np.nanmedian(close * shares, axis=0)

class EBIT_EV(CustomFactor):

    inputs = [morningstar.valuation.enterprise_value,
              morningstar.income_statement.ebit]

    def compute(self, today, assets, out, ev, ebit):
        out[:] = np.nanmean(ebit, axis=0) / np.nanmean(ev, axis=0)

class Spread_EBIT_EV(CustomFactor):

    inputs = [morningstar.valuation.enterprise_value,
              morningstar.income_statement.ebit]

    def compute(self, today, assets, out, ev, ebit):
        spr_ebit = np.nanstd(ebit, axis=0)/np.nanmean(ebit, axis=0)
        spr_ev = np.nanstd(ev, axis=0)/np.nanmean(ev, axis=0)
        out[:] = spr_ebit * spr_ev

class EV_S(CustomFactor):

    inputs = [morningstar.valuation.enterprise_value,
              morningstar.valuation_ratios.sales_per_share,
              morningstar.valuation.shares_outstanding]
    window_length = 1

    def compute(self, today, assets, out, ev, sales, shares):
        out[:] = ev[-1] / (sales[-1] * shares[-1])

class SNOA(CustomFactor):
    inputs = [morningstar.balance_sheet.total_assets,
              morningstar.balance_sheet.total_liabilities,
              morningstar.balance_sheet.cash_and_cash_equivalents,
              morningstar.balance_sheet.current_debt,
              morningstar.balance_sheet.long_term_debt]
    window_length = 1

    def compute(self, today, assets, out, total_assets, total_liabilities, cash, current_debt, long_term_debt):
        out[:] = (total_assets[-1] - total_liabilities[-1] - cash[-1] + long_term_debt[-1] + current_debt[-1])/total_assets[-1]

class Sector(CustomFactor):
    inputs = [morningstar.asset_classification.morningstar_sector_code]
    window_length = 1

    def compute(self, today, assets, out, code):
        out[:] = code

def initialize(context):
    context.position_count = 10
    context.positions_considered = 50
    # Sector mappings
    context.sector_mappings = {
        101.0: "Basic Materials",
        102.0: "Consumer Cyclical",
        103.0: "Financial Services",
        104.0: "Real Estate",
        205.0: "Consumer Defensive",
        206.0: "Healthcare",
        207.0: "Utilites",
        308.0: "Communication Services",
        309.0: "Energy",
        310.0: "Industrials",
        311.0: "Technology"
    }
    # Create and attach an empty Pipeline.
    pipe = Pipeline()
    pipe = attach_pipeline(pipe, name='my_pipeline')
    # Construct Factors.
    dv = AvgDailyDollarVolumeTraded(window_length=20)
    mkt = MarketCap(window_length=10)
    ebit_ev = EBIT_EV(window_length=255)
    spread_ebit_ev = Spread_EBIT_EV(window_length=255)
    snoa = SNOA()
    # Construct a Filter.
    net_filter = mkt.percentile_between(40.0, 100.0) & dv.percentile_between(40.0, 100.0) & snoa.percentile_between(0.0, 95.0)
    ebit_ev_rank = ebit_ev.rank(mask=net_filter, ascending=False)
    spread_ebit_ev_rank = spread_ebit_ev.rank(mask=net_filter, ascending=True)
    # Register outputs.
    pipe.add(ebit_ev, 'ebit_ev')
    pipe.add(ebit_ev_rank, 'ebit_ev_rank')
    pipe.add(spread_ebit_ev_rank, 'spread_ebit_ev_rank')
    # Remove rows for which the Filter returns False.
    pipe.set_screen(net_filter)
    set_long_only()
    schedule_function(rebalance,
                      date_rule=date_rules.week_start(days_offset=2),
                      time_rule=time_rules.market_open(minutes=90))

def before_trading_start(context, data):
    # Access results using the name passed to `attach_pipeline`.
    results = pipeline_output('my_pipeline')
    # Define a universe with the results of a Pipeline.
    # Take the first ten assets by 30-day SMA.
    today = get_datetime()
    results['combined_rank'] = (results['ebit_ev_rank']*results['ebit_ev_rank']*0.90) + (results['spread_ebit_ev_rank']*results['spread_ebit_ev_rank']*0.10)
    best = results.sort('combined_rank', ascending=True).index
    # best = results.sort('ebit_ev_rank', ascending=True).index
    best_non_new = [stock for stock in best if (today - stock.start_date).days >= 730]
    context.stocks = best_non_new[:context.positions_considered]
    update_universe(context.stocks)

def create_weights(context, stocks, max_count):
    """
        Takes in a list of securities and weights them all equally
    """
    if len(stocks) == 0:
        return 0
    elif len(stocks) > max_count:
        return 1.0/max_count
    else:
        return 1.0/len(stocks)

def is_equity(stock):
    return type(stock).__name__.find('Equity') != -1

def log_positions_and_sizes(prefix, positions):
    pns = {}
    for stock in positions:
        if is_equity(stock):
            pns[stock.symbol] = positions[stock].amount
    log.info(prefix + ' Positions: ' + str(pns))

def count_positions(positions):
    result = 0
    for stock in positions:
        if is_equity(stock):
            result += 1
    return result

def rebalance(context, data):
    # Exit all positions before starting new ones
    log_positions_and_sizes('*** Starting Rebalance', context.portfolio.positions)
    desired = 0
    for stock in context.portfolio.positions:
        if stock not in data or stock not in context.stocks:
            log.info('Closing position ' + str(stock))
            order_target_percent(stock, 0.0)
            desired += 1
    # Create weights for each stock
    weight = create_weights(context, context.stocks, context.position_count)
    desired = desired + context.position_count - count_positions(context.portfolio.positions)
    # Rebalance all stocks to target weights
    for stock in context.stocks:
        if desired == 0:
            break
        if stock in data:
            log.info('Opening position ' + str(stock))
            if stock not in context.portfolio.positions:
                desired -= 1
            order_target_percent(stock, weight)
    log_positions_and_sizes('*** Finishing Rebalance', context.portfolio.positions)

def handle_data(context, data):
    record(num_positions = count_positions(context.portfolio.positions))
    #record(value = context.portfolio.portfolio_value)
