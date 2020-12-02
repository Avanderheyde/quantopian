# Sell in May and go away

# this strategy holds EITHER equity (SPY - sid:8554) or fixed income (BSV - sid:33651) 
# depending on the time of year

lastRebalance = None

#initialize the strategy by selecting the two instruments to trade and the notional limits
def initialize(context):
    context.stocks = [sid(8554),sid(33651)]
    context.price={}
    
    context.max_notional = 1000000.1
    context.min_notional = -1000000.0
    
def handle_data(context, data):
    
    global lastRebalance
     
    date = data[data.keys()[0]].datetime
    month = date.strftime("%B") #get current month as a string
    
    if lastRebalance is None or (date-lastRebalance).days >=35: #check if we are due to rebalance the portfolio
       if month == 'October' or month =='May':
                  
        # Initializing the position as zero at the start of each frame
          notional_equity = 0
          notional_fixedIncome = 0
    
         #update the notional amounts in equity and fixed income and the rebalance date
          notional_equity = context.portfolio.positions[sid(8554)].amount * data[sid(8554)].price
          notional_fixedIncome = context.portfolio.positions[sid(33651)].amount * data[sid(33651)].price
          notional_total = notional_equity + notional_fixedIncome
          lastRebalance = data[sid(8554)].datetime
            
          #generate some log entries to keep track of progress
          log.info("Rebalancing Portfolio - %s" %lastRebalance)
          log.info("Starting notional value = %s" %notional_total)
            
          if month == 'October': #in October, sell BSV and buy SPY
             if notional_total == 0: #if this is the first investment use all starting cash
                buyAmount = round(context.portfolio.starting_cash / data[sid(8554)].price)
                log.info("First investment in equity - buying %s shares in SPY" %buyAmount)
                order(sid(8554),buyAmount)
                
             elif notional_total > 0 and notional_fixedIncome > 0:
                  order(sid(33651),-(context.portfolio.positions[sid(33651)].amount))
                  buyAmount = round(notional_total / data[sid(8554)].price)
                  log.info("Rebalancing into equity - buying %s shares in SPY" %buyAmount)
                  order(sid(8554),buyAmount)                
                
          
          if month == 'May': # In May, sell SPY and buy BSV
             if notional_total==0: #if this is the first investment use all starting cash
                buyAmount = round(context.portfolio.starting_cash / data[sid(33651)].price)   
                log.info("First investment in fixed income - buying %s shares in BSV" %buyAmount)
                order(sid(33651),buyAmount)
                
             elif notional_total>0 and notional_equity > 0:
                  order(sid(8554),-(context.portfolio.positions[sid(8554)].amount))
                  buyAmount = round(notional_total / data[sid(33651)].price)
                  log.info("Rebalancing into fixed income - buying %shares in BSV" %buyAmount)
                  order(sid(33651),buyAmount)  
                    
          
          
          