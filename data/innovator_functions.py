
"""
All of these functions are used for the PortfolioStatistics and Optimization Notebook 
"""
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pandas.tseries.offsets import MonthEnd
import collections


from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
from datetime import timedelta
from datetime import date

# Find Common inception date
def find_inception(df, asset_list):
    """"
    Finds common inception date from list of assets using the price dataframe df
    """
    date_list = [ df[df[a].notnull()].index[0] for a in asset_list]
    max_date = max(date_list)
    
    return max_date

def last_business_day(year_date):
    # year date is an automatically calculated day x years prior to last_date
    if year_date.weekday() == 5: # 5 corresponds to saturday
        #print("Changed ", str(year_date), " to ")
        year_date = year_date - relativedelta(days=1)
        #print(str(year_date))
    elif year_date.weekday() == 6: # 6 corresponds to sunday
        #print("Changed ", str(year_date), " to ")
        year_date = year_date - relativedelta(days=2)
        #print(str(year_date))
        
    return year_date


def last_biz_day_of_month(input_date):
    """   
    Function returns the last business day of the month
    Example: Typing in 3/30/2023 will return  3/31/2023
    
    Useful in performance calculations
    For example: say we want to find the last business day of 3 months
    prior to 6/30/2023, If 6/30/2023 is stored as a datetime variable end
    end - relativedelta(month=3) = 3/30/2023
    This is not end of the month
    
    Parameters:
    input_date (datetime):  datetime.datetime(2023, 6, 30, 0, 0)
    
    Returns:
        datetime
    
    
    """
    # Generate a date range for the entire month
    start_date = input_date.replace(day=1)
    end_date = input_date.replace(day=pd.Timestamp(input_date).days_in_month)
    date_range = pd.date_range(start=start_date, end=end_date)
    
    # Create a DataFrame from the date range
    df = pd.DataFrame({'Date': date_range})
    df['weekday'] = df['Date'].dt.weekday
    
    # filter out weekends
    df = df[(df['weekday'] != 5) &(df['weekday']!= 6)] # not equal to saturday or sunday
    last_biz_day_of_month = df['Date'].max()
    
    
    return last_biz_day_of_month


def create_portfolio(df, bgn, end, alloc_dict, initial_investment=100, rebalance_freq="annual", exp_ratio=0.00, exp_days=365):
    """
    Creates portfolio time series based on on portfolio weights over certain time frame 
    
    Parameters:
    
    df (dataframe):      Dataframe of initial prices
    bgn (datetime or string): Starting date to create portfolio "yyyy-mm-dd"
    end (datetime or string): Starting date to create portfolio "yyyy-mm-dd"
    alloc_dict (dict):        key value pair of asset:allocation example {'S&P 500':1}
    initial_investment (int): Beginning portfolio value. Default = 100
    rebalance_freq (string):  Portfolio rebalance frequency. Options: ['annual', 'semi-annual', 'quarterly' 'monthly']
    exp_ratio (float): Management fee to calculate net returns on indices. 0.79% expense ratio should be entered as 0.0079
    exp_days (int): Default should be 365 and should be changed to 252 if the index data does not include weekends or holidays
    
    Returns:
    dataframe: Portfolio values
    
    
    """    
    
    # unpack dictionary to use in portfolio creation
    asset_list = [k for k in alloc_dict.keys()]
    assets_val = [i+'_value' for i in asset_list]
    portfolio_wt = np.array([v for v in alloc_dict.values()])
    
    
    # filter dataframe to date rows and asset columns
    dat = df.loc[bgn:end, asset_list]
    
    # print time frame for created portfolio
    print("Portfolio data from", dat.index[0].strftime('%Y-%m-%d'), "to", dat.index[-1].strftime('%Y-%m-%d'))
    
    # create dataframe of daily returns
    ret = dat.pct_change()
    
    # create flag for rebalancing at beginning of year, we can use other datetimes and flags for other rebal frequency
    if rebalance_freq == "annual":
        ret['year'] = ret.index.year
        ret['rebalance'] = ret['year'].diff()
    elif rebalance_freq == "quarterly":
        ret['quarter'] = ret.index.quarter
        ret['rebalance'] = ret['quarter'].diff()
        ret['rebalance'] = np.where(['rebalance'] != 0, 1, 0)
    elif rebalance_freq == "semi-annual":
        ret['month'] = ret.index.month
        ret['semi'] = np.where(ret['month'] <=6, 1, 2)
        ret['rebalance'] = ret['semi'].diff()
        ret['rebalance'] = np.where(ret['rebalance'] != 0, 1, 0)
    elif rebalance_freq == "monthly":
        ret['month'] = ret.index.month
        ret['rebalance'] = ret['month'].diff()
        ret['rebalance'] = np.where(ret['rebalance'] != 0, 1, 0)
      
    # calculate portfolio return         CONSIDER UPDATING PORTFOLIO RETURN WITH FOR LOOP ON len(asset_list)
    prev_date = None
    for i, date in enumerate(ret.index):
        
         # logic for first row where there is no daily return, should equal target weight x initial investment
        if i == 0:
           
            for idx, val in enumerate(asset_list):
                 ret.loc[date, assets_val[idx]] = portfolio_wt[idx] * initial_investment
            
            ret.loc[date, 'port_value'] = ret.loc[date, assets_val].sum()
        else:
             # logic to rebalance portfolio and redistribute portfolio value to target weights
            if ret.loc[date, 'rebalance'] == 1:
                
                for idx, val in enumerate(asset_list):
                    ret.loc[date, assets_val[idx]] = ret.loc[prev_date, 'port_value'] * portfolio_wt[idx] * (ret.loc[date, asset_list[idx]] + 1) # need previous row port val

            else:
                # logic for all other dates
                for idx, val in enumerate(asset_list):
                    ret.loc[date, assets_val[idx]] = ret.loc[prev_date, assets_val[idx]] * (ret.loc[date, asset_list[idx]] + 1) # need previous row value
            
            ret.loc[date, 'port_value'] = ret.loc[date, assets_val].sum()
            
        # logic to allow us to use prior row    
        prev_date = date
        
    # return ret
    
    if exp_ratio == 0.00: # return gross of fee calculation if no value is input for expense ratio
        
        return ret[['port_value']]
    
    else: # expense ratio is entered we need to return a net of fee index
    
        net_portfolio = ret[['port_value']].copy()
        
        net_portfolio['daily_return'] = net_portfolio['port_value'].pct_change()  # calculate daily returns
        net_portfolio['net_daily_ret'] = net_portfolio['daily_return'] - (exp_ratio / exp_days)  # calculate daily returns less daily expense
        net_portfolio['net_port_value'] = (1 + net_portfolio['net_daily_ret']).cumprod() * net_portfolio['port_value'][0]  # calculate net port value directly

        net_portfolio = net_portfolio[['net_port_value']] # select only net_port_val column for dataframe
        net_portfolio = net_portfolio.rename({'net_port_value': 'port_value'}, axis=1)
        net_portfolio.iloc[0] = initial_investment
        
        return net_portfolio


def create_portfolio_custom(df, bgn, end, rebal_dict, initial_investment=1000, rebalance_freq="annual", exp_ratio=0.00, exp_days=252):
    """
    Creates portfolio time series based on on portfolio weights over certain time frame,
    
    This funcation allows the user to also import a custom
    
    Parameters:
    
    df (dataframe):      Dataframe of initial prices
    bgn (datetime or string): Starting date to create portfolio "yyyy-mm-dd"
    end (datetime or string): Starting date to create portfolio "yyyy-mm-dd"
    rebal_dict (dict):        key value pair of date: {asset:allocation} example {'2023-04-03': {'APRJ': 0.4,
                                                                                                  'APRQ': 0.2,
                                                                                                  'SPAB': 0.15,
                                                                                                  'BILS': 0.1,
                                                                                                  'BIL': 0.1,
                                                                                                  'SPIP': 0.05},
                                                                                 '2023-07-03': {'APRJ': 0.2,
                                                                                                  'JULJ': 0.2,
                                                                                                  'SPAB': 0.15,
                                                                                                  'APRQ': 0.1,
                                                                                                  'JULQ': 0.1,
                                                                                                  'BILS': 0.1,
                                                                                                  'BIL': 0.1,
                                                                                                  'SPIP': 0.05}}
    initial_investment (int): Beginning portfolio value. Default = 1000
    rebalance_freq (string):  Portfolio rebalance frequency. Options: ['annual', 'semi-annual', 'quarterly' 'monthly']
    exp_ratio (float): Management fee to calculate net returns on indices. 0.79% expense ratio should be entered as 0.0079
    exp_days (int): Default should be 365 and should be changed to 252 if the index data does not include weekends or holidays
    
    Returns:
    dataframe: Portfolio values
    
    
    """  
    # bring in rebalance informatin for adding new securities
    rebal_dates = [k for k in rebal_dict.keys()]
    ones = len(rebal_dates) * [1]
    rebal_df = pd.DataFrame({'date':rebal_dates, 'ones':ones})
    rebal_df.index = pd.to_datetime(rebal_df['date'])
    rebal_df = rebal_df.loc[:,['ones']]

    # initial assets and values
    asset_list = [a for a in rebal_dict[bgn.strftime('%Y-%m-%d')].keys()]
    assets_val = [i+'_value' for i in asset_list]
    portfolio_wt = [v for v in rebal_dict[bgn.strftime('%Y-%m-%d')].values()]

    # filter dataframe to date rows and asset columns
    dat = df.loc[bgn:end, asset_list]
    ret = dat.pct_change()

    # create flag for rebalancing at beginning of year, we can use other datetimes and flags for other rebal frequency
    if rebalance_freq == "annual":
        ret['year'] = ret.index.year
        ret['reb'] = ret['year'].diff()
    elif rebalance_freq == "quarterly":
        ret['quarter'] = ret.index.quarter
        ret['reb'] = ret['quarter'].diff()
        ret['reb'] = np.where(['reb'] != 0, 1, 0)
    elif rebalance_freq == "semi-annual":
        ret['month'] = ret.index.month
        ret['semi'] = np.where(ret['month'] <=6, 1, 2)
        ret['reb'] = ret['semi'].diff()
        ret['reb'] = np.where(ret['reb'] != 0, 1, 0)
    elif rebalance_freq == "monthly":
        ret['month'] = ret.index.month
        ret['reb'] = ret['month'].diff()
        ret['reb'] = np.where(ret['reb'] != 0, 1, 0)

    ret = ret.merge(rebal_df, how='left', left_index=True, right_index=True)
    ret['rebalance'] = np.where((ret['reb']==1) | (ret['ones']==1),1,0)
    ret = ret.drop(columns=['reb', 'ones'])

    prev_date = None
    for i, date in enumerate(ret.index):
         # logic for first row where there is no daily return, should equal target weight x initial investment
            if i == 0:
                for idx, val in enumerate(asset_list):
                     ret.loc[date, assets_val[idx]] = portfolio_wt[idx] * initial_investment

                ret.loc[date, 'port_value'] = ret.loc[date, assets_val].sum()
            else:
                 # logic to rebalance portfolio and redistribute portfolio value to target weights
                if ret.loc[date, 'rebalance'] == 1:

                    ### LOGIC TO UPDATE THE WEIGHTS USED
                    asset_list = [a for a in rebal_dict[date.strftime('%Y-%m-%d')].keys()]
                    assets_val = [i+'_value' for i in asset_list]
                    portfolio_wt = [v for v in rebal_dict[date.strftime('%Y-%m-%d')].values()]
#                     print("New Variables assigned as of: ", date)
#                     print(asset_list)
#                     print(portfolio_wt)

                    dat = df.loc[bgn:end, asset_list]
                    updated_ret = dat.pct_change()

                    # convert columns into list so we can create a new list of columns to add where
                    # we add any columns not in updated_ret_cols is added to new ret dataframe
                    ret_cols = list(ret.columns)
                    updated_ret_cols = list(updated_ret.columns)

                    # add new columns if there are new positions based on rebal dae
                    cols_to_add = [c for c in updated_ret_cols if c not in ret_cols]
                    updated_ret = updated_ret.loc[:,cols_to_add]

                    if len(cols_to_add) > 0:
                        # merge new columns to original ret dataframe if there are new columns to add
                        ret= ret.merge(updated_ret, how="left", left_index=True, right_index=True)

                    for idx, val in enumerate(asset_list):
                        ret.loc[date, assets_val[idx]] = ret.loc[prev_date, 'port_value'] * portfolio_wt[idx] * (ret.loc[date, asset_list[idx]] + 1) # need previous row port val
                else:
                    # logic for all other dates
                    for idx, val in enumerate(asset_list):
                        ret.loc[date, assets_val[idx]] = ret.loc[prev_date, assets_val[idx]] * (ret.loc[date, asset_list[idx]] + 1) # need previous row value

                ret.loc[date, 'port_value'] = ret.loc[date, assets_val].sum()

            # logic to allow us to use prior row    
            prev_date = date
            
    # create column to show the day of the week
    ret['weekday'] = ret.index.weekday
    # filter out saturdays and sundays
    ret = ret.loc[(ret['weekday']!=5) & (ret['weekday']!=6)] # weekday integers 5 and 6 are saturday and sunday
    ret = ret.drop(columns=['weekday'])
    
    # return ret
    if exp_ratio == 0.00: # return gross of fee calculation if no value is input for expense ratio
        
        return ret[['port_value']]
        #return ret            # Used to show all the underlying holdings and weights for portfolio turnover calc
    
    else: # expense ratio is entered we need to return a net of fee index
    
        net_portfolio = ret[['port_value']].copy()
        
        net_portfolio['daily_return'] = net_portfolio['port_value'].pct_change()  # calculate daily returns
        net_portfolio['net_daily_ret'] = net_portfolio['daily_return'] - (exp_ratio / exp_days)  # calculate daily returns less daily expense
        net_portfolio['net_port_value'] = (1 + net_portfolio['net_daily_ret']).cumprod() * net_portfolio['port_value'][0]  # calculate net port value directly

        net_portfolio = net_portfolio[['net_port_value']] # select only net_port_val column for dataframe
        net_portfolio = net_portfolio.rename({'net_port_value': 'port_value'}, axis=1)
        net_portfolio.iloc[0] = initial_investment
        
        return net_portfolio
    
    
# Calculate annualized return
def annualized_return(portfolio, all_days=True):
    """
    all_days flag determines annualization
    -analysis done with 365 days like from our database should use 365 for annualizing
    -when calculating model portfolios, the nav and market price data doesn't contain weekends
    therefore use 252 days
    """    
    if all_days:
        # calculate return metric
        period_return = (portfolio.iloc[-1]['port_value'] / portfolio.iloc[0]['port_value'] - 1) # for whole time period
        day_cnt = (portfolio.index[-1] - portfolio.index[0]).days
        total_return = ((period_return+1) ** (365.25 / day_cnt )-1) # annualized using full year average 365.25 days per year
    else:
        # calculate return metric
        period_return = (portfolio.iloc[-1]['port_value'] / portfolio.iloc[0]['port_value'] - 1) # for whole time period
        day_cnt = portfolio.shape[0]
        total_return = ((period_return+1) ** (252 / day_cnt )-1) # annualized using full year average 252 days per year
        
    return total_return

# Calculate annualized return
def annualized_return_monthly(portfolio):
    """
    used for monthly return series
    """    

    # calculate return metric
    period_return = (portfolio.iloc[-1]['port_value'] / portfolio.iloc[0]['port_value'] - 1) # for whole time period
    mo_cnt = portfolio.shape[0]
    total_return = ((period_return+1) ** (12 / mo_cnt )-1) # annualized using full year average 12 months
        
    return total_return

    # ret_df = portfolio.pct_change().dropna() # calc daily returns drop first day that is null
    # ret_df = ret_df.loc[ret_df['port_value'] != 0] # drop weekend days to avoid disrupting annualize calc
    # ret_df['one_plus'] = ret_df['port_value'] + 1
    # product_df = ret_df.product()
    # one_plus = product_df['one_plus']
    # period_return = one_plus - 1
    # annualized_ret = one_plus ** (252/ret_df.shape[0]) - 1
    
    # return annualized_ret

def period_return(portfolio):
    period_return = (portfolio.iloc[-1]['port_value'] / portfolio.iloc[0]['port_value'] - 1) # for whole time period
    
    return period_return

def annualized_volatility(portfolio):
    # calculate annualized volatility metrics
    calc_df = portfolio.pct_change().dropna()
    # remove all days with daily return of zero, weekends and holidays
    calc_df = calc_df.loc[calc_df['port_value']!=0]
    # ddof = 1 for sample standard deviation
    daily_stdev = np.std(calc_df['port_value'], ddof=1) 
    ann_std = daily_stdev * (252**0.5) # 252^(0.5) annualizes the daily std
    
    return ann_std, daily_stdev

def downside_deviation(portfolio, threshold=0):
    """
    -converts portfolio prices to stream of daily returns
    -filters to only returns below threshold(default is zero)
    -Calculates annualized downside deviation

    
    Params:
    portfolio (pandas dataframe or series):          df of prices

    return annualized downside deviation of portfolio
    """
    calc_df = portfolio.pct_change().dropna()
    calc_df = calc_df.loc[calc_df['port_value']<threshold]
    daily_stdev = np.std(calc_df['port_value'],ddof=1)
    annualized_down_dev = daily_stdev * (252**.5)
    return annualized_down_dev
    

def annualized_volatility_monthly(portfolio):
    # convert dataframe to monthly
    stock_monthly = portfolio.dropna()
    stock_monthly['eomonth'] = stock_monthly.index + MonthEnd(0) # add column with month end dates
    stock_monthly = stock_monthly[(stock_monthly.index == stock_monthly['eomonth'])] # filter to only month end days
   
    # drop eomonth column, calc monthly ret, drop nulls
    calc_df = stock_monthly.drop(['eomonth'],axis=1).pct_change().dropna()  
    
    # ddof = 1 for sample standard deviation
    monthly_stdev = np.std(calc_df['port_value'], ddof=1) 
    ann_std = monthly_stdev * (12**0.5) # 12^(0.5) annualizes the monthly std
    
    return ann_std, monthly_stdev

    
def calc_sharpe(df, portfolio, risk_free= 'US Treasury Bills Index'):
    # calculate sharpe ratio
    calc_df = portfolio.pct_change().dropna()
    rf_df = df[[risk_free]].pct_change().dropna()
    
    excess_df = calc_df.merge(rf_df, left_index=True, right_index=True, how='left')
    excess_df['excess_return'] = excess_df['port_value'] - excess_df[risk_free]
    
    excess_df['cum_ret'] = (1 + excess_df['excess_return']).cumprod()-1
    
    ann_excess_return = (excess_df['cum_ret'][-1] + 1)**(365.25/len(portfolio))-1 # annualized excess returns
    # ann_volatility = np.std(excess_df['excess_return'], ddof=1) * (252**0.5) # standard dev of excess returns
    ann_volatility = annualized_volatility(portfolio)[0] # standard dev of excess returns
    
    sharpe = ann_excess_return / ann_volatility
    if ann_excess_return >= 0:
        return sharpe
    else:
        return "-"   

def sortino_ratio(df, portfolio, threshold=0, risk_free= 'US Treasury Bills Index' ):
    
    asset_return = annualized_return(portfolio)
    
    start = portfolio.index[0]
    finish = portfolio.index[-1]
    
    risk_free_port = create_portfolio(df, start, finish, {risk_free:1})
    rf_return = annualized_return(risk_free_port)
    
    down_dev = downside_deviation(portfolio)
    
    sortino = (asset_return - rf_return) / down_dev
    
    return sortino
    
    
def calc_return_risk_ratio(portfolio):
    
    return annualized_return(portfolio) / annualized_volatility(portfolio)[0]

def calc_beta(df, portfolio, bgn, end, benchmark='S&P 500'):
    # calculate beta
    benchmark_df = df.loc[bgn:end, [benchmark]]
    benchmark_df.columns = ['port_value'] # change column name to port value so it can be used in other functions
    benchmark_daily = benchmark_df.pct_change().dropna()
    
    calc_df = portfolio.pct_change().dropna()
    
    # join return streams and remove rows where daily returns were both zero
    # these are likely weekend and holidays that we don't want distorting the calculation
    joined = calc_df.merge(benchmark_daily, left_index=True, right_index=True, how='left')
    joined['zero'] = joined['port_value_x'] + joined['port_value_y']
    joined = joined[joined['zero']!=0]
    
    
    # beta = covariance(a, b) / variance(b)  
    beta = (np.cov(joined['port_value_x'], joined['port_value_y'] )[0,1] ) / (annualized_volatility(benchmark_df)[1]**2)
    return beta


def calc_max_drawdown(portfolio):
    # calculate max drawdown of asset over time period
    calc_df = portfolio.copy()
    calc_df['daily_ret'] = calc_df['port_value'].pct_change()
    
    # count number of rows in dataframe
    last_row = len(calc_df)
    
    # initialize dictionary
    cum_ret_dict = {}
    
    # iterate through all possible combinations of days and store cumulative returns in dictionary
    for i in range(last_row):
        temp_df = calc_df.copy().iloc[i:]
        temp_df['cum_ret'] = (1 + temp_df['daily_ret']).cumprod()-1
        cum_ret_dict[i] = temp_df, temp_df['cum_ret'].min(), temp_df['cum_ret'].idxmin() # store df, min cumulative return, date over period as tuple
        
    # Create list of max drawdowns
    min_rets = [cum_ret_dict[i][1] for i in range(0,last_row)]
    start_date = [cum_ret_dict[i][0].index[0] for i in range(0, last_row)]
    min_date = [cum_ret_dict[i][2] for i in range(0,last_row)]
    
    # Create Dataframe for analysis
    ret_df = pd.DataFrame(data = {"Start date":start_date,
                                  "Minimum date":min_date,
                                  "Max Drawdown":min_rets})
    
    # by using the cumulative return, the peak date appears one day later, adjust this by subtracting a day and
    # applying custom last business day function
    ret_df['Start date'] = ret_df['Start date'] - timedelta(days=1)
    ret_df['Start date'] = ret_df['Start date'].apply(last_business_day)
    
    max_drawdown = ret_df.loc[np.argmin(ret_df['Max Drawdown'])]
    
    return (dt.strftime(max_drawdown['Start date'], "%Y-%m-%d" ),
            dt.strftime(max_drawdown['Minimum date'], "%Y-%m-%d" ),
            max_drawdown['Max Drawdown'])

def check_weights(weights, min_alloc, max_alloc, counter=0): 
    """
    -Checks that the weights in a portfolio sum to 1 
    -Checks that the weights fall between contraints of min_alloc and max_alloc
      - If weights fall outside constraints, reallocate

    
    Params:
    weights (list):          iniital list of weights as decimal
    min_alloc (List):       Assets to include in portfolio. Needs to appear in df.columns
    max_alloc (list)):  Starting date to calculate statistics  for optimization

    return updated list of weights
    """
    length = len(weights)

    # base case: We want all weight to be between their constraints and sum to 1
    if (all(min_alloc[i] <= weights[i] <= max_alloc[i] for i in range(length)) and 0.99 <= abs(sum(weights)) <= 1.001):
        #print("Weights fit constraints:", weights)
        return weights

     # otherwise, find the index of the first weight that is outside the constraints
    try:
        i = next(i for i in range(length) if weights[i] < min_alloc[i] or weights[i] > max_alloc[i])
        #print("i: doesn't fit constraints", i)
    except StopIteration:
        # all weights are within constraints so return weights
        #print("Weights fit constraints:", weights)
        return weights

    # If the weight is below the minimum allocation, set it to the minimum allocation
    if weights[i] < min_alloc[i]:
        #print("i is below min alloc ", i, weights[i])
        diff = weights[i] - min_alloc[i]
        weights[i] = min_alloc[i]

    # otherwise, the weight is above the maximum allocation, so set it to the maximum
    else:
        #print("i is above max alloc ", i, weights[i])
        diff = weights[i] - max_alloc[i]
        weights[i] = max_alloc[i]

    #print("weights: ", weights)
    #print("weights sum: ", sum(weights))
    #print("Reallocate ", diff, " to other assets")

    valid_indices = [i for i in range(len(weights)) if weights[i] < max_alloc[i]]

    # Shuffle the valid indices list randomly
    random.shuffle(valid_indices)

    # Distribute the diff randomly to the valid indices until diff is exhausted
    for i in valid_indices:
        if diff > 0:
            add_amount = min(max_alloc[i], diff)
            amount_to_add = random.uniform(0, add_amount)
            weights[i] += amount_to_add
            diff -= amount_to_add
        else:
            break


    # add any allocatin left in diff back to the last index value in weights
    rand_idx = int(random.uniform(0, length-.1)) # make sure it's only at index 0- (1-lenght)
    #print("rand_idx: ",rand_idx)
    weights[rand_idx] = weights[rand_idx] + diff  
    
    # add 1 to counter
    counter+=1
    
    if counter >= 5:
        #print("counter timeout")
        return [round(w, 4) for w in weights]
    else:
        return check_weights(weights, min_alloc, max_alloc, counter)





def simulate_portfolio(df, asset_list, bgn, end, iterations=5000, alloc_min=None, alloc_max=None, target_return=None, target_volatility=None):
    """
    -Simulates portfolio allocations and creates dataframe of corresponding risk, return, and sharpe ratios
    -Plots efficient frontier.
    
    Params:
    df (dataframe):          Initial database of asset prices
    asset_list (List):       Assets to include in portfolio. Needs to appear in df.columns
    bgn (datetime object):  Starting date to calculate statistics  for optimization
    end (datetime object):  End date to calculate portfolio statistics for optimization
    iterations (int):        Integer number to create many possible portfolios
    alloc_min (list):      Enter list of minimum required weights for allocation in portfolio
                           if asset has no minimum, make = 0 in list example [.4, 0, .3, 0]
    alloc_max (list):      Enter list of minimum required weights for allocation in portfolio
                           if asset has no maximum, make = 1 in list example [.5, 1, .5, 1]
    target_return (float):   float with three decimal places 0.000 
    target_volatility (float):   float with three decimal places 0.000 
    
    """
    print('Portfolio Assets:', asset_list, '\n')
    
    # filter out assets if alloc_max value is zero
    if (alloc_max != None) and (0 in alloc_max):
        alloc_dict = {}
        constraints = list(zip(alloc_min, alloc_max))

        for idx, val in enumerate(asset_list):
            alloc_dict[val] = constraints[idx]

        new_a_list = []
        new_alloc_min = []
        new_alloc_max = []

        for key, val in alloc_dict.items():
            if val[1] != 0:
                new_a_list.append(key)
                new_alloc_min.append(val[0])
                new_alloc_max.append(val[1])

        asset_list = new_a_list
        alloc_min = new_alloc_min
        alloc_max = new_alloc_max
    
    if alloc_max == None:
        alloc_max = [1]*len(asset_list)
    if alloc_min == None:
        alloc_min = [0]*len(asset_list)
            
    price_df = df.loc[bgn:end, asset_list]

    daily_ret = price_df.pct_change()
    daily_ret.iloc[0, :] = 0

    period_return = (price_df.iloc[-1] / price_df.iloc[0] - 1) # for whole time period
    day_cnt = (price_df.index[-1] - price_df.index[0]).days
    total_return = ((period_return+1) ** (365.25 / day_cnt )-1) # annualized using full year average 365.25 days per year

    port_returns = []
    port_vols = []
    port_weights = []
    port_ret_risk = []

    # go through all iterations creating random weights between constraints, calculate return, risk and ratio
    for i in range (iterations):
        weights = []
        threshold = 1
        for mn, mx in zip(alloc_min, alloc_max):
            thresh = min(mx, threshold) # make the threshold minimum value of maximum constraint or portfolio allocation remaining
            rnd_weight = random.uniform(mn, thresh) # pick random number based on constraints and portfolio allocation remaining
            weights.append(rnd_weight)  # add random weight to list of weights
            threshold = 1 - sum(weights) # recalculate threshold for future random weight generation
            # if threshold < 0:
            #     print("To stringent of constraints, please adjust:")
        weights[-1] = weights[-1] + threshold # reassign last weight to make sure total allocation sums to 1
        weights = check_weights(weights, alloc_min, alloc_max) # function works to reallocate weights if they fall outside constraints
        weights = np.array(weights)    
        p_return = np.sum(total_return * weights)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(daily_ret.cov() * 252, weights)))
        ret_risk_ratio = p_return / p_vol
        
        # hold values for certain allocation in various lists
        port_returns.append(p_return)
        port_vols.append(p_vol)
        port_weights.append(weights)
        port_ret_risk.append(ret_risk_ratio)

        
    # Convert lists to arrays for plotting
    plot_returns = np.array(port_returns)
    plot_vols = np.array(port_vols)    
    
    # round metrics
    port_weights = [[np.round(float(i), 3) for i in nested] for nested in port_weights]
    port_returns = [np.round(float(i), 4) for i in port_returns]
    port_vols = [np.round(float(i), 4) for i in port_vols]
    port_ret_risk = [np.round(float(i), 4) for i in port_ret_risk]
    
    # consolidate all portfolios in dataframe
    sim_summary = pd.DataFrame({'Portfolio Return':port_returns,
                               'Portfolio Volatility':port_vols,
                               'Unadjusted Sharpe Ratio': port_ret_risk,
                               'Portfolio Allocation':port_weights})
    

    # create separate columns for asset allocations
    for i in range(iterations):
        for idx, a in enumerate(asset_list):
            sim_summary.loc[i,a] = sim_summary.loc[i, 'Portfolio Allocation'][idx]
            
      
    # NOTE: THIS SECTION FILTERS OUT PORTFOLIOS WHEN AN ASSET FALLS OUTSIDE OF THE CONSTRAINTS
    # IN INSTANCES WHERE THERE ARE MANY TICKERS, THIS CAN LEAD TO FILTERING OUT ALL PORTFOLIOS    
    # CAN I UPDATE THE RANDOM WEIGHT PORTION TO ALWAYS FIT THE CONSTRAINTS
     
    # filter summary dataframe to remove rows that don't match constraints
    # if alloc_min != None:
    #     for am, a in zip(alloc_min, asset_list):
    #         sim_summary = sim_summary[sim_summary[a] >= am]
            
    # if alloc_max != None:
    #     for am, a in zip(alloc_max, asset_list):
    #         sim_summary = sim_summary[sim_summary[a] <= am]
            
    print("Portfolio Minimum Allocations: ", alloc_min)
    print("Portfolio Minimum Allocations: ", alloc_max, '\n')
    
    # organize portfolio options by highest return than best sharpe
    sim_summary = sim_summary.sort_values(by=['Portfolio Return', 'Unadjusted Sharpe Ratio'], ascending=[False, False])
  
    
    print("Max Sharpe Portfolio ")
    print((sim_summary.loc[sim_summary['Unadjusted Sharpe Ratio'] == sim_summary['Unadjusted Sharpe Ratio'].max()]).iloc[0], '\n')
    
    print("Max Return Portfolio ")
    print((sim_summary[sim_summary['Portfolio Return'] == sim_summary['Portfolio Return'].max()]).iloc[0], '\n')
    
    
    print("Minimum Volatility Portfolio ")
    print((sim_summary[sim_summary['Portfolio Volatility'] == sim_summary['Portfolio Volatility'].min()]).iloc[0], '\n')
    
    if target_return != None:
        print("Target Return of ", target_return, " Portfolio")
        try:
            print((sim_summary[sim_summary['Portfolio Return'] >= target_return]).iloc[-1], '\n')
        except:
            print("No portfolios found", "\n")
    
    if target_volatility != None:
        print("Target Volatility of ", target_volatility, " Portfolio")
        try:
            print((sim_summary[sim_summary['Portfolio Volatility'] <= target_volatility]).iloc[0], '\n')
        except:
            print("No portfolios found", "\n")


    # Convert lists to arrays for plotting
    port_returns = np.array(port_returns)
    port_vols = np.array(port_vols)

    # Plot the distribution of portfolio returns and volatilities 
    plt.figure(figsize = (18,10))
    plt.scatter(plot_vols,plot_returns,c = (plot_returns / plot_vols), marker='o')
    plt.xlabel('Portfolio Volatility')
    plt.ylabel('Portfolio Return')
    print()
    plt.colorbar(label = 'Sharpe ratio (not adjusted for risk-free rate)')
    plt.show();
    
    # drop inefficient portfolios (for example, portfolios with the same return and higher risk)
    sim_summary = sim_summary.drop_duplicates(subset='Portfolio Return', keep='first')
    
    # drop inefficient portfolios (for example, portfolios with the same risk and lower return)
    # decided against removing the lower returns with same volatility due to rounding
    # I wanted to have allocation options at each return level.
    # sim_summary = sim_summary.drop_duplicates(subset='Portfolio Volatility', keep='first')
    
    return sim_summary 

# DELETE VERSION OF simulate_portfolio below after confirming version above works

# def simulate_portfolio(df, asset_list, bgn, end, iterations=5000, alloc_min=None, alloc_max=None, target_return=None, target_volatility=None):
#     """
#     -Simulates portfolio allocations and creates dataframe of corresponding risk, return, and sharpe ratios
#     -Plots efficient frontier.
    
#     Params:
#     df (dataframe):          Initial database of asset prices
#     asset_list (List):       Assets to include in portfolio. Needs to appear in df.columns
#     bgn (datetime object):  Starting date to calculate statistics  for optimization
#     end (datetime object):  End date to calculate portfolio statistics for optimization
#     iterations (int):        Integer number to create many possible portfolios
#     alloc_min (list):      Enter list of minimum required weights for allocation in portfolio
#                            if asset has no minimum, make = 0 in list example [.4, 0, .3, 0]
#     alloc_max (list):      Enter list of minimum required weights for allocation in portfolio
#                            if asset has no maximum, make = 1 in list example [.5, 1, .5, 1]
#     target_return (float):   float with three decimal places 0.000 
#     target_volatility (float):   float with three decimal places 0.000 
    
#     """
#     print('Portfolio Assets:', asset_list, '\n')
    
#     # filter out assets if alloc_max value is zero
#     if (alloc_max != None) and (0 in alloc_max):
#         alloc_dict = {}
#         constraints = list(zip(alloc_min, alloc_max))

#         for idx, val in enumerate(asset_list):
#             alloc_dict[val] = constraints[idx]

#         new_a_list = []
#         new_alloc_min = []
#         new_alloc_max = []

#         for key, val in alloc_dict.items():
#             if val[1] != 0:
#                 new_a_list.append(key)
#                 new_alloc_min.append(val[0])
#                 new_alloc_max.append(val[1])

#         asset_list = new_a_list
#         alloc_min = new_alloc_min
#         alloc_max = new_alloc_max
        
       
#     num_assets = len(asset_list)
    
#     price_df = df.loc[bgn:end, asset_list]

#     daily_ret = price_df.pct_change()
#     daily_ret.iloc[0, :] = 0

#     period_return = (price_df.iloc[-1] / price_df.iloc[0] - 1) # for whole time period
#     day_cnt = (price_df.index[-1] - price_df.index[0]).days
#     total_return = ((period_return+1) ** (365.25 / day_cnt )-1) # annualized using full year average 365.25 days per year

#     port_returns = []
#     port_vols = []
#     port_weights = []
#     port_ret_risk = []

#     for i in range (iterations):
#         weights = np.random.dirichlet(np.ones(num_assets),size=1)
#         weights = weights[0]
#         p_return = np.sum(total_return * weights)
#         p_vol = np.sqrt(np.dot(weights.T, np.dot(daily_ret.cov() * 252, weights)))
#         ret_risk_ratio = p_return / p_vol

#         port_returns.append(p_return)
#         port_vols.append(p_vol)
#         port_weights.append(weights)
#         port_ret_risk.append(ret_risk_ratio)
        
#     # Convert lists to arrays for plotting
#     plot_returns = np.array(port_returns)
#     plot_vols = np.array(port_vols)    
    
#     # round metrics
#     port_weights = [[np.round(float(i), 3) for i in nested] for nested in port_weights]
#     port_returns = [np.round(float(i), 3) for i in port_returns]
#     port_vols = [np.round(float(i), 3) for i in port_vols]
#     port_ret_risk = [np.round(float(i), 3) for i in port_ret_risk]
    
#     # consolidate all portfolios in dataframe
#     sim_summary = pd.DataFrame({'Portfolio Return':port_returns,
#                                'Portfolio Volatility':port_vols,
#                                'Unadjusted Sharpe Ratio': port_ret_risk,
#                                'Portfolio Allocation':port_weights})

#     # create separate columns for asset allocations
#     for i in range(iterations):
#         for idx, a in enumerate(asset_list):
#             sim_summary.loc[i,a] = sim_summary.loc[i, 'Portfolio Allocation'][idx]
            
                  
#     # filter summary dataframe to remove rows that don't match constraints
#     if alloc_min != None:
#         for am, a in zip(alloc_min, asset_list):
#             sim_summary = sim_summary[sim_summary[a] >= am]
            
#     if alloc_max != None:
#         for am, a in zip(alloc_max, asset_list):
#             sim_summary = sim_summary[sim_summary[a] <= am]
            
#     print("Portfolio Minimum Allocations: ", alloc_min)
#     print("Portfolio Minimum Allocations: ", alloc_max, '\n')
    
#     # organize portfolio options by highest return than best sharpe
#     sim_summary = sim_summary.sort_values(by=['Portfolio Return', 'Unadjusted Sharpe Ratio'], ascending=[False, False])
  
    
#     print("Max Sharpe Portfolio ")
#     print((sim_summary.loc[sim_summary['Unadjusted Sharpe Ratio'] == sim_summary['Unadjusted Sharpe Ratio'].max()]).iloc[0], '\n')
    
#     print("Max Return Portfolio ")
#     print((sim_summary[sim_summary['Portfolio Return'] == sim_summary['Portfolio Return'].max()]).iloc[0], '\n')
    
    
#     print("Minimum Volatility Portfolio ")
#     print((sim_summary[sim_summary['Portfolio Volatility'] == sim_summary['Portfolio Volatility'].min()]).iloc[0], '\n')
    
#     if target_return != None:
#         print("Target Return of ", target_return, " Portfolio")
#         try:
#             print((sim_summary[sim_summary['Portfolio Return'] >= target_return]).iloc[-1], '\n')
#         except:
#             print("No portfolios found", "\n")
    
#     if target_volatility != None:
#         print("Target Volatility of ", target_volatility, " Portfolio")
#         try:
#             print((sim_summary[sim_summary['Portfolio Volatility'] <= target_volatility]).iloc[0], '\n')
#         except:
#             print("No portfolios found", "\n")


#     # Convert lists to arrays for plotting
#     port_returns = np.array(port_returns)
#     port_vols = np.array(port_vols)

#     # Plot the distribution of portfolio returns and volatilities 
#     plt.figure(figsize = (18,10))
#     plt.scatter(plot_vols,plot_returns,c = (plot_returns / plot_vols), marker='o')
#     plt.xlabel('Portfolio Volatility')
#     plt.ylabel('Portfolio Return')
#     print()
#     plt.colorbar(label = 'Sharpe ratio (not adjusted for risk-free rate)')
#     plt.show();
    
#     # drop inefficient portfolios (for example, portfolios with the same return and higher risk)
#     sim_summary = sim_summary.drop_duplicates(subset='Portfolio Return', keep='first')
    
#     # drop inefficient portfolios (for example, portfolios with the same risk and lower return)
#     # decided against removing the lower returns with same volatility due to rounding
#     # I wanted to have allocation options at each return level.
#     # sim_summary = sim_summary.drop_duplicates(subset='Portfolio Volatility', keep='first')
    
#     return sim_summary 
        
