import streamlit as st

# Set the page layout to wide
st.set_page_config(layout="wide")

import pandas as pd
import numpy as np
import pickle
import urllib.request
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
from datetime import timedelta
from datetime import date

from scipy import stats
from scipy.stats import norm

import plotly.graph_objects as go
import plotly.figure_factory as ff

# ---------------------------------------------------------------------------------------------
# import python functions

import requests
import sys

# GitHub URL for the .py file
py_file_url = 'https://raw.githubusercontent.com/tjboshea/planning-calculator/main/data/innovator_functions.py'

try:
    # Get the raw content of the .py file from GitHub
    response = requests.get(py_file_url)
    response.raise_for_status()  # Raise an exception for HTTP errors

    # Read the content of the .py file into a string
    py_code = response.text

    # Execute the Python code in the string
    exec(py_code)

except Exception as e:
    print("Error:", e)
    sys.exit(1)
# ---------------------------------------------------------------------------------------------------------------
# read in factors

@st.cache_data 
def get_factor_df():
    factor_url = 'https://raw.githubusercontent.com/tjboshea/planning-calculator/main/data/ScenarioFactors.csv'
    factor_df = pd.read_csv(factor_url)
    factor_df = factor_df.loc[~factor_df['date'].isnull()]
    factor_df['date'] = pd.to_datetime(factor_df['date'])
    factor_df = factor_df.loc[factor_df['date'] > "1949-12-29"]
    factor_df = factor_df.set_index('date')

    factors = ['us_credit', 'us_term', 'us_market', 'global_market',
       'global_ex_us_market', 'europe_market', 'north_america_market',
       'pacific_market', 'us_SMB', 'global_SMB', 'global_ex_us_SMB',
       'europe_SMB', 'north_america_SMB', 'pacific_SMB', 'us_HML',
       'global_HML', 'global_ex_us_HML', 'europe_HML', 'north_america_HML',
       'pacific_HML', 'us_UMD', 'global_UMD', 'global_ex_us_UMD', 'europe_UMD',
       'north_america_UMD', 'pacific_UMD', 'us_QMJ', 'global_QMJ',
       'global_ex_us_QMJ', 'north_america_QMJ',
       'Commodity', 'EM']
    
    factor_df = factor_df.loc[:, factors]

    return factor_df

factdb = get_factor_df()

# ------------------------------------------------------------------------------------------------------
# Read in Elastic Net Coefficients

import urllib.request

@st.cache_data
def get_en_coefficients():
    en_url = 'https://raw.githubusercontent.com/tjboshea/planning-calculator/main/data/en_coefficients.pkl'

    # Fetch the pickle file content from the URL
    response = urllib.request.urlopen(en_url)
    
    # Load the pickle file content
    en_results = pickle.load(response)
    
    return en_results

# Call the function to load the pickle data
en_co = get_en_coefficients()


# ------------------------------------------------------------------------------------------------------
# Read in index 

@st.cache_data
def get_index_df():
    idx_url = 'https://raw.githubusercontent.com/tjboshea/planning-calculator/main/data/index_db.csv'

    # Read the CSV file into a DataFrame
    data = pd.read_csv(idx_url)
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')
    return data

df = get_index_df()


# Initialize session state
if 'exp_df' not in st.session_state:
    st.session_state['exp_df'] = pd.DataFrame(columns=['Amount', 'Months Ahead'])
if 'alloc_df' not in st.session_state:
    st.session_state['alloc_df'] = pd.DataFrame(columns=['Asset', 'Weight (%)'])
    
# st.write(st.session_state)  # ALLOWS USER TO VIEW SESSION STATE, COMMENT OUT WHEN DONE TESTING

# Set up the Streamlit app
st.title('Financial Goal Planner')

# --------------------------------------------------------------------------------------------------------
# Establish user inputs for asset allocation

# Establish user inputs
st.sidebar.title("Enter Asset Allocation:")

# dropdown for asset selection 
asset_options = list(df.columns)
selected_asset = st.sidebar.selectbox('Select Asset', asset_options)

# Number input for weight
weight = st.sidebar.number_input('Enter Weight (%)', min_value=0.0, max_value=100.0, step=0.01, value=100.0)

# Add row button
if st.sidebar.button('Add Asset'):
    new_row = pd.DataFrame([{'Asset': selected_asset, 'Weight (%)': weight}])
    st.session_state['alloc_df'] = pd.concat([st.session_state['alloc_df'], new_row])

 # Delete row button
if not st.session_state['alloc_df'].empty:
    delete_index = st.sidebar.selectbox('Delete Asset', st.session_state['alloc_df']['Asset'])
    if st.sidebar.button('Delete Asset'):
         st.session_state['alloc_df'] = st.session_state['alloc_df'].loc[st.session_state['alloc_df']['Asset']!=delete_index]


total_weight = st.session_state['alloc_df']['Weight (%)'].sum()
if total_weight == 100:
    st.sidebar.write(f"Portfolio Weight = 100%")
else:
    st.sidebar.write(f'Portfolio weight must equal 100%. Remaining weight =  {100-total_weight}')

# display DataFrame
st.sidebar.write(st.session_state['alloc_df'].set_index('Asset'))

goal_dict = st.session_state['alloc_df'].set_index('Asset')['Weight (%)'].to_dict()

#----------------------------------------------------------------------------------------------
# Establish user inputs
st.sidebar.title("User inputs:")

# determine if retirement is in growth or withdrawal stage
stage = st.sidebar.selectbox('Planning Stage', ['Pre-retirement', 'Retirement'])

goal_years = st.sidebar.number_input('Goal Years', min_value=1, step=1, value=15)
goal = st.sidebar.number_input('Goal Amount', min_value=0, step=1, value=5000000)
current_savings = st.sidebar.number_input('Current Savings', min_value=0, step=1, value=1250000)
current_house_income = st.sidebar.number_input('Current House Income', min_value=0, step=1, value=100000)
income_growth = st.sidebar.number_input('Income Growth Rate (%, Annual)', min_value=0.0, step=0.25, value=3.0, max_value=100.0)
savings_rate = st.sidebar.slider('Savings Rate (%, Annual)', min_value=0.0, max_value=100.0, step=1.0, value=20.0)

if stage == "Retirement":
    withdrawal_rate = st.sidebar.slider('Withdrawal Rate (%, Annual)', min_value=0.0, max_value=100.0, step=1.0, value=1.0)
else:
    withdrawal_rate = 0

#--------------------------------------------------------------------------------------------------------------------------
# Windfall or Expense inputs
st.sidebar.header("One-off Expense or Windfall")

# dropdown for asset selection 
amount = st.sidebar.number_input('Amount', min_value=-100000000000, step=1, value=0)
months_ahead = st.sidebar.number_input('Months Ahead', min_value=1, value=24, max_value=goal_years*12)

# Add row button
if st.sidebar.button('Add Expense or Windfall'):
    new_row = pd.DataFrame([{'Amount': amount, 'Months Ahead': months_ahead}])
    st.session_state['exp_df'] = pd.concat([st.session_state['exp_df'] , new_row])

 # Delete row button
if not st.session_state['exp_df'].empty:
    delete_index = st.sidebar.selectbox('Delete row', st.session_state['exp_df']['Amount'])
    if st.sidebar.button('Delete row'):
        st.session_state['exp_df'] = st.session_state['exp_df'].loc[st.session_state['exp_df']['Amount']!=delete_index]

lqd_windfall_dict = st.session_state['exp_df'].set_index('Months Ahead')['Amount'].to_dict()

# Display DataFrame in the sidebar
st.sidebar.write(st.session_state.exp_df.set_index('Amount'))
# st.sidebar.write(lqd_windfall_dict)

#---------------------------------------------------------------------------------------------------------
# Function to run retirement simulations

def sim_retirement(stage='Pre-retirement', df=df,factdb=factdb, goal_dict=goal_dict, simulations=1000, goal_years=15,
                   goal=5000000, current_savings=0, current_house_income=0,
                  income_growth=0.03, savings_rate=0.0, withdrawal_rate=0.0, lqd_windfall_dict = {1:0}):
    """
    Function:
     - Simulates monthly market returns based on investor's chosen asset allocation for n simulations.
     - Calculates portfolio value in each simulation.
     - Determines percentage of simulations in which the goal is reached.

    Params:
    - stage (str): Stage of retirement ['pre-retirement', 'retirement']
    - df (DataFrame): DataFrame containing index return.
    - factdb (DataFrame): DataFrame containing factor data driving returns.
    - goal_dict (dict): Dictionary specifying asset allocation weights.
    - simulations (int): Number of simulations (default is 1000).
    - goal_years (int): Number of years for the retirement goal (default is 15).
    - goal (int): Retirement goal amount (default is 5000000). Pre
    - current_savings (float): Current savings amount (default is 0).
    - current_house_income (float): Current household income (default is 0).
    - income_growth (float): Annualized income growth rate (default is 0.03).
    - savings_rate (float): Savings rate as a percentage (default is 0.0).
    - withdrawal_rate (float): Withdrawal rate as a percentage (default is 0.0).
    - lqd_windfall_dict (dict): Dictionary of liquidity/windfall events (default is {1: 0}).

    Returns:
    - success (float): Success rate as a percentage.
    - stat_df_summary (DataFrame): Summary statistics DataFrame.
    - total_dollars (DataFrame): DataFrame containing total portfolio value.
    - stat_df (DataFrame): DataFrame containing portfolio statistics.
    - all_sims (DataFrame): DataFrame containing monthly returns for each simulation.
    - all_holdings (DataFrame): DataFrame containing historical statistics for holdings.
    - init_port_stats (DataFrame): DataFrame containing initial portfolio statistics.
    """
    # update parameters from streamlit to use in our function
    
    income_growth = income_growth / 100
    savings_rate = savings_rate / 100 / 12
    withdrawal_rate = withdrawal_rate / 100 / 12
        
    total_periods = goal_years * 12
    
    # update withdrawl rate depending on stage
    withdrawal_rate = 0 if stage == 'pre-retirement' else withdrawal_rate
    
    # create index for simulation based on goal_years------------------------------------------------------
    start_date = df.index[-1] + pd.DateOffset(months=1) + pd.offsets.MonthEnd(0)
    end_date = start_date + pd.DateOffset(years=goal_years+1) # extend the end data by a little bit to guarantee length of sim index

    sim_index = pd.date_range(start=start_date, end=end_date, freq='M')
    sim_index = sim_index[:total_periods] # reduce sim_index to the total amount of goal years
    
    # Update goal_dict into friendly inputs-----------------------------------------------------------------
    goal_dict = {k:v/100 for k,v in goal_dict.items()}
    a_list = [k for k in goal_dict.keys()]
    
        
    # Create dataframe of cash flow growth-------------------------------------------------------------------
    money_idx = [i for i in range(1, total_periods+1)]

    savings_data = []

    monthly_income = current_house_income * savings_rate

    for year in range(goal_years):
        for month in range(1,13):
            income_for_month = monthly_income * ((1+income_growth)**year)
            print(income_for_month)
            savings_data.append([year+1, month, income_for_month])

    money_df = pd.DataFrame(savings_data, columns=['year', 'month', 'income'], index=money_idx)
    money_df.loc[1,'$ start'] = current_savings
    money_df['$ start'] = money_df['$ start'].fillna(0)
    money_df['expense_or_windfall'] = money_df.index.map(lambda x: lqd_windfall_dict.get(x,0))
    money_df['cash_flow'] = 0

    for i in range(1, total_periods+1):
        if i == 1:
            money_df.loc[i,'cash_flow'] = money_df.loc[i, ['income', '$ start', 'expense_or_windfall']].sum()
        else:
            money_df.loc[i,'cash_flow'] = money_df.loc[i-1, 'cash_flow'] + money_df.loc[i, ['income', '$ start', 'expense_or_windfall']].sum()
            

    # Generate Dependent samples for each Factor----------------------------------------------------------------------
    # specify the number of samples to generate. 
    # Resulting array will have shape nbr_sims x nbr_factors for number of months of simulations
    nbr_sims = total_periods

    # creat dictionary of statistics
    dparams = {}
    for f in factdb.columns:
        vals = factdb[f]
        mean = vals.mean()
        std = vals.std(ddof=1)
        rv = norm(mean, std)
        sample = rv.rvs(size=nbr_sims)
        dparams[f] = {"mean": mean, "std": std, 'rv':sample}
        
        
    # generate correlated random samples

    # Simulating Dependent Draws---------------------------------------------------------------------------------------
    # create vector of means
    means = []
    df_cols = []
    for k,v in dparams.items():
        means.append(v['mean'])
        df_cols.append(k)

    # Bind rference to covariance matrix
    V = factdb.cov().values

    factor_simulations = []
    for i in range(simulations):

        # these are the factors we will use to determine the monthly returns based on chosen asset allocation
        dfsims_corr = pd.DataFrame(np.random.multivariate_normal(means, V, nbr_sims),
                                  columns = df_cols)

    #     # stats on correlated simulated factors
    #     dfsims_corr.describe()
    #     # check if the newly simulated factors align with the original factor database
    #     dfsims_corr.corr()
    #     print(dfsims_corr)
        factor_simulations.append(dfsims_corr)
        
    # Elastic Net Regression IMPLEMENATION ----------------------------------------------------------------------------

    # initialize list of 
    all_preds = [pd.DataFrame() for i in range(simulations)]
    all_sims = pd.DataFrame()

    for asset in a_list:
    #     print(asset)

        for i in range(simulations):
    #         print("Simulation: ", i)
            # pull in simulated months factor returns
            sim_months =  factor_simulations[i].copy()

            # predict return of asset class based on EN fit
            pred_monthly = en_co[asset][0] + np.dot(sim_months, en_co[asset][1])
            pred_df = pd.DataFrame(pred_monthly, columns=[asset])

            # append each assets stream of returns to all preds list
            all_preds[i] = pd.concat([all_preds[i], pred_df], axis=1)

    # add portfolio column that calculates the monthly portfolio return for each simulation
    for idx, a in enumerate(all_preds):
        a['portfolio'] = a.dot(pd.Series(goal_dict))
        all_sims[idx] = a['portfolio']

    all_sims = all_sims.set_index(sim_index) # monthly returns for each simulation
    
    # Calculate portfolio value incorporating asset class simulations, income, savings, withdrawal---------------------
    total_dollars = pd.DataFrame(np.zeros_like(all_sims), index=all_sims.index, columns=all_sims.columns)

    prev_index = start_date
    for row,idx in enumerate(total_dollars.index):
        if row == 0:
            total_dollars.loc[idx,:] = (money_df.loc[row+1,['cash_flow']][0] - (withdrawal_rate * money_df.loc[row+1,['cash_flow']][0]) )* (1+all_sims.loc[idx,:])
        else:
            total_dollars.loc[idx,:] =  (total_dollars.loc[prev_idx,:] + money_df.loc[row+1,['income']][0]
                                         + money_df.loc[row+1,['expense_or_windfall']][0] -
                                        ( total_dollars.loc[prev_idx,:] * withdrawal_rate ))* (1+all_sims.loc[idx,:])
        prev_idx = idx

    total_dollars.loc[df.index[-1], :] = current_savings
    total_dollars = total_dollars.sort_index()
    
    # Cumulative return of each simulation only considering returns -----------------------------------------------------
    all_sims_cum = (1+all_sims).cumprod()
    all_sims_cum.loc[df.index[-1],:] = 1
    all_sims_cum = all_sims_cum.sort_index()
    all_sims_cum = all_sims_cum / all_sims_cum.iloc[0]
    
    # show stats over each time period -----------------------------------------------------------------------------------
    
    # calculate portfolio stats for each simulation only considering asset class changes not income, expenses withdrawal, etc.

    stat_df = pd.DataFrame()
    portfolio_cums = all_sims_cum.copy()

    for col in all_sims_cum:
        new_portfolio = portfolio_cums[[col]].rename(columns={col:'port_value'})
        # stat_df.loc['Total Return',col] = period_return(new_portfolio)
        stat_df.loc['Annualized Return',col] = annualized_return_monthly(new_portfolio)             # needs to be monthly
        stat_df.loc['Volatility',col] = annualized_volatility_monthly(new_portfolio)[0] # needs to be monthly
#         stat_df.loc['Return/Risk Ratio',col] = stat_df.loc['Annualized Return', col] / stat_df.loc['Volatility', col]
#         stat_df.loc['Max Drawdown Peak',col] = calc_max_drawdown(new_portfolio)[0]
#         stat_df.loc['Max Drawdown Trough',col] = calc_max_drawdown(new_portfolio)[1]
        stat_df.loc['Max Drawdown',col] = calc_max_drawdown(new_portfolio)[2]
        stat_df.loc['Ending Portfolio Value',col] = np.round(total_dollars.loc[total_dollars.index[-1], col],0)
        stat_df.loc['Goal Value', col] = goal
        stat_df.loc['Surplus/Shortfall', col] = stat_df.loc['Ending Portfolio Value',col] - stat_df.loc['Goal Value', col]
        stat_df.loc['Goal Reached',col] = np.where(stat_df.loc['Surplus/Shortfall',col] >0, 1, 0)

    # Diagnostics ----------------------------------------------------------------------------------------------------
    # create dataframe that shows the number of times goal was reached
    # min, max, median stats,
    stat_df_summary = pd.DataFrame(columns=['median', 'min', 'max'])

    var_list = ['Annualized Return', 'Volatility', 'Max Drawdown', 'Ending Portfolio Value', 
               'Surplus/Shortfall']

    for var in var_list:
            stat_df_summary.loc[var, 'min'] =  stat_df.loc[var, :].min()
            stat_df_summary.loc[var, 'max'] =  stat_df.loc[var, :].max()
            stat_df_summary.loc[var, 'median'] = stat_df.loc[var, :].median()
            
    # Success rate-------------------------------------------------------------------------------------
    success = stat_df.loc['Goal Reached', :].sum() / simulations * 100
    
    # Return outputs for visualization ---------------------------------------------------------------------

    return (success, stat_df_summary, total_dollars, stat_df, all_sims, all_sims_cum, money_df ) # all_holdings, init_port_stats)

#--------------------------------------------------------------------------------------------------------------------------------------
# Create historical Stats function
def history_stats(df, goal_dict):
    # format inputs
    goal_dict = {k:v/100 for k,v in goal_dict.items()}
    a_list = [k for k in goal_dict.keys()]

    # data calculations below allow for statistic calculations over different time frames
    last_date = df.index[-1]
    inc_date = find_inception(df, a_list)

    # used to override inc_date or last date to earlier date
    # inc_date = dt.strptime("2023-3-31", '%Y-%m-%d') # enter custom beginning date
    # last_date = dt.strptime("2023-3-31", '%Y-%m-%d') # enter custom beginning date

    ytd = date(last_date.year-1, 12, 31)

    # DATA CALCS COMMENTED OUT, UNCOMMENT OUT IF WE DECIDE TO SHOW HISTORICAL TIME PERIOD RETURNS
    # Find dates, 10, 5, 3, and 1 year prior to last date in price dataframe
    ten_years = last_date - relativedelta(years=10)
    five_years = last_date - relativedelta(years=5)
    three_years = last_date - relativedelta(years=3)
    one_year = last_date - relativedelta(years=1)

    # adjust dates above to most recent business day if original day falls on Saturday or Sunday
    ten_years = last_business_day(ten_years)
    five_years = last_business_day(five_years)
    three_years = last_business_day(three_years)
    one_year = last_business_day(one_year)

    # filter dataframe to only relevant tickers, forward fill price data on weekends that didn't match on join
    df = df.ffill()
    
    # HOLDINGS HISTORICAL STATS ------------------------------------------------------------------------------
    # pull stats common period stats based on the latest starting inception date of all tickers

    # iterate through all time periods                  UNCOMMENT OUT WHEN WE WANT TO SHOW STATS AND HOLDINGS
    periods = [inc_date ]
    period_labels = ["Inception"]

    all_stats = []

    for a in a_list:
        print(a)
        for p, pl in zip(periods, period_labels):
            print(p)
            alloc_dict = {a:1}
            index_value = a
            check = round(sum([v for v in alloc_dict.values()]),2)

            new_portfolio = create_portfolio(df, p, last_date, alloc_dict, rebalance_freq="annual") # update this in loop for time period

            summary = pd.DataFrame({"Total Return":period_return(new_portfolio),
                                    "Annualized Return":annualized_return(new_portfolio),
                               "Annualized Volatility":annualized_volatility(new_portfolio)[0],
                               "Sharpe Ratio": calc_sharpe(df, new_portfolio),
                               "Return/Risk Ratio": calc_return_risk_ratio(new_portfolio),
                               "Beta": calc_beta(df, new_portfolio, p, last_date),
                               "Max Drawdown Peak": calc_max_drawdown(new_portfolio)[0],
                               "Max Drawdown Trough": calc_max_drawdown(new_portfolio)[1],
                               "Max Drawdown": calc_max_drawdown(new_portfolio)[2]},
                              index = [index_value]).T
            all_stats.append(summary)

    # concatenate all summary dataframes
    all_holdings = pd.concat(all_stats, axis=1).T  
    
    # HISTORICAL PORTFOLIO STATS -----------------------------------------------------------
    bgn = inc_date
    end = last_date
    # Enter Custom Portfolio Name
    initial_name = 'Proposed Portfolio'

    check = round(sum([v for v in goal_dict.values()]),2)

    if check != 1:
        print("Portfolio Allocation does not sum to 100%")

    else:
        print("Check if allocation sums to 100%: ", check, check==1)

        new_portfolio = create_portfolio(df, bgn, end, goal_dict, rebalance_freq="annual", initial_investment=1)
        
        time_series = new_portfolio.copy()

        # calc stats on core bonds
        summary = pd.DataFrame({"Total Return":period_return(new_portfolio),
                                "Annualized Return":annualized_return(new_portfolio),
                               "Annualized Volatility":annualized_volatility(new_portfolio)[0],
                               "Return/Risk Ratio": calc_return_risk_ratio(new_portfolio),
                               "Max Drawdown Peak": calc_max_drawdown(new_portfolio)[0],
                               "Max Drawdown Trough": calc_max_drawdown(new_portfolio)[1],
                               "Max Drawdown": calc_max_drawdown(new_portfolio)[2]},
                              index = [initial_name]).T  
        
        init_port_stats = summary.copy()

    return (all_holdings, init_port_stats)

#---------------------------------------------------------------------------------------------------------------------------------------
# Create Visualization functions
def create_gauge(value):
    # Define color ranges
    if value <= 25:
        color = 'red'
    elif value <= 40:
        color = 'orange'
    elif value <= 60:
        color = 'yellow'
    elif value <= 80:
        color = 'lightgreen'
    else:
        color = 'limegreen'

    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = value,
        mode = "gauge+number",
        title = {'text': "Probability %"},
        gauge = {'axis': {'range': [None, 100], 'tickvals': [0, 25, 50, 75, 100]},
                 'bar': {'color': color},
                 'steps' : [
                     {'range': [0, 100], 'color': "white"}
                 ]
        },
        number = {'font':{'color':'black'}}
    ))
    return fig

def create_line_chart(total_dollars, goal):
    # Update function to improve the visuals y-axis
    # consider two standard deviations of all the end total_dollars[-1]



    # Calculate median
    line_df = total_dollars.copy()
    line_df['Median Portfolio Value'] = line_df.median(axis=1)

    plt.figure(figsize=(10,6), tight_layout=True)

    for col in line_df.columns:
        color = 'lightgreen' if line_df[col][-1] >= goal else 'lightcoral'
        if col == 'Median Portfolio Value':
            color = 'black'
        plt.plot(line_df.index, line_df[col], label=col, color=color)

    plt.axhline(y=goal, color='gray', linestyle='--', label='Goal')

    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.title('Retirement Possibilities')
    
        # Create custom legend
    custom_legend = [
        plt.Line2D([0], [0], color='black', lw=1, label='Median Outcome'),
        plt.Line2D([0], [0], color='lightgreen', lw=1, label='Successful Outcome'),
        plt.Line2D([0], [0], color='lightcoral', lw=1, label='Unsuccessful Outcome'),
        plt.Line2D([0], [0], color='gray', lw=1, linestyle='--', label='Investment Goal')
    ]
    plt.legend(handles=custom_legend)

    return plt

def plot_distribution(stat_df, stat):


    plt.figure(figsize=(10,6), tight_layout=True)

    arr = stat_df.loc[stat,:].to_numpy() * 100

    bins = np.arange(-100, 10, 10)

    fig, ax = plt.subplots()
    ax.hist(arr, bins=bins)

    plt.xlabel('Max Drawdown (%)')
    plt.ylabel('# of Occurences')

    plt.xticks(np.arange(-100, 10, 10))

    return fig

# --------------------------------------------------------------------------------------------------------------------
# Generate output from sim_retirement()

if st.button('Run Simulation'):

    retire = sim_retirement(stage=stage, df=df, factdb=factdb, goal_dict=goal_dict,
                            goal_years=goal_years, goal=goal, current_savings=current_savings, 
                            current_house_income=current_house_income, income_growth=income_growth, 
                            savings_rate=savings_rate, withdrawal_rate=withdrawal_rate,
                            lqd_windfall_dict = lqd_windfall_dict)
    
    # st.write(retire[2])                                   # DELETE AFTER MAKING SURE WITHDRAWAL WORKS
    # st.write(retire[4])                                   # DELETE AFTER MAKING SURE WITHDRAWAL WORKS
    # st.write(retire[6])                                   # DELETE AFTER MAKING SURE WITHDRAWAL WORKS
    
    st.title('Probability of Reaching Goal (%)')
    fig = create_gauge(retire[0])
    st.plotly_chart(fig, use_container_width=True)

    st.title('Path of Simulations')
    line_chart = create_line_chart(retire[2], goal)
    st.pyplot(line_chart)

    st.title('Distribution of Maximum Drawdowns')
    dist_plot = plot_distribution(retire[3], 'Max Drawdown')
    st.pyplot(dist_plot, use_container_width=True)
                                                                                


                                                # CHECK WITHDRAWAL CALCULATIONS, STILL A PRETTY HIGH SUCCESS RATE WITH HIGH WITHDRAWAL %
    
    # TO DO: update withdrawals in retirement planner to a monthly expense number that has a growth rate for inflation
    # Add expense inflation
    # add tabs, one for simulation outputs, add historgram of final dollar value outputs
    # Add dataframe of median stats
    # add tab for historical returns and growth of 1M
    # Add tab for stress test
    # Add tab for matrix of increments of all selections chosen in middle than two increments up or down on each side maxing an x by 5 matrix
    # update format to look cools
