import pandas as pd
import numpy as np
import streamlit as st
from stratrgy_analysis import StatergyAnalysis
import matplotlib as plt
import re
import glob

#change filepath here in "path"
path = "files/StrategyBacktestingPLBook-*.csv"
print("\nUsing glob.iglob()")
Files = []

for file in glob.glob(path, recursive=True):
    found = re.search('StrategyBacktestingPLBook(.+?)csv', str(file)).group(1)[1:-1]
    Files.append(found)

# Using "with" notation
with st.sidebar:
    option = st.radio(
        "Choose a Stratergy Code",
        ("***" + x + "***" for x in Files)
    )
    option = option[3:-3]
    st.write("\n")
    
    st.write ("Choose Display Options")
    Daily = st.checkbox("Daily Analysis.", True)
    Monthly = st.checkbox("Monthly Analysis", False)
    Yearly = st.checkbox("Yearly Analysis", False)
    Weekly = st.checkbox("Weekly Analysis", False)
    Day = st.checkbox("Analysis based on day of week", False)
    st.write("\n")
    
    st.write ("Show Returns For:")
    threeD = st.checkbox("3 Days", False) 
    twoW = st.checkbox("2 Weeks", False) 
    thirtyD = st.checkbox("30 Days", False) 
    sixM = st.checkbox("6 Months", False) 
    oneY = st.checkbox("1 Years", False) 
    twoY = st.checkbox("2 Years", False) 
    st.write("\n")
    
    st.write ("Show Win Rate For:")
    week = st.checkbox("Last Week", False)
    month= st.checkbox("Last Month", False)
    year= st.checkbox("Last Year", False)
    months6= st.checkbox("Last 6 Months", False)
    quater= st.checkbox("Last Quater", False)
  
st.title(f"Analyis Of Stratrergy ***{option}***")

def daisply(daily_returns, Quant):
    print(Quant)
    if Quant == "Day":
        st.header("Daily Analysis")
    else:
        st.header(f"{Quant}ly Analysis")
        st.write(f"***{Quant}ly Average Returns***: {Alanyze.avgReturns(daily_returns)[0]}")
        st.write(f"***{Quant}ly Average Returns %***: {Alanyze.avgReturns(daily_returns)[1]}%")
    
    days_hist, days_tab = Alanyze.compare_hist(daily_returns, [1000, 2000, 3000, 4000, 5000], Quant)
    st.subheader(f"Number of {Quant} of profit/loss above a threshold") 
    st.pyplot(days_hist)
    with st.expander("More information"):
        st.write(f"Number of ***trading {Quant}s***: {Alanyze.trading_num(daily_returns)}")
        st.write(f"Number of ***Profitable {Quant}s***: {Alanyze.num_profit(daily_returns)} {Quant}")
        st.write(f"Number of ***Loss Making {Quant}s***: {Alanyze.num_loss(daily_returns)} {Quant} ")
        st.write(f"***Most Profitable {Quant}***: {Alanyze.max_profit(daily_returns)[1]}")
        st.write(f"Maximum ***Gains*** in a {Quant}: {Alanyze.max_profit(daily_returns)[0]}")
        st.write(f"***Least Profitable {Quant}***: {Alanyze.min_profit(daily_returns)[1]}")
        st.write(f"Maximum ***Loss*** in a {Quant}: {Alanyze.min_profit(daily_returns)[0]}")
        st.write(f"***Max Win Streak***: {Alanyze.max_consecutive(daily_returns, 1)}")
        st.write(f"***Max Loss streak***: {Alanyze.max_consecutive(daily_returns, -1)}")

    st.subheader(f"Profit/Loss Data per {Quant}")
    st.bar_chart(daily_returns, y=['pnl_absolute'] )
    if 'cum_pnl' in daily_returns.columns:
        st.subheader("Cumulative Profit and loss")
        st.line_chart(daily_returns, y=['cum_pnl'])
    st.write(f"")
    st.divider()

def display(weekday_returns):
    st.subheader(f"Profit/Loss Data per Day of Week")
    st.bar_chart(weekday_returns, y=['pnl_absolute'] )
    st.write(f"***Most Profitable Day*** of the week: {Alanyze.max_profit(weekday_returns)[1]}")
    st.write(f"***Least Profitable Day*** of the week: {Alanyze.min_profit(weekday_returns)[1]}")
    tab = weekday_returns['pnl_absolute']
    st.table(tab)

csv = f"files/StrategyBacktestingPLBook-{option}.csv"
Alanyze = StatergyAnalysis(csv)
daily_returns, monthly_returns, weekday_returns, weekly_returns, yearly_returns = Alanyze.analysis()

st.write(f"***Max Drawdown***: {Alanyze.drawdown_max}")
st.write(f"***Maximum Drawdowm percentage***: {Alanyze.drawdown_pct}")
st.subheader("Drawdown Curve")
st.line_chart(Alanyze.csv_data, y='drawdown_pct', x='Day')
st.line_chart(Alanyze.csv_data, y='drawdown_percentage', x='Day')
st.table
st.write(f"***Average loss per losing trade***: {Alanyze.winCount(Alanyze.csv_data, -1)}")
st.write(f"***Average gain per winning trade***: {Alanyze.winCount(Alanyze.csv_data, 1)}")
st.write(f"***Maximum Gains***: {Alanyze.max_profit(Alanyze.csv_data)[0]}")
st.write(f"***Minimum Gains***: {Alanyze.min_profit(Alanyze.csv_data)[0]}")
st.write(f"Number of ***short trades***: {Alanyze.num_tradeType('short')}")
st.write(f"Number of ***long trades***: {Alanyze.num_tradeType('long')}")
st.write (f"***Average Trades per Day***: {Alanyze.avgTrades(daily_returns)}")
st.write(f"Number of ***wins***: {Alanyze.num_profit(Alanyze.csv_data)}")
st.write(f"Number of ***losses***: {Alanyze.num_loss(Alanyze.csv_data)}")
st.write(f"***HIT Ratio***: {Alanyze.HIT()}")
st.write(f"***ROI***: {Alanyze.roi(monthly_returns)[0]}")
st.write(f"***ROI %***: {Alanyze.roi(monthly_returns)[1]}%")
st.write(f"***Profit Factor***: {Alanyze.ProfitFactor()}")
st.write(f"***Yearly Volatility***: {Alanyze.yearlyVola()}")
st.write(f"***Max Win Streak***: {Alanyze.max_consecutive(Alanyze.csv_data, 1)}")
st.write(f"***Max Loss streak***: {Alanyze.max_consecutive(Alanyze.csv_data, -1)}")
if month:
    last_month_data = daily_returns.iloc[-21:]
    #last_month_data = Alanyze.csv_data[Alanyze.csv_data['Month'] == last_month]
    st.write(f"Win Rate for ***last Month***: {Alanyze.win_rate(last_month_data)}")
if week:
    last_month_data = daily_returns.iloc[-6:]
    #last_month_data = Alanyze.csv_data[Alanyze.csv_data['Week'] == last_month]
    st.write(f"Win Rate for ***last Week***: {Alanyze.win_rate(last_month_data)}")
if year:
    last_month_data = daily_returns.iloc[-213:]
    #last_month_data = Alanyze.csv_data[Alanyze.csv_data['Year'] == last_month]
    st.write(f"Win Rate for ***last Year***: {Alanyze.win_rate(last_month_data)}")
if months6:
    last_month_data = daily_returns.iloc[-101:]
    #last_month_data = Alanyze.csv_data[Alanyze.csv_data['Month'].isin(last_month)]
    st.write(f"Win Rate for ***last 6 Months***: {Alanyze.win_rate(last_month_data)}")
if quater:
    last_month_data = daily_returns.iloc[-59:]
    #last_month_data = Alanyze.csv_data[Alanyze.csv_data['Month'].isin(last_month)]
    st.write(f"Win Rate for ***last Quater:*** {Alanyze.win_rate(last_month_data)}")

st.write(f"***Sharpe Ratio:*** {Alanyze.Sharpe()}")
st.write(f"***Calmar Ratio:*** {Alanyze.Calmar()}")
st.write(f"***Sortino Ratio:*** {Alanyze.Sortino()}")

st.subheader("Equity Curve")
st.line_chart(Alanyze.daily_equity, y='equity_curve')
nan_rows = Alanyze.csv_data[Alanyze.csv_data['equity_curve'] == 0]
st.write(nan_rows)


# Streamlit interface
st.title("Line Chart of Equity Curve Between Given Dates")

# Date input widgets to select the date range
start_date = st.date_input("Start Date", value=pd.to_datetime("2022-10-04"))
end_date = st.date_input("End Date", value=pd.to_datetime("2022-10-31"))

# Filter the DataFrame to include only rows within the specified date range
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Filter the DataFrame to include only rows within the specified date range
if start_date and end_date:
    filtered_df = Alanyze.csv_data[(pd.to_datetime(Alanyze.csv_data['Day']) >= start_date) & (pd.to_datetime(Alanyze.csv_data['Day']) <= end_date)]
    
    # Display the filtered DataFrame for reference
    st.write(f"Data between {start_date.date()} and {end_date.date()}:")
    st.write(filtered_df)

    # Plot the line chart
    st.line_chart(filtered_df.set_index('Day')['equity_curve'])
if threeD:
    st.write(f"Returns for the ***last 3 Days***: {Alanyze.Treturns(3)[1]}%")
if thirtyD:
    st.write(f"Returns for the ***last 30 Days***: {Alanyze.Treturns(30)[1]}%")
if twoW:
    st.write(f"Returns for the ***last 2 Weeks***: {Alanyze.Treturns(14)[1]}%")
if sixM:
    st.write(f"Returns for the ***last 6 Months***: {Alanyze.Treturns(180)[1]}%")
if oneY:
    st.write(f"Returns for the ***last 1 Year***: {Alanyze.Treturns(365)[1]}%")
if twoY:
    st.write(f"Returns for the ***last 2 Years***: {Alanyze.Treturns(365*2)[1]}%")


if Daily:
    daisply(daily_returns, "Day")
if Monthly:
    daisply(monthly_returns, "Month")
if Yearly:
    daisply(yearly_returns, "Year")
if Weekly:
    daisply(weekly_returns, "Week")
if Day:
    display(weekday_returns)

st.subheader("Monthly Returns and ROI% Over Time")

# Plot the data
fig, ax1 = plt.subplots()

# Bar plot for monthly returns
ax1.bar(monthly_returns.index.values, monthly_returns['cum_pnl'].values, color='b', alpha=0.6, label='Monthly Returns')
ax1.set_xlabel('Month')
ax1.set_ylabel('Monthly Returns', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_xticks(monthly_returns.index[::3])
ax1.set_xticklabels(monthly_returns.index[::3], rotation=90)
# Line plot for ROI%
ax2 = ax1.twinx()
ax2.plot(monthly_returns.index.values, monthly_returns['roi'].values, color='r', marker='o', label='ROI%')
ax2.set_ylabel('ROI%', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Add a legend
fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
#fig.xticks(rotation=45)
# Display the plot using Streamlit
st.pyplot(fig)













