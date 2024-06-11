
import streamlit as st
import matplotlib.pyplot as plt  
from stratrgy_analysis import StatergyAnalysis
import pandas as pd
import numpy as np

csv_path = "files/CustomerPLBook.csv"
df = pd.read_csv(csv_path)

dfs = {}

# Get unique id values
unique_ids = df['Startegy Name'].unique()

# Loop through the unique ids and create separate DataFrames
for uid in unique_ids:
    dfs[uid] = df[df['Startegy Name'] == uid]
print(len(unique_ids))
unique_ids = np.insert(unique_ids, 0, "Complete Portfolio Analysis")

with st.sidebar:
    option = st.radio(
        "The stratergies being used are",
        ("***" + x + "***" for x in unique_ids)
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
        st.write(f"Number of ***Profitable {Quant}s***: {Alanyze.num_loss(daily_returns, 1)} {Quant}")
        st.write(f"Number of ***Loss Making {Quant}s***: {Alanyze.num_loss(daily_returns, -1)} {Quant} ")
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

if option == "Complete Portfolio Analysis":
    Alanyze = StatergyAnalysis(df, 1)
else:
    Alanyze = StatergyAnalysis(dfs[option], 1)
daily_returns, monthly_returns, weekday_returns, weekly_returns, yearly_returns = Alanyze.analysis()
Alanyze.initial_investment = Alanyze.initial_investment*2
st.write(f"***Max Drawdown***: {Alanyze.drawdown_max}")
st.write(f"***Maximum Drawdowm percentage***: {Alanyze.drawdown_pct}")
st.subheader("Drawdown Curve")
st.line_chart(Alanyze.csv_data, y='drawdown_percentage', x='Day')
st.write(f"***Average loss per losing trade***: {Alanyze.avgProfit(Alanyze.csv_data, -1)}")
st.write(f"***Average gain per winning trade***: {Alanyze.avgProfit(Alanyze.csv_data, 1)}")
st.write(f"***Maximum Gains***: {Alanyze.max_profit(Alanyze.csv_data)[0]}")
st.write(f"***Minimum Gains***: {Alanyze.min_profit(Alanyze.csv_data)[0]}")
st.write(f"Number of ***short trades***: {Alanyze.num_tradeType('short')}")
st.write(f"Number of ***long trades***: {Alanyze.num_tradeType('long')}")
st.write (f"***Average Trades per Day***: {Alanyze.avgTrades(daily_returns)}")
st.write(f"Number of ***wins***: {Alanyze.num_loss(Alanyze.csv_data, 1)}")
st.write(f"Number of ***losses***: {Alanyze.num_loss(Alanyze.csv_data, -1)}")
st.write(f"***HIT Ratio***: {Alanyze.HIT()}")
st.write(f"***ROI***: {Alanyze.roi(monthly_returns)[0]}")
st.write(f"***ROI %***: {Alanyze.roi(monthly_returns)[1]}%")
st.write(f"***Profit Factor***: {Alanyze.ProfitFactor()}")
#st.write(f"***Yearly Volatility***: {Alanyze.yearlyVola()}")
st.write(f"***Max Win Streak***: {Alanyze.max_consecutive(Alanyze.csv_data, 1)}")
st.write(f"***Max Loss streak***: {Alanyze.max_consecutive(Alanyze.csv_data, -1)}")
if month:
    last_month_data = daily_returns.iloc[-21:]
    st.write(f"Win Rate for ***last Month***: {Alanyze.win_rate(last_month_data)}")
if week:
    last_month_data = daily_returns.iloc[-6:]
    st.write(f"Win Rate for ***last Week***: {Alanyze.win_rate(last_month_data)}")
if year:
    last_month_data = daily_returns.iloc[-213:]
    st.write(f"Win Rate for ***last Year***: {Alanyze.win_rate(last_month_data)}")
if months6:
    last_month_data = daily_returns.iloc[-101:]
    st.write(f"Win Rate for ***last 6 Months***: {Alanyze.win_rate(last_month_data)}")
if quater:
    last_month_data = daily_returns.iloc[-59:]
    st.write(f"Win Rate for ***last Quater:*** {Alanyze.win_rate(last_month_data)}")

st.subheader("Equity Curve")
st.line_chart(Alanyze.csv_data, y='equity_curve', x='Day')
if threeD:
    st.write(f"Returns for the ***last 3 Days***: {Alanyze.Treturns(4)[1]}%")
if thirtyD:
    st.write(f"Returns for the ***last 30 Days***: {Alanyze.Treturns(22)[1]}%")
if twoW:
    st.write(f"Returns for the ***last 2 Weeks***: {Alanyze.Treturns(11)[1]}%")
if sixM:
    st.write(f"Returns for the ***last 6 Months***: {Alanyze.Treturns(101)[1]}%")
if oneY:
    st.write(f"Returns for the ***last 1 Year***: {Alanyze.Treturns(213)[1]}%")
if twoY:
    st.write(f"Returns for the ***last 2 Years***: {Alanyze.Treturns(213*2)[1]}%")

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
