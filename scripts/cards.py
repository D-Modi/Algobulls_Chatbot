import re
import glob
import streamlit as st
import matplotlib.pyplot as plt  
import seaborn as sns
from stratrgy_analysis import StatergyAnalysis
import pandas as pd
import numpy as np

def next_page(Analysis, code):
    st.title("Analysis")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.caption("Stratergy Code")
        st.subheader(code)
    with col2:
        st.caption("Transaction Fee (with Package)")
        st.subheader("0%")
    with col3:
        st.caption("Transaction Fee (w/o Package)")
        st.subheader(""":red[3%]""")
    with col4:
        st.caption("Recomended Days")
        st.subheader(""":green[180]""")
    
    tab1, tab2, tab3 = st.tabs(["Reords", "Analytics", "Returns"])
    with tab3:
        Dur = [213*2, 213, 101,22, 11, 4]
        Duration = ['All Time', ' 2 Years', '1 year', '180 Days', '30 Days', '15Days', '3 Days']
        returns = [f"{Analysis.Treturns(len(Analysis.daily_returnts)+1)[1]}%"]
        for i in Dur:
            returns.append(f"{Analysis.Treturns(i)[1]}%")
        arr = np.array([Duration, returns]).T
        df = pd.DataFrame(arr, columns=["Duration", "Returns"])
        
        st.table(df.assign(hack='').set_index('hack'))
    with tab2:
        bt1 , bt2, bt3, bt4, bt5 = st.tabs(["Stats", "P&L", "ROI%", "Equity Curve", "Drawdown %"])
        with bt5:
            st.line_chart(Analysis.csv_data, y='drawdown_percentage', x='Day')
        with bt4:
            st.line_chart(Analysis.daily_equity, y='equity_curve')
        with bt3:
            st.line_chart( Analysis.daily_returnts, y='roi' )
        with bt2:
            st.subheader("Daily P&L")
            st.bar_chart(Analysis.daily_returnts, y=['pnl_absolute'])
            st.subheader("Cumulative P&L")
            st.line_chart(Analysis.daily_returnts, y=['cum_pnl'])
        with bt1:
            st.write(f"***Max Drawdown***: {Analysis.drawdown_max}")
            st.write(f"***Maximum Drawdowm percentage***: {Analysis.drawdown_pct}")
            st.write(f"***Average loss per losing trade***: {Analysis.winCount(Analysis.csv_data, -1)}")
            st.write(f"***Average gain per winning trade***: {Analysis.winCount(Analysis.csv_data, 1)}")
            st.write(f"***Maximum Gains***: {Analysis.max_profit(Analysis.csv_data)[0]}")
            st.write(f"***Minimum Gains***: {Analysis.min_profit(Analysis.csv_data)[0]}")
            st.write(f"Number of ***short trades***: {Analysis.num_tradeType('short')}")
            st.write(f"Number of ***long trades***: {Analysis.num_tradeType('long')}")
            st.write (f"***Average Trades per Day***: {Analysis.avgTrades(daily_returns)}")
            st.write(f"Number of ***wins***: {Analysis.num_profit(Analysis.csv_data)}")
            st.write(f"Number of ***losses***: {Analysis.num_loss(Analysis.csv_data)}")
            st.write(f"***HIT Ratio***: {Analysis.HIT()}")
            st.write(f"***ROI***: {Analysis.roi(monthly_returns)[0]}")
            st.write(f"***ROI %***: {Analysis.roi(monthly_returns)[1]}%")
            st.write(f"***Profit Factor***: {Analysis.ProfitFactor()}")
            st.write(f"***Yearly Volatility***: {Analysis.yearlyVola()}")
            st.write(f"***Max Win Streak***: {Analysis.max_consecutive(Analysis.csv_data, 1)}")
            st.write(f"***Max Loss streak***: {Analysis.max_consecutive(Analysis.csv_data, -1)}")
            st.write(f"***Sharpe Ratio:*** {Analysis.Sharpe()}")
            st.write(f"***Calmar Ratio:*** {Analysis.Calmar()}")
            st.write(f"***Sortino Ratio:*** {Analysis.Sortino()}")
            last_month_data = daily_returns.iloc[-21:]
            st.write(f"Win Rate for ***last Month***: {Analysis.win_rate(last_month_data)}")
            last_month_data = daily_returns.iloc[-6:]
            st.write(f"Win Rate for ***last Week***: {Analysis.win_rate(last_month_data)}")
            last_month_data = daily_returns.iloc[-213:]
            st.write(f"Win Rate for ***last Year***: {Analysis.win_rate(last_month_data)}")
            last_month_data = daily_returns.iloc[-101:]
            st.write(f"Win Rate for ***last 6 Months***: {Analysis.win_rate(last_month_data)}")
            last_month_data = daily_returns.iloc[-59:]
            st.write(f"Win Rate for ***last Quater:*** {Analysis.win_rate(last_month_data)}")
        with tab1:
            Analysis.htmap()

st.set_page_config(layout="wide")

path = "files/StrategyBacktestingPLBook-*.csv"
print("\nUsing glob.iglob()")
Files = []

for file in glob.glob(path, recursive=True):
    found = re.search('StrategyBacktestingPLBook(.+?)csv', str(file)).group(1)[1:-1]
    Files.append(found)

row1 = st.columns(3)
row2 = st.columns(3)
i = 0
for col in row1 + row2:
    tile = col.container(height=400, border=True)
    tile.write("By Algobulls")
    if i < len(Files):
        stratergy = Files[i]
        csv_path = f"files/StrategyBacktestingPLBook-{stratergy}.csv"
        Analysis = StatergyAnalysis(csv_path)
        i += 1
        daily_returns, monthly_returns, weekday_returns, weekly_returns, yearly_returns = Analysis.analysis()

        custom_aligned_text = f"""
            <div style="display: flex; justify-content: space-between;">
            <span style="text-align: left;">{stratergy}</span>
            <span style="text-align: right; color: green">{Analysis.roi(monthly_returns)[1]}</span>
            </div>
            """
        mark = f"""
            <div style="display: flex; justify-content: space-between; color: grey; font-size: 12px;">
            <span style="text-align: left;">   </span>
            <span style="text-align: right;">ROI% | All Time</span>
            </div>
            """
        tile.write(custom_aligned_text, unsafe_allow_html=True)
        tile.markdown(mark, unsafe_allow_html=True)
        pnl, cum_pnl = Analysis.daily_returns_hist(monthly_returns)
        tile.pyplot(pnl)
        #tile.bar_chart(monthly_returns, y=['pnl_absolute'])

        ratios = f"""
            <div style="display: flex; justify-content: space-between;">
            <span style="text-align: left;">{Analysis.Sharpe()}</span>
            <span style="text-align: center; flex-grow: 1; text-align: center;"> {Analysis.Calmar()}</span>
            <span style="text-align: right;">{Analysis.Sortino()}</span>
            </div>
            """
        cap = f"""
            <div style="display: flex; justify-content: space-between; color: grey; font-size: 12px;">
            <span style="text-align: left;">Sharpe Ratio</span>
            <span style="text-align: center; flex-grow: 1; text-align: center;"> Calmar Ratio</span>
            <span style="text-align: right;">Sortino Ratio</span>
            </div>
            """
        tile.write(ratios, unsafe_allow_html=True)
        tile.markdown(cap, unsafe_allow_html=True)

        head_ = f"""
            <div style="display: flex; justify-content: space-between; color: grey; font-size: 12px;">
            <span style="text-align: left;">Initial Investment</span>
            <span style="text-align: center; flex-grow: 1; text-align: center;"> HIT Ratio</span>
            <span style="text-align: right;"> max. Drawdown</span>
            </div>
            """

        disp = f"""
            <div style="display: flex; justify-content: space-between;">
            <span style="text-align: left;">{Analysis.initial_investment}</span>
            <span style="text-align: center; flex-grow: 1; text-align: center;"> {Analysis.HIT()}</span>
            <span style="text-align: right; color: red;">{Analysis.drawdown_pct}%</span>
            </div>
            """
        tile.write("\n")
        tile.markdown(head_, unsafe_allow_html=True)
        tile.write(disp, unsafe_allow_html=True)
        tile.write("\n")
        on = tile.toggle(f"Execute {stratergy}")
        if on:
            tile.write("#########################")
            next_page(Analysis, stratergy)











