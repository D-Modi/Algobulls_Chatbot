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
    daily_returns, monthly_returns, weekday_returns, weekly_returns, yearly_returns = Analysis.analysis()

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
# import streamlit as st
# import plotly.express as px
# import pandas as pd
# import numpy as np

# df = pd.DataFrame({
#     'Position': range(1, 10001),  # Example range, adjust according to your data
#     'Value': np.random.randn(10000)  # Random data for demonstration
# })

# fig = px.line(df, x='Position', y='Value')

# fig.update_layout(
#     xaxis=dict(
#         rangeslider=dict(
#             visible=True
#         ),
#         type="linear"
#     )
# )

# st.plotly_chart(fig, use_container_width=True)
