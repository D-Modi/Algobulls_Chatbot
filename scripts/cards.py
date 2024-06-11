
import re
import glob
import streamlit as st
import matplotlib.pyplot as plt  
import seaborn as sns
from stratrgy_analysis import StatergyAnalysis
import pandas as pd
import numpy as np

if 'clicked' not in st.session_state:
    st.session_state.clicked = False 
if 'ana' not in st.session_state:
    st.session_state['ana'] = None
if 'stra' not in st.session_state:
    st.session_state['stra'] = None
if 'button' not in st.session_state:
    st.session_state.button = False 
if 'Time' not in st.session_state:
    st.session_state['Time'] = None
if 'rt' not in st.session_state:
    st.session_state['rt'] = None
    
def click_button():
    st.session_state.clicked = not st.session_state.clicked
    
def click_button_disp(s, r):
    st.session_state.button = True
    st.session_state['Time'] = s
    st.session_state['rt'] = r
    
def click_button_arg(a,b):
    st.session_state.clicked = not st.session_state.clicked
    st.session_state['ana'] = a
    st.session_state['stra'] = b
    
def daisply(daily_returns, Quant, Alanyze):
    if Quant == "Day":
        st.header("Daily Analysis")
    else:
        st.header(f"{Quant}ly Analysis")
        st.write(f"***{Quant}ly Average Returns***: {Alanyze.avgReturns(daily_returns)[0]}")
        st.write(f"***{Quant}ly Average Returns %***: {Alanyze.avgReturns(daily_returns)[1]}%")
    
    days_hist, days_tab = Alanyze.compare_hist(daily_returns, [1000, 2000, 3000, 4000, 5000], Quant)
    freq_hist = Alanyze.freq_hist(daily_returns, [-5000,-4000, -3000, -2000, -1000, 0, 1000, 2000, 3000, 4000, 5000])

    st.subheader(f"Number of {Quant} of profit/loss above a threshold") 
    st.pyplot(days_hist)
    st.subheader("Frequency of profits")
    st.pyplot(freq_hist)
    
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
    st.bar_chart(daily_returns, y=['pnl_absolute'], width=500, height=800)
    if 'cum_pnl' in daily_returns.columns:
        st.subheader("Cumulative Profit and loss")
        st.line_chart(daily_returns, y=['cum_pnl'])
    st.write(f"")
    st.divider()
 
def display(weekday_returns, Alanyze):
    st.subheader(f"Profit/Loss Data per Day of Week")
    st.bar_chart(weekday_returns, y=['pnl_absolute'] )
    st.write(f"***Most Profitable Day*** of the week: {Alanyze.max_profit(weekday_returns)[1]}")
    st.write(f"***Least Profitable Day*** of the week: {Alanyze.min_profit(weekday_returns)[1]}")
    tab = weekday_returns['pnl_absolute']
    st.table(tab)
        
def next_page( Analysis, code):
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
        
        st.dataframe(df, hide_index=True, use_container_width=True)
        
        
    with tab2:
        bt1 , bt2, bt3, bt4, bt5 = st.tabs(["Stats", "P&L", "ROI%", "Equity Curve", "Drawdown %"])
        
        with bt5:
            st.subheader("Drawdown Curve")
            st.line_chart(Analysis.csv_data, y='drawdown_percentage', x='Day')
            st.write(f"***Max Drawdown***: {Analysis.drawdown_max}")
            st.write(f"***Maximum Drawdowm percentage***: {Analysis.drawdown_pct}")
            
        with bt4:
            st.subheader("Equity Curve")
            st.line_chart(Analysis.daily_equity, y='equity_curve')
            
        with bt3:
            st.subheader("ROI% Curve")
            st.line_chart( Analysis.daily_returnts, y='roi' )
            st.write(f"***ROI***: {Analysis.roi(monthly_returns)[0]}")
            st.write(f"***ROI %***: {Analysis.roi(monthly_returns)[1]}%")
            
            st.subheader("Monthly Returns and ROI% Over Time")
            fig, ax1 = plt.subplots()

            ax1.bar(monthly_returns.index.values, monthly_returns['cum_pnl'].values, color='b', alpha=0.6, label='Monthly Returns')
            ax1.set_xlabel('Month')
            ax1.set_ylabel('Monthly Returns', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.set_xticks(monthly_returns.index[::3])
            ax1.set_xticklabels(monthly_returns.index[::3], rotation=90)
            
            ax2 = ax1.twinx()
            ax2.plot(monthly_returns.index.values, monthly_returns['roi'].values, color='r', marker='o', label='ROI%')
            ax2.set_ylabel('ROI%', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
            st.pyplot(fig)
            
        with bt2:
            st.subheader("Daily P&L")
            st.bar_chart(Analysis.daily_returnts, y=['pnl_absolute'])
            st.subheader("Cumulative P&L")
            st.line_chart(Analysis.daily_returnts, y=['cum_pnl'])
            
        with bt1:
            last_month_data = daily_returns.iloc[-6:]
            week = Analysis.win_rate(last_month_data)
            last_month_data = daily_returns.iloc[-21:]
            month = Analysis.win_rate(last_month_data)
            last_month_data = daily_returns.iloc[-59:]
            quat = Analysis.win_rate(last_month_data)
            last_month_data = daily_returns.iloc[-101:]
            half = Analysis.win_rate(last_month_data)
            last_month_data = daily_returns.iloc[-213:]
            yr = Analysis.win_rate(last_month_data)
            
            c1, c2 = st.columns(2)
            with c1:
                fig0, ax0 = plt.subplots(figsize=(10,2))
                values = [Analysis.max_profit(Analysis.csv_data)[0], Analysis.min_profit(Analysis.csv_data)[0]]
                x = ["Maximum Profit", "Minimum Profit"]
                colors = ['r' if value < 0 else 'g' for value in values]
                ind = np.arange(2)
                width = 0.05
                
                bars = ax0.barh(ind, values, color=colors)
                ax0.bar_label(bars, label_type="center")
                avg_profit = Analysis.winCount(Analysis.csv_data, 1)
                avg_loss = Analysis.winCount(Analysis.csv_data, -1)
                ax0.axvline(avg_profit, ls='--', ymax=0.5, color='k', label='Avg Profit')
                ax0.axvline(avg_loss*(-1), ymin= 0.5, ls='--', color='k', label='Avg Loss')
                ax0.text(avg_profit, 0.75, f'Avg Profit: {avg_profit}', color='k', va='top', ha='left')
                ax0.text(avg_loss * -1, 0.25, f'Avg Loss: {avg_loss}', color='k', va='bottom', ha='right')
                ax0.set_yticks(ind+width/2)
                ax0.set_yticklabels(x, minor=False)
                ax0.set_title('Profit/Loss Analysis')
                st.pyplot(fig0)
                
                st.write(f"***Average loss per losing trade***: {Analysis.winCount(Analysis.csv_data, -1)}")
                st.write(f"***Average gain per winning trade***: {Analysis.winCount(Analysis.csv_data, 1)}")
                st.write(f"***Maximum Gains***: {Analysis.max_profit(Analysis.csv_data)[0]}")
                st.write(f"***Minimum Gains***: {Analysis.min_profit(Analysis.csv_data)[0]}")
                st.write (f"***Average Trades per Day***: {Analysis.avgTrades(daily_returns)}")
                st.write("\n")
                st.write(f"***HIT Ratio***: {Analysis.HIT()}")
                st.write(f"***Profit Factor***: {Analysis.ProfitFactor()}")
                st.write(f"***Yearly Volatility***: {Analysis.yearlyVola()}")
                st.write(f"***Max Win Streak***: {Analysis.max_consecutive(Analysis.csv_data, 1)}")
                st.write(f"***Max Loss streak***: {Analysis.max_consecutive(Analysis.csv_data, -1)}")
                st.write(f"***Sharpe Ratio:*** {Analysis.Sharpe()}")
                st.write(f"***Calmar Ratio:*** {Analysis.Calmar()}")
                st.write(f"***Sortino Ratio:*** {Analysis.Sortino()}")
                st.write(f"Win Rate for ***last Week***: {week}")                
                st.write(f"Win Rate for ***last Month***: {month}")                
                st.write(f"Win Rate for ***last Quater:*** {quat}")                
                st.write(f"Win Rate for ***last 6 Months***: {half}")                
                st.write(f"Win Rate for ***last Year***: {yr}")
                
            with c2:        
                labels = np.array(['Profitable Trades', 'Loss Making Trades', 'Short Trades', 'Long Trades'])
                vals = np.array([[Analysis.num_loss(Analysis.csv_data, 1), Analysis.num_loss(Analysis.csv_data, -1)],[ Analysis.num_tradeType('short'), Analysis.num_tradeType('long')]])
                size = 0.3
                
                figp, axp = plt.subplots()
                outer, _ = axp.pie(vals[0], radius=1, wedgeprops=dict(width=size, edgecolor='w'))
                inner,_ = axp.pie(vals[1], radius=1-size, wedgeprops=dict(width=size, edgecolor='w'))
                axp.set(aspect="equal", title='Trade Analysis')
                wedge = outer + inner
                axp.legend(wedge, labels, title="Trade Types", loc="upper left")
                st.pyplot(figp)
                
                st.write(f"Number of ***short trades***: {Analysis.num_tradeType('short')}")
                st.write(f"Number of ***long trades***: {Analysis.num_tradeType('long')}")
                st.write(f"Number of ***wins***: {Analysis.num_loss(Analysis.csv_data, 1)}")
                st.write(f"Number of ***losses***: {Analysis.num_loss(Analysis.csv_data, -1)}")
                st.write(f"***Total*** Number of Trades{len(Analysis.csv_data)}")   
                
            categories = ['Last Week', 'Last Month', 'Last Quater', 'Last 6 Months', 'Last Year']
            win_rates = [week, month, quat, half, yr]
            fp, ap = plt.subplots(figsize=(10, 3))
            ap.bar(categories, win_rates, color='b')
            ap.set_xlabel('Time Period')
            ap.set_ylabel('Win Rate (%)')
            ap.set_title('Win Rate for Different Time Periods')
            for i in range(len(categories)):
                ap.text(i, win_rates[i], f'{win_rates[i]:.2f}%', ha='center', va='bottom')
            ap.set_xticklabels(labels=categories, rotation=45)  # Rotate x-axis labels for better visibility
            st.pyplot(fp)
            #plt.tight_layout()  # Adjust layout to prevent overlapping labels
 
    with tab1:
        co1, co2,co3,co4,co5,co6 = st.columns([5,1,1,1,1,1])
        with co2:
            a = st.button("Daily", use_container_width=True, on_click=click_button_disp, args=["Daily", daily_returns])
        with co3:
            a = st.button("Monthly", use_container_width=True, on_click=click_button_disp, args=["Monthly", monthly_returns])
        with co4:
            a = st.button("Weekly", use_container_width=True, on_click=click_button_disp, args=["Weekly", weekly_returns])
        with co5:
            a = st.button("Yearly", use_container_width=True, on_click=click_button_disp, args=["Yearly", yearly_returns])
        with co6:
            a = st.button("Day", use_container_width=True, on_click=click_button_disp, args=["Day"])          
        if st.session_state.button:
            if st.session_state["Time"] != "Day":
                daisply(st.session_state['rt'], st.session_state["Time"], Analysis)
            else:
                display(weekday_returns, Analysis)
        else:
            Analysis.htmap()
            
if not st.session_state.clicked:
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
            analysis = Analysis
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
            tile.button("Execute", key=stratergy, use_container_width=True, on_click=click_button_arg, args=[Analysis, stratergy])
        
if st.session_state.clicked:
    next_page(st.session_state['ana'], st.session_state['stra'])
    with st.sidebar:
        st.button("Return to cards", on_click=click_button)
                    











