
import re
import glob
import streamlit as st
import matplotlib.pyplot as plt  
import seaborn as sns
from stratrgy_analysis import StatergyAnalysis
import pandas as pd
import numpy as np
import base64
import sqlite3
import pickle
import seaborn as sn
from  matplotlib.colors import LinearSegmentedColormap
from sql import insert_sql, delete_id

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
if 'i' not in st.session_state:
    st.session_state['i'] = None
    
def click_button():
    st.session_state.clicked = not st.session_state.clicked
    
def click_button_return():
    st.session_state.clicked = not st.session_state.clicked
    st.session_state.button = False
    
def click_button_disp(s, r, i):
    st.session_state.button = True
    st.session_state['Time'] = s
    st.session_state['rt'] = r
    st.session_state['i'] = i
    
def click_button_arg(a,b):
    st.session_state.clicked = not st.session_state.clicked
    st.session_state['ana'] = a
    st.session_state['stra'] = b

def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
    
def htmap(data, days):        
    data = np.array(data['pnl_absolute'].tolist())
    data = data[-1 * days:]
    m = 5 * int(len(data)/ 5)
    data = data[:m]
    data = np.reshape(data, (5, -1))
    line_width = 0.8
    linecolor = "White"
    box_width = int(m*2/25) + 2

    c = ["darkred","red","lightcoral","white", "palegreen","green","darkgreen"]
    v = [0,.15,.4,.5,0.6,.9,1.]
    l = list(zip(v,c))
    cm=LinearSegmentedColormap.from_list('rg',l, N=256)

    hm, ax = plt.subplots(figsize=(box_width,2), dpi=400)
    sn.heatmap(data=data, linecolor=linecolor, linewidths=line_width, cmap=cm, center=0, ax=ax) 

    st.write(hm)
    return box_width

def freq_hist(profit, num):
    num.insert(0, "")  
    num.append("")  
    
    n = len(profit)
    r = np.arange(1, n + 1)  
    width = 0.75

    fig, ax = plt.subplots()
    ax.bar(r - 0.5, profit, color='b', width=width, align='center', label='Profit')
    for index, value in enumerate(profit):
        ax.text(index + 0.5, value + 1, str(value), ha='center')
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_xticks(np.arange(len(num)))
    ax.set_xticklabels(num, rotation=45)
    ax.legend()

    return fig

def daisply(daily_returns, Quant, col):
    if Quant == "Day":
        st.header("Daily Analysis")
    else:
        st.header(f"{Quant}ly Analysis")
        st.write(f"***{Quant}ly Average Returns***: {daily_returns[1][0]}")
        st.write(f"***{Quant}ly Average Returns %***: {daily_returns[1][1]}%")
    
    #days_hist, days_tab = Alanyze.compare_hist(daily_returns, [1000, 2000, 3000, 4000, 5000], Quant)
    freq_hist_py = freq_hist(daily_returns[2], [-5000,-4000, -3000, -2000, -1000, 0, 1000, 2000, 3000, 4000, 5000])


    st.subheader("Frequency of profits")
    st.pyplot(freq_hist_py)
    
    with st.expander("More information"):
        st.write(f"Number of ***trading {Quant}s***: {daily_returns[3]}")
        st.write(f"Number of ***Profitable {Quant}s***: {daily_returns[4]} {Quant}")
        st.write(f"Number of ***Loss Making {Quant}s***: {daily_returns[5]} {Quant} ")
        st.write(f"***Most Profitable {Quant}***: {daily_returns[6][1]}")
        st.write(f"Maximum ***Gains*** in a {Quant}: {daily_returns[6][0]}")
        st.write(f"***Least Profitable {Quant}***: {daily_returns[7][1]}")
        st.write(f"Maximum ***Loss*** in a {Quant}: {daily_returns[7][0]}")
        if Quant == "Day":
            st.write(f"***Max Win Streak***: {daily_returns[8]}")
            st.write(f"***Max Loss streak***: {daily_returns[9]}")

    st.subheader(f"Profit/Loss Data per {Quant}")
    st.bar_chart(col, y=['pnl_absolute'], width=500, height=800)
    if 'cum_pnl' in col.columns:
        st.subheader("Cumulative Profit and loss")
        st.line_chart(col, y=['cum_pnl'])
    st.write(f"")
    st.divider()
 
def display(weekday_returns, q):
    st.subheader(f"Profit/Loss Data per Day of Week")
    st.bar_chart(q[5], y=['pnl_absolute'] )
    st.write(f"***Most Profitable Day*** of the week: {weekday_returns[0][1]}")
    st.write(f"***Least Profitable Day*** of the week: {weekday_returns[1][1]}")
    tab = q[5]['pnl_absolute']
    st.table(tab)

def daily_returns_hist(daily_returns):
        fig1, ax1 = plt.subplots(figsize=(10, 2))  
        ax1.bar(daily_returns.index, daily_returns['pnl_absolute'])
        ax1.set_xticks([])
        ax1.set_yticks([])

        fig2, ax2 = plt.subplots(figsize=(10, 5))  
        ax2.bar(daily_returns.index, daily_returns['cum_pnl'])
        ax2.set_xticklabels([])
        
        return fig1, fig2
    
def next_page( q, code):
    
    st.title("Analysis")
    #daily_returns, q[3], weekday_returns, weekly_returns, yearly_returns = Analysis.analysis()

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
        Dur = [252*2, 252, 101,11, 22, 4]
        Duration = ['All Time', ' 2 Years', '1 year', '180 Days', '15 Days', '30 Days', '3 Days']
        returns = [f"{q[19][1]}%"]
        for i in range(len(Dur)):
            returns.append(f"{q[44-i]}%")
        arr = np.array([Duration, returns]).T
        df = pd.DataFrame(arr, columns=["Duration", "Returns"])
        
        st.dataframe(df, hide_index=True, use_container_width=True)
        
        
    with tab2:
        bt1 , bt2, bt3, bt4, bt5 = st.tabs(["Stats", "P&L", "ROI%", "Equity Curve", "Drawdown %"])
        
        with bt5:
            st.subheader("Drawdown Curve")
            st.line_chart(q[1], y='drawdown_percentage', x='Day')
            st.write(f"***Max Drawdown***: {q[7]}")
            st.write(f"***Maximum Drawdowm percentage***: {q[8]}")
            
        with bt4:
            st.subheader("Equity Curve")
            st.line_chart(q[1], y='equity_curve')
            
        with bt3:
            st.subheader("ROI% Curve")
            st.line_chart( q[2], y='roi' )
            st.write(f"***ROI***: {q[19][0]}")
            st.write(f"***ROI %***: {q[19][1]}%")
            
            st.subheader("Monthly Returns and ROI% Over Time")
            fig, ax1 = plt.subplots()

            ax1.bar(q[3].index.values, q[3]['cum_pnl'].values, color='b', alpha=0.6, label='Monthly Returns')
            ax1.set_xlabel('Month')
            ax1.set_ylabel('Monthly Returns', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.set_xticks(q[3].index[::3])
            ax1.set_xticklabels(q[3].index[::3], rotation=90)
            
            ax2 = ax1.twinx()
            ax2.plot(q[3].index.values, q[3]['roi'].values, color='r', marker='o', label='ROI%')
            ax2.set_ylabel('ROI%', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
            st.pyplot(fig)
            
        with bt2:
            st.subheader("Daily P&L")
            st.bar_chart(q[2], y=['pnl_absolute'])
            st.subheader("Cumulative P&L")
            st.line_chart(q[2], y=['cum_pnl'])
            
        with bt1:
            #last_month_data = daily_returns.iloc[-6:]
            week = q[32]
            #last_month_data = daily_returns.iloc[-21:]
            month = q[31]
            #last_month_data = daily_returns.iloc[-59:]
            quat = q[35]
            #last_month_data = daily_returns.iloc[-101:]
            half = q[34]
            #last_month_data = daily_returns.iloc[-252:]
            yr = q[33]
            
            c1, c2 = st.columns(2)
            with c1:
                fig0, ax0 = plt.subplots(figsize=(10,2))
                values = [q[11], q[12]]
                x = ["Maximum Profit", "Minimum Profit"]
                colors = ['r' if value < 0 else 'g' for value in values]
                ind = np.arange(2)
                width = 0.05
                
                bars = ax0.barh(ind, values, color=colors)
                ax0.bar_label(bars, label_type="center")
                avg_profit = q[10][0]
                avg_loss =q[9][0]
                ax0.axvline(avg_profit, ls='--', ymax=0.5, color='k', label='Avg Profit')
                ax0.axvline(avg_loss, ymin= 0.5, ls='--', color='k', label='Avg Loss')
                ax0.text(avg_profit, 0.75, f'Avg Profit: {avg_profit}', color='k', va='top', ha='left')
                ax0.text(avg_loss, 0.25, f'Avg Loss: {avg_loss}', color='k', va='bottom', ha='right')
                ax0.set_yticks(ind+width/2)
                ax0.set_yticklabels(x, minor=False)
                ax0.set_title('Profit/Loss Analysis')
                st.pyplot(fig0)
                max_win_streak = int.from_bytes(q[24], byteorder='little')   
                max_loss_streak = int.from_bytes(q[25], byteorder='little')
                
                st.write(f"***Average loss per losing trade***: {avg_profit}")
                st.write(f"***Average gain per winning trade***: {avg_loss}")
                st.write(f"***Maximum Gains***: {values[0]}")
                st.write(f"***Minimum Gains***: {values[1]}")
                st.write (f"***Average Trades per Day***: {q[15]}")
                st.write("\n")
                st.write(f"***HIT Ratio***: {q[18]}")
                st.write(f"***Profit Factor***: {q[20]}")
                st.write(f"***Yearly Volatility***: {q[23]}")
                st.write(f"***Max Win Streak***: {max_win_streak}")
                st.write(f"***Max Loss streak***: {max_loss_streak}")
                st.write(f"***Sharpe Ratio:*** {q[36]}")
                st.write(f"***Calmar Ratio:*** {q[37]}")
                st.write(f"***Sortino Ratio:*** {q[38]}")
                st.write(f"Win Rate for ***last Week***: {week}")                
                st.write(f"Win Rate for ***last Month***: {month}")                
                st.write(f"Win Rate for ***last Quater:*** {quat}")                
                st.write(f"Win Rate for ***last 6 Months***: {half}")                
                st.write(f"Win Rate for ***last Year***: {yr}")
                
            with c2:        
                labels = np.array(['Profitable Trades', 'Loss Making Trades', 'Short Trades', 'Long Trades'])
                vals = np.array([[q[16], q[17]],[q[13], q[14]]])
                size = 0.3
                
                figp, axp = plt.subplots()
                outer, _ = axp.pie(vals[0], radius=1, wedgeprops=dict(width=size, edgecolor='w'))
                inner,_ = axp.pie(vals[1], radius=1-size, wedgeprops=dict(width=size, edgecolor='w'))
                axp.set(aspect="equal", title='Trade Analysis')
                wedge = outer + inner
                axp.legend(wedge, labels, title="Trade Types", loc="upper left")
                st.pyplot(figp)
                
                st.write(f"Number of ***short trades***: {vals[1][0]}")
                st.write(f"Number of ***long trades***: {vals[1][1]}")
                st.write(f"Number of ***wins***: {vals[0][1]}")
                st.write(f"Number of ***losses***: {vals[0][0]}")
                st.write(f"***Total*** Number of Trades{vals[1][1] + vals[1][0]}")   
                
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
        
        htmap(q[2], 90)        
        
        co1, co2,co3,co4,co5,co6 = st.columns([5,1,1,1,1,1])
        with co2:
            a = st.button("Daily", use_container_width=True, on_click=click_button_disp, args=["Day", q[47], q[2]])
        with co3:
            a = st.button("Monthly", use_container_width=True, on_click=click_button_disp, args=["Month", q[48], q[3]])
        with co4:
            a = st.button("Weekly", use_container_width=True, on_click=click_button_disp, args=["Week", q[49], q[4]])
        with co5:
            a = st.button("Yearly", use_container_width=True, on_click=click_button_disp, args=["Year", q[51], q[6]])
        with co6:
            a = st.button("Day", use_container_width=True, on_click=click_button_disp, args=["WeekDay"])          
        if st.session_state.button:
            if st.session_state["Time"] != "WeekDay":
                daisply(st.session_state['rt'], st.session_state["Time"], st.session_state['i'])
            else:
                display(q[50], q)
        
            
if not st.session_state.clicked:
    st.set_page_config(layout="wide")
    # path = "files/StrategyBacktestingPLBook-*.csv"
    # print("\nUsing glob.iglob()")
    # Files = []

    # for file in glob.glob(path, recursive=True):
    #     found = re.search('StrategyBacktestingPLBook(.+?)csv', str(file)).group(1)[1:-1]
    #     Files.append(found)
    conn = sqlite3.connect('strategy_analysis.db')
    cursor = conn.cursor()
    cursor.execute('SELECT Id FROM StrategyData ')
    names  = cursor.fetchall()
    Files = [row[0] for row in names]

    default = 150000
    row1 = st.columns(3)
    row2 = st.columns(3)
    i = 0
    for col in row1 + row2:
        tile = col.container(height=400, border=True)
        tile.write("By Algobulls")
        if i < len(Files):
            stn = Files[i]
            #csv_path = f"files/StrategyBacktestingPLBook-{stratergy}.csv"
            #Analysis = StatergyAnalysis(csv_path)
            #analysis = Analysis
            i += 1
            cursor.execute('SELECT * FROM StrategyData WHERE Id = ?', (stn,))
            q  = cursor.fetchone()

            pick = [1,2,3,4,5,6,9,10,19,30,47,48,49,50,51]
            q = list(q)
            for p in pick: 
                q[p] = pickle.loads(q[p])
                
            #daily_returns, q[3], weekday_redddddcturns, weekly_returns, yearly_returns = Analysis.analysis()

            custom_aligned_text = f"""
            <div style="display: flex; justify-content: space-between;">
            <span style="text-align: left;">{stn}</span>
            <span style="text-align: right; color: green">{q[19][1]}</span>
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
            pnl, cum_pnl = daily_returns_hist(q[3])
            tile.pyplot(pnl)

            ratios = f"""
            <div style="display: flex; justify-content: space-between;">
            <span style="text-align: left;">{q[36]}</span>
            <span style="text-align: center; flex-grow: 1; text-align: center;"> {q[37]}</span>
            <span style="text-align: right;">{q[38]}</span>
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
            <span style="text-align: left;">{q[28]}</span>
            <span style="text-align: center; flex-grow: 1; text-align: center;"> {q[18]}</span>
            <span style="text-align: right; color: red;">{q[8]}%</span>
            </div>
            """
            tile.write("\n")
            tile.markdown(head_, unsafe_allow_html=True)
            tile.write(disp, unsafe_allow_html=True)
            tile.write("\n")
            tile.button("Execute", key=stn, use_container_width=True, on_click=click_button_arg, args=[q, stn])
        
if st.session_state.clicked:
    with st.sidebar:
        st.button("Return to cards", on_click=click_button_return)
        # number = st.number_input("Enter initial investment", value=150000)
        # st.write (f"Inital investment {number}") 
        # Analysis.initial_investment = number 
        # st.write()
    next_page(st.session_state['ana'], st.session_state['stra'])












