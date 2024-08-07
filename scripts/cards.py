
import streamlit as st
import matplotlib.pyplot as plt  
from stratrgy_analysis import StatergyAnalysis
import pandas as pd
import numpy as np
import base64
import sqlite3
import pickle
import seaborn as sn
from  matplotlib.colors import LinearSegmentedColormap
from datetime import datetime
import os
from sql import insert_sql, delete_id, calc, append_sql
import warnings
from customerPLBook_analysis import customerPLBook_Analysis

warnings.filterwarnings("ignore", category=UserWarning, message=".*experimental_allow_widgets.*")
st.set_page_config(layout="wide")


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
if 'index' not in st.session_state:
    st.session_state['index'] = None 
    
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

def click_button_arg(a,b,c):
    st.session_state.clicked = not st.session_state.clicked
    st.session_state['ana'] = a
    st.session_state['stra'] = b
    st.session_state['index'] = c

@st.cache_data(show_spinner=False, ttl=86400)
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

@st.cache_data(show_spinner=False, ttl=86400)
def htmap(data, days):
    data = np.array(data['pnl_absolute'].tolist())
    data = data[-1 * days:]
    m = 5 * int(len(data) / 5)
    data = data[:m]
    data = np.reshape(data, (5, -1))
    line_width = 0.8
    linecolor = "White"
    box_width = int(m * 2 / 25) + 2

    c = ["darkred", "red", "lightcoral", "white", "palegreen", "green", "darkgreen"]
    v = [0, .15, .4, .5, 0.6, .9, 1.]
    l = list(zip(v, c))
    cm = LinearSegmentedColormap.from_list('rg', l, N=256)

    hm, ax = plt.subplots(figsize=(box_width, 2), dpi=400)
    sn.heatmap(data=data, linecolor=linecolor, linewidths=line_width, cmap=cm, center=0, ax=ax)
    plt.savefig("temp_plot.png", bbox_inches='tight', pad_inches=0)

    # Load the saved image
    with open("temp_plot.png", "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode()

    
    height = 200
    container = st.container(height = 220, border=True)
    # HTML to display image with horizontal scroll
    html_code = f'''
    <div style="height:{height}px; overflow-x: sroll;">
            <img src="data:image/png;base64,{img_base64}"; height="100%"; width="auto">
    </div>
    '''
    container.write(html_code, unsafe_allow_html=True)

@st.cache_data(show_spinner=False, ttl=86400)
def freq_hist(profit, num):
    num.insert(0, "")  
    num.append("")  
    
    n = len(profit)
    r = np.arange(1, n + 1)  
    width = 0.75

    fig, ax = plt.subplots()
    ax.bar(r - 0.5, profit, color='b', width=width, align='center', label='Profit')
    for index, value in enumerate(profit):
        if value != 0:
            ax.text(index + 0.5, value, str(value), ha='center')
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_xticks(np.arange(len(num)))
    ax.set_xticklabels(num, rotation=45)
    ax.legend()

    return fig

@st.cache_data(show_spinner=False, ttl=86400)
def daisply(daily_returns, period, csv):
    strings = []
    values = []
    if period == "Day":
        st.header("Daily Analysis")
    else:
        strings.extend([f"{period}ly Average Returns", f"{period}ly Average Returns %" ])
        values.extend([daily_returns[1][0], daily_returns[1][1]])
        st.header(f"{period}ly Analysis")
        st.write(f"***{period}ly Average Returns***: {daily_returns[1][0]}")
        st.write(f"***{period}ly Average Returns %***: {daily_returns[1][1]}%")
    
    freq_hist_py = freq_hist(daily_returns[2], [-5000,-4000, -3000, -2000, -1000, 0, 1000, 2000, 3000, 4000, 5000])
    st.subheader("Frequency of profits")
    st.pyplot(freq_hist_py)
    
    with st.expander("More information"):
        strings.extend([f"Number of trading {period}s", f"Number of Profitable {period}s", f"Number of Loss Making {period}s", f"Most Profitable {period}", f"Maximum Gains in a {period}", f"Least Profitable {period}", f"Maximum Loss in a {period}"])
        values.extend([daily_returns[3], daily_returns[4], daily_returns[5], daily_returns[6][1], daily_returns[6][0], daily_returns[7][1], daily_returns[7][0]])

        if period == "Day":
            strings.extend(["Max Win Streak", "Max Loss Streak"])
            values.extend([daily_returns[8], daily_returns[9]])
            
        arr = np.array([strings, values]).T
        df = pd.DataFrame(arr, columns=["", " "])
        st.table(df.assign(hack='').set_index('hack'))
        
    if daily_returns[3]>=252:
        st.subheader(f"Profit/Loss Data per {period} for last year")
        st.bar_chart(csv[-252:], y=['pnl_absolute'], width=500, height=400)
    else:
        st.subheader(f"Profit/Loss Data per {period}")
        st.bar_chart(csv, y=['pnl_absolute'], width=500, height=400)
    if 'cum_pnl' in csv.columns:
        st.subheader("Cumulative Profit and loss")
        st.line_chart(csv, y=['cum_pnl'])
 
@st.cache_data(show_spinner=False, ttl=86400)
def display(weekday_returns, q):
    st.subheader(f"Profit/Loss Data per Day of Week")
    st.bar_chart(q[5], y=['pnl_absolute'] )
    st.write(f"***Most Profitable Day*** of the week: {weekday_returns[0][1]}")
    st.write(f"***Least Profitable Day*** of the week: {weekday_returns[1][1]}")
    table_data = q[5]['pnl_absolute']
    st.table(table_data)

@st.cache_data(show_spinner=False, ttl=86400)
def daily_returns_hist(daily_returns):
        fig1, ax1 = plt.subplots(figsize=(10, 2))  
        ax1.bar(daily_returns.index, daily_returns['pnl_absolute'])
        ax1.set_xticks([])
        ax1.set_yticks([])

        fig2, ax2 = plt.subplots(figsize=(10, 5))  
        ax2.bar(daily_returns.index, daily_returns['cum_pnl'])
        ax2.set_xticklabels([])
        
        return fig1, fig2


@st.cache_data(show_spinner=False, ttl=86400)
def is_valid_datetime(input_str):
    try:
        datetime.strptime(input_str, "%Y-%m-%d")
        return True
    except ValueError:
        try:
            datetime.strptime(input_str, "%d-%m-%Y")
            return True
        except ValueError:
            return False
        
@st.cache_data(show_spinner=False, ttl=86400)
def entry_find_nearest_date(data, target_date, entry_data_col_index):
    target_date_str = target_date.strftime("%Y-%m-%d %H:%M:%S")
    date_col = list(data[entry_data_col_index])
    for i in range(len(date_col)):
        if date_col[i] >= target_date_str:
            return i

@st.cache_data(show_spinner=False, ttl=86400)     
def exit_find_nearest_date(data, target_date, entry_data_col_index):
    target_date_str = target_date.strftime("%Y-%m-%d %H:%M:%S")
    date_col = list(data[entry_data_col_index])[::-1]
    for i in range(len(date_col)):
        if date_col[i] <= target_date_str:
            return len(date_col)-i-1;

@st.cache_data(show_spinner=False, ttl=86400)
def get_data_using_path(csv_path):
    data = pd.read_csv(csv_path)
    return data;

@st.cache_data(show_spinner=False, ttl=86400)
def get_analysis_obj(data, stn):
    row = calc(data, is_dataframe=1, filename=stn)
    return row;        

@st.cache_data(show_spinner=False, ttl=86400)     
def get_analysis_with_initial_invest(data, initial_investment, stn):
    Analysis = calc(data, is_dataframe=1, initial_inestent=initial_investment, filename=stn)
    return Analysis;    

def next_page(q, stratergy, i):
    
    data = q[1]
    print(data.columns)
    row = list(data.iloc[0])
    entry_data_col_index = "entry_timestamp";
    for j in range(len(row)):
        tempstr = str(row[j]).split(" ")
        if is_valid_datetime(tempstr[0]):
            entry_data_col_index = data.columns[j]
            break;
    data = data.sort_values(by=entry_data_col_index)
    try:
        date_format = "%Y-%m-%d"
        startdate = datetime.strptime(data.loc[:, entry_data_col_index].iloc[0].split(" ")[0], date_format)
        enddate = datetime.strptime(data.loc[:, entry_data_col_index].iloc[-1].split(" ")[0], date_format)
    except:
        date_format = "%d-%m-%Y"
        startdate = datetime.strptime(data.loc[:, entry_data_col_index].iloc[0].split(" ")[0], date_format)
        enddate = datetime.strptime(data.loc[:, entry_data_col_index].iloc[-1].split(" ")[0], date_format)
        
    custom_aligned_text = f"""
        <div style="display: flex; justify-content: space-between; color: green; font-size: 24px;">
        <span style="text-align: left;">   </span>
        <span style="text-align: center; margin-top: -30px;"><strong>{stratergy}</strong></span>
        <span style="text-align: right; color: green"> </span>
        </div>
        """
    mark = f"""
        <div style="display: flex; justify-content: space-between; color: grey; font-size: 12px;">
        <span style="text-align: left;">   </span>
        <span style="text-align: right;"></span>
        </div>
        """
    st.write(custom_aligned_text, unsafe_allow_html=True)
    st.markdown(mark, unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1.7, 3])
    with col1:
        st.write("")
    with col2:
        
        st.markdown(
            """
            <style>
            .stDateInput > label {
                display: flex;
                justify-content: center;
                font-size: 18px;
                margin-top: -30px;
                
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        selected_date_entry = st.date_input(
        "Entry Date",
        startdate,
        key=f"entrydate{i}",
        min_value=startdate,
        max_value=enddate,

    )
    with col3:
        st.markdown(
            """
            <style>
            .stDateInput > label {
                display: flex;
                justify-content: center;
                font-size: 18px;
                margin-top: -30px;
                
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        selected_date_exit = st.date_input(
        "Exit Date",
        enddate,
        key=f"exitdate{i}",
        min_value=startdate,
        max_value=enddate,

    )
    with col4:
        # Create the number input box
        # Inject custom CSS to center the label
        st.markdown(
            """
            <style>
            .stNumberInput > label {
                display: flex;
                justify-content: center;
                font-size: 18px;
                margin-top: -30px;
                
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Create the number input
        initial_investment = st.number_input('Initial Investment', key=f"numbox{i}", min_value=0, step=10000, format='%d', value=150000)

    with col5:
        st.write("")

    entry_date_index = entry_find_nearest_date(data, selected_date_entry, entry_data_col_index)
    exit_date_index = exit_find_nearest_date(data, selected_date_exit, entry_data_col_index)
    if entry_date_index > exit_date_index:
        entry_date_index = 0
        exit_date_index = len(data)-1

    subcol1, subcol2, subcol3 = st.columns([1.8, 1, 1])
    with subcol1:
        st.write("")
    with subcol2:
        if st.button("Submit"):
            if entry_date_index != 0 or exit_date_index != len(data) -1 or initial_investment != 150000:
                data = data.iloc[entry_date_index:exit_date_index+1, :].copy()
                q = get_analysis_with_initial_invest(data, initial_investment, stratergy)
    with subcol3:
        st.write("")

    st.title("Analysis")
    # Analysis = get_analysis(Analysis)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.caption("Stratergy Code")
        st.subheader(stratergy)
    with col2:
        st.caption("Transaction Fee (with Package)")
        st.subheader("0%")
    with col3:
        st.caption("Transaction Fee (w/o Package)")
        st.subheader(""":red[3%]""")
    with col4:
        st.caption("Recomended Days")
        st.subheader(""":green[180]""")
    
    tab1, tab2, tab3 = st.tabs(["Records", "Analytics", "Returns"])
    
    with tab3:
        Dur = [252*2, 252, 101,11, 22, 4]
        Duration = ['All Time', ' 2 Years', '1 year', '180 Days', '30 Days', '15 Days', '3 Days']
        returns = [f"{q[19][1]}%", f"{q[44]}%", f"{q[43]}%", f"{q[42]}%", f"{q[40]}%", f"{q[41]}%", f"{q[39]}%"]
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
            week = q[32]
            month = q[31]
            quat = q[35]
            half = q[34]
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
                
                if isinstance(q[24], bytes):   
                    q[24] = int.from_bytes(q[24], byteorder='little')      
                if isinstance(q[25], bytes):   
                    q[25] = int.from_bytes(q[25], byteorder='little')
                
                strings = [
                    "Average loss per losing trade: ",
                    "Average gain per winning trade: ",
                    "Maximum Gains: ",
                    "Minimum Gains: ",
                    "Average Trades per Day: ",
                    "HIT Ratio: ",
                    "Profit Factor: ",
                    "Yearly Volatility: ",
                    "Max Win Streak: ",
                    "Max Loss streak: ",
                    "Sharpe Ratio: ",
                    "Calmar Ratio: ",
                    "Sortino Ratio: ",
                    "Win Rate for last Week: ",
                    "Win Rate for last Month: ",
                    "Win Rate for last Quarter: ",
                    "Win Rate for last 6 Months: ",
                    "Win Rate for last Year: "
                ]


                variables = [
                    avg_profit,
                    avg_loss,
                    values[0],
                    values[1],
                    q[15],
                    q[18],
                    q[20],
                    q[23],
                    q[24],
                    q[25],
                    q[36],
                    q[37],
                    q[38],
                    week,
                    month,
                    quat,
                    half,
                    yr
                ]

                arr = np.array([strings, variables]).T
                df = pd.DataFrame(arr, columns=["Duration", "Returns"])

                st.table(df.assign(hack='').set_index('hack'))                
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
                
                some_numbers = [
                    "Number of short trades: ",
                    "Number of long trades: ",
                    "Number of wins: ",
                    "Number of losses: ",
                    "Total Number of Trades: "
                ]

                num_data_x = [
                    vals[1][0],
                    vals[1][1],
                    vals[0][0],
                    vals[0][1],
                    vals[1][0] + vals[1][1]
                ]

                arr = np.array([some_numbers, num_data_x]).T
                df = pd.DataFrame(arr, columns=["Duration", "Returns"])

                st.table(df.assign(hack='').set_index('hack'))    
                
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
        st.subheader("All Time Heatmap")
        days = st.slider("Select number of days", min_value=5, max_value=1000, value=100, step=5)
        htmap(q[2], days)        
        
        co1, co2,co3,co4,co5,co6 = st.columns([5,1,1,1,1,1])
        with co2:
            a = st.button("Daily", use_container_width=True, on_click=click_button_disp, args=["Day", q[47], q[2]])
        with co4:
            a = st.button("Monthly", use_container_width=True, on_click=click_button_disp, args=["Month", q[48], q[3]])
        with co3:
            a = st.button("Weekly", use_container_width=True, on_click=click_button_disp, args=["Week", q[49], q[4]])
        with co5:
            a = st.button("Yearly", use_container_width=True, on_click=click_button_disp, args=["Year", q[51], q[6]])
        with co6:
            a = st.button("Weekday", use_container_width=True, on_click=click_button_disp, args=["WeekDay",q[50],q[5]])          
        if st.session_state.button:
            if st.session_state["Time"] != "WeekDay":
                daisply(st.session_state['rt'], st.session_state["Time"], st.session_state['i'])
            else:
                display(q[50], q)

def save_uploaded_file(uploaded_file, save_directory, file_name):
    # Create the save directory if it does not exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    # Save the uploaded file to the specified directory
    file_path = os.path.join(save_directory, file_name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

@st.cache_data(show_spinner=False, ttl=86400)
def get_files(names):
    Files = [row[0] for row in names]
    return Files
        
def CustomerPLBook():
    customerPLBook_analysis_streamlit = customerPLBook_Analysis()
    customerPLBook_analysis_streamlit.run()

def home():
    if not st.session_state.clicked:
        conn = sqlite3.connect('strategy_analysis.db')
        cursor = conn.cursor()
        cursor.execute('SELECT Id FROM StrategyData ')
        names  = cursor.fetchall()
        Files = get_files(names)
        
        num_file = len(Files)+1
        num_row = num_file//3
        if num_file%3 != 0:
            num_row += 1

        rows = None
        for i in range(num_row):
            row1 = st.columns(3)
            if rows is None:
                rows = row1
            else:
                rows += row1

        if rows is not None:
            i = 0
            for col in rows:
                if i < len(Files):
                    stratergy = Files[i]
                    tile = col.container(height=410, border=True)
                    with tile:
                        col1,col2, col3, col4 = st.columns([0.35,0.15, 0.3, 0.2]) 

                        with col1:
                            st.write("By Algobulls") 

                        with col3:
                            with st.popover("Append Data"):
                                uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key=f"append{i}")        
                                if st.button("Submit", key=f"append_{stratergy}"):
                                        if uploaded_file is not None:
                                            csv_data = pd.read_csv(uploaded_file)
                                            append_sql(csv_data, is_dataframe=1, filename=stratergy)
                                            st.rerun()    

                        with col4:
                            delete_button = st.button("Delete", key=f"delete{i}")
                            if delete_button:
                                delete_id(Files[i])
                                st.rerun()
                    
                    # csv_path = f"files/StrategyBacktestingPLBook-{stratergy}.csv"
                    # data = get_data_using_path(csv_path)
                    # Analysis = get_analysis_obj(data)
                    i += 1
                    cursor.execute('SELECT * FROM StrategyData WHERE Id = ?', (stratergy,))
                    q  = cursor.fetchone()

                    pick = [1,2,3,4,5,6,9,10,19,30,47,48,49,50,51]
                    q = list(q)
                    for p in pick: 
                        q[p] = pickle.loads(q[p])

                    data = q[1]
                    custom_aligned_text = f"""
                    <div style="display: flex; justify-content: space-between;">
                    <span style="text-align: left;">{stratergy}</span>
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
                    tile.button("Execute", key=stratergy, use_container_width=True, on_click=click_button_arg, args=[q, stratergy, i])
                
                else:
                    tile = col.container(height=410, border=True)
                    centered_red_bold_large_text = """
                    <div style='display: flex; justify-content: center;'>
                        <span style='color:green; font-size:30px;'><strong>ADD YOUR STRATEGY</strong></span>
                    </div>
                    """
                    tile.markdown(centered_red_bold_large_text, unsafe_allow_html=True)
                    
                    with tile:
                        st.markdown("""
                            <style>
                            ::placeholder {
                                text-align: center;
                            }
                            </style>
                        """, unsafe_allow_html=True)

                        user_input = st.text_input("", placeholder="Enter Name of the Strategy")
                    
                    with tile:
                        uploaded_file = st.file_uploader("", type="csv")
                        col1, col2 = st.columns([0.37, 0.63]) 
                        with col2:
                            if st.button("Submit"):
                                if uploaded_file is not None and user_input is not None:
                                    file_name = f"StrategyBacktestingPLBook-{user_input}.csv"
                                    data = pd.read_csv(uploaded_file)
                                    insert_sql(data, 1, user_input)
                                    st.rerun()



                    tile.write("\n")
                    break

    if st.session_state.clicked:
        next_page(st.session_state['ana'], st.session_state['stra'], st.session_state['index'])
        with st.sidebar:
            st.button("Return to cards", on_click=click_button_return)


tab_home, tab_customer = st.tabs(["Home", "CustomerPLBook Analysis"])
with tab_home:
    home()
with tab_customer:
    CustomerPLBook()



