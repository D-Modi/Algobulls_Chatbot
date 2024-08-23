
from streamlit.errors import StreamlitAPIException
import streamlit as st
import matplotlib.pyplot as plt  
import pandas as pd
from  matplotlib.colors import LinearSegmentedColormap
import os
import warnings
import base64
from datetime import datetime
from calculations import calc
import numpy as np
import seaborn as sn
import re
import glob
from stratrgy_analysis import StatergyAnalysis

warnings.filterwarnings("ignore", category=UserWarning, message=".*experimental_allow_widgets.*")
def set_page_config():
    if 'page_config_set' not in st.session_state:
        st.set_page_config(layout="wide")
        st.session_state['page_config_set'] = True
        
set_page_config()
    
def click_button():
    st.session_state.clicked = not st.session_state.clicked
    
def click_button_return():
    print("Return to Home")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    st.session_state.clicked = False
    st.session_state.button = False
    st.session_state['new_q'] = None
    # st.session_state['ana'] = None
    # st.session_state['stra'] = None
    # st.session_state['index'] = None

def click_button_disp(s, r, i):
    st.session_state.button = True
    st.session_state['Time'] = s
    st.session_state['rt'] = r
    st.session_state['i'] = i

def click_button_arg(a,b,c):
    st.session_state.clicked = True
    st.session_state['ana'] = a
    st.session_state['stra'] = b
    st.session_state['index'] = c
    print("#########################")
    print("Execute bittom clicked")
    
def click_button_done(): 
    if(len(st.session_state['options']) == 0):
        st.write("No strategies Seleted")  
    else:       
        st.session_state.clicked = True
        return
    
# @st.cache(allow_output_mutation=True)
def create_ROI_plot(q):        
        fig, ax1 = plt.subplots()
        print("plotting")
        print(q[3])
        ax1.bar(q[3].index.values, q[3]['cum_pnl'].values, color='b', alpha=0.6, label='Monthly Returns')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Monthly Returns', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_xticks(q[3].index[::3])
        ax1.set_xticklabels(q[3].index[::3], rotation=90)
        print("Subheader 2")
        ax2 = ax1.twinx()
        print("L1")
        ax2.plot(q[3].index.values, q[3]['roi'].values, color='r', marker='o', label='ROI%')
        print("l2")
        ax2.set_ylabel('ROI%', color='r')
        print("l3")
        ax2.tick_params(axis='y', labelcolor='r')
        print("l4")
        fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
        print("l5")
        st.pyplot(fig)
        # filename = f"roi_plot_{q[0]}.png"
        # plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        # plt.close(fig)  # Close the figure to free up memory
        # print("l6")
        # # Read the image file and encode it
        # with open(filename, "rb") as img_file:
        #     img_base64 = base64.b64encode(img_file.read()).decode()
        # print("l7")
        # # Display the image using Streamlit
        # html_code = f'''
        #     <div style="width: 100%; height: auto; overflow: hidden;">
        #         <img src="data:image/png;base64,{img_base64}" style="width: 100%; height: auto;">
        #     </div>
        #     '''

        # # Display the image using Streamlit
        # st.write(html_code, unsafe_allow_html=True)
        # print("Image displayed successfully")
        print("l8")
        
#@st.cache_data(show_spinner=False, ttl=86400)
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

#@st.cache_data(show_spinner=False, ttl=86400)
def htmap(data, days):
    data = np.array(data['pnl_absolute'].tolist())
    data = data[-1 * days:]
    m = 5 * int(len(data) / 5)
    data = data[:m]
    data = np.reshape(data, (5, -1))
    line_width = 0.8
    linecolor = "White"
    box_width = int(m * 4 / 25) + 2

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

#@st.cache_data(show_spinner=False, ttl=86400)
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

#@st.cache_data(show_spinner=False, ttl=86400)
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
 
#@st.cache_data(show_spinner=False, ttl=86400)
def display(weekday_returns, q):
    st.subheader(f"Profit/Loss Data per Day of Week")
    st.bar_chart(q[5], y=['pnl_absolute'] )
    st.write(f"***Most Profitable Day*** of the week: {weekday_returns[0][1]}")
    st.write(f"***Least Profitable Day*** of the week: {weekday_returns[1][1]}")
    table_data = q[5]['pnl_absolute']
    st.table(table_data)

#@st.cache_data(show_spinner=False, ttl=86400)
def daily_returns_hist(daily_returns):
        fig1, ax1 = plt.subplots(figsize=(10, 2))  
        ax1.bar(daily_returns.index, daily_returns['pnl_absolute'])
        ax1.set_xticks([])
        ax1.set_yticks([])

        fig2, ax2 = plt.subplots(figsize=(10, 5))  
        ax2.bar(daily_returns.index, daily_returns['cum_pnl'])
        ax2.set_xticklabels([])
        
        return fig1, fig2


#@st.cache_data(show_spinner=False, ttl=86400)
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
        
#@st.cache_data(show_spinner=False, ttl=86400)
def entry_find_nearest_date(data, target_date, entry_data_col_index):
    target_date_str = target_date.strftime("%Y-%m-%d %H:%M:%S")
    date_col = list(data[entry_data_col_index])
    for i in range(len(date_col)):
        if date_col[i] >= target_date_str:
            return i

#@st.cache_data(show_spinner=False, ttl=86400)     
def exit_find_nearest_date(data, target_date, entry_data_col_index):
    target_date_str = target_date.strftime("%Y-%m-%d %H:%M:%S")
    date_col = list(data[entry_data_col_index])[::-1]
    for i in range(len(date_col)):
        try:
            if date_col[i] <= target_date_str:
                return len(date_col)-i-1;
        except:
            continue

#@st.cache_data(show_spinner=False, ttl=86400)
def get_data_using_path(csv_path):
    data = pd.read_csv(csv_path)
    return data;

#@st.cache_data(show_spinner=False, ttl=86400)
def get_analysis_obj(data, stn):
    row = calc(data, is_dataframe=1, filename=stn)
    return row;        

#@st.cache_data(show_spinner=False, ttl=86400)     
def get_analysis_with_initial_invest(data, initial_investment, stn):
    Analysis = calc(data, is_dataframe=1, initial_investment=initial_investment, filename=stn)
    return Analysis;    

def next_page(q, stratergy, i):
    
    st.write("\n")
    data = q[1]
    from datetime import datetime

    # Get the current time from your computer's clock
    current_time = datetime.now()

    # Format the time to include hours, minutes, and seconds
    formatted_time = current_time.strftime('%H:%M:%S')

    # Print the current time
    print("Current Time:", formatted_time)
    print(stratergy)
    session_state_df = pd.DataFrame({
        'Key': list(st.session_state.keys()),
        'Value': list(st.session_state.values())
    })
    # print(session_state_df.head())
    st.table()
    row = list(data.iloc[0])
    entry_data_col_index = "entry_timestamp";
    for j in range(len(row)):
        tempstr = str(row[j]).split(" ")
        if is_valid_datetime(tempstr[0]):
            entry_data_col_index = data.columns[j]
            break;

    def parse_datetime(date_str):
        formats = [
            "%d-%m-%Y %H:%M",
            "%d-%m-%Y %H:%M:%S",
            "%d/%m/%Y %H:%M",
            "%d/%m/%Y %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
            "%Y/%m/%d %H:%M:%S"
        ]
        for fmt in formats:
            try:
                return datetime.strptime(str(date_str), fmt)
            except ValueError:
                continue
        raise ValueError(f"Date format not recognized: {date_str}")


    # Format datetime to string in desired format
    data.loc[:, entry_data_col_index] = data.loc[:, entry_data_col_index].apply(parse_datetime)
    for i in range(len(data)):
        try:
            data.loc[i, entry_data_col_index] = data.loc[i, entry_data_col_index].strftime('%Y-%m-%d %H:%M')
        except:
            pass

    
    data = data.sort_values(by=entry_data_col_index)
    try:
        date_format = "%Y-%m-%d"
        startdate = datetime.strptime(data.loc[:, entry_data_col_index].iloc[0].split(" ")[0], date_format)
        i = -1
        while(True):
            try:
                enddate = datetime.strptime(data.loc[:, entry_data_col_index].iloc[i].split(" ")[0], date_format)
                break;
            except:
                i -= 1
    except:
        date_format = "%d-%m-%Y"
        startdate = datetime.strptime(data.loc[:, entry_data_col_index].iloc[0].split(" ")[0], date_format)
        i = -1
        while(True):
            try:
                enddate = datetime.strptime(data.loc[:, entry_data_col_index].iloc[i].split(" ")[0], date_format)
                break;
            except:
                i -= 1     
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
    print("Inital Layout Ready")
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
                st.session_state['ana'] = q
    with subcol3:
        st.write("")

    st.title("Analysis")
    # Analysis = get_analysis(Analysis)
    print("calander rendered")
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
        Duration = ['All Time', ' 2 Years', '1 year', '180 Days', '30 Days', '14 Days', '3 Days']
        returns = [f"{q[19][1]}%", f"{q[44]}%", f"{q[43]}%", f"{q[42]}%", f"{q[40]}%", f"{q[41]}%", f"{q[39]}%"]
        arr = np.array([Duration, returns]).T
        df = pd.DataFrame(arr, columns=["Duration", "Returns"])
        
        st.dataframe(df, hide_index=True, use_container_width=True)
    print("tab3 Done")
        
    with tab2:
        print("in tab2")
        bt1 , bt2, bt3, bt4, bt5 = st.tabs(["Stats", "P&L", "ROI%", "Equity Curve", "Drawdown %"])
        
        with bt5:
            st.subheader("Drawdown Curve")
            st.line_chart(q[1], y='drawdown_percentage', x='Day')
            st.write(f"***Max Drawdown***: {q[7]}")
            st.write(f"***Maximum Drawdowm percentage***: {q[8]}")
        print("DrawDown Donw")
        with bt4:
            st.subheader("Equity Curve")
            st.line_chart(q[1], y='equity_curve')
        print("Equity Curve")
        with bt3:
            st.subheader("ROI% Curve")
            st.line_chart( q[2], y='roi' )
            st.write(f"***ROI***: {q[19][0]}")
            st.write(f"***ROI %***: {q[19][1]}%")
            print("Data written")
            try:
                st.subheader("Monthly Returns and ROI% Over Time")
                # create_ROI_plot(q)
                fig, ax1 = plt.subplots()
                print("plotting")
                ax1.bar(q[3].index.values, q[3]['cum_pnl'].values, color='b', alpha=0.6, label='Monthly Returns')
                ax1.set_xlabel('Month')
                ax1.set_ylabel('Monthly Returns', color='b')
                ax1.tick_params(axis='y', labelcolor='b')
                ax1.set_xticks(q[3].index[::3])
                ax1.set_xticklabels(q[3].index[::3], rotation=90)
                print("Subheader 2")
                ax2 = ax1.twinx()
                print("L1")
                ax2.plot(q[3].index.values, q[3]['roi'].values, color='r', marker='o', label='ROI%')
                print("l2")
                ax2.set_ylabel('ROI%', color='r')
                print("l3")
                ax2.tick_params(axis='y', labelcolor='r')
                print("l4")
                fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
                print("fig returned")
                st.pyplot(fig)
                print("PLOT")
                print("ROI CURVE")
            except:
                print("Some Error Occured")
                import traceback
                traceback.print_exc()
                
        with bt2:
            st.subheader("Daily P&L")
            st.bar_chart(q[2], y=['pnl_absolute'])
            st.subheader("Cumulative P&L")
            st.line_chart(q[2], y=['cum_pnl'])
            print("daily pnl")
        with bt1:
            week = q[32][0]
            month = q[31][0]
            quat = q[35][0]
            half = q[34][0]
            yr = q[33][0]
            
            
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
                print("Befor pyplot")
                st.pyplot(fig=fig0)
                print("After pyplot")
                
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
                    avg_loss,
                    avg_profit,
                    values[0],
                    values[1],
                    q[15],
                    q[18],
                    q[20],
                    f"{q[23]}%",
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
                variables = [str(v) for v in variables]
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
                
            categories = ['Last Week', 'Last Month', 'Last Quarter', 'Last 6 Months', 'Last Year']
            win_rates = [week, month, quat, half, yr]

            # Create the bar plot
            fp, ap = plt.subplots(figsize=(10, 3))
            ap.bar(categories, win_rates, color='b')
            ap.set_xlabel('Time Period')
            ap.set_ylabel('Win Rate (%)')
            ap.set_title('Win Rate for Different Time Periods')

            # Annotate the bars with the win rates
            for i in range(len(categories)):
                ap.text(i, win_rates[i], f'{win_rates[i]:.2f}%', ha='center', va='bottom')

            # Rotate x-axis labels for better visibility
            ap.set_xticklabels(labels=categories, rotation=45)
            st.pyplot(fp)
    # Display the 
            #plt.tight_layout()  # Adjust layout to prevent overlapping labels
 
    print("tab2 done")
    with tab1:
        print("In tab1")
        st.subheader("Heatmap")
        htmap(q[2], 180)        
        
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
            if st.session_state["Time"] == "Day":
                daisply(q[47], st.session_state['Time'] , q[2])
            elif st.session_state["Time"] == "Month":
                daisply(q[48], st.session_state['Time'], q[3])    
            elif st.session_state["Time"] == "Week":
                daisply(q[49], st.session_state['Time'],q[4])    
            elif st.session_state["Time"] == "Year":
                daisply(q[51], st.session_state['Time'], q[6])    
            else:
                display(q[50], q)
    print("tab1 Done")
    
def save_uploaded_file(uploaded_file, save_directory, file_name):
    # Create the save directory if it does not exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    # Save the uploaded file to the specified directory
    file_path = os.path.join(save_directory, file_name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def get_files():
    path = "files/StrategyBacktestingPLBook-*.csv"
    Files = []

    for file in glob.glob(path, recursive=True):
        found = re.search('StrategyBacktestingPLBook(.+?)csv', str(file)).group(1)[1:-1]
        Files.append(found)
    
    return Files
    

def home():
    if st.session_state.warning_message:
        st.warning(st.session_state.warning_message)
    if not st.session_state.clicked:

        Files = get_files()
        num_file = len(Files)
        if st.session_state['account_details']['add_cards'] == "Yes":
            num_file +=1
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
                    csv_path = f"files/StrategyBacktestingPLBook-{stratergy}.csv"
                    with tile:
                        col1,col2, col3, col4 = st.columns([0.35,0.15, 0.3, 0.2]) 

                        with col1:
                            st.write("By Algobulls") 

                        with col3:
                            if st.session_state['account_details']['add_cards'] == "Yes":
                                with st.popover("Append Data"):
                                    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key=f"append{i}")        
                                    if st.button("Submit", key=f"append_{stratergy}"):
                                            st.session_state.warning_message = ""
                                            if uploaded_file is not None:       
                                                data = pd.read_csv(uploaded_file)
                                                csv_data = StatergyAnalysis.new_csv(data, is_dataframe=1)  
                                                start_date = csv_data['date'].iat[0]    
                                                last_date = csv_data['date'].iat[-1]  
                                                original_data = pd.read_csv(csv_path)
                                                
                                                if True:
                                                #if(start_date > q[1]['date'].iat[-1] and last_date > q[1]['date'].iat[-1]):  
                                                    try:
                                                        csv_data = csv_data[original_data.columns]
                                                        result = pd.concat([original_data, data], ignore_index=True)
                                                        result.to_csv(csv_path)

                                                    except:
                                                        try:
                                                            original_data = original_data[csv_data.columns]
                                                            result = pd.concat([original_data, data], ignore_index=True)
                                                            result.to_csv(csv_path)
                                                        except: 
                                                            st.session_state.warning_message = "**Error:** Columns of new csv don't match with previous one"
                                                else:  
                                                    st.session_state.warning_message = "**Error** Data in uploaded csv file preceeds last date in previous csv"
                                                st.rerun() 
                                            
                        with col4:
                            if st.session_state['account_details']['add_cards'] == "Yes":
                                delete_button = st.button("Delete", key=f"delete{i}")
                                if delete_button:
                                    os.remove(csv_path)
                                    st.rerun()
                    
                    # csv_path = f"files/StrategyBacktestingPLBook-{stratergy}.csv"
                    
                    data = get_data_using_path(csv_path)
                    q = get_analysis_obj(data, stratergy)
                    i += 1

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
                    if st.session_state['sidebar']=="Home":
                        tile.button("Execute", key=stratergy, use_container_width=True, on_click=click_button_arg, args=[q, stratergy, i])
                    if st.session_state['sidebar']=="PortfolioOptimization":
                        on = tile.toggle("Add", key=f"add_{stratergy}")
                        if on:  
                            if not stratergy in st.session_state['options']:
                                st.session_state['options'].append(stratergy)
                        else:
                            if stratergy in st.session_state['options']:
                                st.session_state['options'].remove(stratergy)
                        # tile.button("Add", key=f"add_{stratergy}", use_container_width=True, on_click=click_button_add, args=[stratergy])
                
                elif st.session_state['sidebar']=="Home" and st.session_state['account_details']['add_cards'] == "Yes":
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

                        if uploaded_file is not None and user_input != '':
                            # save_directory = "files"
                            file_name = f"files/StrategyBacktestingPLBook-{user_input}.csv"
                            # file_path = save_uploaded_file(uploaded_file, save_directory, file_name)
                            inserted_data = pd.read_csv(uploaded_file)
                            inserted_data.to_csv(file_name)
                        
                    with tile:
                        col1, col2 = st.columns([0.37, 0.63]) 

                        with col1:
                            tile.write("") 

                        with col2:
                            if st.button("Submit"):
                                if uploaded_file is not None and user_input is not None:
                                    st.rerun()

                    tile.write("\n")
                    break
                
        if st.session_state['sidebar']=="PortfolioOptimization":
            st.button("Done", on_click=click_button_done)
                     
    if st.session_state.clicked:
        if st.session_state['sidebar']=="Home":
            print("Next Page called")
            next_page(st.session_state['ana'], st.session_state['stra'], st.session_state['index'])
            with st.sidebar:
                st.button("Return to cards", on_click=click_button_return)