import streamlit as st
from stratrgy_analysis import StatergyAnalysis
import pandas as pd
import numpy as np
from calculations import calc_for_customer_plb
from utils import *

if 'card' not in st.session_state:
    st.session_state['card']= "no card" 


class customerPLBook_Analysis:
    
    def __init__(self):
        self.Daily = None
        self.Weekly = None
        self.Monthly = None
        self.Yearly = None
        self.Day = None
        self.three_days = None
        self.month = None
        self.week = None
        self.year = None
        self.month6 = None
        self.quater = None
        self.two_week = None
        self.thirty_days = None
        self.six_months = None
        self.one_year = None
        self.Two_years = None
        self.df = None
        self.unique_ids = None
        self.unique_dfs = None
        self.uploaded_file = None
            
    def customerPLBook_analysis_display(self, Analysis, option):   
        if 'card' not in st.session_state:
            st.session_state['card']= "no card" 

        with st.sidebar:
            st.write ("Choose Display Options")
            self.Daily = st.checkbox("Daily Analysis.", True)
            self.Monthly = st.checkbox("Monthly Analysis", False)
            self.Yearly = st.checkbox("Yearly Analysis", False)
            self.Weekly = st.checkbox("Weekly Analysis", False)
            self.Day = st.checkbox("Analysis based on day of week", False)
            st.write("\n")
        
            st.write ("Show Returns For:")
            self.three_days = st.checkbox("3 Days", False) 
            self.two_week = st.checkbox("2 Weeks", False) 
            self.thirty_days = st.checkbox("30 Days", False) 
            self.six_months = st.checkbox("6 Months", False) 
            self.one_year = st.checkbox("1 Years", False) 
            self.Two_years = st.checkbox("2 Years", False) 
            st.write("\n")
        
            st.write ("Show Win Rate For:")
            self.week = st.checkbox("Last Week", False)
            self.month= st.checkbox("Last Month", False)
            self.year= st.checkbox("Last Year", False)
            self.month6= st.checkbox("Last 6 Months", False)
            self.quater= st.checkbox("Last Quater", False)
 
        daily_returns, monthly_returns, weekday_returns, weekly_returns, yearly_returns = Analysis.analysis()
        if(option is not None):
            col1, col2, col3 = st.columns([1, 2, 0.6])

            with col1:
                if st.button("Strategy Card"):
                    st.session_state["card"] = "card"

            with col3:
                if st.button("Return to Analysis"):
                    st.session_state["card"] = "no card"


        if st.session_state['card'] == "no card":
            # Collecting the data
            data = {
                "Metric": [
                    "Max Drawdown",
                    "Maximum Drawdown Percentage"
                ],
                "Value": [
                    round(Analysis.drawdown_max, 2),
                    round(Analysis.drawdown_pct, 2)
                ]
            }

            # Create a DataFrame
            df = pd.DataFrame(data)                                                                                                                                                                     

            # Display the table
            st.write(df)   
            st.subheader("Drawdown Curve")
            if(len(Analysis.csv_data)>4000):
                st.line_chart(Analysis.csv_data[-4000:], y='drawdown_pct', x='Day')
            else:
                st.line_chart(Analysis.csv_data, y='drawdown_pct', x='Day')
            data = {
                "Metric": [
                    "Average Loss per Losing Trade",
                    "Average Gain per Winning Trade",
                    "Maximum Gains",
                    "Minimum Gains",
                    "Number of Short Trades",
                    "Number of Long Trades",
                    "Average Trades per Day",
                    "Number of Wins",
                    "Number of Losses",
                    "HIT Ratio",
                    "ROI",
                    "ROI %",
                    "Profit Factor",
                    "Max Win Streak",
                    "Max Loss Streak"
                ],
                "Value": [
                    Analysis.avgProfit(Analysis.csv_data, -1)[0],
                    Analysis.avgProfit(Analysis.csv_data, 1)[0],
                    Analysis.max_profit(Analysis.csv_data)[0],
                    Analysis.min_profit(Analysis.csv_data)[0],
                    Analysis.num_tradeType('short'),
                    Analysis.num_tradeType('long'),
                    Analysis.avgTrades(daily_returns),
                    Analysis.num_loss(Analysis.csv_data, 1),
                    Analysis.num_loss(Analysis.csv_data, -1),
                    Analysis.HIT(),
                    Analysis.roi(monthly_returns)[0],
                    f"{Analysis.roi(monthly_returns)[1]}%",
                    Analysis.ProfitFactor()[0],
                    Analysis.max_consecutive(Analysis.csv_data, 1),
                    Analysis.max_consecutive(Analysis.csv_data, -1)
                ]
            }

            # Create a DataFrame
            df = pd.DataFrame(data)

            # Display the table
            st.table(df)
            d = [30, 7, 365, 180, 90]
            row = []
            for t in d:
                start_date = Analysis.date_calc(day=t)
                index_number = Analysis.daily_returnts.index.get_loc(start_date)
                num_rows = len(Analysis.daily_returnts)-index_number
                last_month_data = Analysis.daily_returnts.iloc[index_number:]
        
                row.append([Analysis.win_rate(last_month_data), num_rows])
            if self.month:
                last_month_data = daily_returns.iloc[-21:]
                st.write(f"Win Rate for ***last Month***: {row[0][0]}")
            if self.week:
                last_month_data = daily_returns.iloc[-6:]
                st.write(f"Win Rate for ***last Week***: {row[1][0]}")
            if self.year:
                last_month_data = daily_returns.iloc[-213:]
                st.write(f"Win Rate for ***last Year***: {row[2][0]}")
            if self.month6:
                last_month_data = daily_returns.iloc[-101:]
                st.write(f"Win Rate for ***last 6 Months***: {row[3][0]}")
            if self.quater:
                last_month_data = daily_returns.iloc[-59:]
                st.write(f"Win Rate for ***last Quater:*** {row[4][0]}")

            st.subheader("Equity Curve")
            if(len(Analysis.csv_data)>4000):
                st.line_chart(Analysis.csv_data[-4000:], y='equity_calculated')
            else:
                st.line_chart(Analysis.csv_data, y='equity_calculated')
            
            T = [3,30,14,180,365,730]
            row = []
            for a in T:
                row.append(Analysis.Treturns(day=a)[1])
                
            if self.three_days:
                st.write(f"Returns for the ***last 3 Days***: {row[0]}%")
            if self.thirty_days:
                st.write(f"Returns for the ***last 30 Days***: {row[1]}%")
            if self.two_week:
                st.write(f"Returns for the ***last 2 Weeks***: {row[2]}%")
            if self.six_months:
                st.write(f"Returns for the ***last 6 Months***: {row[3]}%")
            if self.one_year:
                st.write(f"Returns for the ***last 1 Year***: {row[4]}%")
            if self.Two_years:
                st.write(f"Returns for the ***last 2 Years***: {row[5]}%")


            if self.Daily:
                self.daisply(daily_returns, "Day", Analysis)
            if self.Monthly:
                self.daisply(monthly_returns, "Month", Analysis)
            if self.Yearly:
                self.daisply(yearly_returns, "Year", Analysis)
            if self.Weekly:
                self.daisply(weekly_returns, "Week", Analysis)
            if self.Day:
                self.display(weekday_returns, Analysis)
                
        if(option is not None):
            if st.session_state['card'] == 'card':
                q = calc_for_customer_plb(option, Analysis)
                next_page(q, option, 0)

    def daisply(self, daily_returns, Quant, Analysis):
        print(Quant)
        if Quant == "Day":
            st.header("Daily Analysis")
        else:
            st.header(f"{Quant}ly Analysis")
            data = {
                "Metric": [
                    f"{Quant}ly Average Returns",
                    f"{Quant}ly Average Returns %"
                ],
                "Value": [
                    Analysis.avgReturns(daily_returns)[0],
                    f"{Analysis.avgReturns(daily_returns)[1]}%"
                ]
            }

            # Create a DataFrame
            df = pd.DataFrame(data)

            # Display the table
            st.table(df)

        #days_hist, days_tab = Analysis.compare_hist(daily_returns, [1000, 2000, 3000, 4000, 5000], Quant)
        freq_hist = Analysis.freq_hist(daily_returns, [-5000,-4000, -3000, -2000, -1000, 0, 1000, 2000, 3000, 4000, 5000])
        st.subheader("Frequency of profits")
        st.pyplot(freq_hist)
        if len(daily_returns) > 1:
            with st.expander("More information"):
                # Data for the table
                data = {
                    "Metric": [
                        f"Number of trading {Quant}s",
                        f"Number of Profitable {Quant}s",
                        f"Number of Loss Making {Quant}s",
                        f"Most Profitable {Quant}",
                        f"Maximum Gains in a {Quant}",
                        f"Least Profitable {Quant}",
                        f"Maximum Loss in a {Quant}",
                        "Max Win Streak",
                        "Max Loss Streak"
                    ],
                    "Value": [
                        Analysis.trading_num(daily_returns),
                        Analysis.num_loss(daily_returns, 1),
                        Analysis.num_loss(daily_returns, -1),
                        Analysis.max_profit(daily_returns)[1],
                        Analysis.max_profit(daily_returns)[0],
                        Analysis.min_profit(daily_returns)[1],
                        Analysis.min_profit(daily_returns)[0],
                        Analysis.max_consecutive(daily_returns, 1),
                        Analysis.max_consecutive(daily_returns, -1)
                    ]
                }

                # Create a DataFrame
                df = pd.DataFrame(data)

                # Display the table
                st.table(df)

            st.subheader(f"Profit/Loss Data per {Quant}")
            st.bar_chart(daily_returns, y=['pnl_absolute'] )
            if 'cum_pnl' in daily_returns.columns:
                st.subheader("Cumulative Profit and loss")
                st.line_chart(daily_returns, y=['cum_pnl'])
        st.write(f"")
        st.divider()

    def display(self, weekday_returns, Analysis):
        st.subheader(f"Profit/Loss Data per Day of Week")
        st.bar_chart(weekday_returns, y=['pnl_absolute'] )
        st.write(f"***Most Profitable Day*** of the week: {Analysis.max_profit(weekday_returns)[1]}")
        st.write(f"***Least Profitable Day*** of the week: {Analysis.min_profit(weekday_returns)[1]}")
        tab = weekday_returns['pnl_absolute']
        st.table(tab)
        
    def run(self):
        if self.uploaded_file is None:
            st.title("CSV File Uploader")
            self.uploaded_file  = st.file_uploader("Choose a CSV file", type="csv")
            default = 150000
            if self.uploaded_file  is not None:
                st.header("Analysis of Uploaded File")
                self.df = pd.read_csv(self.uploaded_file )
        
                self.unique_dfs = {}

            # Get unique id values
                self.unique_ids = self.df['Startegy Name'].unique()

            # Loop through the unique ids and create separate DataFrames
                for uid in self.unique_ids:
                    self.unique_dfs[uid] = self.df[self.df['Startegy Name'] == uid]
                print(len(self.unique_ids))
                self.unique_ids = np.insert(self.unique_ids, 0, "Complete Portfolio Analysis")
        if self.uploaded_file  is not None:
            return 1
        return 0

    def sidebar(self):
            with st.sidebar:
                number = st.number_input("Enter initial investment", value=150000, placeholder="Initial value taken as 150000")
                st.write (f"Inital investment {number}")
                option = st.radio(
                "The stratergies being used are",
                ("***" + x + "***" for x in self.unique_ids)
            )
                option = option[3:-3]
                st.write("\n")
            
            if option == "Complete Portfolio Analysis":
                Analysis = StatergyAnalysis(self.df, is_dataframe=1, number=number)
                st.session_state['card'] = 'no card'
            else:
                Analysis = StatergyAnalysis(self.unique_dfs[option], is_dataframe=1, number=number, customerPLBook=True)
                st.session_state['card'] = 'no card'
                st.subheader(f"Analysis Of Strategy ***{option}***")
            self.customerPLBook_analysis_display(Analysis, option)

