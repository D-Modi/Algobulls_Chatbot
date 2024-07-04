import streamlit as st
import matplotlib.pyplot as plt  
from stratrgy_analysis import StatergyAnalysis
import pandas as pd
import numpy as np

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

            
    def customerPLBook_analysis_display(self, Analysis):    
        daily_returns, monthly_returns, weekday_returns, weekly_returns, yearly_returns = Analysis.analysis()

        st.write(f"***Max Drawdown***: {Analysis.drawdown_max}")
        st.write(f"***Maximum Drawdowm percentage***: {Analysis.drawdown_pct}")
        st.subheader("Drawdown Curve")
        st.line_chart(Analysis.csv_data, y='drawdown_pct', x='Day')
        analysis_data1 = {
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
                "ROI Percentage",
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

        analysis_df1 = pd.DataFrame(analysis_data1)

        st.table(analysis_df1)
        if self.month:
            last_month_data = daily_returns.iloc[-21:]
            st.write(f"Win Rate for ***last Month***: {Analysis.win_rate(last_month_data)}")
        if self.week:
            last_month_data = daily_returns.iloc[-6:]
            st.write(f"Win Rate for ***last Week***: {Analysis.win_rate(last_month_data)}")
        if self.year:
            last_month_data = daily_returns.iloc[-213:]
            st.write(f"Win Rate for ***last Year***: {Analysis.win_rate(last_month_data)}")
        if self.month6:
            last_month_data = daily_returns.iloc[-101:]
            st.write(f"Win Rate for ***last 6 Months***: {Analysis.win_rate(last_month_data)}")
        if self.quater:
            last_month_data = daily_returns.iloc[-59:]
            st.write(f"Win Rate for ***last Quater:*** {Analysis.win_rate(last_month_data)}")

        st.subheader("Equity Curve")
        st.line_chart(Analysis.csv_data, y='equity_curve', x='Day')
        if self.three_days:
            st.write(f"Returns for the ***last 3 Days***: {Analysis.Treturns(4)[1]}%")
        if self.thirty_days:
            st.write(f"Returns for the ***last 30 Days***: {Analysis.Treturns(22)[1]}%")
        if self.two_week:
            st.write(f"Returns for the ***last 2 Weeks***: {Analysis.Treturns(11)[1]}%")
        if self.six_months:
            st.write(f"Returns for the ***last 6 Months***: {Analysis.Treturns(127)[1]}%")
        if self.one_year:
            st.write(f"Returns for the ***last 1 Year***: {Analysis.Treturns(253)[1]}%")
        if self.Two_years:
            st.write(f"Returns for the ***last 2 Years***: {Analysis.Treturns(505)[1]}%")

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

    def daisply(self, daily_returns, Quant, Analysis):
        print(Quant)
        if Quant == "Day":
            st.header("Daily Analysis")
        else:
            st.header(f"{Quant}ly Analysis")
            st.write(f"***{Quant}ly Average Returns***: {Analysis.avgReturns(daily_returns)[0]}")
            st.write(f"***{Quant}ly Average Returns %***: {Analysis.avgReturns(daily_returns)[1]}%")

        #days_hist, days_tab = Analysis.compare_hist(daily_returns, [1000, 2000, 3000, 4000, 5000], Quant)
        freq_hist = Analysis.freq_hist(daily_returns, [-5000,-4000, -3000, -2000, -1000, 0, 1000, 2000, 3000, 4000, 5000])
        st.subheader("Frequency of profits")
        st.pyplot(freq_hist)
        if len(daily_returns) > 1:
            with st.expander("More information"):
                analysis_data2 = {
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
                        f"{Analysis.num_loss(daily_returns, 1)} {Quant}",
                        f"{Analysis.num_loss(daily_returns, -1)} {Quant}",
                        Analysis.max_profit(daily_returns)[1],
                        Analysis.max_profit(daily_returns)[0],
                        Analysis.min_profit(daily_returns)[1],
                        Analysis.min_profit(daily_returns)[0],
                        Analysis.max_consecutive(daily_returns, 1),
                        Analysis.max_consecutive(daily_returns, -1)
                    ]
                }

                # Convert the dictionary to a DataFrame
                analysis_df2 = pd.DataFrame(analysis_data2)

                # Display the DataFrame as a table
                st.table(analysis_df2)

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
        st.title("CSV File Uploader")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        default = 150000
        if uploaded_file is not None:
            st.title("Analysis of Uploaded File")
            df = pd.read_csv(uploaded_file)
    
            dfs = {}

        # Get unique id values
            unique_ids = df['Startegy Name'].unique()

        # Loop through the unique ids and create separate DataFrames
            for uid in unique_ids:
                dfs[uid] = df[df['Startegy Name'] == uid]
            print(len(unique_ids))
            unique_ids = np.insert(unique_ids, 0, "Complete Portfolio Analysis")
        
            with st.sidebar:
                number = st.number_input("Enter initial investment", min_value=0, step=10000,  value=default, placeholder="Initial value taken as 150000")
                st.write (f"Inital investment {number}")
                option = st.radio(
                "The stratergies being used are",
                ("***" + x + "***" for x in unique_ids)
            )
                option = option[3:-3]
                st.write("\n")
            
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

            if option == "Complete Portfolio Analysis":
                Analysis = StatergyAnalysis(df, is_dataframe=1, number=number)
            else:
                Analysis = StatergyAnalysis(dfs[option], is_dataframe=1, number=number, customerPLBook=True)
                st.title(f"Analyis Of Stratrergy ***{option}***")
            self.customerPLBook_analysis_display(Analysis)


