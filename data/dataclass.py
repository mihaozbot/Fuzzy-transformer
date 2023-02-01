
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class StockData():
    def __init__(self, input_length, output_length,output_steps_ahead):
        self.datasnp = yf.download( 
                tickers = "^GSPC",
                period = "max",
                interval = "1d",
                ignore_tz = False,
                group_by = 'ticker',
                auto_adjust = True,
                repair = False,
                prepost = True,
                threads = True,
                proxy = None,
                start="2001-01-01",
                end="2023-01-01"    
        )
        self.data = yf.download( 
                tickers = "^GSPC,^VIX,^FVX,GC=F",
                period = "max",
                interval = "1d",
                ignore_tz = False,
                group_by = 'ticker',
                auto_adjust = True,
                repair = False,
                prepost = True,
                threads = True,
                proxy = None,
                start="2001-01-01",
                end="2023-01-01"    
            )
        
        self.input_seq_len = input_length
        self.output_seq_len = output_length
        self.output_steps_ahead = output_steps_ahead
        self.clean_data()
        self.norm_data()
        self.create_dataset()
        
        self.dates_datasnp()
        self.clean_datasnp()
        self.norm_datasnp()
        
    #print(data.info)
    #print(data.describe())
    def display_data(self):
        sns.set_style('whitegrid')
        plt.style.use("fivethirtyeight")
        fig, axes = plt.subplots(nrows=2, ncols=2,figsize = (10, 10))
        plt.subplots_adjust(top=1.25, bottom=1.2)
        fig.tight_layout(pad=1.0)
        plt.rcParams.update({'font.size': 12})
        self.data.plot( y = [('^GSPC','Close')], kind='line',linewidth=1.0,ax=axes[0,0])
        axes[0,0].legend(['S&P500'])
        self.data.plot(y = [('^VIX','Close')], kind='line',linewidth=1.0,ax=axes[0,1])
        axes[0,1].legend(['VIX Volatility Index'])
        self.data.plot(y = [('^FVX','Close')], kind='line',linewidth=1.0,ax=axes[1,0])
        axes[1,0].legend(['Treasury Yield 5 Years'])
        self.data.plot(y = [('GC=F','Close')], kind='line',linewidth=1.0,ax=axes[1,1])
        axes[1,1].legend(['Gold'])

    def save_data(self):
        self.datasnp.to_csv('snp500_data.csv', index = False)
        self.datasnp.to_csv('snp500_dates.csv', index = True)
        self.data.to_csv('stock_data.csv', index = False)
        self.data.to_csv('stock_dates.csv', index = True)
        
    def clean_data(self):
        self.data_dropped = self.data.loc[:, self.data.columns.get_level_values(1) == 'Close'] #self.data.drop(['open', 'high', 'low','volume'], axis=1)
        self.order = [('^GSPC', 'Close'),( 'GC=F', 'Close'),( '^VIX', 'Close'),( '^FVX', 'Close')]
        self.names = [["S&P500"],["Gold"],["VIX Volatility Index"],["Treasury Yield 5 Years"]]
        self.data_dropped = self.data_dropped[self.order[:]]
        self.data_dropped.columns = ["SNP","GOLD","VIX","BONDS"]
        if self.data_dropped.isnull().values.any():
            self.data_dropped = self.data_dropped.fillna(method='ffill')
            print("Nani?! in data")   
        
    def norm_data(self):
        self.scalar = MinMaxScaler(feature_range = (0,1))
        self.data_norm = self.scalar.fit_transform(np.array(self.data_dropped))
    
    def clean_datasnp(self):
        self.datasnp_dropped = self.datasnp.loc[:, (self.datasnp.columns.get_level_values(0) == 'Close')] 
        self.datasnp_dropped.columns = ["SNP"]
        
    def dates_datasnp(self): 
        self.datasnp_dates = self.datasnp.reset_index()    
        self.datasnp_dates = self.datasnp_dates.loc[:, (self.datasnp_dates.columns.get_level_values(0) == 'Close') | (self.datasnp_dates.columns.get_level_values(0) == 'Date')] 
        self.datasnp_dates.columns = ["Date","SNP"] 
    
    def norm_datasnp(self):
        self.scalarsnp = MinMaxScaler(feature_range = (0,1))
        self.datasnp_norm = self.scalarsnp.fit_transform(np.array(self.datasnp_dropped))

    def display_data_norm(self):
        sns.set_style('whitegrid')
        plt.style.use("fivethirtyeight")
        fig, axes = plt.subplots(nrows=2, ncols=2,figsize = (10, 8))
        plt.subplots_adjust(top=1.25, bottom=1.2)
        fig.tight_layout(pad=1.0)
        plt.rcParams.update({'font.size': 12})
        axes[0,0].plot(self.data_norm[:,0], linewidth=1.0)
        axes[0,0].legend((self.names[0][0],))
        axes[0,1].plot(self.data_norm[:,1], linewidth=1.0)
        axes[0,1].legend((self.names[1][0],))
        axes[1,0].plot(self.data_norm[:,2], linewidth=1.0)
        axes[1,0].legend((self.names[2][0],))
        axes[1,1].plot(self.data_norm[:,3], linewidth=1.0)
        axes[1,1].legend((self.names[3][0],))
    
    def create_dataset(self):
        dataset_length = len(self.data_norm)-self.input_seq_len-self.output_seq_len-1-self.output_steps_ahead
        self.dataset_input = np.empty((dataset_length,self.input_seq_len,4))
        self.dataset_output = np.empty((dataset_length,self.output_seq_len))
        for i in range(dataset_length):
            self.dataset_input[i,:,:] = self.data_norm[i : i+self.input_seq_len]
            try:
                self.dataset_output[i,:] = self.data_norm[i+self.input_seq_len+1+self.output_steps_ahead: i+self.input_seq_len+self.output_seq_len+1+self.output_steps_ahead,0]
                #self.dataset_output[i,:] = self.data_norm[i+self.input_seq_len+self.output_steps_ahead+1,0]
            except:
                print("Something didn't match!")