import pandas as pd
import matplotlib.pyplot as plt

class InventoryVisualizer:
    def __init__(self, df):
        self.df = df

    def plot_stock_levels(self):
        """Plot stock levels by category if possible"""
        if 'Catagory' not in self.df.columns or 'Stock_Quantity' not in self.df.columns:
            print("Insufficient data for stock level visualization")
            return
            
        stock_by_category = self.df.groupby('Catagory')['Stock_Quantity'].sum()
        stock_by_category.plot(kind='bar', title='Stock Levels by Category')
        plt.xlabel('Category')
        plt.ylabel('Stock Quantity')
        plt.tight_layout()
        plt.show()

    def plot_seasonal_turnover(self):
        """Plot seasonal turnover patterns if possible"""
        date_col = None
        for col in ['Date_Received', 'Last_Order_Date', 'Expiration_Date']:
            if col in self.df.columns and pd.api.types.is_datetime64_any_dtype(self.df[col]):
                date_col = col
                break
        
        if date_col is None or 'Sales_Volume' not in self.df.columns:
            print("Insufficient data for seasonal turnover visualization")
            return
            
        df = self.df.copy()
        df['Month'] = df[date_col].dt.month
        
        if 'Catagory' in df.columns:
            # Plot by category if available
            monthly_avg = df.groupby(['Month', 'Catagory'])['Sales_Volume'].mean().unstack()
            monthly_avg.plot(
                kind='line', 
                marker='o', 
                title='Average Monthly Turnover by Category'
            )
        else:
            # Overall plot if no category
            monthly_avg = df.groupby('Month')['Sales_Volume'].mean()
            monthly_avg.plot(
                kind='line', 
                marker='o', 
                title='Average Monthly Turnover'
            )
            
        plt.xlabel('Month')
        plt.ylabel('Average Sales Volume')
        plt.grid(True)
        plt.tight_layout()
        plt.show()