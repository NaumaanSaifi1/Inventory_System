import pandas as pd
from datetime import datetime, timedelta

class InventoryAnalyzer:
    def __init__(self, inventory_df):
        self.df = inventory_df
        
    def analyze_seasonal_turnover(self, season_choice):
        """Analyze turnover by season (weekly/monthly) if possible"""
        df = self.df.copy()
        
        # Try different possible date columns
        date_columns = ['Date_Received', 'Last_Order_Date', 'Expiration_Date']
        date_col = None
        
        for col in date_columns:
            if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
                date_col = col
                break
        
        if date_col is None:
            raise ValueError("No suitable date column found in the data")
        
        if 'Sales_Volume' not in df.columns:
            raise ValueError("Data is missing required 'Sales_Volume' column")
        
        df['Season'] = pd.NaT
        if season_choice == '1':  # Weekly
            df['Season'] = df[date_col].dt.to_period('W').apply(lambda r: r.start_time)
        elif season_choice == '2':  # Monthly
            df['Season'] = df[date_col].dt.to_period('M').apply(lambda r: r.start_time)
        else:
            raise ValueError("Invalid season choice. Choose 1 (Weekly) or 2 (Monthly)")
        
        group_cols = ['Season']
        if 'Catagory' in df.columns:
            group_cols.append('Catagory')
        
        turnover = df.groupby(group_cols)['Sales_Volume'].sum().reset_index()
        return turnover
    
    def get_low_stock_items(self, threshold_days=7):
        """Identify items that need reordering soon if possible"""
        df = self.df.copy()
        
        # Check minimum required columns
        if 'Stock_Quantity' not in df.columns:
            raise ValueError("Missing required 'Stock_Quantity' column")
        
        # Initialize result columns
        result_cols = ['Product_Name', 'Stock_Quantity']
        if 'Catagory' in df.columns:
            result_cols.append('Catagory')
        if 'Supplier_Name' in df.columns:
            result_cols.append('Supplier_Name')
        
        # Simple low stock detection based on reorder level if available
        if 'Reorder_Level' in df.columns:
            low_stock = df[df['Stock_Quantity'] <= df['Reorder_Level']]
            result_cols.append('Reorder_Level')
        else:
            # Fallback: use arbitrary threshold if no reorder level
            low_stock = df[df['Stock_Quantity'] <= 10]  # Default threshold
            
        # Add days until stockout if sales data available
        if 'Sales_Volume' in df.columns:
            df['days_until_stockout'] = (df['Stock_Quantity'] / df['Sales_Volume']) * 30
            low_stock = df[df['days_until_stockout'] <= threshold_days]
            result_cols.append('days_until_stockout')
        
        # Filter by status if available
        if 'Status' in df.columns:
            low_stock = low_stock[low_stock['Status'] == 'Active']
        
        return low_stock[result_cols]
    
    def get_expiring_soon(self, days=30):
        """Identify items expiring soon if expiration data exists"""
        if 'Expiration_Date' not in self.df.columns:
            raise ValueError("Data is missing required 'Expiration_Date' column")
            
        today = datetime.now()
        expiring = self.df[
            (self.df['Expiration_Date'] <= today + timedelta(days=days)) & 
            (self.df['Expiration_Date'] >= today)
        ]
        
        result_cols = ['Product_Name', 'Expiration_Date']
        if 'Catagory' in self.df.columns:
            result_cols.append('Catagory')
        if 'Stock_Quantity' in self.df.columns:
            result_cols.append('Stock_Quantity')
            
        return expiring[result_cols]
    
    def analyze_turnover(self):
        """Analyze inventory turnover rates if possible"""
        if 'Inventory_Turnover_Rate' in self.df.columns:
            turnover_col = 'Inventory_Turnover_Rate'
        elif all(col in self.df.columns for col in ['Sales_Volume', 'Stock_Quantity']):
            self.df['Inventory_Turnover_Rate'] = self.df['Sales_Volume'] / self.df['Stock_Quantity']
            turnover_col = 'Inventory_Turnover_Rate'
        else:
            raise ValueError("Insufficient data to calculate turnover rates")
            
        group_col = 'Catagory' if 'Catagory' in self.df.columns else None
        
        if group_col:
            turnover_stats = self.df.groupby(group_col)[turnover_col].agg(['mean', 'median', 'std'])
        else:
            turnover_stats = pd.DataFrame({
                'mean': [self.df[turnover_col].mean()],
                'median': [self.df[turnover_col].median()],
                'std': [self.df[turnover_col].std()]
            }, index=['Overall'])
            
        return turnover_stats