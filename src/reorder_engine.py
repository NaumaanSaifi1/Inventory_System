import pandas as pd
from datetime import datetime, timedelta

class ReorderEngine:
    def __init__(self, inventory_df):
        self.df = inventory_df
        
    def generate_reorder_list(self):
        """Generate list of items to reorder with suggested quantities if possible"""
        required_cols = ['Stock_Quantity', 'Reorder_Level', 'Status']
        if not all(col in self.df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {', '.join(required_cols)}")
            
        reorder_list = []
        
        for _, row in self.df.iterrows():
            if row['Stock_Quantity'] <= row['Reorder_Level'] and row['Status'] == 'Active':
                item = {
                    'Product_ID': row.get('Product_ID', 'N/A'),
                    'Product_Name': row.get('Product_Name', 'N/A'),
                    'Current_Stock': row['Stock_Quantity'],
                    'Reorder_Level': row['Reorder_Level']
                }
                
                # Calculate suggested quantity if possible
                if 'Reorder_Quantity' in row:
                    item['Suggested_Order_Qty'] = row['Reorder_Quantity']
                elif 'Sales_Volume' in row:
                    item['Suggested_Order_Qty'] = max(10, (row['Sales_Volume'] / 30) * 14)  # 2 weeks supply
                else:
                    item['Suggested_Order_Qty'] = max(10, row['Reorder_Level'] * 2)
                
                # Add supplier info if available
                if 'Supplier_ID' in row:
                    item['Supplier_ID'] = row['Supplier_ID']
                if 'Supplier_Name' in row:
                    item['Supplier_Name'] = row['Supplier_Name']
                
                # Add pricing info if available
                if 'Unit_Price' in row:
                    item['Unit_Price'] = row['Unit_Price']
                    item['Estimated_Cost'] = item['Suggested_Order_Qty'] * row['Unit_Price']
                
                # Add location info if available
                if 'Warehouse_Location' in row:
                    item['Warehouse'] = row['Warehouse_Location']
                
                reorder_list.append(item)
        
        return pd.DataFrame(reorder_list)
    
    def predict_future_demand(self, product_id, days=30):
        """Simple demand prediction based on historical sales if available"""
        if 'Product_ID' not in self.df.columns or 'Sales_Volume' not in self.df.columns:
            return None
            
        product_data = self.df[self.df['Product_ID'] == product_id]
        if len(product_data) == 0:
            return None
            
        daily_sales = product_data['Sales_Volume'].values[0] / 30
        return round(daily_sales * days)
    
    def seasonal_reorder(self):
        """Seasonal reorder suggestions if seasonal data is available"""
        if not all(col in self.df.columns for col in ['Season', 'Quantity Sold']):
            return None
            
        seasonal_suggestions = []
        for season, group in self.df.groupby("Season"):
            reorder_qty = group['Quantity Sold'].sum() * 1.1  # Add 10% buffer
            seasonal_suggestions.append({
                'Season': season,
                'Suggested_Reorder_Quantity': reorder_qty
            })
        
        return pd.DataFrame(seasonal_suggestions)