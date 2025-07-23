import pandas as pd
import numpy as np
from datetime import datetime

def load_inventory_data(filepath):
    """
    Load and preprocess inventory data from CSV file
    
    Parameters:
        filepath (str): Path to the inventory CSV file
        
    Returns:
        pd.DataFrame: Cleaned and processed inventory data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If data is invalid
    """
    try:
        # Load the CSV file
        df = pd.read_csv(filepath)
        
        # Clean and transform data based on available columns
        df = clean_numeric_columns(df)
        df = clean_date_columns(df)
        df = clean_status_column(df)
        
        # Calculate additional metrics if possible
        df = calculate_additional_metrics(df)
        
        return df
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Inventory data file not found at: {filepath}")
    except pd.errors.EmptyDataError:
        raise ValueError("The inventory data file is empty")
    except Exception as e:
        raise ValueError(f"Error loading inventory data: {str(e)}")

def clean_numeric_columns(df):
    """Clean and convert numeric columns that exist"""
    # Clean percentage column if exists
    if 'percentage' in df.columns:
        df['percentage'] = (
            df['percentage'].astype(str).str.rstrip('%').astype(float) / 100
        )
    
    # Clean unit price if exists
    if 'Unit_Price' in df.columns:
        df['Unit_Price'] = (
            df['Unit_Price'].astype(str).str.replace('[$€£,]', '', regex=True).astype(float)
        )
    
    # Convert numeric columns to proper types
    numeric_cols = ['Stock_Quantity', 'Reorder_Level', 'Sales_Volume', 'Reorder_Quantity']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df

def clean_date_columns(df):
    """Clean and convert date columns that exist"""
    date_cols = ['Date_Received', 'Last_Order_Date', 'Expiration_Date']
    
    for col in date_cols:
        if col in df.columns:
            # Try multiple date formats
            df[col] = pd.to_datetime(
                df[col],
                format='mixed',
                errors='coerce',
                dayfirst=False
            )
    
    return df

def clean_status_column(df):
    """Clean and standardize status column if exists"""
    if 'Status' in df.columns:
        df['Status'] = (
            df['Status'].astype(str).str.strip().str.title()
        )
        # Standardize status values
        status_map = {
            'Active': 'Active',
            'Inactive': 'Inactive',
            'Discontinued': 'Inactive'
        }
        df['Status'] = df['Status'].map(status_map).fillna('Active')
    
    return df

def calculate_additional_metrics(df):
    """Calculate additional inventory metrics if possible"""
    # Calculate days of inventory remaining if possible
    if all(col in df.columns for col in ['Stock_Quantity', 'Sales_Volume']):
        df['Days_Of_Inventory'] = (df['Stock_Quantity'] / df['Sales_Volume']).replace([np.inf, -np.inf], 0)
    
    # Calculate inventory value if possible
    if all(col in df.columns for col in ['Stock_Quantity', 'Unit_Price']):
        df['Inventory_Value'] = df['Stock_Quantity'] * df['Unit_Price']
    
    # Calculate simple turnover if possible
    if all(col in df.columns for col in ['Sales_Volume', 'Stock_Quantity']):
        df['Inventory_Turnover_Rate'] = df['Sales_Volume'] / df['Stock_Quantity']
    
    return df