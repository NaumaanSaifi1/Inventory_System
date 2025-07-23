import pandas as pd
from datetime import datetime
from src.data_loader import load_inventory_data
from src.inventory_analyzer import InventoryAnalyzer
from src.reorder_engine import ReorderEngine
from src.visualization import InventoryVisualizer
from src.demand_forecaster import DemandForecaster

def display_menu(available_features):
    """Display the main menu options based on available features"""
    print("\n" + "="*50)
    print("AI-Driven Inventory Management System")
    print("="*50)
    print("1. View Current Inventory Status")
    
    if 'low_stock' in available_features:
        print("2. Check Low Stock Items")
    if 'reorder' in available_features:
        print("3. Generate Reorder List")
    if 'expiring' in available_features:
        print("4. View Expiring Soon Items")
    if 'turnover' in available_features:
        print("5. Analyze Inventory Turnover")
    if 'visualization' in available_features:
        print("6. Visualize Stock Levels")
    if 'seasonal' in available_features:
        print("7. Seasonal Turnover Analysis")
    if 'forecasting' in available_features:
        print("8. Forecast Demand")
    print("9. Exit")
    print("="*50)

def detect_available_features(df):
    """Detect which features are available based on columns in the dataframe"""
    features = {}
    
    # Basic inventory features
    features['low_stock'] = all(col in df.columns for col in ['Stock_Quantity', 'Reorder_Level'])
    features['reorder'] = all(col in df.columns for col in ['Stock_Quantity', 'Reorder_Level', 'Status'])
    features['expiring'] = 'Expiration_Date' in df.columns
    features['turnover'] = 'Sales_Volume' in df.columns and 'Stock_Quantity' in df.columns
    features['visualization'] = 'Catagory' in df.columns and 'Stock_Quantity' in df.columns
    
    # Seasonal analysis requires date and sales data
    features['seasonal'] = any(col in df.columns for col in ['Date_Received', 'Last_Order_Date']) and 'Sales_Volume' in df.columns
    
    # Forecasting requires some time-based data and demand metric
    features['forecasting'] = 'Stock_Quantity' in df.columns and any(col in df.columns for col in ['Date_Received', 'Last_Order_Date'])
    
    return features

def main():
    # Load data 
    excel_path = ("Enter your file path here" )
    
    try:
        inventory_df = load_inventory_data(excel_path)
        print(f"\nSuccessfully loaded inventory data from: {excel_path}")
        print("\nAvailable columns in the dataset:")
        print(inventory_df.columns.tolist())
    except Exception as e:
        print(f"Failed to load inventory data: {e}")
        return

    # Detect available features based on columns
    available_features = detect_available_features(inventory_df)
    
    analyzer = InventoryAnalyzer(inventory_df)
    reorder_engine = ReorderEngine(inventory_df)
    visualizer = InventoryVisualizer(inventory_df)
    forecaster = DemandForecaster()

    while True:
        display_menu(available_features)
        choice = input("Enter your choice (1-9): ")

        if choice == "1":
            print("\nCurrent Inventory Status:")
            print(inventory_df.head())

        elif choice == "2" and 'low_stock' in available_features:
            print("\nLow Stock Items:")
            low_stock = analyzer.get_low_stock_items()
            print(low_stock)

        elif choice == "3" and 'reorder' in available_features:
            print("\nReorder List:")
            reorder_list = reorder_engine.generate_reorder_list()
            print(reorder_list)

        elif choice == "4" and 'expiring' in available_features:
            print("\nItems Expiring Soon:")
            expiring_items = analyzer.get_expiring_soon()
            print(expiring_items)

        elif choice == "5" and 'turnover' in available_features:
            print("\nInventory Turnover Analysis:")
            turnover = analyzer.analyze_turnover()
            print(turnover)

        elif choice == "6" and 'visualization' in available_features:
            print("\nGenerating Stock Level Visualization...")
            visualizer.plot_stock_levels()

        elif choice == "7" and 'seasonal' in available_features:
            print("\nGenerating Seasonal Turnover Visualization...")
            visualizer.plot_seasonal_turnover()

        elif choice == "8" and 'forecasting' in available_features:
            try:
                df = inventory_df.copy()
                
                # Determine demand column
                demand_col = 'Stock_Quantity' if 'Stock_Quantity' in df.columns else 'Sales_Volume'
                df = df.rename(columns={demand_col: "demand"})
                
                # Create features based on available data
                feature_columns = []
                
                # Time-based feature
                if any(col in df.columns for col in ['Date_Received', 'Last_Order_Date']):
                    date_col = 'Date_Received' if 'Date_Received' in df.columns else 'Last_Order_Date'
                    df[date_col] = pd.to_datetime(df[date_col])
                    df['day_index'] = (df[date_col] - df[date_col].min()).dt.days
                    feature_columns.append('day_index')
                
                # Add category if available
                if 'Catagory' in df.columns:
                    feature_columns.append('Catagory')
                
                # Fallback if no features created
                if not feature_columns:
                    df['day_index'] = range(len(df))
                    feature_columns = ['day_index']
                
                # Prepare features and target
                X = df[feature_columns]
                y = df['demand']
                
                # Train model
                training_result = forecaster.train(X, y)
                print(f"\nModel trained successfully.")
                print(f"R² score: {training_result['r2_score']:.2f}")
                print(f"Mean Absolute Error: {training_result['mae']:.2f}")
                print("\nFeature Importance:")
                for feature, importance in training_result['feature_importance'].items():
                    print(f"{feature}: {importance:.2f}")
                
                # Forecast for next period
                future_days = 7
                if 'day_index' in feature_columns:
                    future_X = pd.DataFrame({
                        "day_index": range(df['day_index'].max()+1, df['day_index'].max()+1+future_days)
                    })
                    
                    # Handle category if it was used in training
                    if 'Catagory' in feature_columns:
                        median_category = df['Catagory'].mode()[0]
                        future_X['Catagory'] = median_category
                        print(f"\nUsing category '{median_category}' for forecasting")
                    
                    try:
                        future_predictions = forecaster.forecast_future(future_X)
                        
                        print("\n📈 Forecast for Next 7 Days:")
                        for i, val in enumerate(future_predictions, 1):
                            print(f"Day {i}: {val:.2f} units")
                    except Exception as e:
                        print(f"\n⚠️ Forecasting error: {str(e)}")
                        print("Trying without category features...")
                        if 'Catagory' in future_X.columns:
                            future_X = future_X.drop('Catagory', axis=1)
                            future_predictions = forecaster.forecast_future(future_X)
                            for i, val in enumerate(future_predictions, 1):
                                print(f"Day {i}: {val:.2f} units")
                
                # Detect trend
                try:
                    trend_result = forecaster.detect_trend(df)
                    print(f"\n📊 Overall Trend Detected: {trend_result['trend'].title()}")
                    print(f"📈 Trend Strength: {trend_result['strength']:.2%}")
                except Exception as e:
                    print(f"\n⚠️ Trend detection error: {str(e)}")

            except Exception as e:
                print(f"\n❌ Error during forecasting: {e}")

        elif choice == "9":
            print("\nExiting Inventory Management System. Goodbye!")
            break

        else:
            print("\nInvalid choice or feature not available with current data. Please try again.")

if __name__ == "__main__":
    main()