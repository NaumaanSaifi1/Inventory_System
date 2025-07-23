from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import warnings

class DemandForecaster:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            min_samples_leaf=5
        )
        self.feature_columns = []
        self.target_column = 'demand'
        self.encoder = None
        self.numeric_features = []
        self.categorical_features = []
        self.trained = False

    def train(self, X, y):
        """Train the model with available features"""
        if len(X) == 0:
            raise ValueError("No features available for training")
            
        original_columns = X.columns.tolist()
        self.numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        self.categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
        
        X_processed = self._preprocess_features(X, training=True)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X_train, y_train)
        
        self.feature_columns = X_processed.columns.tolist()
        self.trained = True
        
        y_pred = self.model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Return metrics as a dictionary with properly formatted values
        return {
            'r2_score': float(r2),  # Convert numpy float to Python float
            'mae': float(mae),
            'feature_importance': self._format_feature_importance()
        }

    def _format_feature_importance(self):
        """Format feature importance for safe string representation"""
        importance = dict(zip(
            self.feature_columns,
            [float(x) for x in self.model.feature_importances_]  # Convert numpy floats
        ))
        # Sort by importance
        return dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))

    def predict(self, X):
        if not self.trained:
            raise ValueError("Model has not been trained yet")
        X_prepared = self._preprocess_features(X)
        return self.model.predict(X_prepared)

    def forecast_future(self, X_future):
        predictions = self.predict(X_future)
        # Convert numpy array to list for better serialization
        return [float(x) for x in predictions]

    def _preprocess_features(self, X, training=False):
        X_processed = X.copy()
        
        date_cols = [col for col in X_processed.columns 
                    if pd.api.types.is_datetime64_any_dtype(X_processed[col])]
        for col in date_cols:
            X_processed[col] = X_processed[col].astype(np.int64) // 10**9
        
        if self.categorical_features:
            if training:
                self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                encoded_data = self.encoder.fit_transform(X_processed[self.categorical_features])
            else:
                try:
                    encoded_data = self.encoder.transform(X_processed[self.categorical_features])
                except AttributeError:
                    raise ValueError("Model has not been trained with categorical features")
                
            encoded_cols = self.encoder.get_feature_names_out(self.categorical_features)
            encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=X_processed.index)
            
            X_processed = X_processed.drop(columns=self.categorical_features)
            X_processed = pd.concat([X_processed, encoded_df], axis=1)
        
        for col in self.numeric_features:
            if col not in X_processed.columns and col not in self.categorical_features:
                X_processed[col] = 0
        
        if training:
            return X_processed
        else:
            available_cols = [col for col in self.feature_columns if col in X_processed.columns]
            return X_processed[available_cols].reindex(columns=self.feature_columns, fill_value=0)

    def detect_trend(self, df, column='demand', window_size=5):
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")
            
        window_size = min(window_size, len(df))
        if window_size < 2:
            return {
                'trend': "insufficient data",
                'strength': 0,
                'data': df.to_dict()  # Convert DataFrame to dict for safe serialization
            }
            
        trend_df = df.copy()
        trend_df['rolling_avg'] = trend_df[column].rolling(window=window_size).mean()
        
        valid_values = trend_df['rolling_avg'].dropna()
        if len(valid_values) < 2:
            return {
                'trend': "insufficient data",
                'strength': 0,
                'data': trend_df.to_dict()
            }
            
        start_val = valid_values.iloc[0]
        end_val = valid_values.iloc[-1]
        
        if end_val > start_val:
            direction = "increasing"
            strength = float((end_val - start_val) / start_val)
        else:
            direction = "decreasing"
            strength = float((start_val - end_val) / start_val)
            
        return {
            'trend': direction,
            'strength': strength,
            'data': trend_df.to_dict()
        }

    def get_feature_importance(self):
        if not self.trained:
            raise ValueError("Model has not been trained yet")
        return self._format_feature_importance()