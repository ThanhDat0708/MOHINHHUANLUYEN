import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dữ liệu (thay đường dẫn nếu cần)
df = pd.read_csv('Housing.csv')   # hoặc đường dẫn đầy đủ đến file bạn tải về

print(df.head())
print(df.info())
print(df.describe())

# 2. Tách features và target
X = df.drop('price', axis=1)
y = df['price']

# 3. Phân loại columns
numeric_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                        'airconditioning', 'prefarea', 'furnishingstatus']

# 4. Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ])

# 5. Hàm đánh giá model
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n=== {model_name} ===")
    print(f"MAE : {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R²  : {r2:.4f}")
    return mae, rmse, r2

# 6. Split dữ liệu (80% train - 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Huấn luyện và so sánh 3 mô hình

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=8),   # giới hạn depth tránh overfitting
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10)
}

results = {}

for name, model in models.items():
    # Tạo pipeline đầy đủ
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred = pipeline.predict(X_test)
    
    # Đánh giá
    mae, rmse, r2 = evaluate_model(y_test, y_pred, name)
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

# 8. So sánh tổng hợp
print("\n" + "="*50)
print("KẾT QUẢ SO SÁNH CÁC MÔ HÌNH")
print("="*50)
comparison = pd.DataFrame(results).T
print(comparison.round(2))
print("\nMô hình tốt nhất dựa trên R² cao nhất và RMSE thấp nhất.")