# train_model.py
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("india_housing_prices.csv")

# Features & Target
X = df.drop("Price_in_Lakhs", axis=1)
y = df["Price_in_Lakhs"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model
with open("house_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as house_price_model.pkl")
