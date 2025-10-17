import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
data = {
    'Size_sqft': [1500, 1800, 2400, 3000, 3500, 4000, 4200, 2500, 2700, 3200],
    'Bedrooms': [3, 4, 3, 5, 4, 5, 5, 3, 4, 4],
    'Age': [10, 15, 20, 8, 5, 2, 1, 18, 12, 6],
    'Price': [400000, 500000, 600000, 650000, 700000, 750000, 780000, 580000, 630000, 710000]
}
df = pd.DataFrame(data)
print("Dataset:\n", df)
X = df[['Size_sqft', 'Bedrooms', 'Age']]  # independent variables
y = df['Price']  # dependent variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2=r2_score(y_test,y_pred)
mse = mean_squared_error(y_test, y_pred)
print("\nModel Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
print("R² Score:", round(r2, 3))
print("Mean Squared Error:", round(mse, 2))
plt.figure(figsize=(8, 5))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Prices', marker='o')
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted Prices', marker='x')
plt.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--',
label="Prediction Line")
plt.title("Actual vs Predicted House Prices")
plt.xlabel("Test Sample Index")
plt.ylabel("House Price (in $)")
plt.legend()
plt.grid(True)
plt.show()
new_house = [[2800, 4, 10]] 
new_house_df = pd.DataFrame(new_house, columns=['Size_sqft', 'Bedrooms', 'Age'])
predicted_price = model.predict(new_house_df)

print("\nPredicted Price for new house (2800 sqft, 4 bed, 10 years): $", round(predicted_price[0], 2))
