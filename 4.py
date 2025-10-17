import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    "AGE": [1, 2, 3, 4, 5, 6],
    "HEIGHT": [2, 6, 8, 9, 11, 15],
}
df = pd.DataFrame(data)
print("Dataset:\n", df)
X = df[["AGE"]]
Y = df["HEIGHT"]
model = LinearRegression()
model.fit(X, Y)
m = model.coef_[0]
c = model.intercept_
print(f"\nRegression equation: Y = {m:.2f}x + {c:.2f}")
Y_pred = model.predict(X)
plt.scatter(X, Y, color="green", label="Actual Data")
plt.plot(X, Y_pred, color="red", label="Regression Line")
plt.xlabel("AGE")
plt.ylabel("HEIGHT")
plt.title("Simple Linear Regression (scikit-learn)")
plt.legend()
plt.show()  
age_input = input("Enter age to predict height: ")
try:
    age_val = float(age_input)
    AGE = pd.DataFrame({"AGE": [age_val]})
    predicted_height = model.predict(AGE)
    print(f"Predicted Height for age {age_val} = {predicted_height[0]:.2f}")
except ValueError:
    print("Please enter a valid number.")
