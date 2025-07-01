# Step 1: Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 2: Sample Data (Years of Experience vs Salary)
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Independent variable (Years)
y = np.array([30000, 35000, 50000, 60000, 65000])  # Dependent variable (Salary)

# Step 3: Create and train the model
model = LinearRegression()
model.fit(X, y)

# Step 4: Predict
y_pred = model.predict(X)

# Step 5: Visualize
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()

# Step 6: Print model parameters
print("Intercept (b0):", model.intercept_)
print("Slope (b1):", model.coef_[0])
