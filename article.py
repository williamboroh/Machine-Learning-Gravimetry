import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, r2_score
from deap import base, creator, tools, algorithms
import random

# Load data
data = pd.read_excel("validation.xlsx")
X = data[['Direction', 'Range', 'Nugget', 'Sill']]
y_rmse = data['RMSE']
y_r = data['R']

# Define 3-fold cross-validation
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Linear regression models
model_rmse = LinearRegression()
model_r = LinearRegression()

# Predictions with cross-validation
predicted_rmse = cross_val_predict(model_rmse, X, y_rmse, cv=kf)
predicted_r = cross_val_predict(model_r, X, y_r, cv=kf)

# Compute performance metrics
rmse_score = mean_squared_error(y_rmse, predicted_rmse, squared=False)
r2_score_value = r2_score(y_r, predicted_r)

print(f"3-Fold Cross-Validation: RMSE = {rmse_score:.4f}, R² = {r2_score_value:.4f}")

# Create subplots for validation
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram of errors
errors = y_rmse - predicted_rmse
axes[0].hist(errors, bins=20, edgecolor='black', alpha=0.7)
axes[0].set_title("Error Histogram")
axes[0].set_xlabel("Error (Actual - Predicted Value)")
axes[0].set_ylabel("Frequency")

# Scatter plot of actual vs predicted values
axes[1].scatter(y_r, predicted_r, alpha=0.6, edgecolors='k')
axes[1].plot([min(y_r), max(y_r)], [min(y_r), max(y_r)], '--r', lw=2)
axes[1].set_title("Correlation between Actual and Predicted Values")
axes[1].set_xlabel("Actual Values")
axes[1].set_ylabel("Predicted Values")

plt.tight_layout()
plt.show()

# Genetic Algorithm Implementation
creator.create("FitnessMax", base.Fitness, weights=(1.0, -1.0))  # Maximize R, minimize RMSE
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Get variable bounds
min_vals = X.min()
max_vals = X.max()

def create_individual():
    return [random.uniform(min_vals[i], max_vals[i]) for i in X.columns]

toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    X_test = np.array([individual])
    predicted_rmse = model_rmse.predict(X_test)[0]
    predicted_r = model_r.predict(X_test)[0]
    return predicted_r, -predicted_rmse

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=100)
NGEN = 50
for gen_num in range(NGEN):
    population, _ = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=1, verbose=False)

best_individual = tools.selBest(population, 1)[0]
print("Best individual:", best_individual)

