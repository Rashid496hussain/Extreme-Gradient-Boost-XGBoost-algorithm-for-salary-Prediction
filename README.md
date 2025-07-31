XGBoost Salary Prediction Project


Overview
This project analyzes a synthetic dataset of employee attributes (Age, Qualification, Salary) and develops an Extreme Gradient Boosting (XGBoost) model to predict salaries based on Age and Qualification. The dataset is augmented to 100 samples to ensure robust model training and evaluation.

Dataset
The dataset is synthetically generated with the following features:

Age: Integer values between 20 and 35, representing employee age.
Qualification: Categorical variable ('Yes' or 'No'), indicating whether the employee has a qualification (e.g., degree). 60% 'Yes', 40% 'No'.
Salary: Continuous values between 30 and 100, generated based on Age (linear trend) and Qualification (higher base for 'Yes'), with added Gaussian noise.

Requirements

Python 3.6+
Required libraries:
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn



Installation

Clone or download the project repository.
Ensure Python is installed (python --version).
Install required libraries using pip:pip install pandas numpy scikit-learn xgboost matplotlib seaborn

Or in a Jupyter Notebook:!pip install pandas numpy scikit-learn xgboost matplotlib seaborn



Usage

Place the Python script (xgboost_analysis.py) in your working directory.
Run the script:python xgboost_analysis.py


The script will:
Generate and analyze a synthetic dataset.
Display dataset statistics, correlation matrix, and visualizations (scatter and box plots).
Train an XGBoost regressor to predict Salary.
Output evaluation metrics (MSE, RMSE, R²) and feature importance.
Show sample predictions vs. actual values.



Code Structure

Data Generation: Creates a synthetic dataset with 100 samples.
Data Analysis: Computes summary statistics, checks for missing values, and visualizes relationships (Age vs. Salary, Salary by Qualification).
Preprocessing: Encodes categorical variable (Qualification) and splits data into training (80%) and testing (20%) sets.
XGBoost Model: Trains an XGBoost regressor with tuned hyperparameters (n_estimators=100, learning_rate=0.05, max_depth=4).
Evaluation: Calculates Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R² score, and plots feature importance.
Predictions: Displays predictions vs. actual salaries for a sample of test data.

Notes

The dataset is synthetic but mimics realistic patterns (e.g., higher salaries for qualified employees, positive Age-Salary correlation).
Hyperparameters are tuned for a small-to-medium dataset to prevent overfitting.
Visualizations require a graphical environment to display plots.
For larger datasets or additional features, modify the data generation section in the script.

Troubleshooting

ModuleNotFoundError: Ensure all required libraries are installed. If xgboost is missing, install it with pip install xgboost.
Environment Issues: Use a virtual environment or Conda to manage dependencies. For Conda, install xgboost with conda install xgboost.
Plot Issues: Ensure a graphical backend is available (e.g., run in Jupyter or a local IDE with matplotlib support).

License
This project is for educational purposes and does not include a specific license.
