

### Overview
This notebook provides a comprehensive pipeline for data loading, preprocessing, exploratory data analysis (EDA), model training, and evaluation on a dataset that involves predictive modeling. The notebook uses Python libraries such as `pandas`, `numpy`, `scikit-learn`, `seaborn`, and `matplotlib` to handle and visualize data, and it applies various machine learning algorithms, including Linear Regression, Random Forest, and XGBoost Regressor.

### Structure and Key Steps

1. **Data Loading and Preprocessing**
   - **Data Loading**: Loads two datasets (`calories.csv` and `exercise.csv`) and merges them based on a common identifier (`User_ID`).
   - **Data Information and Summary**: The notebook provides functions to display data information, statistics, null values, and check for duplicates.
   - **Feature and Target Separation**: Features (`X`) and target (`y`) are separated for training purposes.
   - **Train-Test Split**: The data is split into training and testing sets with a default ratio of 80:20 and a specified random state for reproducibility.

2. **Exploratory Data Analysis (EDA)**
   - **Visualization**: 
     - Pair plots and scatter plots are used to understand relationships among features.
     - Histograms for numerical columns and count plots for categorical columns are generated to analyze the data distribution.
   - **Insights**: EDA insights provide a foundation for selecting relevant features and preprocessing techniques for the model.

3. **Data Preprocessing Pipeline**
   - **Standard Scaling and Ordinal Encoding**: Uses `StandardScaler` for numerical features and `OrdinalEncoder` for categorical features within a `ColumnTransformer` pipeline.
   - **Pipeline Structure**: A `Pipeline` is used to apply preprocessing steps and modeling sequentially, ensuring reproducible and systematic data transformations.

4. **Model Selection and Training**
   - The notebook includes several regression models:
     - **Linear Regression**
     - **Random Forest Regressor**
     - **XGBoost Regressor**
   - Each model’s pipeline is defined, where hyperparameters such as estimators for Random Forest and learning rate for XGBoost are specified. The models are evaluated using cross-validation with metrics like R² score.

5. **Hyperparameters and Cross-Validation**
   - **Hyperparameter Tuning**: Specific hyperparameters are set, such as:
     - Random Forest (`n_estimators`, `max_depth`)
     - XGBoost (`learning_rate`, `max_depth`)
   - **Cross-Validation**: Uses `KFold` cross-validation with scoring metrics to evaluate model performance systematically across different data splits.

6. **Evaluation Metrics**
   - **R² Score**: The notebook computes the R² score on the test set to assess model performance, providing insights into how well each model captures variance in the target variable.

### Requirements
- **Python Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`
- **Installation**:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   ```

### Usage
To use this notebook, load the datasets (`calories.csv` and `exercise.csv`) in the specified directory and run each cell sequentially. The notebook will preprocess the data, perform EDA, train each model, and evaluate their performance.

### Interactive GUI using Tkinter framework from Python
![Predictor_GUI](https://github.com/user-attachments/assets/8d705d33-4e1e-4a58-b979-38aa65bba7eb)

![Personal Goal Tracker GUI](https://github.com/user-attachments/assets/0af6af8e-403b-4567-a3a9-da3d93374cf7)

---
