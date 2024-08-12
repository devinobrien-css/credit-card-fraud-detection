
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

def main():
    # Load the dataset
    print("-------------------------------------------------------")
    print("\tDATA IMPORT")
    print("-------------------------------------------------------")
    
    url = '../data/creditcard.csv'
    df = pd.read_csv(url)

    print("\n\n")
    print("-------------------------------------------------------")
    print("\tDATA HEAD, INFO, DESCRIBE")
    print("\n-------------------------------------------------------\n")
    print(df.head())
    print("\n-------------------------------------------------------\n")
    print(df.info())
    print("\n-------------------------------------------------------\n")
    print(df.describe())
    print(f"Number of fraudulent transactions: {df['Class'].sum()}")

    # Check for missing values
    if df.isnull().sum().any():
        print("Dataset contains missing values.")
    else:
        print("No missing values found in the dataset.")

    print("-------------------------------------------------------")
    print("\n\n")

    # Basic exploration
    print("-------------------------------------------------------")
    print("\tPREPROCESSING")
    print("-------------------------------------------------------")

    # Feature scaling
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])
    df['Time'] = scaler.fit_transform(df[['Time']])

    # Splitting the data into training and testing sets
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Handling class imbalance with SMOTE
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # Verify the resampling
    print(f"Original training set class distribution: \n{y_train.value_counts()}")
    print("\n-------------------------------------------------------")
    print(f"Resampled training set class distribution: \n{y_train_res.value_counts()}")
    print("\n-------------------------------------------------------")
    print("\n\n")
          
    # The preprocessed data is now ready for model training

    # Train and evaluate models
    print("-------------------------------------------------------")
    print("\tMODEL TRAINING & EVALUATION")
    print("-------------------------------------------------------")

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    # Evaluate models using cross-validation
    for model_name, model in models.items():
        print("\n")
        print("-------------------------------------------------------")
        print(f"\t{model_name}")
        print("-------------------------------------------------------")
        print(f"Evaluating...\n")
        scores = cross_val_score(model, X_train_res, y_train_res, cv=5, scoring='roc_auc')
        print(f"Mean ROC AUC Score: {scores.mean():.4f}\n")
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        print(f"Test ROC AUC Score: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}\n")
        print("-------------------------------------------------------")



    print("\n\n")
    print("-------------------------------------------------------")
    print("\tMODEL SELECTION: RANDOM FOREST")
    print("-------------------------------------------------------")

    # Selecting Best Model and Refining

    # Define the model
    model = RandomForestClassifier(random_state=42)

    # Define an even smaller hyperparameter grid for Grid Search
    param_grid = {
        'n_estimators': [50, 100],            # 2 options
        'max_depth': [10, 20],                # 2 options
        'min_samples_split': [5],             # 1 option
        'min_samples_leaf': [1, 2],           # 2 options
    }

    # Setup the Grid Search with cross-validation
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                            cv=3, scoring='roc_auc', n_jobs=-1, verbose=2)

    # Fit the Grid Search to the data
    grid_search.fit(X_train_res, y_train_res)

    # Output the best parameters and best score
    print("Best Parameters found: ", grid_search.best_params_)
    print("Best ROC AUC Score: ", grid_search.best_score_)

    # Evaluate the best model on the test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("Test ROC AUC Score: ", roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]))
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()