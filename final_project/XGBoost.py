import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import random
import time
from datetime import datetime

random.seed(1993)

df = pd.read_csv('final_project\\water_pollution_disease_fixed_column_titles.csv')

targets = [
    "Diarrheal Cases per 100,000 people",
    "Cholera Cases per 100,000 people",
    "Typhoid Cases per 100,000 people",
    "Infant Mortality Rate (per 1,000 live births)"
]

X = df.drop(columns=targets)
y_dict = {target: df[target] for target in targets}

categorical_cols = X.select_dtypes(include=["object"]).columns
numeric_cols = X.select_dtypes(exclude=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)


bestModels = {}
bestResults = {}

n_estimators_range=450
learning_rate_range=0.495
max_depth_range=2
subsample_range=0.25
colsample_bytree_range=0.25

n_estimators_absolute_min=100
learning_rate_absolute_min=0.01
max_depth_absolute_min=2
subsample_absolute_min=0.5
colsample_bytree_absolute_min=0.5

n_estimators_absolute_max=1000
learning_rate_absolute_max=1
max_depth_absolute_max=6
subsample_absolute_max=1
colsample_bytree_absolute_max=1

n_estimators_current={}
learning_rate_current={}
max_depth_current={}
subsample_current={}
colsample_bytree_current={}

for target in targets:
    n_estimators_current[target] = 550
    learning_rate_current[target] = 0.505
    max_depth_current[target] = 4
    subsample_current[target] = 0.75
    colsample_bytree_current[target] = 0.75

n_estimators_best={}
learning_rate_best={}
max_depth_best={}
subsample_best={}
colsample_bytree_best={}

for target in targets:
    n_estimators_best[target] = 550
    learning_rate_best[target] = 0.505
    max_depth_best[target] = 4
    subsample_best[target] = 0.75
    colsample_bytree_best[target] = 0.75

    bestResults[target] = {"mse": 10000, "r2": -100000}


print(f'Start Time:\n{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
next_update = time.time() + 15 * 60
next_thinning = next_update + 105 * 60

while True:
    for target in targets:

        n_estimators=random.randint(max(n_estimators_absolute_min, n_estimators_current[target] - n_estimators_range),min(n_estimators_absolute_max, n_estimators_current[target] + n_estimators_range))
        learning_rate=random.uniform(max(learning_rate_absolute_min, learning_rate_current[target] - learning_rate_range),min(learning_rate_absolute_max, learning_rate_current[target] + learning_rate_range))
        max_depth=random.randint(max(max_depth_absolute_min, max_depth_current[target] - max_depth_range),min(max_depth_absolute_max, max_depth_current[target] + max_depth_range))
        subsample=random.uniform(max(subsample_absolute_min, subsample_current[target] - subsample_range),min(subsample_absolute_max, subsample_current[target] + subsample_range))
        colsample_bytree=random.uniform(max(colsample_bytree_absolute_min, colsample_bytree_current[target] - colsample_bytree_range),min(colsample_bytree_absolute_max, colsample_bytree_current[target] + colsample_bytree_range))
        

        model = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("xgb", XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=1993
            ))
        ])

        cv_scores = cross_validate(model, X, y_dict[target], cv=5, scoring=('r2', 'neg_mean_squared_error'))

        r2 = cv_scores['test_r2'].mean()
        mse = -cv_scores['test_neg_mean_squared_error'].mean()

        if r2 > bestResults[target]['r2']:
            bestModels[target] = model
            bestResults[target] = {"mse": mse, "r2": r2}
            n_estimators_best[target] = n_estimators
            learning_rate_best[target] = learning_rate
            max_depth_best[target] = max_depth
            subsample_best[target] = subsample
            colsample_bytree_best[target] = colsample_bytree
            print(f'\n\nNew best {target} model found!\nTime: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n  MSE: {mse:.4f}\n  R² : {r2:.4f}')

    current_time = time.time()

    if current_time >= next_update:
        print(f'\n\n\n\n\n\nBest model performance (as of {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}):')
        for target, metrics in bestResults.items():
            print(f"{target}")
            print(f"  MSE: {metrics['mse']:.4f}")
            print(f"  R² : {metrics['r2']:.4f}\n")
            print(f'    Hyperparameters:\n{n_estimators_best[target]}\n{learning_rate_best[target]}\n{max_depth_best[target]}\n{subsample_best[target]}\n{colsample_bytree_best[target]}')
        next_update += 15*60

    if current_time >= next_thinning:
        n_estimators_range /= 2
        learning_rate_range /= 2
        subsample_range /= 2
        colsample_bytree_range /= 2
        if max_depth_range > 1:
            max_depth_range /= 2
        else:
            max_depth_range = 0

        for target in targets:
            n_estimators_current[target] = n_estimators_best[target]
            learning_rate_current[target] = learning_rate_best[target]
            max_depth_current[target] = max_depth_best[target]
            subsample_current[target] = subsample_best[target]
            colsample_bytree_current[target] = colsample_bytree_best[target]

        next_thinning += 90*60