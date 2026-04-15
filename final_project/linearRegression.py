import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np


def stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    elif p < 0.1:
        return '.'
    else:
        return ''


data = pd.read_csv('water_pollution_disease_fixed_column_titles.csv')

data = data.dropna()

data.columns = (
    data.columns
    .str.strip()
    .str.replace(r"[^\w]+", "_", regex=True)
)

targets = ['Diarrheal_Cases_per_100_000_people','Cholera_Cases_per_100_000_people','Typhoid_Cases_per_100_000_people','Infant_Mortality_Rate_per_1_000_live_births_']


for a in targets:
    loops = 0
    worst_pval = 1
    features = [col for col in data.columns if col not in targets and col != 'Country' and col != 'Region' and col != 'Water_Treatment_Method']
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=7)
    while loops < 15 and worst_pval > 0.05:
        loops += 1
        formula = a + '~' + '+'.join(features)

        model = smf.ols(formula = formula, data = train_data).fit()

        y_true = test_data[a]
        y_pred = model.predict(test_data)

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        summary_df = model.summary2().tables[1]

        summary_df['signif'] = summary_df['P>|t|'].apply(stars)

        numeric_features = [
            col for col in features
            if not col.startswith("C(")
        ]

        numeric_table = summary_df[summary_df.index.isin(numeric_features)]

        worst_feature = numeric_table['P>|t|'].idxmax()
        worst_pval = numeric_table['P>|t|'].max()

        features.remove(worst_feature)
    
    formula = a + '~' + '+'.join(features)

    model = smf.ols(formula = formula, data = train_data).fit()

    y_true = test_data[a]
    y_pred = model.predict(test_data)

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    summary_df = model.summary2().tables[1]

    summary_df['signif'] = summary_df['P>|t|'].apply(stars)

    print(f'\n\n\n\n\nTarget Variable - {a}:')

    print(summary_df)

    print(f"R^2: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")