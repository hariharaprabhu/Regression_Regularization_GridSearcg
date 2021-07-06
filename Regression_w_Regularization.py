# Import dependencies
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import PolynomialFeatures  # Adding Polynomial Features
import pandas as pd
from xgboost.sklearn import XGBRegressor
import numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_rows", None, "display.max_columns", None)


# Extract the boston data set
boston_data = load_boston()
boston_df = pd.DataFrame(data=boston_data.data, columns=boston_data.feature_names)
input_feat = boston_df.columns
boston_df['prices'] = boston_data.target

# Perform some inspection
print(boston_df.info())

# Perform Pre-Processing
# Check to see if any columns can be reduced and identify the relationship to the target variable price
corr = boston_df.corr()
corr_price = corr.loc['prices'].abs().sort_values(ascending=False)

# Selecting variables with higher correlation to see if these variable influence model output
index_col = np.array(corr_price.index)[(corr_price > 0.3) & (corr_price < 1.0)]

# Split the data into Training and Testing Set
X_train, X_test, y_train, y_test = train_test_split(boston_df[input_feat], boston_df['prices'], test_size=0.3,
                                                    random_state=12)

# Steps to Do Perform Pre-Processing
# Apply Normalization
# Compare Model Performance With and Without Normalization
# Perform Regularization to see if it minimizes bias
# Try Different Models and Report the Findings
# Use Same accuracy metric consistent across all models

models_try = [LinearRegression(), Lasso(), Ridge(),XGBRegressor()]
model_params = [{}, {'alpha': np.arange(1, 100, 5)}, {'alpha': np.arange(0, 1, 0.01)},
                    {'nthread':[4], 'objective':['reg:squarederror'], 'learning_rate': [.03, 0.05, .07], 'max_depth': [5, 6, 7],
                        'min_child_weight': [4],'subsample': [0.7],'colsample_bytree': [0.7],'n_estimators': [100]
                     }
               ]
# lasso_params = {'fit__alpha': np.arange(0, 1, 0.01)}

BEST_MODEL_SCORE = {}
BEST_TEST_ACC = 0


def get_best_model(train_df, test_df, models_try):
    """

    This function will return the best model pipeline that maximizes the r2 score on the holdout dataset

    :param train_df:  Provide the training data set & target label in this format [df1.features, df1.target]
    :param test_df: Provide the testing data set & target label in this format [df1.features, df1.target]
    :param models_try: Provide the models you would like to try in an array format
    :return: BEST_MODEL_SCORE: This variable would be of type array it would return the best pipeline along with the accuracy metric

    """

    for i in range(0, len(models_try)):
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=12)
        steps = [('scalar', StandardScaler()), ('Polynomial', PolynomialFeatures(degree=2))]

        if i > 0:
            steps.append((str(models_try[i]), GridSearchCV(models_try[i], param_grid=model_params[i], cv=cv)))
        else:
            steps.append((str(models_try[i]), models_try[i]))

        model = Pipeline(steps)

        # Added line

        model.fit(train_df[0], train_df[1])


        train_accuracy = model.score(train_df[0], train_df[1])
        # print("Model Accuracy =", model.score(X_train, y_train))
        test_accuracy = model.score(test_df[0], test_df[1])

        print("Model = ", str(models_try[i]))
        print("Test Accuracy = ", test_accuracy, "\n", "Training accuracy =", train_accuracy)

        global BEST_TEST_ACC
        if test_accuracy > BEST_TEST_ACC:
            BEST_TEST_ACC = test_accuracy
            BEST_MODEL_SCORE[0] = [model, BEST_TEST_ACC]

    return BEST_MODEL_SCORE


def plot_graph(model, df, df_type='Testing'):
    X, y = df[0], df[1]
    y_pred = model[0][0].predict(X)
    score = model[0][1]
    plt.scatter(y, y_pred)
    label_name = 'Actual vs Prediction Model Accuracy on ' + df_type + ' dataset =' + str(score)
    plt.title(label=label_name)
    plt.show()
    # print("Model Test Accuracy =", model.score(X_test, y_test))


best_model = get_best_model([X_train, y_train], [X_test, y_test], models_try)
plot_graph(best_model, [X_test, y_test], 'Testing')
