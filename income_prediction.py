import pandas as pd

def vis_check(data):
    print(data.describe())
    print()
    print(data.columns)
    print()
    print(data.head())
    return

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state = 0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae

# load data onto pandas dataframe
household_data_path = 'C:/Users/Eo/Desktop/Coding/filipino_household/Family Income and Expenditure.csv'
household_data = pd.read_csv(household_data_path)

# look at meaningful metrics
vis_check(household_data)

# drop missing values
household_data = household_data.dropna(axis=0)

# pick prediction target
y = household_data['Total Household Income']

# choose arbitrary features
household_features = ['Total Food Expenditure', 
                  'Clothing, Footwear and Other Wear Expenditure', 'Housing and water Expenditure',
                  'Medical Care Expenditure', 'Transportation Expenditure', 'Communication Expenditure',
                  'Education Expenditure', 'Total Income from Entrepreneurial Acitivites',
                  'Total Number of Family members', 'Total number of family members employed',
                  'House Floor Area', 'House Age', 'Number of bedrooms', 'Electricity']

X = household_data[household_features]
vis_check(X)

###

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# split data into training and validation
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# define model
household_model = DecisionTreeRegressor()
# fit model
household_model.fit(train_X, train_y)

from sklearn.metrics import mean_absolute_error

predicted_income_prices = household_model.predict(val_X)
print(mean_absolute_error(val_y, predicted_income_prices))

### test split with least error
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
for max_leaf_nodes in candidate_max_leaf_nodes:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d \t\t Mean Absolute Error: %d" %(max_leaf_nodes, my_mae))

# store best value of max_leaf_nodes
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores, key = scores.get)
print(best_tree_size)

# fit final model
final_household_model = DecisionTreeRegressor(max_leaf_nodes = best_tree_size, random_state = 1)
final_household_model.fit(train_X, train_y)

# predict
predicted_with_leaves = final_household_model.predict(val_X)
print(mean_absolute_error(val_y, predicted_with_leaves))

###

# using a random forest model

from sklearn.ensemble import RandomForestRegressor

# define the model

rf_model = RandomForestRegressor(random_state = 1)

# fit model
rf_model.fit(train_X, train_y)

# calculate mae
rf_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_predictions, val_y)

print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))