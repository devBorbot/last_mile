from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def train_models(X_train, y_train):
    lr = LinearRegression().fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    return lr, rf, gb

def predict_models(models, X_test):
    return [model.predict(X_test) for model in models]