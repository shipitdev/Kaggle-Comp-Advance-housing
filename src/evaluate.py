import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from pipeline import AdvHousingPipeline

def load_and_align_data():
    raw_train = pd.read_csv("datasets/train.csv")
    raw_test = pd.read_csv("datasets/test.csv")
    
    train_cleaned = raw_train[raw_train['BsmtFinSF1'].fillna(0) < 5000].copy()
    train_cleaned = train_cleaned[train_cleaned['GrLivArea'].fillna(0) < 4000].copy()
    
    processor = AdvHousingPipeline()
    processor.fit(train_cleaned)
    
    piped_train = processor.transform(train_cleaned)
    piped_test = processor.transform(raw_test)
    
    y_train = np.log1p(piped_train['SalePrice'])
    submission_ids = piped_test['Id']
    
    nominal_targets = ['MSZoning', 'Foundation', 'BsmtExposure', 'GarageType', 'Neighborhood', 'BldgType', 'HouseStyle', 'Exterior1st', 'SaleType', 'SaleCondition']
    piped_train['is_train'] = 1
    piped_test['is_train'] = 0
    
    combined = pd.concat([piped_train, piped_test], axis=0, ignore_index=True)
    combined_encoded = pd.get_dummies(combined, columns=nominal_targets, drop_first=True, dtype=int)
    
    X_train = combined_encoded[combined_encoded['is_train'] == 1].drop(columns=['is_train', 'Id', 'SalePrice'], errors='ignore')
    X_test = combined_encoded[combined_encoded['is_train'] == 0].drop(columns=['is_train', 'Id', 'SalePrice'], errors='ignore')
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    return X_train, X_test, y_train, submission_ids

def run_cross_validation(X, y, X_test):
    price_bins = pd.qcut(y, q=5, labels=False)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        "Ridge": Ridge(alpha=10.0),
        "XGBoost": xgb.XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1),
        "LightGBM": lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
    }
    
    oof_preds = {name: np.zeros(len(X)) for name in models}
    test_preds = {name: np.zeros(len(X_test)) for name in models}
    
    for train_idx, val_idx in skf.split(X, price_bins):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)
        X_te_scaled = scaler.transform(X_test)
        
        for name, model in models.items():
            model.fit(X_tr_scaled, y_tr)
            oof_preds[name][val_idx] = model.predict(X_va_scaled)
            test_preds[name] += model.predict(X_te_scaled) / skf.n_splits
            
    #weighted ensemble
    blended_oof = (0.60 * oof_preds["Ridge"]) + (0.25 * oof_preds["XGBoost"]) + (0.15 * oof_preds["LightGBM"])
    print(f"Blended Ensemble Out-Of-Fold Validation RMSE: {root_mean_squared_error(y, blended_oof):.5f}")
    
    final_test_preds = (0.60 * test_preds["Ridge"]) + (0.25 * test_preds["XGBoost"]) + (0.15 * test_preds["LightGBM"])
    return np.expm1(final_test_preds)

if __name__ == "__main__":
    X_train, X_test, y_train, sub_ids = load_and_align_data()
    final_prices = run_cross_validation(X_train, y_train, X_test)
    
    submission = pd.DataFrame({"Id": sub_ids, "SalePrice": final_prices})
    submission.to_csv("final.csv", index=False)
    print("csv file generated..")