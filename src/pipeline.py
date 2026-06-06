import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class AdvHousingPipeline(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.modes = {}
        self.lot_frontage_median = 70.0
        
    def fit(self, X, y=None):
        for col in ['MSZoning', 'Exterior1st', 'SaleType']:
            if col in X.columns:
                self.modes[col] = X[col].mode()[0]
        if 'LotFrontage' in X.columns:
            self.lot_frontage_median = X['LotFrontage'].median()
        return self
        
    def transform(self, X):
        df_out = X.copy()
        
        if 'LotFrontage' in df_out.columns:
            df_out['LotFrontage'] = df_out['LotFrontage'].fillna(self.lot_frontage_median)
        df_out['LotArea_log'] = np.log1p(df_out['LotArea'])
        df_out['LotFrontage_log'] = np.log1p(df_out['LotFrontage'])
        
        for col, mode_val in self.modes.items():
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna(mode_val)
                
        if 'KitchenQual' in df_out.columns:
            df_out['KitchenQual'] = df_out['KitchenQual'].fillna('TA')
        

        df_out['OverallGrade'] = (df_out['OverallQual'] ** 2) + df_out['OverallCond']
        df_out['Age'] = df_out['YrSold'] - df_out['YearBuilt']
        df_out['AvgRoomSize'] = df_out['GrLivArea'] / df_out['TotRmsAbvGrd'].fillna(6)
        
        for zero_col in ['BsmtFinSF1', 'TotalBsmtSF', 'BsmtUnfSF', 'MasVnrArea']:
            if zero_col in df_out.columns:
                df_out[zero_col] = df_out[zero_col].fillna(0.0)
        
        f_bath = df_out['FullBath'].fillna(0)
        h_bath = df_out['HalfBath'].fillna(0)
        bf_bath = df_out['BsmtFullBath'].fillna(0)
        bh_bath = df_out['BsmtHalfBath'].fillna(0)
        df_out['TotalBathrooms'] = f_bath + (h_bath * 0.5) + bf_bath + (bh_bath * 0.5)
        
        porch_features = ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
        df_out['TotalOutdoorSF'] = df_out[porch_features].fillna(0).sum(axis=1)
        
        # 4. Binary Hurdle Flags (Zero-Inflation Controls)
        df_out['Has_MasVnr'] = (df_out['MasVnrArea'] > 0).astype(int)
        df_out['Has_Basement'] = (df_out['TotalBsmtSF'] > 0).astype(int)
        df_out['Has2ndFloor'] = (df_out['2ndFlrSF'].fillna(0) > 0).astype(int)
        df_out['Has_OutdoorSpace'] = (df_out['TotalOutdoorSF'] > 0).astype(int)
        
        # 5. High-Homogeneity Structural Hurdle Binary Splits
        df_out['Is_Functional'] = (df_out['Functional'].fillna('Typ') == 'Typ').astype(int)
        df_out['Is_Normal_Proximity'] = (df_out['Condition1'].fillna('Norm') == 'Norm').astype(int)
        df_out['Is_Hip_Roof'] = (df_out['RoofStyle'].fillna('Gable') == 'Hip').astype(int)
        df_out['Is_Standard_Electrical'] = (df_out['Electrical'].fillna('SBrkr') == 'SBrkr').astype(int)
        df_out['Is_Premium_MasVnr'] = df_out['MasVnrType'].fillna('None').isin(['BrkFace', 'Stone']).astype(int)
        
        qual_scale = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
        fin_scale = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0}
        
        ordinal_blueprints = {
            'ExterQual': qual_scale, 'BsmtQual': qual_scale, 'KitchenQual': qual_scale, 'HeatingQC': qual_scale,
            'BsmtFinType1': fin_scale, 'BsmtFinType2': fin_scale, 'FireplaceQu': qual_scale,
            'GarageFinish': {'Fin': 3, 'RFn': 2, 'Unf': 1, 'NA': 0},
            'PavedDrive': {'Y': 3, 'P': 2, 'N': 1},
            'LotShape': {'Reg': 1, 'IR1': 2, 'IR2': 3, 'IR3': 4},
            'LandSlope': {'Gtl': 1, 'Mod': 2, 'Sev': 3},
            'Fence': {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'NA': 0}
        }
        
        for col, target_mapping in ordinal_blueprints.items():
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna('NA').map(target_mapping).fillna(0).astype(int)
                
        if 'GarageType' in df_out.columns:
            df_out['GarageType'] = df_out['GarageType'].fillna('NA')
        if 'BsmtExposure' in df_out.columns:
            df_out['BsmtExposure'] = df_out['BsmtExposure'].fillna('No')
            
        redundant_drops = [
            'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath', 'YearBuilt', 'YrSold',
            'TotRmsAbvGrd', 'KitchenAbvGr', 'GarageArea', 'GarageYrBlt', 'BsmtFinSF2',
            'LowQualFinSF', '2ndFlrSF', 'PoolArea', 'MiscVal', 'PoolQC', 'MiscFeature',
            'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
            'Utilities', 'LandContour', 'LotConfig', 'Condition2', 'Exterior2nd', 
            'RoofMatl', 'Heating', 'BsmtCond', 'GarageQual', 'GarageCond', 'Alley', 
            'Street', 'Functional', 'Condition1', 'RoofStyle', 'Electrical', 'MasVnrType'
        ]
        df_out = df_out.drop(columns=[c for c in redundant_drops if c in df_out.columns], errors='ignore')
        return df_out