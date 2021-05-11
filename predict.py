import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import ast
import pickle
import optuna.integration.lightgbm as lgb
import sklearn.metrics
# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('tsv_path', type=str, help='tsv file path')
args = parser.parse_args()

# Reading input TSV
data_raw = pd.read_csv(args.tsv_path, sep="\t")

#####
with open('preprocessor.pkl', 'rb') as input:
	prep = pickle.load(input)
data = prep.transform(data_raw)

# logtarget = np.log1p(data.revenue)
# data = data.drop(['revenue'],axis = 1)

bst_lgbm = lgb.Booster(model_file='lgbm_model.txt')  # init model
preds_lgbm = bst_lgbm.predict(data)

bst_xgb = xgb.Booster()
bst_xgb.load_model("xgb_model.txt")
dtest = xgb.DMatrix(data)
preds_xgb = bst_xgb.predict(dtest)

preds = (preds_lgbm + preds_xgb)/2
preds = np.expm1(preds)


prediction_df = pd.DataFrame(columns=['id', 'revenue'])
prediction_df['id'] = data_raw['id']
prediction_df['revenue'] = preds

prediction_df.to_csv("prediction.csv", index=False, header=False)
