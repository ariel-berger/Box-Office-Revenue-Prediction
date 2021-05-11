import numpy as np
import sklearn.datasets
from sklearn.model_selection import KFold
from preprocess import preprocessor
import pandas as pd

import optuna.integration.lightgbm as lgb

if __name__ == "__main__":

	### Load the data ###
	train = pd.read_csv("train.tsv", sep='\t')
	test = pd.read_csv("test.tsv", sep='\t')

	### preprocess the data ###
	prep = preprocessor()
	train = prep.fit_transform(train)
	test = prep.transform(test)
	data_test = test.drop(['revenue'], axis=1)
	logtarget_test = np.log1p(test.revenue)
	data = train.drop(['revenue'], axis=1)
	target = train.revenue
	logtarget = np.log1p(target)
	dtrain = lgb.Dataset(data, label=logtarget)

	### set the parameters and optimize the hiper-parameters ####
	params = {
		"objective": "rmse",
		"metric": "rmse",
		"verbosity": -1,
		"boosting_type": "gbdt",
	}

	tuner = lgb.LightGBMTunerCV(
		params, dtrain, verbose_eval=100, early_stopping_rounds=100, folds=KFold(n_splits=10), return_cvbooster=True
	)
	tuner.run()
	### Print the results ###
	print("Best score:", tuner.best_score)
	best_params = tuner.best_params
	print("Best params:", best_params)
	print("  Params: ")
	for key, value in best_params.items():
		print("    {}: {}".format(key, value))

	### Save the best model ####
	# model = tuner.get_best_booster()
	# model.save_model('lgbm_model.txt')


	#### Run the trained (best) model ####
	# bst = lgb.Booster(model_file='lgbm_model.txt')  # init model
	# preds = bst.predict(data_test)
	# rmsle = sklearn.metrics.mean_squared_error(logtarget_test, preds, squared=False)
	# print(f"rmsle:{rmsle}")
