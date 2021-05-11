import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import KFold
import xgboost as xgb
import pandas as pd
from preprocess import preprocessor
import ast
import pickle
import optuna


def objective(trial, df):

	data = df.drop(['revenue'], axis=1)
	logtarget = np.log1p(df.revenue)

	### Kfold cross validation ###
	kf = KFold(n_splits=10, shuffle=True, random_state=42)

	### set the param to optimize ###
	param = {
		"verbosity": 0,
		"objective": "reg:squarederror",
		"eval_metric": "rmse",
		# use exact for small dataset.
		"tree_method": "exact",
		# defines booster, gblinear for linear functions.
		"booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
		# L2 regularization weight.
		"lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
		# L1 regularization weight.
		"alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
		# sampling ratio for training data.
		"subsample": trial.suggest_float("subsample", 1e-8, 1.0),
		# sampling according to each tree.
		"colsample_bytree": trial.suggest_float("colsample_bytree", 1e-8, 1.0),
	}
	if param["booster"] in ["gbtree", "dart"]:
		# maximum depth of the tree, signifies complexity of the tree.
		param["max_depth"] = trial.suggest_int("max_depth", 1, 50, step=2)
		# minimum child weight, larger the term more conservative the tree.
		param["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 30)
		param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
		# defines how selective algorithm is.
		param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
		param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
	if param["booster"] == "dart":
		param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
		param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
		param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
		param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

	models = []
	scores = []
	validScore = 0
	for trainIdx, validIdx in kf.split(data, logtarget):
		### for every fold in the Kfold
		train_x = data.iloc[trainIdx, :]
		train_y = logtarget[trainIdx]
		valid_x = data.iloc[validIdx, :]
		valid_y = logtarget[validIdx]
		dtrain = xgb.DMatrix(train_x, label=train_y)
		dvalid = xgb.DMatrix(valid_x, label=valid_y)
		bst = xgb.train(param, dtrain)
		preds = bst.predict(dvalid)
		rmsle = sklearn.metrics.mean_squared_error(valid_y, preds, squared=False)
		scores.append(rmsle)

	validScore = np.mean(scores)
	return validScore


if __name__ == '__main__':

	### Load the data ###
	train = pd.read_csv("train.tsv", sep='\t')
	test = pd.read_csv("test.tsv", sep='\t')

	### preprocess the data ###
	prep = preprocessor()
	train = prep.fit_transform(train)
	test = prep.transform(test)

	### Find The best param for the model ####
	study = optuna.create_study(direction="minimize")
	study.optimize(lambda trial: objective(trial, train), n_trials=100, timeout=600)

	print("Number of finished trials: ", len(study.trials))
	print("Best trial:")
	trial = study.best_trial

	### Print the results ###
	print("  Value: {}".format(trial.value))
	print("  Params: ")
	bst_prm = ""
	for key, value in trial.params.items():
		bst_prm += "    \"{}\": \"{}\", \n".format(key, value)
	print(bst_prm)

	# ### Save the best model ####
	# with open("xgb_bst_param.txt", "w") as text_file:
	# 	print(f"{bst_prm}", file=text_file)

	# #### Run the trained (best) model ####
	# with open("xgb_bst_param.txt", "r") as text_file:
	# 	param = text_file.read()
	# data = train.drop(['revenue'], axis=1)
	# logtarget = np.log1p(train.revenue)
	# data_test = test.drop(['revenue'], axis=1)
	# logtarget_test = np.log1p(test.revenue)
	# dtrain = xgb.DMatrix(data, label=logtarget)
	# dtest = xgb.DMatrix(data_test, label=logtarget_test)
	# param = ast.literal_eval("{" + param + "}")
	# bst = xgb.train(param, dtrain)
	# preds = bst.predict(dtest)
	# rmsle = sklearn.metrics.mean_squared_error(logtarget_test, preds, squared=False)
	# print(f"rmsle:{rmsle}")
