import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import torch.utils.data
# from torchvision import transforms
from preprocess import preprocessor

import optuna
from optuna.trial import TrialState
from sklearn.model_selection import train_test_split

DEVICE = torch.device("cpu")
BATCHSIZE = 64
DIR = os.getcwd()
EPOCHS = 10
LOG_INTERVAL = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 80
N_VALID_EXAMPLES = BATCHSIZE * 16


def define_model(trial, num_columns):
	# We optimize the number of layers, hidden units and dropout ratio in each layer.
	n_layers = trial.suggest_int("n_layers", 1, 3)
	layers = []

	in_features = num_columns
	for i in range(n_layers):
		out_features = trial.suggest_int("n_units_l{}".format(i), 4, 1024)
		layers.append(nn.Linear(in_features, out_features))
		layers.append(nn.ReLU())
		p = trial.suggest_float("dropout_l{}".format(i), 0.0, 0.8)
		layers.append(nn.Dropout(p))

		in_features = out_features
	layers.append(nn.Linear(in_features, 1))

	return nn.Sequential(*layers)


def my_model(num_columns):
	### the model with the hyper-parameters from the optimizition  ###
	layers = []
	in_features = num_columns

	layers.append(nn.Linear(in_features, 861))
	layers.append(nn.ReLU())
	p = 0.26079935489264505
	layers.append(nn.Dropout(p))

	layers.append(nn.Linear(861, 34))
	layers.append(nn.ReLU())
	p = 0.13432807417875603
	layers.append(nn.Dropout(p))

	layers.append(nn.Linear(34, 1))

	return nn.Sequential(*layers)


class RMSLELoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.mse = nn.MSELoss()

	def forward(self, pred, actual):
		pred[pred < 0] = 1e-6
		return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))


def objective(trial, num_columns, train_loader, valid_loader):
	# Generate the model.
	model = define_model(trial, num_columns).to(DEVICE)

	# Generate the optimizers.
	optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "Adam"])
	lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
	optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

	loss_func = RMSLELoss()
	# Training of the model.
	for epoch in range(EPOCHS):
		model.train()
		for batch_idx, (data, target) in enumerate(train_loader):
			# Limiting training data for faster epochs.
			if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
				break

			data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE).squeeze()

			optimizer.zero_grad()
			output = model(data)
			output[output < 0] = 1e-6
			loss = loss_func(output, target)

			loss.backward()
			optimizer.step()
		# Validation of the model.
		model.eval()
		predictions = []
		true_label = []
		with torch.no_grad():
			for batch_idx, (data, target) in enumerate(valid_loader):
				# Limiting validation data.
				if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
					break

				data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE).squeeze()
				output = model(data)
				predictions.append(output)
				true_label.append(target)

		rmsle = loss_func(torch.stack(predictions).view(-1), torch.stack(true_label).view(-1))

		trial.report(rmsle, epoch)

		# Handle pruning based on the intermediate value.
		if trial.should_prune():
			raise optuna.exceptions.TrialPruned()

	return rmsle


if __name__ == "__main__":
	### Load the data ###
	train_raw = pd.read_csv("train.tsv", sep='\t')
	test_raw = pd.read_csv("test.tsv", sep='\t')

	### preprocess the data ###
	prep = preprocessor()
	train_raw = prep.fit_transform(train_raw)
	test_raw = prep.transform(test_raw)

	### Split to train and validation set ###
	X_train, X_valid, y_train, y_valid = train_test_split(train_raw.drop("revenue", axis=1), train_raw.revenue,
														  test_size=0.20,
														  random_state=1)

	### To check to results on the test ###
	# X_train = train_raw.drop("revenue", axis=1)
	# X_valid = test_raw.drop("revenue", axis=1)
	# y_train = train_raw.revenue
	# y_valid = test_raw.revenue

	### dataframe to dataloader ###
	train_target = torch.tensor(y_train.values.astype(np.float32))
	train = torch.tensor(X_train.values.astype(np.float32))
	train_tensor = torch.utils.data.TensorDataset(train, train_target)
	train_dataloader = torch.utils.data.DataLoader(dataset=train_tensor, batch_size=BATCHSIZE, shuffle=True)

	valid_target = torch.tensor(y_valid.values.astype(np.float32))
	valid = torch.tensor(X_valid.values.astype(np.float32))
	valid_tensor = torch.utils.data.TensorDataset(valid, valid_target)
	valid_dataloader = torch.utils.data.DataLoader(dataset=valid_tensor, batch_size=BATCHSIZE, shuffle=True)

	###### Find The best param for the model #############

	study = optuna.create_study(direction="minimize")
	study.optimize(lambda trial: objective(trial, len(test_raw.columns) - 1, train_dataloader, valid_dataloader),
				   n_trials=200, timeout=1200)

	pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
	complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

	print("Study statistics: ")
	print("  Number of finished trials: ", len(study.trials))
	print("  Number of pruned trials: ", len(pruned_trials))
	print("  Number of complete trials: ", len(complete_trials))

	print("Best trial:")
	trial = study.best_trial

	print("  Value: ", trial.value)

	print("  Params: ")
	for key, value in trial.params.items():
		print("    {}: {}".format(key, value))

	# #### init the best model ####
	# model = my_model(len(test_raw.columns) - 1).to(DEVICE)

	# #### Load the best model and predict on test ####
	# loss_func = RMSLELoss()
	# model.load_state_dict(torch.load("./ann_model.pkl"))
	# model.eval()
	# predictions = []
	# true_label = []
	# for batch_idx, (data, target) in enumerate(valid_dataloader):
	# 	# Limiting validation data.
	# 	if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
	# 		break
	#
	# 	data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE).squeeze()
	# 	output = model(data)
	# 	predictions.append(output)
	# 	true_label.append(target)
	#
	# rmsle = loss_func(torch.stack(predictions).view(-1), torch.stack(true_label).view(-1))
	# print(f"rmlse: {rmsle}")

	# ######### Train the Chosen Model ########
	# # Generate the optimizers.
	# optimizer_name = "SGD"
	# lr = 0.03463793058271601
	# optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
	#
	# loss_func = RMSLELoss()
	# best_rmsle = np.inf
	# best_model = None
	# # Training of the model.
	# for epoch in range(EPOCHS):
	# 	model.train()
	# 	for batch_idx, (data, target) in enumerate(train_dataloader):
	# 		# Limiting training data for faster epochs.
	# 		if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
	# 			break
	#
	# 		data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE).squeeze()
	#
	# 		optimizer.zero_grad()
	# 		output = model(data)
	# 		output[output < 0] = 1e-6
	# 		loss = loss_func(output, target)
	#
	# 		loss.backward()
	# 		optimizer.step()
	# 	# Validation of the model.
	# 	model.eval()
	# 	# correct = 0
	# 	predictions = []
	# 	true_label = []
	# 	with torch.no_grad():
	# 		for batch_idx, (data, target) in enumerate(valid_dataloader):
	# 			# Limiting validation data.
	# 			if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
	# 				break
	#
	# 			data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE).squeeze()
	# 			output = model(data)
	# 			predictions.append(output)
	# 			true_label.append(target)
	#
	# 	rmsle = loss_func(torch.stack(predictions).view(-1), torch.stack(true_label).view(-1))
	# 	if rmsle < best_rmsle:
	# 		best_rmsle = rmsle
	# 		best_model = model
	# print(f"rmlse: {best_rmsle}")
	# # Save the net
	# torch.save(best_model.state_dict(), "./ann_model.pkl")
