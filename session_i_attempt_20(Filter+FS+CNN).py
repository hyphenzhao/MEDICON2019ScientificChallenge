from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from BasicDataProcess import BasicDataProcess
from sklearn import svm
from sklearn.svm import LinearSVC
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from keras.utils import to_categorical
import time
import numpy as np
import os
import traceback
from scipy.fftpack import rfft, irfft

timestr = time.strftime("%Y%m%d-%H%M%S")
result_file_name = "Result-" + timestr + ".csv"
result_file = open(result_file_name, "w+")
result_file.write("SUBJECT,SESSION,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,\n")

all_results = []
successful = True
def GetNonTargetsAverage(train_inputs, train_targets):
	print("--Get non-targets average matrix--")
	non_targets = None
	count = 0
	for i in range(len(train_targets)):
		if train_targets[i] == 0:
			count += 1
			if non_targets is None:
				non_targets = train_inputs[i]
			else:
				non_targets = np.add(non_targets, train_inputs[i])
	non_targets = non_targets / float(count)
	print("--Done--")
	return non_targets

def ApplySpecialFilter(inputs, filter_feature, reshaped):
	print("--Apply special filter to the inputs--")
	result = []
	for single_input in inputs:
		epoch = []
		for j in range(8):
			input_fft = rfft(single_input[j])
			filter_fft = rfft(filter_feature[j])
			output = irfft(input_fft - filter_fft)
			epoch.append(output)
		if reshaped:
			result.append(np.array(epoch).reshape(-1))
		else:
			result.append(np.array(epoch))
	print("--Done--")
	return np.array(result)

def PreprocessData(data_dir, filter_applied, pca_applied, pca_threshold, reshaped, data_size, input_data, width):
	print("--Load " + data_dir + " Successfully! Now start processing--")
	print("--Applying Low Pass Filter and reshape to input data(Train)...--")
	if filter_applied:
		print("Low pass filter: YES")
	else:
		print("Low pass filter: NO")
	if pca_applied:
		result = BasicDataProcess.GetFeatureByPCA(input_data, data_size, filter_applied, pca_threshold, reshaped, width)
		print("PCA Applied with threshold: " + str(pca_threshold))
	else:
		result = BasicDataProcess.GetP300Inputs(input_data, filter_applied, data_size, reshaped, width)
		print("PCA Applied: NO")
	if reshaped:
		print("Data set has been reshaped to 1D.")
	print("--Input data preprocessed!--")
	return result
try:
	for sbj_no in range(1, 16):
		print("===================SBJ%02d===================" % sbj_no)
		sbj_folder = "./SBJ%02d" % sbj_no
		for session_no in range(1, 4):
			# File IO
			data_dir = sbj_folder + "/S0" + str(session_no)
			train_data = BasicDataProcess.LoadEEGFromFile(data_dir, True)
			train_targets = BasicDataProcess.LoadDataFromFile(data_dir + "/Train/trainTargets.txt")
			


			# Apply basic preprocess methods
			sample_width = 200
			train_inputs = PreprocessData(data_dir, 
				filter_applied = True,
				pca_applied = False,
				pca_threshold = 20,
				reshaped = False,
				data_size = len(train_targets),
				input_data = train_data,
				width = sample_width)
			filter_feature = GetNonTargetsAverage(train_inputs, train_targets)
			train_inputs = ApplySpecialFilter(train_inputs, filter_feature, reshaped = True)
			print(train_inputs.shape)
			# Modeling
			train_targets = np.array(train_targets)
			lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(train_inputs, train_targets)
			sel_model = SelectFromModel(lsvc, prefit=True)
			train_inputs = sel_model.transform(train_inputs)
			print(train_inputs.shape)
			train_inputs = train_inputs.reshape(train_inputs.shape[0], train_inputs.shape[1], 1)
			#train_inputs = BasicDataProcess.TransposeElements(train_inputs)
			train_targets = to_categorical(train_targets)
			#create model
			model = Sequential()
			#add model layers
			model.add(Conv1D(128, kernel_size=4, activation='relu'))
			model.add(Conv1D(64, kernel_size=4, activation='relu'))
			model.add(Conv1D(32, kernel_size=4, activation='relu'))
			model.add(Conv1D(16, kernel_size=4, activation='relu'))
			model.add(Flatten())
			model.add(Dense(2, activation='softmax'))
			#compile model using accuracy to measure model performance
			model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
			model.fit(train_inputs, train_targets, epochs=30)


			# Prediction
			test_events = BasicDataProcess.LoadDataFromFile(data_dir + "/Test/testEvents.txt")
			test_data = BasicDataProcess.LoadEEGFromFile(data_dir, False)
			test_inputs = PreprocessData(data_dir, 
				filter_applied = True,
				pca_applied = False,
				pca_threshold = 20,
				reshaped = False,
				data_size = len(test_events),
				input_data = test_data,
				width = sample_width)
			test_inputs = ApplySpecialFilter(test_inputs, filter_feature, reshaped = True)
			test_inputs = sel_model.transform(test_inputs)
			test_inputs = test_inputs.reshape(test_inputs.shape[0], test_inputs.shape[1], 1)
			print(test_inputs.shape)
			raw_results = model.predict_proba(test_inputs)
			test_targets = BasicDataProcess.GetTargetResults(raw_results)
			test_votes = BasicDataProcess.GetLabelResults(test_targets, test_events)
			test_runs_per_block = BasicDataProcess.LoadDataFromFile(data_dir + "/Test/runs_per_block.txt")[0]
			test_labels = BasicDataProcess.GetLabels(test_votes, test_runs_per_block)
			all_results.append(test_labels)
			print(test_labels)
			result_file.write(str(sbj_no) + "," + str(session_no))
			for i in test_labels:
				result_file.write("," + str(i))
			result_file.write("\n")
			print("-------------------------------------------")

		print("========================================")
except Exception as e:
	traceback.print_exc()
	print(e)
	result_file.close()
	os.remove(result_file_name)
	successful = False
if successful:
	print("Saving to files..." + result_file_name + " Saved! Assessing result...")
	result_file.close()
	os.system("python3 result_compare.py " + result_file_name)