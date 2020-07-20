import scipy.io
import numpy as np
from scipy import signal
class BasicDataProcess:
	@staticmethod
	def TransposeElements(inputs):
		outputs = []
		for i in inputs:
			outputs.append(i.transpose())
		return np.array(outputs)
	@staticmethod
	def LoadEEGFromFile(data_dir, is_train):
		if is_train:
			train_data = scipy.io.loadmat(data_dir + "/Train/trainData.mat")
		else:
			test_data = scipy.io.loadmat(data_dir + "/Test/testData.mat")
		channels = []
		if is_train:
			for i in train_data['trainData']:
				channels.append(np.array(i).transpose())
		else:
			for i in test_data['testData']:
				channels.append(np.array(i).transpose())
		return channels
	@staticmethod
	def LoadDataFromFile(filename):
		file = open(filename)
		raw_data = file.readlines()
		file.close()
		data = []
		for i in raw_data:
			data.append(int(i))
		return data
	@staticmethod
	def LowPassFilter(fc, data):
		fs = 250.0
		w = fc / (fs / 2.0) # Normalize the frequency
		b, a = signal.butter(9, w, 'low', analog=False)
		output = signal.filtfilt(b, a, data)
		return output
	@staticmethod
	def GetFeatureByPCA(channels, data_size, with_filter, pca_threshold, reshaped, width):
		train_input = []
		for i in range(data_size):
			p300_matrix = []
			for channel in channels:
				left = 125 - width / 2
				right = 125 + width / 2
				raw_channel_data = channel[i][int(left):int(right)]
				if with_filter:
					filtered_data = BasicDataProcess.LowPassFilter(15, raw_channel_data)
					p300_matrix.append(filtered_data.tolist())
				else:
					p300_matrix.append(raw_channel_data)
			pca = PCA()
			p300_pca = pca.fit_transform(p300_matrix)
			if reshaped:
				train_input.append(p300_pca[0:pca_threshold].reshape(-1).tolist())
			else:
				train_input.append(p300_pca[0:pca_threshold].tolist())
		return train_input
	@staticmethod
	def GetP300Inputs(channels, with_filter, data_size, reshaped, width):
		result = []
		for i in range(data_size):
			p300_matrix = []
			for channel in channels:
				left = 125 - width / 2
				right = 125 + width / 2
				raw_channel_data = channel[i][int(left):int(right)]
				if(with_filter):
					filtered_data = BasicDataProcess.LowPassFilter(20, raw_channel_data)
					p300_matrix.append(filtered_data.tolist())
				else:
					p300_matrix.append(raw_channel_data)
			if reshaped:
				result.append(np.array(p300_matrix).reshape(-1).tolist())
			else:
				result.append(np.array(p300_matrix))
		return result
	@staticmethod
	def GetTargetResults(proba_results):
		label_results = np.zeros(len(proba_results), dtype=np.int8)
		pos = 0
		max_proba = 0
		record = 0
		for i in proba_results:
			if(i[1] > max_proba):
				max_proba = i[1]
				record = pos
			pos += 1
			if pos % 8 == 0:
				label_results[record] = 1
				max_proba = 0
		return label_results.tolist()
	@staticmethod
	def GetLabelResults(targets, events):
		result = []
		pos = 0
		for i in targets:
			if i == 1:
				result.append(events[pos])
			pos += 1
		return result
	@staticmethod
	def GetDifferences(a, b, size):
		result = 0
		for i in range(size):
			if a[i] != b[i]:
				result += 1
		return result
	@staticmethod
	def GetMostFrequent(List): 
		counter = 0
		num = List[0] 

		for i in List: 
			curr_frequency = List.count(i) 
			if(curr_frequency > counter): 
				counter = curr_frequency 
				num = i 
		return num
	@staticmethod
	def GetLabels(votes, block_size):
		test_labels = []
		for i in range(len(votes) // block_size):
			block = np.array(votes[i * block_size : (i + 1) * block_size])
			test_labels.append(BasicDataProcess.GetMostFrequent(block.tolist()))
		return test_labels