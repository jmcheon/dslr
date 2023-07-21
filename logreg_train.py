import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os, itertools, pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data(path):
	print(f"path: {path}")
	try:
		df = pd.read_csv(path, index_col=0)
	except:
		print("Invalid file error.")
		sys.exit()
	print("df shape:", df.shape)
	features = df.columns.tolist()

	return (df, features)

def normalization(data):
	data_min = np.min(data, axis=0)
	data_max = np.max(data, axis=0)
	normalized_data = (data - data_min) / (data_max - data_min)
	return normalized_data, data_min, data_max

def denormalization(normalized_data, data_min, data_max):
	x = normalized_data * (data_max - data_min)
	denormalized_data = normalized_data * (data_max - data_min) + data_min
	return denormalized_data

def denormalize_thetas(thetas, data_max, data_min):
	# Recover the slope of the line
	slope = thetas[1] * (data_max[1] - data_min[1]) / (data_max[0] - data_min[0])
	# Recover the intercept of the line
	intercept = thetas[0] * (data_max[1] - data_min[1]) + data_min[1] - slope * data_min[0]
	denormalized_thetas = np.array([intercept, slope]).reshape(-1, 1)
	return denormalized_thetas

def label_data(y, house):
	y_ = np.zeros(y.shape)
	y_[np.where(y == int(house))] = 1
	y_labelled = y_.reshape(-1, 1)
	#print("y_labelled shape:", y_labelled.shape)
	#print("y_labelled[:5]:", y_labelled[:5])
	return y_labelled

def data_spliter_by(x, y, house):
	#print("y:", y, "house:", house)
	y_ = np.zeros(y.shape)
	y_[np.where(y == (house))] = 1
	y_labelled = y_.reshape(-1, 1)
	#print("y_labelled shape:", y_labelled.shape)
	#print("y_labelled[:5]:", y_labelled[:5])
	return train_test_split(x, y_labelled, test_size=0.2, random_state=42)

def gradient(x, y, theta):
	"""
	Computes a gradient vector from three non-empty numpy.ndarray, without any for-loop. The three arrays must have compatible
	"""
	for v in [x, y, theta]:
		if not isinstance(v, np.ndarray):
			print(f"Invalid input: argument {v} of ndarray type required")
			return None

	if not x.ndim == 2:
		print(f"Invalid input: wrong shape of x", x.shape)
		return None

	if y.ndim == 1:
		y = y.reshape(y.size, 1)
	elif not (y.ndim == 2 and y.shape[1] == 1):
		print(f"Invalid input: wrong shape of y", y.shape)
		return None

	if x.shape[0] != y.shape[0]:
		print(f"Invalid input: two vectors of compatible shape are required")
		return None

	if theta.ndim == 1 and theta.size == x.shape[1] + 1:
		theta = theta.reshape(x.shape[1] + 1, 1)
	elif not (theta.ndim == 2 and theta.shape == (x.shape[1] + 1, 1)):
		print(f"Invalid input: wrong shape of theta", theta.shape)
		return None

	X = np.hstack((np.ones((x.shape[0], 1)), x))
	y_hat = np.array(1 / (1 + np.exp(-X.dot(theta))))
	gradient = np.dot(X.T, (y_hat - y))
	return gradient / x.shape[0]

def loss_(y, y_hat, eps=1e-15):
	"""
	Compute the logistic loss value.
	"""
	for v in [y, y_hat]:
		if not isinstance(v, np.ndarray):
			print(f"Invalid input: argument {v} of ndarray type required")
			return None

	if not isinstance(eps, float):
		print(f"Invalid input: argument esp of float type required")	
		return None
	
	v = [y, y_hat]
	for i in range(len(v)): 
		if v[i].ndim == 1:
			v[i] = v[i].reshape(v[i].size, 1)
		elif not (v[i].ndim == 2 and v[i].shape[1] == 1):
			print(f"Invalid input: wrong shape of {v[i]}", v[i].shape)
			return None
	y, y_hat = v
	if y.shape != y_hat.shape:
		print(f"Invalid input: two vectors of compatible shape are required")
		return None
	# Clip values to avoid numerical instability
	y_hat = np.clip(y_hat, eps, 1 - eps)  
	loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
	return float(loss)

def batch_fit(x, y, thetas, alpha, max_iter):
	for v in [x, y, thetas]:
		if not isinstance(v, np.ndarray):
			print(f"Invalid input: argument {v} of ndarray type required")
			return None

	if not x.ndim == 2:
		print(f"Invalid input: wrong shape of x", x.shape)
		return None

	if y.ndim == 1:
		y = y.reshape(y.size, 1)
	elif not (y.ndim == 2 and y.shape[1] == 1):
		print(f"Invalid input: wrong shape of y", y.shape)
		return None
	
	if x.shape[0] != y.shape[0]:
		print(f"Invalid input: x, y matrices should be compatible.", x.shape[0], y.shape[0])
		return None

	if thetas.ndim == 1 and thetas.size == x.shape[1] + 1:
		thetas = thetas.reshape(x.shape[1] + 1, 1)
	elif not (thetas.ndim == 2 and thetas.shape == (x.shape[1] + 1, 1)):
		print(f"Invalid input: wrong shape of {thetas}", thetas.shape)
		return None

	if not isinstance(alpha, float) or alpha <= 0:
		print(f"Invalid input: argument alpha of positive float type required")	
		return None
	if not isinstance(max_iter, int):
		print(f"Invalid input: argument max_iter of int type required")
		return None

	accuracy_list = []
	loss_list = []
	epoch_list = []

	X = np.hstack((np.ones((x.shape[0], 1)), x))
	new_theta = np.copy(thetas.astype("float64"))

	for i in range(max_iter):
		# Compute gradient descent
		y_pred = np.array(1 / (1 + np.exp(-X.dot(new_theta))))
		grad = np.dot(X.T, (y_pred - y)) / len(y)
        # Handle invalid values in the gradient
		if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
			#print("Warning: Invalid values encountered in the gradient. Skipping update.")
			continue
		# Update new_theta
		new_theta -= (alpha * grad)
		#print(y.shape, y_pred.shape, type(y), type(y_pred))
		#print(y[:5], y_pred[:5])
		binary_predictions = (y_pred >= 0.5).astype(int)
		accuracy = accuracy_score(y, binary_predictions)
		loss = loss_(y, y_pred)
		if i % 10 == 0: 
			accuracy_list.append(accuracy)
			loss_list.append(loss)
			epoch_list.append(i)
	thetas = new_theta
	return thetas, accuracy, accuracy_list, loss_list, epoch_list

def fit_(x, y, thetas, alpha, max_iter):
	for v in [x, y, thetas]:
		if not isinstance(v, np.ndarray):
			print(f"Invalid input: argument {v} of ndarray type required")
			return None

	if not x.ndim == 2:
		print(f"Invalid input: wrong shape of x", x.shape)
		return None

	if y.ndim == 1:
		y = y.reshape(y.size, 1)
	elif not (y.ndim == 2 and y.shape[1] == 1):
		print(f"Invalid input: wrong shape of y", y.shape)
		return None
	
	if x.shape[0] != y.shape[0]:
		print(f"Invalid input: x, y matrices should be compatible.", x.shape[0], y.shape[0])
		return None

	if thetas.ndim == 1 and thetas.size == x.shape[1] + 1:
		thetas = thetas.reshape(x.shape[1] + 1, 1)
	elif not (thetas.ndim == 2 and thetas.shape == (x.shape[1] + 1, 1)):
		print(f"Invalid input: wrong shape of {thetas}", thetas.shape)
		return None

	if not isinstance(alpha, float) or alpha <= 0:
		print(f"Invalid input: argument alpha of positive float type required")	
		return None
	if not isinstance(max_iter, int):
		print(f"Invalid input: argument max_iter of int type required")
		return None

	# Weights to update: alpha * mean((y_hat - y) * x) 
	# Bias to update: alpha * mean(y_hat - y)
	X = np.hstack((np.ones((x.shape[0], 1)), x))
	new_theta = np.copy(thetas.astype("float64"))
	i = 0
	for _ in range(max_iter):
		# Compute gradient descent
		y_hat = np.array(1 / (1 + np.exp(-X.dot(new_theta))))
		grad = np.dot(X.T, (y_hat - y)) / len(y)
		#grad = gradient(x, y ,new_theta)
        # Handle invalid values in the gradient
		if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
			#print("Warning: Invalid values encountered in the gradient. Skipping update.")
			continue
		# Update new_theta
		new_theta -= (alpha * grad)
	thetas = new_theta
	return thetas

def predict_(x, thetas):
	for v in [x, thetas]:
		if not isinstance(v, np.ndarray):
			print(f"Invalid input: argument {v} of ndarray type required")
			return None

	if not x.ndim == 2:
		print("Invalid input: wrong shape of x", x.shape)
		return None

	if thetas.ndim == 1 and thetas.size == x.shape[1] + 1:
		thetas = thetas.reshape(x.shape[1] + 1, 1)
	elif not (thetas.ndim == 2 and thetas.shape == (x.shape[1] + 1, 1)):
		print(f"p Invalid input: wrong shape of {thetas}", thetas.shape)
		return None
	
	X = np.hstack((np.ones((x.shape[0], 1)), x))
	return np.array(1 / (1 + np.exp(-X.dot(thetas))))

def train(data, x_features, target_categories):
	print(f"Starting training each classifier for logistic regression...")
	x = data[:,:-1]
	weights = []
	fig, axes = plt.subplots(1, 2, figsize=(10, 8))
	for house in range(len(target_categories)):
		print(f"Current house: {house}")
		y_labelled = label_data(data[:,-1], house)
		#theta = fit_(x, y_labelled, np.random.rand(x.shape[1] + 1, 1), 1e-1, 1000)
		theta, accuracy, accuracy_list, loss_list, epoch_list = batch_fit(x, y_labelled, np.random.rand(x.shape[1] + 1, 1), 1e-1, 1000)
		print(accuracy)
		#plt.plot(epoch_list, accuracy_list)
		axes[0].plot(epoch_list, loss_list, label=target_categories[house])
		axes[1].plot(epoch_list, accuracy_list, label=target_categories[house])

		# Save the weights for the current class
		weights.append(theta)
		#print(weights)
		#print("Theta:", theta)
	axes[0].set_xlabel('epoch')
	axes[0].set_ylabel('loss')
	axes[0].set_title('[Batch] Loss vs Epoch by Hogwarts House')
	axes[0].legend()
	axes[1].set_xlabel('epoch')
	axes[1].set_ylabel('accuracy')
	axes[1].set_title('[Batch] Accuracy vs Epoch by Hogwarts House')
	axes[1].legend()
	plt.show()
		
	# Get the directory of the script
	script_dir = os.path.dirname(os.path.realpath(__file__))
	
	# Transpose the theta arrays
	transposed_thetas = [theta.T for theta in weights]
	
	# Concatenate the transposed thetas vertically
	all_thetas = np.concatenate(transposed_thetas, axis=0)
	
	# Create an array of class numbers
	class_numbers = np.arange(4).reshape(-1, 1)
	
	# Create a DataFrame to store the class numbers and theta values
	df = pd.DataFrame(np.concatenate((class_numbers, all_thetas), axis=1), columns=[
	                  'Class'] + ['Theta_' + str(i) for i in range(data[:,:-1].shape[1] + 1)])
	
	# Save the DataFrame to a CSV file
	df.to_csv('weights.csv', index=False)

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print(f"Usage: python {sys.argv[0]} [data path]")
	else:
		# Load the data
		df, features = load_data(sys.argv[1])
		df = df.drop(columns=['Arithmancy', 'Potions',
 								'Care of Magical Creatures'])
		df.dropna(inplace=True)
		target_categories = df['Hogwarts House'].unique()

		# Map unique values to numbers
		mapping = {'Ravenclaw': 0, 'Slytherin': 1, 'Gryffindor': 2, 'Hufflepuff': 3}

		x = df.select_dtypes(include='number')
		normalized_x, data_min, data_max = normalization(x.values)
		y = df['Hogwarts House'].replace(mapping).values
		new_data = np.column_stack((normalized_x, y))
		#print(target_categories)
		train(new_data, features, target_categories)
