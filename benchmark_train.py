import pandas as pd
import numpy as np
import sys, os, itertools, pickle
from sklearn.model_selection import train_test_split

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
	new_loss = 0.0
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

def benchmark_train(data, house):
	y_labelled = label_data(data[:,-1], house)
	x = data[:,:-1]
	
	# Define the range of hyperparameter values to search
	thetas_range = [np.zeros((x.shape[1] + 1, 1)), np.random.rand(x.shape[1] + 1, 1)]
	#alpha_range = [1e-2, 3e-2, 1e-1]
	alpha_range = [1e-1]
	#max_iter_range = [50000, 100000, 300000]
	max_iter_range = [1000]
	
	classifiers = []
	predictions = np.zeros(y_labelled.shape)
	# Perform grid search to find the best hyperparameters
	for thetas, alpha, max_iter in itertools.product(thetas_range, alpha_range, max_iter_range):
		new_thetas = fit_(x, y_labelled, thetas, alpha, max_iter)
		print(f"house: {house}, alpha: {alpha}, max_iter: {max_iter}, thetas: {new_thetas}")
		classifiers.append({'thetas': new_thetas, 'alpha': alpha, 'max_iter': max_iter})
	return classifiers

def benchmark(data, x_features, target_categories):
	print(f"Starting training each classifier for logistic regression...")
	classifiers = {}
	for house in range(len(target_categories)):
		print(f"Current house: {house}")
		classifiers[house] = benchmark_train(data, house)

	filename = "models.pickle"
	with open(filename, 'wb') as file:
        	pickle.dump(classifiers, file)

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
		benchmark(new_data, features, target_categories)
