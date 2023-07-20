import numpy as np
import pandas as pd
import sys, os, itertools, pickle
from logreg_train import load_data, label_data, normalization, fit_, predict_

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
