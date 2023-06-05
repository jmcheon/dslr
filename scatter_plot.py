import sys
import matplotlib.pyplot as plt
from describe import load_data

def display_scatter(df):
	numerical_df = df.select_dtypes(include='number')
	print("numerical_df shape:", numerical_df.shape)

	# Select two features that are not the same but have similar values
	selected_features = None
	best_similarity = 0.0
	
	# Iterate over each pair of features in the numerical dataframe
	for i, feature1 in enumerate(numerical_df.columns):
		for j, feature2 in enumerate(numerical_df.columns):
			# Check if the features are not the same
			if feature1 != feature2:
				#print("f1:", feature1, "f2:", feature2)
				# Compute similarity between the two features
				similarity = numerical_df[feature1].corr(numerical_df[feature2])
				#print(similarity)

				if similarity > best_similarity:
					best_similarity = similarity
					selected_features = [feature1, feature2]
	print(f"Selected features: {selected_features}")
	
	plt.scatter(df[selected_features[0]], df[selected_features[1]], color='blue')
	plt.xlabel(f'{selected_features[0]}')
	plt.ylabel(f'{selected_features[1]}')
	plt.title(f'Scatter Plot of Similar Features: [correlation {best_similarity:.2f}]')
	plt.show()


if __name__ == "__main__":
	if len(sys.argv) != 2:
		print(f"Usage: python {sys.argv[0]} [data path]")
	else:
		df, features = load_data(sys.argv[1])
		display_scatter(df)
