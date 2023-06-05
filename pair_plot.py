import sys
import seaborn as sns
import matplotlib.pyplot as plt
from describe import load_data

def display_pair_plot(df):
	# Select relevant columns representing Hogwarts House and course scores
	numerical_df = df.select_dtypes(include='number')
	print("numerical_df shape:", numerical_df.shape)
	numerical_features = numerical_df.columns.tolist()

	# Calculate the correlation matrix
	correlation_matrix = df[numerical_features].corr()
	print("correlation matrix:\n", correlation_matrix)

	# Select the features with the highest absolute correlation
	selected_features = correlation_matrix.abs().sum().nlargest(5).index.tolist()

	subset_data = df[['Hogwarts House'] + selected_features]
	sns.pairplot(subset_data, hue='Hogwarts House', diag_kind='hist')
	plt.show()

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print(f"Usage: python {sys.argv[0]} [data path]")
	else:
		df, features = load_data(sys.argv[1])
		display_pair_plot(df)
