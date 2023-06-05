import pandas as pd
import numpy as np
import sys, os
from TinyStatistician import TinyStatistician as Tstat

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

def describe(data, features):
	statistics = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
	lst_stat = []
	for _ in range(len(statistics)):
		lst_stat.append([])

	for column in range(data.shape[1]):
		feature_data = data[:, column]
		feature_data = feature_data[np.logical_and(~np.isnan(feature_data), feature_data is not None)]

		feature_name = f"{features[column]}"
		feature_stats = [len(feature_data), Tstat().mean(feature_data), Tstat().std(feature_data), Tstat().min(feature_data)]
		feature_stats.append(Tstat().percentile(feature_data, 25))
		feature_stats.append(Tstat().percentile(feature_data, 50))
		feature_stats.append(Tstat().percentile(feature_data, 75))
		feature_stats.append(Tstat().max(feature_data))

		lst_stat[0].append(feature_stats[0])
		lst_stat[1].append(feature_stats[1])
		lst_stat[2].append(feature_stats[2])
		lst_stat[3].append(feature_stats[3])
		lst_stat[4].append(feature_stats[4])
		lst_stat[5].append(feature_stats[5])
		lst_stat[6].append(feature_stats[6])
		lst_stat[7].append(feature_stats[7])
		
	return pd.DataFrame(lst_stat, index=statistics, columns=features)


def compare_df(real, mine):
	diff = real.compare(mine)
	if diff.empty:
		print("The DataFrames are identical.")
	else:
		print("The DataFrames are not identical. Differences:")
		print(diff)
	# Check for missing or null values in each DataFrame
	real_null = real.isnull()
	mine_null = mine.isnull()
	
	# Compare the results to identify any differences
	if real_null.equals(mine_null):
		print("The DataFrames have the same missing or null values.")
	else:
		print("The DataFrames have different missing or null values.")
		print("Missing or null values in real:")
		print(real_null)
		print("Missing or null values in mine:")
		print(mine_null)
	
if __name__ == "__main__":
	if len(sys.argv) != 2:
		print(f"Usage: {sys.argv[0]} [data path]")
	else:
		# Load the data
		df, features = load_data(sys.argv[1])
		#print(f"features: {features}")

		numerical_df = df.select_dtypes(include='number')
		print("numerical_df shape:", numerical_df.shape)

		numerical_features = numerical_df.columns.tolist()
		#print(f"numerical_features: {numerical_features}")

		numerical_data = numerical_df.values
		#print("numerical_data shape:", numerical_data.shape)

		real = numerical_df.describe()
		mine = describe(numerical_data, numerical_features)
		#print(real)
		print(mine)
		#compare_df(real, mine)
