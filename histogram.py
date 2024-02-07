import sys
import matplotlib.pyplot as plt
from hogwarts_mapping import colors
from utils import load_data


def display_histogram(df):
    # Select relevant columns representing Hogwarts House and course scores
    numerical_df = df.select_dtypes(include='number')
    print("numerical_df shape:", numerical_df.shape)

    numerical_features = numerical_df.columns.tolist()
    df_scores = df[['Hogwarts House'] + numerical_features]

    # Calculate the variance of scores for each course
    variances = numerical_df.var()
    print("variances:\n", variances)

    # Identify the course with the lowest variance
    homogeneous_course = variances.idxmin()
    print("homogeneous course between all four houses:", homogeneous_course)

    # Group the DataFrame by the course column
    grouped = df_scores.groupby('Hogwarts House')

    # Plot histograms for each homogeneous course
    plt.figure(figsize=(8, 6))
    for house, group in grouped:
        group[homogeneous_course].plot.hist(
            alpha=0.5, legend=True, bins=10, color=colors[house], label=house)

    plt.title(f'Histogram of {homogeneous_course} Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} [data path]")
    else:
        df, features = load_data(sys.argv[1])
        display_histogram(df)
