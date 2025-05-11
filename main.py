
from ExploratoryDA import run_exploratory_da
from EqualFrequencyBinning import run_classification_analysis


def exploratory_analysis(datasets):
    print("Starting Exploratory Data Analysis!")
    print("=" * 50)

    analyses = []

    for dataset in datasets:
        print(f"\n=== Running Exploratory Data Analysis for {dataset['file_path']} ===")
        analysis_info = run_exploratory_da(dataset['file_path'], dataset['target'])

        analyses.append(analysis_info)


    print("\n" + "=" * 50)
    print("Exploratory Data Analysis Completed!")
    print("=" * 50)


def price_binning(datasets, bins):
    print("Starting Price Category Binning Analysis!")
    print("=" * 50)

    analyses = {}

    for dataset in datasets:
        analysis_info = run_classification_analysis(
            dataset['file_path'],
            dataset['target'],
            bins=bins
        )

        dataset_name = dataset['file_path'].split('/')[-1].split('.')[0]
        analyses[dataset_name] = analysis_info

    print("\n" + "=" * 50)
    print("Price Category Binning Analysis Completed!")
    print("=" * 50)

    return analyses



if __name__ == "__main__":
    datasets = [
        {'file_path': 'Data/train.csv', 'target': 'Price'},
        {'file_path': 'Data/training_extra.csv', 'target': 'Price'}
    ]

    # exploratory_analysis(datasets)

    # Price Classification with Equal Frequency
    prices = price_binning(datasets, 5)

    for name, analysis in prices.items():
        print(f"\nDataset: {name}")
        print(f"Categories edges: {analysis['bin_edges']}")
        print(f"Number of samples in each category:")
        print(analysis['dataframe']['Price_category'].value_counts())

        output_file = f"Data/{name}_with_categories.csv"
        analysis['dataframe'].to_csv(output_file, index=False)
        print(f"Saved categorized data to {output_file}")