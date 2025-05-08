from exploratoryDA import run_exploratory_da


def run_full_analysis(datasets):
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


if __name__ == "__main__":
    datasets = [
        {'file_path': 'Data/train.csv', 'target': 'Price'},
        {'file_path': 'Data/training_extra.csv', 'target': 'Price'}
    ]

    run_full_analysis(datasets)