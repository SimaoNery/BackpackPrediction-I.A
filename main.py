from PriceClassification import price_classification, plot_price_distribution
from ExploratoryDA import run_exploratory_da
from Preprocessing import run_preprocessing_pipeline

if __name__ == "__main__":
    original_datasets = [
        {'file_path': 'Data/train.csv', 'target': 'Price'},
        {'file_path': 'Data/training_extra.csv', 'target': 'Price'}
    ]

    # Turn the price into a categorical feature
    # processed_datasets = price_classification(original_datasets, 5)

    # Plot distribution for the first dataset
    # plot_price_distribution(processed_datasets[0])
    # plot_price_distribution(processed_datasets[1])

    price_datasets = [
        {'file_path': 'Data/train_with_categories.csv', 'target': 'Price_Category'},
        {'file_path': 'Data/training_extra_with_categories.csv', 'target': 'Price_Category'}
    ]
    
    # Exploratory Data Analysis
    run_exploratory_da(price_datasets, 'Price_Category', 60)

    """
    # Data Preprocessing
    processed_data = preprocess_datasets(
        price_datasets,
        feature_selection=True,
        selection_method='mutual_info',
        k=8
    )

    print("\nPreprocessing Summary:")
    for name, data in processed_data.items():
        print(f"\nDataset: {name}")
        print(f"- Final features: {data['X'].columns.tolist()}")
        print(f"- Processed data shape: {data['X'].shape}")
        print(f"- Output file: {data['output_file']}") 
    """


def preprocess_datasets(datasets, feature_selection=True, selection_method='mutual_info', k=None):
    print("Starting Data Preprocessing!")
    print("=" * 50)

    processed_datasets = {}

    for dataset in datasets:
        dataset_name = dataset['file_path'].split('/')[-1].split('.')[0]
        print(f"\n=== Preprocessing {dataset_name} dataset ===")

        input_file = f"Data/{dataset_name}_with_categories.csv"
        output_file = f"Data/{dataset_name}_preprocessed.csv"

        x, y, preprocessing_info = run_preprocessing_pipeline(
            file_path=input_file,
            output_path=output_file,
            feature_selection=feature_selection,
            selection_method=selection_method,
            k=k
        )

        processed_datasets[dataset_name] = {
            'X': x,
            'y': y,
            'preprocessing_info': preprocessing_info,
            'output_file': output_file
        }

    print("\n" + "=" * 50)
    print("Data Preprocessing Completed!")
    print("=" * 50)

    return processed_datasets