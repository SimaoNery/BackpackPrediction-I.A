import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def price_classification(datasets, bins):
    processed_datasets = []

    all_prices = []
    for dataset in datasets:
        df = pd.read_csv(dataset['file_path'])
        all_prices.extend(df['Price'].tolist())

        min_price = np.floor(min(all_prices))
        max_price = np.ceil(max(all_prices))

        bin_width = (max_price - min_price) / bins
        bin_edges = [min_price + i * bin_width for i in range(bins + 1)]

        print(f"Overall Price Range: {min_price} to {max_price}")
        print(f"Price Bin Edges: {bin_edges}")

        df = pd.read_csv(dataset['file_path'])

        # Add a new category
        df['Price_Category'] = pd.cut(df['Price'],
                                      bins=bin_edges,
                                      labels=[f'Class_{i + 1}' for i in range(bins)],
                                      include_lowest=True)

        output_path = dataset['file_path'].replace('.csv', '_with_categories.csv')

        if not os.path.exists(output_path):
            df.to_csv(output_path, index=False)
            print(f"Saved to {output_path}")
        else:
            print(f"File already exists at {output_path}. Skipping save.")

        processed_datasets.append(df)

    return processed_datasets


def plot_price_distribution(df):
    plt.figure(figsize=(10, 6))

    category_counts = df['Price_Category'].value_counts().sort_index()

    bars = plt.bar(category_counts.index, category_counts.values,
                   color='skyblue', edgecolor='black')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')

    plt.title('Price Category Distribution')
    plt.xlabel('Price Categories')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()