import pandas as pd


def run_classification_analysis(file_path, target, bins):
    print(f"\n=== Processing {file_path} ===")

    df = pd.read_csv(file_path)

    # Ensure consistent return structure
    result = {'dataframe': (bin_frequency(df, target, bins))[0], 'bin_labels': (bin_frequency(df, target, bins))[1],
              'bin_edges': (bin_frequency(df, target, bins))[2]}

    # Print statistics instead of plotting
    print_category_statistics(result['dataframe'], target)

    return result


def bin_frequency(df, target, bins):
    # Create bins based on quantiles
    bin_edges = list(pd.qcut(df[target], q=bins, retbins=True, duplicates='drop')[1])

    # Generate labels based on price ranges
    bin_labels = []
    for i in range(len(bin_edges) - 1):
        label = f"{int(bin_edges[i])}-{int(bin_edges[i + 1])}"
        bin_labels.append(label)

    # Create the bin column
    df[f'{target}_bin'] = pd.cut(
        df[target],
        bins=bin_edges,
        labels=bin_labels,
        include_lowest=True
    )

    # Map the bins to intuitive labels
    # Make sure we have the right number of labels based on how many bins were actually created
    actual_bins = len(bin_labels)
    if actual_bins == 5:
        mapping = {
            bin_labels[0]: 'Super-Cheap',
            bin_labels[1]: 'Cheap',
            bin_labels[2]: 'Normal',
            bin_labels[3]: 'Expensive',
            bin_labels[4]: 'Super-Expensive'
        }
    else:
        # Dynamically create labels if we don't have exactly 5 bins
        mapping = {}
        categories = ['Super-Cheap', 'Cheap', 'Normal', 'Expensive', 'Super-Expensive']
        for i in range(min(actual_bins, len(categories))):
            mapping[bin_labels[i]] = categories[i]

        # Handle any additional bins if necessary
        for i in range(len(categories), actual_bins):
            mapping[bin_labels[i]] = f'Category {i + 1}'

    df[f'{target}_category'] = df[f'{target}_bin'].map(mapping)

    return df, bin_labels, bin_edges


def print_category_statistics(df, target_column):
    # Print statistics
    print(f"\n=== {target_column} Category Statistics ===")
    category_stats = df.groupby(f'{target_column}_category', observed=False)[target_column].agg(['count', 'min', 'max', 'mean'])
    print(category_stats)

    # Calculate percentage distribution
    category_counts = df[f'{target_column}_category'].value_counts(normalize=True).mul(100).round(1)
    print("\nPercentage Distribution:")
    for category, percentage in category_counts.items():
        print(f"{category}: {percentage}%")