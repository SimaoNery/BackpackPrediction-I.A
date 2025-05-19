import pandas as pd
from scipy.stats import spearmanr, chi2_contingency, pointbiserialr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import os
import time
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

def analyze_basic_info(df, output_dir):
    print("\n" + "=" * 50)
    print("BASIC DATASET INFORMATION")
    print("=" * 50)

    print("\n- First 5 rows:")
    print(df.head())

    print("\n- Data types:")
    print(df.dtypes)


    type_counts = df.dtypes.value_counts()
    print(f"\n- Column type distribution:")
    for dtype, count in type_counts.items():
        print(f"   - {dtype}: {count} columns")

    print("\n- Basic numerical statistics:")
    non_id_cols = [col for col in df.columns if col.lower() != 'id']
    print(df[non_id_cols].describe())

    info_dir = os.path.join(output_dir, "basic_info")
    os.makedirs(info_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%')
    plt.title('Data Types Distribution', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(info_dir, 'data_types_distribution.png'), dpi=300)
    plt.close()

def analyze_missing_values(df, output_dir):
    print("\n" + "=" * 50)
    print("MISSING VALUES ANALYSIS")
    print("=" * 50)

    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_data = pd.concat([missing, missing_percent], axis=1)
    missing_data.columns = ['Count', 'Percent']

    missing_values = missing_data[missing_data['Count'] > 0].sort_values('Count')
    if not missing_values.empty:
        print("\n- Columns with missing values:")
        print(missing_values)
    else:
        print("\nNo missing values found in the dataset.")


    missing_dir = os.path.join(output_dir, "missing_values")
    os.makedirs(missing_dir, exist_ok=True)

    if not missing_data[missing_data['Count'] > 0].empty:
        plt.figure(figsize=(12, 6))
        plt.title('Missing Values by Column (%)', fontsize=15)
        missing_values['Percent'].sort_values().plot(kind='bar')
        plt.ylabel('Missing Values (%)')
        plt.tight_layout()
        plt.savefig(os.path.join(missing_dir, 'missing_values_percent.png'), dpi=300)
        plt.close()

    return missing_data[missing_data['Count'] > 0]

def analyze_duplicates(df, output_dir):
    print("\n" + "=" * 50)
    print("DUPLICATE ROWS ANALYSIS")
    print("=" * 50)

    duplicates_dir = os.path.join(output_dir, "duplicates")
    os.makedirs(duplicates_dir, exist_ok=True)


    duplicates1 = df[df.duplicated(subset=["id"], keep=False)]
    print(f"\nChecking duplicates based on ID column")

    duplicates2 = df[df.duplicated(keep=False)]
    print("\nChecking for fully duplicated rows")

    duplicates = duplicates1 + duplicates2

    if not duplicates.empty:
        print(f"\nFound {len(duplicates)} duplicate rows")
        print("\nSample of duplicate rows:")
        print(duplicates.head())

        dup_file = os.path.join(duplicates_dir, "duplicates.csv")
        duplicates.to_csv(dup_file, index=False)
    else:
        print("\nNo duplicate rows found")

    return duplicates


def analyze_target_variable(df, target_col, output_dir):
    print("\n" + "=" * 50)
    print(f"TARGET VARIABLE ANALYSIS: {target_col}")
    print("=" * 50)

    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in the dataset.")
        return

    print("\n- Target variable statistics:")
    print(df[target_col].describe())
    df[target_col].value_counts()

    target_dir = os.path.join(output_dir, "target_analysis")
    os.makedirs(target_dir, exist_ok=True)

    # Count plot
    plt.figure(figsize=(10, 6))
    plt.title(f'Distribution of {target_col}', fontsize=15)
    value_counts = df[target_col].value_counts()

    sns.countplot(y=target_col, data=df, order=value_counts.index)

    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, f'{target_col}_countplot.png'), dpi=300)
    plt.close()

    # Pie chart
    plt.figure(figsize=(10, 8))
    plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%')
    plt.title(f'Distribution of {target_col}', fontsize=15)

    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, f'{target_col}_pie_chart.png'), dpi=300)
    plt.close()


    print("\n- Price statistics:")
    print(df['Price'].describe())

    # Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Price'], kde=True)
    plt.title('Distribution of Price', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, 'price_distribution.png'), dpi=300)
    plt.close()

    # Boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=df['Price'])
    plt.title('Box Plot of Price', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, 'price_boxplot.png'), dpi=300)
    plt.close()

def analyze_feature_distributions(df, target_col, output_dir):
    print("\n" + "=" * 50)
    print("FEATURE DISTRIBUTIONS ANALYSIS")
    print("=" * 50)

    dist_dir = os.path.join(output_dir, "distributions")
    os.makedirs(dist_dir, exist_ok=True)

    numerical_cols = [
        col for col in df.select_dtypes(include=['int64', 'float64']).columns
        if col not in [target_col, "id", "Price"]
    ]

    if numerical_cols:
        print(f"\n- Found {len(numerical_cols)} numerical features")

        rows = 1
        cols = 2

        # Histogram
        plt.figure(figsize=(5 * cols, 4 * rows))
        for i, col in enumerate(numerical_cols):
            plt.subplot(rows, cols, i + 1)
            sns.histplot(data=df, x=col, kde=True)
            plt.title(col)

        plt.tight_layout()
        plt.savefig(os.path.join(dist_dir, 'numerical_distributions.png'), dpi=300)
        plt.close()

        # Boxplot
        plt.figure(figsize=(5 * cols, 4 * rows))
        for i, col in enumerate(numerical_cols):
            plt.subplot(rows, cols, i + 1)
            sns.boxplot(data=df, y=col)
            plt.title(col)

        plt.tight_layout()
        plt.savefig(os.path.join(dist_dir, 'numerical_boxplots.png'), dpi=300)
        plt.close()


    else:
        print("\n- No numerical features found (excluding target)")

    columns_to_analyze = [col for col in df.columns if col != target_col and col != 'id' and col != 'Price']
    categorical_cols = [col for col in df[columns_to_analyze].select_dtypes(include=['object', 'category']).columns]

    if target_col in categorical_cols:
        categorical_cols = categorical_cols.remove(target_col)

    if categorical_cols:
        print(f"\n- Found {len(categorical_cols)} categorical features")

        for col in categorical_cols:
            value_counts = df[col].value_counts()
            unique_count = len(value_counts)

            print(f"   - {col}: {unique_count} unique values")

            plt.figure(figsize=(12, 6))
            sns.countplot(y=col, data=df, order=value_counts.index)
            plt.title(f'Count of {col}', fontsize=15)

            plt.tight_layout()
            plt.savefig(os.path.join(dist_dir, f'{col}_countplot.png'), dpi=300)
            plt.close()

        if len(categorical_cols) > 10:
            print(f"   (and {len(categorical_cols) - 10} more categorical features)")

    else:
        print("\n- No categorical features found (excluding target)")


def analyze_price_correlations(df, target_col, output_dir, correlation_timeout):
    print("\n" + "=" * 50)
    print("PRICE CORRELATION ANALYSIS")
    print("=" * 50)

    results = {}

    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in the dataset.")
        return results

    columns_to_analyze = [col for col in df.columns if col not in [target_col, 'id', 'Price']]

    numerical_cols = [col for col in df[columns_to_analyze].select_dtypes(include=['int64', 'float64']).columns]
    categorical_cols = [col for col in df[columns_to_analyze].select_dtypes(include=['object', 'category']).columns]

    biserial_cols = ['Laptop Compartment', 'Waterproof']

    start_time = time.time()

    # Map the Price_Category to numbers
    if df[target_col].dtype == 'object' or df[target_col].dtype == 'category':
        unique_categories = df[target_col].unique()
        mapping = {cat: i for i, cat in enumerate(unique_categories)}
        target_numeric = df[target_col].map(mapping)
    else:
        target_numeric = df[target_col]

    correlation_results = []

    print("\n" + "-" * 50)
    print("NUMERICAL FEATURES - SPEARMAN CORRELATION")
    print("-" * 50)
    print(f"{'Feature':<20} {'Correlation':<12} {'p-value':<12} {'Significance'}")
    print("-" * 70)

    # Analyze numerical columns using Spearman
    numerical_results = []
    for col in numerical_cols:
        if df[col].isna().any():
            valid_data = df[[col, target_col]].dropna()
            if len(valid_data) < 2:
                continue
            corr, p_value = spearmanr(valid_data[col], target_numeric[valid_data.index])
        else:
            corr, p_value = spearmanr(df[col], target_numeric)

        sig_status = "significant" if p_value < 0.05 else "not significant"
        print(f"{col:<20} {corr:>10.4f}   {p_value:>10.4f}   {sig_status}")

        numerical_results.append({
            'feature': col,
            'method': 'Spearman',
            'correlation': corr,
            'p_value': p_value,
            'significance': p_value < 0.05,
            'feature_type': 'Numerical'
        })
        correlation_results.append(numerical_results[-1])


    print("\n" + "-" * 50)
    print("ORDINAL FEATURES - SPEARMAN CORRELATION")
    print("-" * 50)
    print(f"{'Feature':<20} {'Correlation':<12} {'p-value':<12} {'Significance'}")
    print("-" * 70)

    # Analyze ordinal columns using Spearman
    ordinal_results = []
    for col in ['Size']:
        valid_data = df[[col, target_col]].dropna()
        nan_count = len(df) - len(valid_data)
        if nan_count > 0:
            print(f"{col:<20} - Processing without {nan_count} NaN values ({nan_count / len(df) * 100:.1f}%)")

        if df[col].dtype == 'object' or df[col].dtype == 'category':
            unique_values = valid_data[col].unique()
            mapping = {val: i for i, val in enumerate(unique_values)}
            col_numeric = valid_data[col].map(mapping)
        else:
            col_numeric = valid_data[col]

        valid_target = target_numeric[valid_data.index]
        corr, p_value = spearmanr(col_numeric, valid_target)

        sig_status = "significant" if p_value < 0.05 else "not significant"
        print(f"{col:<20} {corr:>10.4f}   {p_value:>10.4f}   {sig_status}")

        ordinal_results.append({
            'feature': col,
            'method': 'Spearman',
            'correlation': corr,
            'p_value': p_value,
            'significance': p_value < 0.05,
            'feature_type': 'Ordinal'
        })
        correlation_results.append(ordinal_results[-1])


    print("\n" + "-" * 50)
    print("CATEGORICAL FEATURES - CHI-SQUARE TEST")
    print("-" * 50)
    print(f"{'Feature':<20} {'Cramer\'s V':<12} {'p-value':<12} {'Significance'}")
    print("-" * 70)

    # Analyze categorical columns using Chi-square test
    categorical_results = []
    for col in categorical_cols:
        if col == 'Size':
            continue

        valid_data = df[[col, target_col]].dropna()
        nan_count = len(df) - len(valid_data)
        if nan_count > 0:
            print(f"{col:<20} - Processing without {nan_count} NaN values ({nan_count / len(df) * 100:.1f}%)")

        try:
            # Create contingency table
            contingency_table = pd.crosstab(valid_data[col], valid_data[target_col])

            # Check if contingency table is valid for chi-square test
            if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                print(f"{col:<20} {'N/A':<12} {'N/A':<12} {'N/A - Not enough unique values'}")
                continue

            # Check if expected frequencies are too small
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)

            if (expected < 5).any():
                print(f"{col:<20} - Warning: Some expected frequencies < 5")

            # Calculate Cramer's V as a measure of association strength
            n = contingency_table.sum().sum()
            min_dim = min(contingency_table.shape) - 1
            if min_dim == 0:
                cramers_v = 0
            else:
                cramers_v = np.sqrt(chi2 / (n * min_dim))

            sig_status = "significant" if p_value < 0.05 else "not significant"
            print(f"{col:<20} {cramers_v:>10.4f}   {p_value:>10.4f}   {sig_status}")

            categorical_results.append({
                'feature': col,
                'method': 'Chi-square',
                'correlation': cramers_v,
                'p_value': p_value,
                'significance': p_value < 0.05,
                'feature_type': 'Nominal'
            })
            correlation_results.append(categorical_results[-1])
        except Exception as e:
            print(f"{col:<20} {'Error':<12} {'Error':<12} {'Error: ' + str(e)}")


    print("\n" + "-" * 50)
    print("BINARY FEATURES - POINT-BISERIAL CORRELATION")
    print("-" * 50)
    print(f"{'Feature':<20} {'Correlation':<12} {'p-value':<12} {'Significance'}")
    print("-" * 70)

    # Analyze binary columns using Point-Biserial correlation
    binary_results = []
    for col in biserial_cols:
        valid_data = df[[col, target_col]].dropna()
        nan_count = len(df) - len(valid_data)
        if nan_count > 0:
            print(f"{col:<20} - Processing without {nan_count} NaN values ({nan_count / len(df) * 100:.1f}%)")

        if valid_data[col].nunique() != 2:
            print(f"{col:<20} {'N/A':<12} {'N/A':<12} {'N/A - Not a binary column'}")
            continue

        try:
            # Convert binary column to 0/1
            binary_col = pd.factorize(valid_data[col])[0]
            valid_target = target_numeric[valid_data.index]
            corr, p_value = pointbiserialr(binary_col, valid_target)

            sig_status = "significant" if p_value < 0.05 else "not significant"
            print(f"{col:<20} {corr:>10.4f}   {p_value:>10.4f}   {sig_status}")

            binary_results.append({
                'feature': col,
                'method': 'Point-Biserial',
                'correlation': corr,
                'p_value': p_value,
                'significance': p_value < 0.05,
                'feature_type': 'Binary'
            })
            correlation_results.append(binary_results[-1])
        except Exception as e:
            print(f"{col:<20} {'Error':<12} {'Error':<12} {'Error: ' + str(e)}")

    if time.time() - start_time > correlation_timeout:
        print(f"\nCorrelation analysis timed out after {correlation_timeout} seconds")
        return results

    if correlation_results:
        results_df = pd.DataFrame(correlation_results)
        results_df['abs_corr'] = results_df['correlation'].abs()
        results_df = results_df.sort_values('abs_corr', ascending=False)

        all_correlations = results_df.copy()
        results_df = results_df.drop('abs_corr', axis=1)
        results['all_correlations'] = all_correlations

        print("\n" + "=" * 50)
        print("CORRELATIONS SUMMARY")
        print("=" * 50)

        results = results_df.head(10)
        for _, row in results.iterrows():
            sig = "significant" if row['significance'] else "not significant"
            print(f"{row['feature']} ({row['feature_type']}): {row['correlation']:.4f} ({row['method']}, {sig})")

    return results


def run_exploratory_da(file_path, target_col, correlation_timeout):
    df = pd.read_csv(file_path)
    dataset_name = os.path.basename(file_path)

    print(f"\nDataset Loaded: {dataset_name}")
    print(f"\nRows: {df.shape[0]}     Columns: {df.shape[1]}")

    main_dir = "EDA_Resutls"
    os.makedirs(main_dir, exist_ok=True)

    dataset_dir = os.path.join(main_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    print("-------------  Exploratory Data Analysis  -------------")
    print("\n" + "=" * 50)

    analyze_basic_info(df, dataset_dir)
    analyze_missing_values(df, dataset_dir)
    analyze_duplicates(df, dataset_dir)
    analyze_target_variable(df, target_col, dataset_dir)
    analyze_feature_distributions(df, target_col, dataset_dir)
    analyze_price_correlations(df, target_col, dataset_dir, correlation_timeout)

    print("\n" + "=" * 50)
    print(f"\nExploratory Data Analysis Completed! Results saved to {dataset_dir}")
    print("=" * 50)