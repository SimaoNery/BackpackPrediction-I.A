import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings


class ExploratoryDA:

    def __init__(self, file_path, target_col, correlation_timeout=60):
        self.file_path = file_path
        self.target_col = target_col
        self.dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        self.frame = None
        self.output_dir = self._setup_output_directory()
        self.correlation_timeout = correlation_timeout

        # Set the style for the plots
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("viridis")

        # Suppress warnings
        warnings.filterwarnings('ignore')

    def _setup_output_directory(self):
        main_dir = "EDA_Results"
        os.makedirs(main_dir, exist_ok=True)

        dataset_dir = os.path.join(main_dir, self.dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        return dataset_dir

    def load_data(self):
        self.frame = pd.read_csv(self.file_path)
        print(f"\nDataset loaded: {self.dataset_name}")
        print(f"   Rows: {self.frame.shape[0]}, Columns: {self.frame.shape[1]}")

        return self.frame

    def analyze_basic_info(self):
        print("\n" + "=" * 50)
        print("BASIC DATASET INFORMATION")
        print("=" * 50)

        print("\n- First 5 rows:")
        print(self.frame.head())

        print("\n- Data types:")
        print(self.frame.dtypes)

        # Count data types
        type_counts = self.frame.dtypes.value_counts()
        print(f"\n- Column type distribution:")
        for dtype, count in type_counts.items():
            print(f"   - {dtype}: {count} columns")

        print("\n- Basic numerical statistics:")
        # Exclude ID-like columns (columns with all unique values or named 'id')
        non_id_cols = []
        for col in self.frame.columns:
            if (col.lower() not in ['id', 'identifier', 'index']) and \
                    (not (self.frame[col].nunique() == len(self.frame) and
                          pd.api.types.is_numeric_dtype(self.frame[col]))):
                non_id_cols.append(col)

        # If we found columns to exclude
        if len(non_id_cols) < len(self.frame.columns):
            excluded_cols = set(self.frame.columns) - set(non_id_cols)
            print(f"   (Excluded ID-like columns: {', '.join(excluded_cols)})")
            print(self.frame[non_id_cols].describe())
        else:
            print(self.frame.describe())

        # Create basic info directory
        info_dir = os.path.join(self.output_dir, "basic_info")
        os.makedirs(info_dir, exist_ok=True)

        # Create data type distribution pie chart
        plt.figure(figsize=(10, 6))
        plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Column Data Type Distribution', fontsize=15)
        plt.tight_layout()
        plt.savefig(os.path.join(info_dir, 'data_type_distribution.png'), dpi=300)
        plt.close()

    def analyze_missing_values(self):
        """Check and visualize missing values"""
        print("\n" + "=" * 50)
        print("MISSING VALUES ANALYSIS")
        print("=" * 50)

        missing = self.frame.isnull().sum()
        missing_percent = (missing / len(self.frame)) * 100
        missing_data = pd.concat([missing, missing_percent], axis=1)
        missing_data.columns = ['Count', 'Percent']

        missing_values = missing_data[missing_data['Count'] > 0].sort_values('Count', ascending=False)
        if not missing_values.empty:
            print("\n- Columns with missing values:")
            print(missing_values)
        else:
            print("\nNo missing values found in the dataset.")

        # Create missing values directory
        missing_dir = os.path.join(self.output_dir, "missing_values")
        os.makedirs(missing_dir, exist_ok=True)

        # Visualize missing values if there are any
        if not missing_data[missing_data['Count'] > 0].empty:
            plt.figure(figsize=(12, 6))
            plt.title('Missing Values Heatmap', fontsize=15)
            sns.heatmap(self.frame.isnull(), cbar=False, cmap='viridis')
            plt.tight_layout()
            plt.savefig(os.path.join(missing_dir, 'missing_values_heatmap.png'), dpi=300)

            # Create bar chart of missing values percentage
            plt.figure(figsize=(12, 6))
            plt.title('Missing Values by Column (%)', fontsize=15)
            missing_values['Percent'].sort_values(ascending=False).plot(kind='bar')
            plt.axhline(y=5, color='r', linestyle='-', alpha=0.3, label='5% threshold')
            plt.legend()
            plt.ylabel('Missing Values (%)')
            plt.tight_layout()
            plt.savefig(os.path.join(missing_dir, 'missing_values_percent.png'), dpi=300)
            plt.close()

            print(f"    Saved missing values visualizations to {missing_dir}")

        return missing_data[missing_data['Count'] > 0]

    def check_duplicates(self, id_col=None):
        print("\n" + "=" * 50)
        print("DUPLICATE ROWS ANALYSIS")
        print("=" * 50)

        # Create duplicates directory
        dup_dir = os.path.join(self.output_dir, "duplicates")
        os.makedirs(dup_dir, exist_ok=True)

        if id_col:
            duplicates = self.frame[self.frame.duplicated(subset=[id_col], keep=False)]
            print(f"\nChecking duplicates based on ID column: {id_col}")
        else:
            duplicates = self.frame[self.frame.duplicated(keep=False)]
            print("\nChecking for fully duplicated rows")

        if not duplicates.empty:
            print(f"\nFound {len(duplicates)} duplicate rows")
            print("\nSample of duplicate rows:")
            print(duplicates.head())

            dup_file = os.path.join(dup_dir, "duplicates.csv")
            duplicates.to_csv(dup_file, index=False)
            print(f"\nSaved duplicate rows to {dup_file}")
        else:
            print("\nNo duplicate rows found")

        return duplicates


    def analyze_target_variable(self):
        """Analyze and visualize the target variable"""
        print("\n" + "=" * 50)
        print(f"TARGET VARIABLE ANALYSIS: {self.target_col}")
        print("=" * 50)

        # Check if target variable exists in the dataframe
        if self.target_col not in self.frame.columns:
            print(f"Error: Target column '{self.target_col}' not found in the dataset.")
            return

        print("\n- Target variable statistics:")
        print(self.frame[self.target_col].describe())

        # Create target directory
        target_dir = os.path.join(self.output_dir, "target_analysis")
        os.makedirs(target_dir, exist_ok=True)

        # Check if target is numerical or categorical
        if pd.api.types.is_numeric_dtype(self.frame[self.target_col]):
            print(f"\n- Target variable is NUMERICAL")

            # Histogram
            plt.figure(figsize=(10, 6))
            plt.title(f'Distribution of {self.target_col}', fontsize=15)
            sns.histplot(self.frame[self.target_col], kde=True)
            plt.tight_layout()
            plt.savefig(os.path.join(target_dir, f'{self.target_col}_distribution.png'), dpi=300)
            plt.close()

            # Box plot
            plt.figure(figsize=(8, 6))
            plt.title(f'Box Plot of {self.target_col}', fontsize=15)
            sns.boxplot(y=self.frame[self.target_col])
            plt.tight_layout()
            plt.savefig(os.path.join(target_dir, f'{self.target_col}_boxplot.png'), dpi=300)
            plt.close()

            # QQ plot to check normality
            plt.figure(figsize=(10, 6))
            stats.probplot(self.frame[self.target_col].dropna(), plot=plt)
            plt.title(f'Q-Q Plot of {self.target_col}', fontsize=15)
            plt.tight_layout()
            plt.savefig(os.path.join(target_dir, f'{self.target_col}_qqplot.png'), dpi=300)
            plt.close()

            print(f"    Saved target variable visualizations to {target_dir}")
        else:
            print(f"\n- Target variable is CATEGORICAL")

            # Count plot
            plt.figure(figsize=(10, 6))
            plt.title(f'Distribution of {self.target_col}', fontsize=15)
            value_counts = self.frame[self.target_col].value_counts()

            # If too many categories, limit display
            if len(value_counts) > 20:
                print(f"   Note: Target has {len(value_counts)} unique values (showing top 20)")
                top_categories = value_counts.nlargest(20).index
                temp_df = self.frame.copy()
                temp_df.loc[~temp_df[self.target_col].isin(top_categories), self.target_col] = 'Other'
                sns.countplot(y=self.target_col, data=temp_df, order=temp_df[self.target_col].value_counts().index)
                plt.title(f'Count of {self.target_col} (Top 20 + Other)', fontsize=15)
            else:
                sns.countplot(y=self.target_col, data=self.frame, order=value_counts.index)

            plt.tight_layout()
            plt.savefig(os.path.join(target_dir, f'{self.target_col}_countplot.png'), dpi=300)
            plt.close()

            # Pie chart for target distribution
            plt.figure(figsize=(10, 8))
            plt.pie(value_counts, labels=value_counts.index if len(value_counts) <= 10 else None,
                    autopct='%1.1f%%' if len(value_counts) <= 10 else None)
            plt.title(f'Distribution of {self.target_col}', fontsize=15)
            if len(value_counts) > 10:
                plt.legend(value_counts.nlargest(10).index.tolist() + ['Others'],
                           loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
            plt.tight_layout()
            plt.savefig(os.path.join(target_dir, f'{self.target_col}_pie_chart.png'), dpi=300)
            plt.close()

            print(f"    Saved target variable visualizations to {target_dir}")

    def analyze_feature_distributions(self):
        """Analyze and visualize feature distributions"""
        print("\n" + "=" * 50)
        print("FEATURE DISTRIBUTIONS ANALYSIS")
        print("=" * 50)

        # Create distributions directory
        dist_dir = os.path.join(self.output_dir, "distributions")
        os.makedirs(dist_dir, exist_ok=True)

        # Analyze numerical features - exclude ID columns and duplicates
        numerical_cols = self.frame.select_dtypes(include=['int64', 'float64']).columns
        numerical_cols = numerical_cols.drop(self.target_col, errors='ignore')  # Remove target if it's numerical

        # Remove ID-like columns and duplicates
        numerical_cols = [col for col in numerical_cols
                          if not (col.lower().startswith('id') or
                                  self.frame[col].nunique() == len(self.frame))]
        numerical_cols = list(dict.fromkeys(numerical_cols))  # Remove duplicates while preserving order

        if len(numerical_cols) > 0:
            print(f"\n- Found {len(numerical_cols)} numerical features")

            # Histograms for numerical features (up to 16)
            plot_cols = min(16, len(numerical_cols))
            rows = (plot_cols + 3) // 4  # Ceiling division

            plt.figure(figsize=(16, 4 * rows))
            for i, col in enumerate(numerical_cols[:plot_cols]):
                plt.subplot(rows, 4, i + 1)
                sns.histplot(self.frame[col], kde=True)
                plt.title(col)
            plt.tight_layout()
            plt.savefig(os.path.join(dist_dir, 'numerical_distributions.png'), dpi=300)
            plt.close()

            # Box plots for numerical features (up to 16)
            plt.figure(figsize=(16, 4 * rows))
            for i, col in enumerate(numerical_cols[:plot_cols]):
                plt.subplot(rows, 4, i + 1)
                sns.boxplot(y=self.frame[col])
                plt.title(col)
            plt.tight_layout()
            plt.savefig(os.path.join(dist_dir, 'numerical_boxplots.png'), dpi=300)
            plt.close()

            print(f"    Saved numerical feature distributions to {dist_dir}")
        else:
            print("\n- No numerical features found (excluding target)")

        # Analyze categorical features
        categorical_cols = self.frame.select_dtypes(include=['object']).columns
        if self.target_col in categorical_cols:
            categorical_cols = categorical_cols.drop(self.target_col)

        if len(categorical_cols) > 0:
            print(f"\n- Found {len(categorical_cols)} categorical features")

            for col in categorical_cols[:10]:  # Limit to 10 categories for readability
                value_counts = self.frame[col].value_counts()
                unique_count = len(value_counts)

                print(f"   - {col}: {unique_count} unique values")

                # For high cardinality features, only plot top categories
                plt.figure(figsize=(12, 6))
                if unique_count > 10:
                    top_categories = value_counts.nlargest(10).index
                    temp_df = self.frame.copy()
                    temp_df.loc[~temp_df[col].isin(top_categories), col] = 'Other'
                    sns.countplot(y=col, data=temp_df, order=temp_df[col].value_counts().index)
                    plt.title(f'Count of {col} (Top 10 + Other)', fontsize=15)
                else:
                    sns.countplot(y=col, data=self.frame, order=value_counts.index)
                    plt.title(f'Count of {col}', fontsize=15)

                plt.tight_layout()
                plt.savefig(os.path.join(dist_dir, f'{col}_countplot.png'), dpi=300)
                plt.close()

            if len(categorical_cols) > 10:
                print(f"   (and {len(categorical_cols) - 10} more categorical features)")

            print(f"    Saved categorical feature distributions to {dist_dir}")
        else:
            print("\n- No categorical features found (excluding target)")

    def analyze_price_correlations(self):
        """Analyze and visualize correlations between features and price target"""
        print("\n" + "=" * 50)
        print("PRICE CORRELATION ANALYSIS")
        print("=" * 50)

        # Create correlations directory
        corr_dir = os.path.join(self.output_dir, "price_correlations")
        os.makedirs(corr_dir, exist_ok=True)

        results = {}
        target_col = 'Price'  # Explicitly set target to price

        # Check if price exists in the dataframe
        if target_col not in self.frame.columns:
            print(f"Error: Target column '{target_col}' not found in the dataset.")
            return results

        print(f"Target column {target_col} found in dataset. Proceeding with analysis.")

        # Get numerical and categorical columns, excluding ID columns
        numerical_cols = [col for col in self.frame.select_dtypes(include=['int64', 'float64']).columns
                          if 'id' not in col.lower() and col != 'index']
        categorical_cols = self.frame.select_dtypes(include=['object', 'category']).columns

        print(f"Found {len(numerical_cols)} numerical columns and {len(categorical_cols)} categorical columns")
        print(f"Numerical columns: {numerical_cols}")

        # Start timer for correlation analysis
        start_time = time.time()

        # 1. Pearson correlation (for numerical-numerical relationships)
        if len(numerical_cols) >= 2 and target_col in numerical_cols:
            print(f"Computing Pearson correlations with {target_col}")
            # Calculate correlations only with the price target
            pearson_correlations = {}
            for col in numerical_cols:
                if col != target_col:
                    pearson_correlations[col] = self.frame[col].corr(self.frame[target_col], method='pearson')

            # Convert to Series and sort
            target_pearson = pd.Series(pearson_correlations).sort_values(ascending=False)
            results['target_pearson'] = target_pearson

            print("\nPearson Correlations with price")
            print("--------------------------------")
            for feature, corr_value in target_pearson.head(10).items():
                print(f"{feature}: {corr_value:.4f}")

            # SIMPLIFIED: Basic horizontal bar chart with minimal styling
            plt.figure(figsize=(8, 5))
            plt.barh(target_pearson.head(5).index, target_pearson.head(5).values)
            plt.title('Top 5 Pearson Correlations with Price')
            plt.savefig(os.path.join(corr_dir, 'pearson_correlations_price.png'))
            plt.close()

        else:
            if len(numerical_cols) < 2:
                print(f"Not enough numerical columns found. Need at least 2, but found {len(numerical_cols)}")
            if target_col not in numerical_cols:
                print(f"Target column '{target_col}' not found in numerical columns.")

        # Check timeout
        if time.time() - start_time > self.correlation_timeout:
            print(f"\nCorrelation analysis timed out after {self.correlation_timeout} seconds")
            return results

        # 2. Spearman correlation (for numerical-numerical with non-linear relationships)
        if len(numerical_cols) >= 2 and target_col in numerical_cols:
            print(f"Computing Spearman correlations with {target_col}")
            # Calculate correlations only with the price target
            spearman_correlations = {}
            for col in numerical_cols:
                if col != target_col:
                    spearman_correlations[col] = self.frame[col].corr(self.frame[target_col], method='spearman')

            # Convert to Series and sort
            target_spearman = pd.Series(spearman_correlations).sort_values(ascending=False)
            results['target_spearman'] = target_spearman

            print("\nSpearman Correlations with price")
            print("--------------------------------")
            for feature, corr_value in target_spearman.head(10).items():
                print(f"{feature}: {corr_value:.4f}")

            # SIMPLIFIED: Basic horizontal bar chart with minimal styling
            plt.figure(figsize=(8, 5))
            plt.barh(target_spearman.head(5).index, target_spearman.head(5).values)
            plt.title('Top 5 Spearman Correlations with Price')
            plt.savefig(os.path.join(corr_dir, 'spearman_correlations_price.png'))
            plt.close()

        # Check timeout
        if time.time() - start_time > self.correlation_timeout:
            print(f"\nCorrelation analysis timed out after {self.correlation_timeout} seconds")
            return results

        # 3. Analysis for categorical features with price target
        if target_col in numerical_cols and len(categorical_cols) >= 1:
            print(f"Computing categorical feature impacts on {target_col}")
            cat_impact_results = []
            print("\nCategorical Features Impact on Price")
            print("--------------------------------")

            for col in categorical_cols:
                if self.frame[col].nunique() <= 30:  # Skip high cardinality features
                    try:
                        # ANOVA test to see if categories have different price means
                        categories = self.frame[col].dropna().unique()
                        samples = [self.frame[self.frame[col] == cat][target_col].values
                                   for cat in categories if len(self.frame[self.frame[col] == cat]) > 0]

                        if len(samples) > 1:  # Need at least 2 categories for ANOVA
                            f_stat, p_value = stats.f_oneway(*samples)

                            # Calculate max difference in means between categories
                            grouped = self.frame.groupby(col)[target_col]
                            means = grouped.mean().sort_values(ascending=False)
                            value_range = means.max() - means.min()

                            cat_impact_results.append({
                                'feature': col,
                                'f_statistic': f_stat,
                                'p_value': p_value,
                                'category_count': len(categories),
                                'price_range': value_range
                            })
                    except Exception as e:
                        print(f"Error analyzing categorical feature {col}: {str(e)}")

            if cat_impact_results:
                cat_impact_df = pd.DataFrame(cat_impact_results).sort_values('f_statistic', ascending=False)
                results['categorical_impact'] = cat_impact_df

                for idx, row in cat_impact_df.head(5).iterrows():
                    sig = "significant" if row['p_value'] < 0.05 else "not significant"
                    print(f"{row['feature']}: F={row['f_statistic']:.2f}, p-value={row['p_value']:.4e} ({sig})")
                    print(f"  {row['category_count']} categories, price range: {row['price_range']:.2f}")

                # SIMPLIFIED: Basic horizontal bar chart showing only top 5
                plt.figure(figsize=(8, 5))
                plt.barh(cat_impact_df.head(5)['feature'], cat_impact_df.head(5)['f_statistic'])
                plt.title('Top 5 Categorical Features Impact')
                plt.savefig(os.path.join(corr_dir, 'categorical_impact_price.png'))
                plt.close()
            else:
                print("No categorical features had significant impact on price")

        print(f"\nPrice correlation analysis completed in {time.time() - start_time:.2f} seconds")
        print(f"    Saved correlation visualizations to {corr_dir}")

        return results



    def run_analysis(self):
        print("\n" + "=" * 50)
        print(f"STARTING EXPLORATORY DATA ANALYSIS: {self.dataset_name}")
        print("=" * 50)
        print(f"Using correlation timeout: {self.correlation_timeout} seconds")

        self.load_data()
        self.analyze_basic_info()
        self.analyze_missing_values()
        self.check_duplicates(id_col='id')
        self.analyze_target_variable()
        self.analyze_feature_distributions()
        self.analyze_price_correlations()

        print("\n" + "=" * 50)
        print(f"EDA COMPLETED: Visualizations saved to {self.output_dir}")
        print("=" * 50)

        return self.frame, self.output_dir


def run_exploratory_da(file_path, target_col, correlation_timeout=60):
    analyzer = ExploratoryDA(file_path, target_col, correlation_timeout)
    return analyzer.run_analysis()