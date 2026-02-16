from configuration import *

class PreprocessingAgent:
    def __init__(self):
        self.pipeline = None
        self.target_col = None
        self.numeric_features = []
        self.categorical_features = []

    def validate_target(self, df: pd.DataFrame, target_col: str):
        """
        Validates the target column to ensure it is suitable for ML.
        Returns: A cleaned DataFrame (rows with missing targets removed).
        """
        # 1. Check Existence
        if target_col not in df.columns:
            raise ValueError(f"CRITICAL ERROR: Target column '{target_col}' not found in the dataset. "
                             f"Available columns are: {list(df.columns)}")

        # 2. Check for Missing Values in Target
        missing_count = df[target_col].isnull().sum()
        if missing_count > 0:
            print(f"WARNING: Target column '{target_col}' contains {missing_count} missing values. "
                  f"Dropping these rows to prevent errors.")
            df = df.dropna(subset=[target_col])
        
        # 3. Check for Variance 
        if df[target_col].nunique() < 2:
            raise ValueError(f"CRITICAL ERROR: Target column '{target_col}' has only 1 unique value. "
                             f"The model cannot learn to predict constant data.")

        # 4. Check for High Cardinality
        if df[target_col].dtype == 'object' or df[target_col].dtype == 'int64':
             if df[target_col].nunique() == len(df) and len(df) > 50:
                 print(f"WARNING: Target column '{target_col}' has unique values for every row. "
                       f"Are you sure this isn't an ID column?")

        return df

    def remove_outliers(self, df, multiplier=1.5):
        """
        Removes outliers using IQR from the Training set only.
        """
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        cols_to_check = [c for c in numeric_cols if c != self.target_col]

        for col in cols_to_check:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - multiplier * IQR
            upper = Q3 + multiplier * IQR
            
            # Filter
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
            
        print(f"Outlier removal: Dropped {len(df) - len(df_clean)} rows.")
        return df_clean

    def create_pipeline(self, X_train):
        """
        Creates the sklearn transformation pipeline based on X_train columns.
        """
        
        self.numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

        # 1. Numeric Pipeline
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # 2. Categorical Pipeline
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) 
        ])

        # Combine
        self.pipeline = ColumnTransformer(
            transformers=[
                ('num', num_transformer, self.numeric_features),
                ('cat', cat_transformer, self.categorical_features)
            ],
            verbose_feature_names_out=False 
        )

    def run(self, df, target_col, test_size=0.2):
        print(f"--- Starting Preprocessing for Target: '{target_col}' ---")
        self.target_col = target_col
        
        df = self.validate_target(df, target_col)
        
        try:
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df[target_col])
        except ValueError:
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        
        train_df_clean = self.remove_outliers(train_df)
        
        # Separate X and y
        X_train = train_df_clean.drop(columns=[target_col])
        y_train = train_df_clean[target_col]
        
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        
        self.create_pipeline(X_train)
        
        # Fit on Train, Transform Train
        print("Fitting Pipeline...")
        X_train_processed = self.pipeline.fit_transform(X_train)
        
        
        X_test_processed = self.pipeline.transform(X_test)
        
        try:
            feat_names = self.pipeline.get_feature_names_out()
        except:
            # Fallback if specific sklearn version issues occur
            feat_names = [f"feat_{i}" for i in range(X_train_processed.shape[1])]
            
        X_train_df = pd.DataFrame(X_train_processed, columns=feat_names, index=X_train.index)
        X_test_df = pd.DataFrame(X_test_processed, columns=feat_names, index=X_test.index)

        print("Preprocessing Complete.")
        return X_train_df, X_test_df, y_train, y_test