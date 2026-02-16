from configuration import *

class EDAAgent:
    """
    EDA Agent:
    - Runs basic exploratory data analysis
    - Detects outliers
    - Summarizes dataset statistics
    - Can produce text report via LLM
    """
        
    def __init__(self, llm=None):
        self.llm = llm 
    
    # 1. Basic stats
    def summary_stats(self, df: pd.DataFrame):
        stats = df.describe(include='all').transpose()
        return stats
    
    # 2. Missing values
    def missing_values(self, df: pd.DataFrame):
        missing = df.isnull().sum()
        percent_missing = df.isnull().mean() * 100
        mv_df = pd.DataFrame({'missing_count': missing, 'missing_percent': percent_missing})
        return mv_df
    
    # 3. Outlier detection 
    def detect_outliers(self, df: pd.DataFrame):
            
        num_cols = df.select_dtypes(include=[np.number]).columns
        outlier_results = {}

        for col in num_cols:
                
            data = df[col].dropna()
            if data.empty:
                continue
            
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            
            outlier_count = ((data < lower_bound) | (data > upper_bound)).sum()
            outlier_results[col] = outlier_count

        return outlier_results
    
    # 4. Generate textual summary via LLM
    # 4. Generate textual summary via LLM
    def generate_summary(self, df: pd.DataFrame):
        if self.llm is None:
            raise ValueError("LLM not provided for generating textual summary.")

        # Prepare context data
        missing_values = self.missing_values(df)
        outliers = self.detect_outliers(df)
        stats_summary = df.describe().transpose().to_string()
        
        # Convert missing/outliers to string or a message if empty
        missing_str = missing_values.to_string() if not missing_values.empty else "None detected."
        outliers_str = str(outliers) if outliers else "None detected."

        # UPDATED PROMPT: Explicitly asks for NO markdown and clean formatting
        prompt_template = PromptTemplate(
            input_variables=["stats_summary", "missing_values", "outliers"],
            template="""
You are a Senior Data Scientist presenting a report to stakeholders. 
Generate a comprehensive executive summary based on the dataset analysis below.

STYLE GUIDELINES:
- Do NOT use Markdown formatting (no **, ##, or italics).
- Do NOT use code blocks.
- Use UPPERCASE labels for section headers.
- Use standard bullets (-) for lists.
- The output must be ready to be copied into a plain text email or document.

DATA CONTEXT:
1. Statistical Summary:
{stats_summary}

2. Missing Values:
{missing_values}

3. Outlier Counts:
{outliers}

REPORT STRUCTURE:
EXECUTIVE SUMMARY
[Brief high-level overview of the dataset size and scope]

KEY DATA INSIGHTS
[Analyze specific columns. Explain what the mean, min/max, and spread imply about the customer base or data subject. Translate numbers into business context.]

DATA QUALITY & RISKS
[Discuss missing values and outliers. Explain the potential impact on analysis or modeling.]

CONCLUSION
[One sentence wrap-up]
"""
        )
        
        # UPDATED CHAIN: Uses StrOutputParser to ensure string return, not dict
        chain = prompt_template | self.llm | StrOutputParser()
        
        result = chain.invoke({
            "stats_summary": stats_summary,
            "missing_values": missing_str,
            "outliers": outliers_str
        })
        
        return result
    
    def plot_distributions(self, df: pd.DataFrame, output_dir: str):
        """
        Generates and saves EDA plots to the specified output_dir.
        Returns a list of saved file paths.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        saved_plot_paths = []
        sns.set_style("whitegrid")
        
        # Identify columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Limit to top 10 numeric cols to avoid cluttering PDF if dataset is huge
        for col in numeric_cols[:10]: 
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            
            # Histogram
            sns.histplot(df[col].dropna(), kde=True, bins=30, color='skyblue', ax=ax[0])
            ax[0].set_title(f'Distribution of {col}')
            
            # Boxplot
            sns.boxplot(x=df[col], color='lightgreen', ax=ax[1])
            ax[1].set_title(f'Boxplot of {col}')
            
            plt.tight_layout()
            filename = os.path.join(output_dir, f"dist_{col}.png")
            plt.savefig(filename, dpi=100)
            plt.close() # Important: Close plot to free memory
            saved_plot_paths.append(filename)

        # Categorical Plots (Limit to top 5)
        for col in categorical_cols[:5]:
            plt.figure(figsize=(8, 4))
            counts = df[col].value_counts().head(10) # Top 10 categories only
            sns.barplot(x=counts.index, y=counts.values, palette='pastel')
            plt.title(f'Count of {col}')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            filename = os.path.join(output_dir, f"cat_{col}.png")
            plt.savefig(filename, dpi=100)
            plt.close()
            saved_plot_paths.append(filename)

        # Correlation Heatmap (Crucial for ML)
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
            plt.title("Correlation Matrix")
            plt.tight_layout()
            
            filename = os.path.join(output_dir, "correlation_matrix.png")
            plt.savefig(filename, dpi=100)
            plt.close()
            saved_plot_paths.insert(0, filename) # Put heatmap first

        return saved_plot_paths