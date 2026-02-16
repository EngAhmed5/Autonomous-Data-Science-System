from configuration import *

class ErrorAnalysisAgent:
    """
    Error Analysis Agent:
    - Takes trained model and test data.
    - Calculates technical error metrics.
    - Uses LLM to generate a professional, qualitative report.
    - Identifies specific failure points.
    """
    def __init__(self, llm):
        self.llm = llm

    def _analyze_classification(self, y_test, preds):
        """
        Prepares technical context for classification problems.
        """
        # Get dict for logic
        report_dict = classification_report(y_test, preds, output_dict=True)
        # Get string for LLM context
        report_str = classification_report(y_test, preds)
        
        # Convert keys to strings to ensure '0' and 0 are handled consistently
        report_dict = {str(k): v for k, v in report_dict.items()}
        
        # Identify the worst performing class (lowest F1-score)
        # Filter out summary keys
        classes = [k for k in report_dict.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
        
        if classes:
            worst_class = min(classes, key=lambda x: report_dict[x]['f1-score'])
            worst_metric = f"{report_dict[worst_class]['f1-score']:.2f}"
            
            # Extract precision/recall specifically for the worst class for better context
            worst_prec = f"{report_dict[worst_class]['precision']:.2f}"
            worst_rec = f"{report_dict[worst_class]['recall']:.2f}"
            specific_issue = f"Class '{worst_class}' (F1: {worst_metric}, Precision: {worst_prec}, Recall: {worst_rec})"
        else:
            worst_class = "Unknown"
            specific_issue = "N/A"

        context = {
            "task_type": "Classification",
            "overall_accuracy": f"{report_dict.get('accuracy', 0):.2%}",
            "detailed_report": report_str,
            "specific_issue": specific_issue,
            "summary_stats": "N/A (Classification)"
        }
        return context

    def _analyze_regression(self, y_test, preds):
        """
        Prepares technical context for regression problems.
        """
        residuals = y_test - preds
        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        
        # Analyze residuals
        max_error = np.max(np.abs(residuals))
        mean_error = np.mean(residuals)
        std_error = np.std(residuals)
        
        if mean_error > 0:
            bias = "The model tends to UNDER-predict (actual values are higher than predicted)."
        else:
            bias = "The model tends to OVER-predict (actual values are lower than predicted)."
            
        context = {
            "task_type": "Regression",
            "overall_accuracy": f"RMSE: {np.sqrt(mse):.2f}",
            "detailed_report": f"MAE: {mae:.2f}\nMax Error: {max_error:.2f}",
            "specific_issue": f"Residual Standard Deviation: {std_error:.2f}",
            "summary_stats": f"Residual Mean: {mean_error:.2f}\nBias Analysis: {bias}"
        }
        return context

    def analyze(self, model, X_test, y_test, task_type):
        """
        Main entry point. Generates the narrative report.
        """
        preds = model.predict(X_test)
        
        # 1. Gather Technical Context
        # Ensure robust string comparison for task type
        if task_type and 'class' in str(task_type).lower():
            context = self._analyze_classification(y_test, preds)
        else:
            context = self._analyze_regression(y_test, preds)

        # 2. Define the Prompt for the LLM
        prompt_template = PromptTemplate(
            input_variables=["task_type", "overall_accuracy", "detailed_report", "specific_issue", "summary_stats"],
            template="""
            You are a Senior Machine Learning Engineer writing an Error Analysis section for a formal PDF report.
            
            INPUT DATA:
            - Task Type: {task_type}
            - Overall Performance: {overall_accuracy}
            - Primary Issue Detected: {specific_issue}
            - Stats: {summary_stats}
            
            FULL METRICS:
            {detailed_report}

            INSTRUCTIONS:
            Write a professional, readable analysis.
            Do NOT use Markdown formatting (no asterisks **, no hash signs ##). 
            Do NOT include a title/header at the very top (the report already has one).
            Use standard newlines for paragraph breaks.

            STRUCTURE YOUR RESPONSE INTO THESE 3 SECTIONS:

            EXECUTIVE SUMMARY
            [Brief assessment of model reliability and overall score.]

            DIAGNOSTIC ANALYSIS
            [Deep dive into the problem. If Classification: discuss the specific class failing and if it is a Precision vs Recall issue. If Regression: discuss the bias/variance.]

            RECOMMENDATIONS
            [Provide 2-3 actionable, technical suggestions to improve the model based on the specific error detected above.]
            """
        )

        # 3. Run the Chain with Output Parser
        # StrOutputParser ensures we get a clean string, not a dict
        chain = prompt_template | self.llm | StrOutputParser()
        
        report = chain.invoke({
            "task_type": context['task_type'],
            "overall_accuracy": context['overall_accuracy'],
            "detailed_report": context['detailed_report'],
            "specific_issue": context['specific_issue'],
            "summary_stats": context['summary_stats']
        })
        
        return report