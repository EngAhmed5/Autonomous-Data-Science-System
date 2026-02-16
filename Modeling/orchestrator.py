from configuration import *
from email_agent import *
from model_setup import llm, embedding_model
from data_loader import DataLoaderAgent
from eda_agent import EDAAgent
from preprocessing_agent import PreprocessingAgent 
from ml_tuning_agent import MLTuningAgent
from error_analysis_agent import ErrorAnalysisAgent
from documentation_agent import MLReportAgent  

class MLOrchestrator:
    def __init__(self):
        self.llm = llm
        self.email_agent = EmailAgent(bot_email=BOT_MAIL, bot_app_password=BOT_PASS, llm=self.llm)
        self.results_dir = RESULTS_DIR
        self.models_dir = MODELS_DIR

    # 1. ADD status_callback argument
    def run_pipeline(self, file_path, target_col, recipient_email, status_callback=None):
        
        # Helper function to handle both printing to console and updating Streamlit
        def update_status(message, progress=None):
            print(message) # Keep console working
            if status_callback:
                status_callback(message, progress)

        update_status("="*60)
        update_status(f"STARTING ML PIPELINE FOR: {os.path.basename(file_path)}")
        update_status("="*60)

        # Initialize Agents
        self.loader = DataLoaderAgent()
        self.eda_agent = EDAAgent(llm=self.llm)
        self.preprocessor = PreprocessingAgent()
        self.tuner = MLTuningAgent(n_iter=N_ITER, cv=CV, random_state=RANDOM_STATE)
        self.error_agent = ErrorAnalysisAgent(llm=self.llm)
        self.report_agent = MLReportAgent(llm=self.llm)

        # [1/7] Loading
        update_status("\n[1/7] Loading Data...", 10)
        df = self.loader.load_df(file_path)

        # [2/7] EDA
        update_status("\n[2/7] Running EDA...", 25)
        plots_dir = os.path.join(self.results_dir, "plots")
        if not os.path.exists(plots_dir): os.makedirs(plots_dir)
            
        eda_summary_text = self.eda_agent.generate_summary(df)
        update_status("      Generating visualizations...")
        plot_images = self.eda_agent.plot_distributions(df, output_dir=plots_dir)
        
        stats_df = self.eda_agent.summary_stats(df)
        missing_df = self.eda_agent.missing_values(df)
        outliers_dict = self.eda_agent.detect_outliers(df)
        
        df_summary = {
            "text_summary": eda_summary_text,
            "stats": stats_df,
            "missing": missing_df,
            "outliers": outliers_dict,
            "plot_paths": plot_images
        }
        update_status("EDA Complete.")

        # [3/7] Preprocessing
        update_status("\n[3/7] Preprocessing Data...", 40)
        X_train, X_test, y_train, y_test = self.preprocessor.run(df, target_col=target_col)
        
        preproc_info = {
            "num_cols": self.preprocessor.numeric_features,
            "cat_cols": self.preprocessor.categorical_features,
            "target": target_col,
            "train_shape": X_train.shape,
            "test_shape": X_test.shape
        }
        update_status("Preprocessing Complete.")

        # [4/7] Tuning
        update_status("\n[4/7] Tuning Models...", 55)
        best_model, history = self.tuner.tune_and_evaluate(X_train, y_train, X_test, y_test)
        
        # Save Model
        update_status("      Saving Best Model...")
        if not os.path.exists(self.models_dir): os.makedirs(self.models_dir)
            
        dataset_name = os.path.basename(file_path).split('.')[0]
        model_filename = f"BestModel_{dataset_name}.pkl"
        model_save_path = os.path.join(self.models_dir, model_filename)
        joblib.dump(best_model, model_save_path)
        update_status(f"      Model saved at: {model_save_path}")

        update_status("      Generating ML performance plots...")
        ml_plot_paths = self.tuner.plot_performance(X_test, y_test, output_dir=plots_dir)
        
        ml_summary = {
            "best_model_name": best_model.__class__.__name__,
            "best_score": self.tuner.best_score,
            "best_params": self.tuner.best_params,
            "leaderboard": pd.DataFrame([{'Model': h['Model'], 'Score': h['Test_Score'], 'Metrics': h['Metrics']} for h in history]),
            "ml_plots": ml_plot_paths,
            "model_path": model_save_path
        }

        # [5/7] Error Analysis
        update_status("\n[5/7] Analyzing Errors...", 70)
        task_type = self.tuner.task_type
        analysis_report_text = self.error_agent.analyze(model=best_model, X_test=X_test, y_test=y_test, task_type=task_type)

        # [6/7] Report
        update_status("\n[6/7] Generating PDF Report...", 85)
        if not os.path.exists(self.results_dir): os.makedirs(self.results_dir)

        base_filename = f"Report_{os.path.basename(file_path).split('.')[0]}.pdf"
        full_pdf_path = os.path.join(self.results_dir, base_filename)
        
        self.report_agent.generate_pdf_report(
            filename=full_pdf_path,
            data_summary=df_summary,
            preproc_info=preproc_info,
            ml_results=ml_summary,
            error_analysis=analysis_report_text
        )
        update_status(f"PDF Saved at: {full_pdf_path}")

        # [7/7] Email
        update_status("\n[7/7] Sending Email...", 95)
        self.email_agent.send_report(user_recipient_email=recipient_email, pdf_path=full_pdf_path, ml_results=ml_summary)

        update_status("\n" + "="*60)
        update_status("PIPELINE FINISHED SUCCESSFULLY")
        update_status("="*60, 100)