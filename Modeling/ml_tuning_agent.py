from configuration import *

class MLTuningAgent:
    """
    Advanced Agent that:
    1. Detects Task (Classif/Reg)
    2. Selects Candidate Models
    3. TUNES Hyperparameters (RandomizedSearch)
    4. Evaluates on Test Set
    """
    def __init__(self, n_iter=N_ITER, cv=CV, random_state=RANDOM_STATE):
        self.n_iter = n_iter        # How many hyperparam combinations to try
        self.cv = cv                # Cross-validation folds
        self.random_state = random_state
        self.best_model = None
        self.best_params = {}
        self.best_score = -float('inf')
        self.task_type = None
        self.history = []

    def _detect_task(self, y):
        """Determine if Classification or Regression based on target data."""
        if y.dtype == 'object' or y.dtype.name == 'category':
            return 'classification'
        
        if y.nunique() <= 20:
            return 'classification'
        return 'regression'

    def _get_candidates(self):
        """
        Returns a dictionary of models AND their hyperparameter grids.
        """
        if self.task_type == 'classification':
            return {
                "RandomForest": {
                    "model": RandomForestClassifier(random_state=self.random_state, class_weight='balanced'),
                    "params": {
                        "n_estimators": [50, 100, 200],
                        "max_depth": [None, 10, 20, 30],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4]
                    }
                },
                "GradientBoosting": {
                    "model": GradientBoostingClassifier(random_state=self.random_state),
                    "params": {
                        "n_estimators": [50, 100, 200],
                        "learning_rate": [0.01, 0.1, 0.2],
                        "max_depth": [3, 5, 7]
                    }
                },
                "LogisticRegression": {
                    "model": LogisticRegression(max_iter=2000, random_state=self.random_state),
                    "params": {
                        "C": [0.1, 1, 10],
                        "solver": ['liblinear', 'lbfgs']
                    }
                }
            }
        else: # Regression
            return {
                "RandomForest": {
                    "model": RandomForestRegressor(random_state=self.random_state),
                    "params": {
                        "n_estimators": [50, 100, 200],
                        "max_depth": [None, 10, 20],
                        "min_samples_split": [2, 5]
                    }
                },
                "GradientBoosting": {
                    "model": GradientBoostingRegressor(random_state=self.random_state),
                    "params": {
                        "n_estimators": [50, 100],
                        "learning_rate": [0.01, 0.1],
                        "max_depth": [3, 5]
                    }
                },
                "Ridge": {
                    "model": Ridge(),
                    "params": {
                        "alpha": [0.1, 1.0, 10.0]
                    }
                }
            }

    def tune_and_evaluate(self, X_train, y_train, X_test, y_test):
        """
        Main execution workflow.
        """
        self.task_type = self._detect_task(y_train)

        candidates = self._get_candidates()
        
        # Metric to optimize during tuning
        scoring_metric = 'f1_weighted' if self.task_type == 'classification' else 'r2'
        
        for name, config in candidates.items():
            # RandomizedSearchCV setup
            search = RandomizedSearchCV(
                estimator=config['model'],
                param_distributions=config['params'],
                n_iter=self.n_iter,
                scoring=scoring_metric,
                cv=self.cv,
                n_jobs=-1, # Use all CPU cores
                random_state=self.random_state,
                verbose=0
            )
            
            # Fit tuning
            search.fit(X_train, y_train)
            tuned_model = search.best_estimator_
            
            # Evaluate on Holdout Test Set
            if self.task_type == 'classification':
                preds = tuned_model.predict(X_test)
                # Handle probability for AUC if available
                try:
                    probs = tuned_model.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, probs) if y_test.nunique() == 2 else 0
                except:
                    auc = 0
                
                # --- ADDED: Classification Report ---
                cls_report = classification_report(y_test, preds, zero_division=0)
                
                score = f1_score(y_test, preds, average='weighted')
                metrics = {
                    "Accuracy": accuracy_score(y_test, preds),
                    "F1 Score": score,
                    "AUC": auc,
                    "Best Params": search.best_params_,
                    "Classification Report": cls_report # Stored here
                }
            else: # Regression
                preds = tuned_model.predict(X_test)
                score = r2_score(y_test, preds)
                metrics = {
                    "R2 Score": score,
                    "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
                    "MAE": mean_absolute_error(y_test, preds),
                    "Best Params": search.best_params_
                }

            # Save history
            self.history.append({
                "Model": name,
                "Test_Score": score,
                "Metrics": metrics,
                "Object": tuned_model
            })
            
            # Update Best Model Logic
            if score > self.best_score:
                self.best_score = score
                self.best_model = tuned_model
                self.best_params = search.best_params_

        return self.best_model, self.history

    def plot_performance(self, X_test, y_test, output_dir):
        """
        Generates and SAVES performance plots.
        Returns a dictionary of generated file paths.
        """
        if self.best_model is None:
            return {}

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        generated_plots = {}
        sns.set_style("whitegrid")
        preds = self.best_model.predict(X_test)
        
        # 1. Specific Performance Plot (Confusion Matrix or Residuals)
        plt.figure(figsize=(8, 6))
        
        if self.task_type == 'classification':
            # Confusion Matrix
            cm = confusion_matrix(y_test, preds)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=1)
            plt.title(f"Confusion Matrix: {self.best_model.__class__.__name__}")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            filename = "confusion_matrix.png"
        else:
            # Residual Plot
            residuals = y_test - preds
            plt.scatter(preds, residuals, alpha=0.5, color='green')
            plt.axhline(0, color='red', linestyle='--')
            plt.xlabel("Predicted Values")
            plt.ylabel("Residuals (Error)")
            plt.title(f"Residual Plot: {self.best_model.__class__.__name__}")
            filename = "residuals_plot.png"
            
        filepath = os.path.join(output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=100)
        plt.close() # Close to free memory
        generated_plots['performance_plot'] = filepath

        # 2. Model Comparison Bar Chart
        if self.history:
            model_names = [h['Model'] for h in self.history]
            scores = [h['Test_Score'] for h in self.history]
            
            plt.figure(figsize=(10, 5))
            sns.barplot(x=model_names, y=scores, palette='viridis')
            plt.title("Model Comparison (Test Score)")
            plt.ylabel("Score")
            plt.ylim(0, 1.1 if max(scores) <= 1 else max(scores)*1.1)
            
            filepath_comp = os.path.join(output_dir, "model_comparison.png")
            plt.tight_layout()
            plt.savefig(filepath_comp, dpi=100)
            plt.close()
            generated_plots['comparison_plot'] = filepath_comp

        return generated_plots