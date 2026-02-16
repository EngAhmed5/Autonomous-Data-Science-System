# 🤖 Autonomous Data Science System (ADSS)
> **Data Science on Autopilot: An end-to-end agentic ML orchestrator designed to transform raw data into production-ready models and executive-level PDF reports.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange)](https://scikit-learn.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Agentic-green)](https://langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)](https://streamlit.io/)
[![Orange](https://img.shields.io/badge/ODC_Egypt-Final_Project-black?labelColor=FF7900)](https://www.orangedigitalcenters.com/)

---

## 👥 The Team
This project was collaboratively developed as a **Team Project** for the **Orange Digital Center (ODC) Egypt** (AI Agents & LLMs Track) by:
*   **Ahmed Ismail El Sayed**
*   **Ahmed Mohamed Hussein**

---

## 📌 Project Overview
The **Autonomous Data Science System (ADSS)** is a sophisticated multi-agent framework that automates the entire machine learning lifecycle. By combining the reasoning capabilities of **Large Language Models (LLMs)** with the computational power of **Scikit-Learn**, ADSS handles everything from data ingestion to sending a professional diagnostic report directly to your inbox.

### 🏗️ Agentic Workflow
The system employs an **Orchestrator-Agent Architecture** consisting of specialized autonomous units:
*   **🔍 EDA Agent:** Generates statistical summaries, detects outliers, and creates a comprehensive visualization suite.
*   **🛠️ Preprocessing Agent:** Dynamically engineers features, handles scaling, and encodes variables based on real-time data inspection.
*   **🏎️ ML Tuning Agent:** Optimizes model performance via `RandomizedSearchCV`, comparing multiple algorithms to find the best fit.
*   **⚖️ Error Analysis Agent:** Interprets complex technical metrics into high-level, human-readable diagnostic insights using LLMs.
*   **📑 Documentation Agent:** Programmatically builds professional multi-page PDF reports using `ReportLab`.
*   **📧 Email Agent:** Automates the final delivery of reports and artifacts via Gmail.

---

## 🎥 Project Demos
### Streamlit Pipeline Interface
*Witness the system logs, progress tracking, and autonomous decision-making in real-time.*

https://github.com/user-attachments/assets/3143b04d-ac54-4ec4-92f1-10be3989e50f

---

## 📁 Repository Structure
```bash
│   requirements.txt
│
├───Data                    # Sample datasets used for testing
│       Bank Customer Churn Prediction.csv
│       Car details v3.csv
│
├───Modeling                # Core System Logic & Agents
│       app.py              # Streamlit Web Application
│       orchestrator.py     # Central System Brain
│       eda_agent.py        # Exploratory Data Analysis Agent
│       preprocessing_agent.py # Dynamic Pipeline Agent
│       ml_tuning_agent.py  # Hyperparameter Optimization Agent
│       error_analysis_agent.py # LLM Diagnostic Agent
│       documentation_agent.py # PDF Reporting Agent
│       email_agent.py      # Automated Delivery Agent
│       configuration.py    # Environment & API Credentials
│       model_setup.py      # LLM Initialization
│
├───Results                 # Output Artifacts
│   │   Report_Bank_Churn.pdf
│   └───plots               # Embedded Visualizations (Correlation, Residuals, etc.)
│           confusion_matrix.png
│           model_comparison.png
│           dist_credit_score.png
│
└───Save Models             # Serialized Best Models (.pkl)
        BestModel_Churn_Prediction.pkl
```

---

## 🚀 Key Technical Features
- **Dynamic Task Detection:** Automatically distinguishes between Classification and Regression tasks.
- **Auto-Report Generation:** Generates a professional 8-section PDF including TOC, Executive Summary, and Visual Analysis.
- **LLM Diagnostic Reasoning:** Provides qualitative explanations of model failure points (Precision/Recall/Bias).
- **Terminal-Style UI:** A specialized Streamlit log console that tracks agent activity line-by-line.
- **Secure Model Export:** Saves the final optimized model as a `.pkl` for immediate deployment.

---

## 💻 Installation & Setup

1.  **Clone the Repo**
    ```bash
    git clone https://github.com/yourusername/autonomous-data-science-system.git
    cd autonomous-data-science-system
    ```

2.  **Install Requirements**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Configuration**
    Set your API keys and Gmail credentials in `Modeling/configuration.py`.

4.  **Run the App**
    ```bash
    streamlit run Modeling/app.py
    ```

---

## 🎓 Acknowledgments
Developed as the final graduation project for the **Orange Digital Center (ODC) Egypt**. We extend our gratitude to our mentors in the **AI Agents & LLMs track** for their technical guidance and support.

**License:** MIT License