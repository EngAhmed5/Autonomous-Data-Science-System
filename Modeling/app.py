import streamlit as st
import pandas as pd
import os
import time
import base64
import shutil

# Import your Orchestrator
from orchestrator import MLOrchestrator
from configuration import *

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="Autonomous Data Science System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Terminal Look
st.markdown("""
<style>
    .main { background-color: #f5f5f5; }
    .stButton>button {
        width: 100%;
        background-color: #FF7900; 
        color: white;
        border-radius: 5px;
        font-weight: bold;
    }
    /* Terminal Style for Logs */
    .terminal-box {
        background-color: #1e1e1e;
        color: #00ff00;
        font-family: 'Courier New', Courier, monospace;
        padding: 10px;
        border-radius: 5px;
        height: 300px;
        overflow-y: scroll;
        white-space: pre-wrap;
        font-size: 14px;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Session State Management
# ==========================================
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None
if "report_path" not in st.session_state:
    st.session_state.report_path = None
if "model_path" not in st.session_state:
    st.session_state.model_path = None

# ==========================================
# Sidebar: Configuration
# ==========================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Orange_logo.svg/1200px-Orange_logo.svg.png", width=50)
st.sidebar.title("Configuration")

recipient_email = st.sidebar.text_input("Recipient Email", value="your_email@example.com")
uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file:
    if st.session_state.last_uploaded_file != uploaded_file.name:
        st.session_state.analysis_done = False
        st.session_state.report_path = None
        st.session_state.model_path = None
        st.session_state.last_uploaded_file = uploaded_file.name
        if os.path.exists("temp_data"):
            shutil.rmtree("temp_data")
            os.makedirs("temp_data")

# ==========================================
# Main Layout
# ==========================================
st.title("🤖 Autonomous Data Science System")
st.markdown("### Powered by LLM Agents & Scikit-Learn")
st.write("Upload your dataset, select the target variable, and let the AI agents handle the rest.")

if uploaded_file:
    # 1. Save File Temporarily
    if not os.path.exists("temp_data"):
        os.makedirs("temp_data")
        
    file_path = os.path.join("temp_data", uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # 2. Preview Data
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        st.subheader("📊 Data Preview")
        st.dataframe(df.head())
        
        all_columns = df.columns.tolist()
        
        # 3. Target Selection
        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox("Select Target Column (Prediction Goal)", all_columns)
        
        with col2:
            st.info(f"The system will automatically detect if this is a **Classification** or **Regression** task based on '{target_col}'.")

        # 4. Execution Button
        if st.button("🚀 Start AI Analysis Pipeline"):
            
            # Create UI Elements for Progress
            progress_label = st.empty()
            progress_bar = st.progress(0)
            
            # Create a dedicated container for logs that looks like a terminal
            st.markdown("### 🖥️ System Logs")
            log_container = st.empty()
            
            # Helper to manage logs state locally during execution
            logs = []

            # -------------------------------------------------------------
            # THE CALLBACK FUNCTION
            # -------------------------------------------------------------
            def stream_callback(message, percentage=None):
                # 1. Update Progress Bar if percentage is provided
                if percentage is not None:
                    progress_bar.progress(percentage)
                    progress_label.markdown(f"**Status:** {message.strip()}")
                
                # 2. Update Logs (Append new message)
                # Add timestamp for extra professionalism
                timestamp = time.strftime("%H:%M:%S")
                logs.append(f"[{timestamp}] {message}")
                
                # Join logs and render in the container
                # We use a code block to simulate a terminal
                log_text = "\n".join(logs)
                log_container.code(log_text, language="bash")
            # -------------------------------------------------------------

            try:
                # Instantiate Orchestrator FRESH
                orchestrator = MLOrchestrator()
                
                # Run Pipeline passing the callback
                orchestrator.run_pipeline(
                    file_path=file_path, 
                    target_col=target_col, 
                    recipient_email=recipient_email,
                    status_callback=stream_callback  # <--- PASSING THE CALLBACK
                )
                
                # Success Logic
                st.session_state.analysis_done = True
                
                dataset_name = uploaded_file.name.split('.')[0]
                
                # Set Paths
                st.session_state.report_path = os.path.join(orchestrator.results_dir, f"Report_{dataset_name}.pdf")
                st.session_state.model_path = os.path.join(orchestrator.models_dir, f"BestModel_{dataset_name}.pkl")
                
                st.balloons()
                st.rerun() 
                    
            except Exception as e:
                st.error(f"❌ An error occurred during execution: {e}")
                import traceback
                traceback.print_exc()

    except Exception as e:
        st.error(f"Error reading file: {e}")

# ==========================================
# Results Display
# ==========================================
if st.session_state.analysis_done:
    st.divider()
    st.header("📈 Analysis Results")
    
    col_res1, col_res2 = st.columns(2)
    
    # --- PDF DOWNLOAD ---
    with col_res1:
        st.subheader("📄 Project Report")
        if st.session_state.report_path and os.path.exists(st.session_state.report_path):
            with open(st.session_state.report_path, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_bytes,
                    file_name=os.path.basename(st.session_state.report_path),
                    mime="application/pdf",
                    key="btn_pdf"
                )
            st.success(f"Report emailed to: **{recipient_email}**")
        else:
            st.warning("PDF Report not found.")

    # --- MODEL DOWNLOAD ---
    with col_res2:
        st.subheader("🧠 Best Model")
        if st.session_state.model_path and os.path.exists(st.session_state.model_path):
            with open(st.session_state.model_path, "rb") as model_file:
                model_bytes = model_file.read()
                st.download_button(
                    label="💾 Download Trained Model (.pkl)",
                    data=model_bytes,
                    file_name=os.path.basename(st.session_state.model_path),
                    mime="application/octet-stream",
                    key="btn_model"
                )
            st.success("Model is ready for deployment.")
        else:
            st.warning("Model file not found.")

else:
    if not uploaded_file:
        st.info("👈 Please upload a dataset to begin.")