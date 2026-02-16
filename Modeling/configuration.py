# ===========================
# Suppress Warnings
# ===========================
import warnings
warnings.filterwarnings("ignore")

# ===========================
# Core Python & Data Handling
# ===========================
import numpy as np
import pandas as pd
import os
import re
import datetime
import torch

# ===========================
# Email Handling
# ===========================
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# ===========================
# LangChain & RAG Components
# ===========================
from langchain_community.vectorstores import FAISS             
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter     
from langchain_core.documents import Document                  
from langchain_community.document_loaders import CSVLoader      
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser             
from langchain_community.llms import Ollama                    

# ===========================
# PDF & File Processing
# ===========================
import pdfplumber                                            

# ===========================
# Visualization Libraries
# ===========================
import matplotlib.pyplot as plt                                 
import seaborn as sns                                           

# ===========================
# Scikit-Learn: Preprocessing
# ===========================
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.compose import ColumnTransformer                  
from sklearn.pipeline import Pipeline                           
from sklearn.impute import SimpleImputer                        

# ===========================
# Scikit-Learn: Model Training Utilities
# ===========================
from sklearn.model_selection import train_test_split            
import joblib
# ===========================
# Scikit-Learn: Models
# ===========================
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)

# ===========================
# Scikit-Learn: Hyperparameter Tuning
# ===========================
from sklearn.model_selection import RandomizedSearchCV          

# ===========================
# Scikit-Learn: Evaluation Metrics
# ===========================
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    mean_squared_error, r2_score, mean_absolute_error, 
    classification_report, confusion_matrix
)

# ===========================
# PDF Report Generation (ReportLab)
# ===========================
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, 
    Table, TableStyle, Image, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.pdfgen import canvas


# ===========================
# Configuration Constants
CHUNKS_SIZE = 250
OVERLAP = 50
N_ITER = 7
CV = 5
RANDOM_STATE = 42

# For Main.py 
CHURN_CSV_PATH = r"D:\Ahmed\Study\Orange Digital Hub ODC\Autonomous Data Science System\Data\Bank Customer Churn Prediction.csv"
CAR_CSV_PATH = r"D:\Ahmed\Study\Orange Digital Hub ODC\Autonomous Data Science System\Data\Car details v3.csv"

# ===========================
# Paths 
RESULTS_DIR = r"D:\Ahmed\Study\Orange Digital Hub ODC\Autonomous Data Science System\Results"
MODELS_DIR = r"D:\Ahmed\Study\Orange Digital Hub ODC\Autonomous Data Science System\Save Models"