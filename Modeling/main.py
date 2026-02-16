from configuration import *
from orchestrator import *

def main():
    # 1. Instantiate the Orchestrator
    orchestrator = MLOrchestrator()

    # Define your Recipient
    USER_EMAIL = "ahmed202201998@gmail.com" 

    # --- SCENARIO 1: Bank Customer Churn (Classification) ---
    print(">>> RUNNING SCENARIO 1: BANK CHURN")
    orchestrator.run_pipeline(
        file_path=CHURN_CSV_PATH,
        target_col="Exited",  # Target column for Churn dataset
        recipient_email=USER_EMAIL
    )

    # --- SCENARIO 2: Car Price Prediction (Regression) ---
    # Uncomment below to run the second dataset
    """
    print("\n\n>>> RUNNING SCENARIO 2: CAR PRICES")
    orchestrator.run_pipeline(
        file_path=CAR_CSV_PATH,
        target_col="selling_price", # Target column for Car dataset
        recipient_email=USER_EMAIL
    )
    """

if __name__ == "__main__":
    orchestrator = MLOrchestrator()

    # Define your Recipient
    USER_EMAIL = "ahmed202201998@gmail.com" 

    # --- SCENARIO 1: Bank Customer Churn (Classification) ---
    print(">>> RUNNING SCENARIO 1: BANK CHURN")
    orchestrator.run_pipeline(
        file_path=CHURN_CSV_PATH,
        target_col="churn",  # Target column for Churn dataset
        recipient_email=USER_EMAIL
    )