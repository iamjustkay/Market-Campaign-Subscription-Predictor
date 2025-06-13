import subprocess
import threading
import time
import os

# Step 1: Run Predictive_Model.py
def run_predictive_model():
    print("🚀 Running Predictive_Model.py...")
    subprocess.run([r"venv\Scripts\python.exe", "Predictive_Model.py"], check=True)
    print("✅ Finished running Predictive_Model.py.\n")

# Step 2: Start FastAPI backend
def run_fastapi():
    print("🌐 Starting FastAPI backend...")
    subprocess.run(["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"])

# Step 3: Start Streamlit frontend
def run_streamlit():
    print("🎨 Starting Streamlit frontend...")
    subprocess.run(["streamlit", "run", "frontend.py", "--server.port=8501", "--server.address=0.0.0.0"])

if __name__ == "__main__":
    # Run predictive model first
    run_predictive_model()

    # Start FastAPI and Streamlit in parallel threads
    t1 = threading.Thread(target=run_fastapi)
    t2 = threading.Thread(target=run_streamlit)

    t1.start()
    time.sleep(2)  # Small delay to let backend initialize
    t2.start()

    t1.join()
    t2.join()