from datetime import datetime, timedelta
from pathlib import Path
import papermill as pm
from airflow import DAG
import os
from airflow.operators.python import PythonOperator



BASE_DIR = "/opt/airflow"
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_notebooks")


default_args = {
    "owner": "mouad",
    "retries": 1,
    "retry_delay": timedelta(minutes=2)
}



def execute_notebook(notebook_name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    pm.execute_notebook(
        input_path=f"{NOTEBOOKS_DIR}/{notebook_name}",
        output_path=f"{OUTPUT_DIR}/out_{notebook_name}",
        cwd=NOTEBOOKS_DIR,
        kernel_name="python3",
    )


with DAG(
    dag_id="mental_health_two_notebooks",
    default_args=default_args,
    description="EDA->PREPROCESSING",
    schedule="* * * * *",
    start_date=datetime(2026,11,1),
    catchup=False,
) as dag:

    task_eda = PythonOperator(
        task_id="run_eda",
        python_callable=execute_notebook,
        op_kwargs={"notebook_name": "01_eda.ipynb"}
    )

    task_preprocessing = PythonOperator(
        task_id="run_preprocessing",
        python_callable=execute_notebook,
        op_kwargs={"notebook_name": "02_preprocessing.ipynb"},   
    )

    task_eda >> task_preprocessing

