from datetime import datetime, timedelta
import os
import papermill as pm
from airflow import DAG
from airflow.operators.python import PythonOperator

BASE_DIR = "/opt/airflow"
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")

default_args = {
    "owner": "mouad",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

def execute_notebook(notebook_name):
    pm.execute_notebook(
        input_path=os.path.join(NOTEBOOKS_DIR, notebook_name),
        output_path=None,
        cwd=NOTEBOOKS_DIR,
        kernel_name="python3",
    )

with DAG(
    dag_id="mental_health_two_notebooks",
    default_args=default_args,
    description="EDA->PREPROCESSING",
    schedule="* * * * *",
    start_date=datetime(2026, 11, 1),
    catchup=False,
) as dag:

    task_eda = PythonOperator(
        task_id="run_eda",
        python_callable=execute_notebook,
        op_kwargs={"notebook_name": "01_eda.ipynb"},
    )

    task_preprocessing = PythonOperator(
        task_id="run_preprocessing",
        python_callable=execute_notebook,
        op_kwargs={"notebook_name": "02_preprocessing.ipynb"},
    )

    task_eda >> task_preprocessing