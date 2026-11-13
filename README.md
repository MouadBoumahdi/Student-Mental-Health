# Student Mental Health Monitor

## Run Airflow locally

From the project directory in WSL, start Docker Desktop with WSL integration enabled, then run:

```bash
docker compose up --build
```

Open [http://localhost:8080](http://localhost:8080). Airflow prints the generated
`admin` password in the container logs on first startup. Trigger
`mental_health_two_notebooks` from the UI when you want to run the notebooks.

The DAG runs the notebooks in order and saves executed copies under
`airflow/output_notebooks/`. Model files are written to `models/`.

## Run the Streamlit app

Install the project dependencies, then launch the app from the project directory:

```bash
python -m pip install -r requirement.txt
python -m streamlit run app/streamlit_app.py
```

The app has two sections: model-performance analysis and a form for predictions
on a new student profile.

## Project layout

```text
airflow/
  dags/mental_health_dag.py
app/streamlit_app.py
data/raw/student_mental_health.csv
models/                         # fitted pipeline, label encoder, and metrics
notebooks/
  01_eda.ipynb
  02_preprocessing.ipynb        # training, tuning, evaluation, and saving
Dockerfile
docker-compose.yml
requirement.txt
```