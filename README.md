# Student Mental Health Monitor

## Run Airflow locally

From the project directory in WSL, start Docker Desktop with WSL integration enabled, then run:

```bash
docker compose up --build
```

Open [http://localhost:8080](http://localhost:8080). Airflow prints the generated
`admin` password in the container logs on first startup. If the DAG is paused,
unpause `mental_health_two_notebooks` in the Airflow UI.

The DAG runs the notebooks in order and saves executed copies under
`airflow/output_notebooks/`. Model files are written to `models/`.

## Run the Streamlit app

Install the project dependencies, then launch the app from the project directory:

```bash
pip install -r requirement.txt
streamlit run app/streamlit_app.py
```

The app has two sections: model-performance analysis and a form for predictions
on a new student profile.