from datetime import datetime

from minio import Minio
from airflow.models import DAG
from airflow.hooks.base import BaseHook
from airflow.models.param import Param
from airflow.operators.python import PythonOperator
from airflow.providers.http.hooks.http import HttpHook

from _minio.s3 import upload_file, download_file
from _label_studio.task import create_task
from _label_studio.project import create_project
from _label_studio.prediction import add_prediction
from preprocessing.split import split_tif_file


with DAG(
    "create_labeling_set",
    schedule=None,
    start_date=datetime(2021, 1, 1),
    catchup=False,
    params={
        "type": Param(
            type="string",
            description="If type segmentation is selected, the selected model will be used to create a segmentation mask. If object detection is selected, the selected model will be used to create a bounding box.",
            default="ObjectDetection",
            enum=["Segmentation", "ObjectDetection"],
        ),
        "model": Param(
            [],
            type="string",
            enum=[
                f"models/{p.object_name}"
                for p in Minio(
                    BaseHook.get_connection(conn_id="minio").host,
                    access_key=BaseHook.get_connection(conn_id="minio").login,
                    secret_key=BaseHook.get_connection(
                        conn_id="minio").password,
                    secure=True,
                ).list_objects("models", recursive=True)
            ],
        ),
        "s3_file": Param(
            [],
            type="string",
            enum=[
                f"raw-tree/{p.object_name}"
                for p in Minio(
                    BaseHook.get_connection(conn_id="minio").host,
                    access_key=BaseHook.get_connection(conn_id="minio").login,
                    secret_key=BaseHook.get_connection(
                        conn_id="minio").password,
                    secure=True,
                ).list_objects("raw-tree", recursive=True)
            ],
        ),
        "chunk_size": Param(640, type="integer", minimum=1, maximum=10_000),
        "overlap_factor": Param(0.2, type="number", minimum=0.0, maximum=1.0),
        "create_project": Param(
            default=False,
            type="boolean",
            description="Create a new project in Label Studio",
        ),
        "project": Param(
            [],
            type=["null", "string"],
            description="Select a Label Studio project or nothing if create a new project is selected",
            enum=[
                f"{p['id']}/{p['title']}"
                for p in HttpHook(http_conn_id="label-studio", method="GET")
                .run(
                    endpoint="/api/projects",
                    headers={"Content-Type": "application/json"},
                )
                .json()["results"]
            ],
        ),
    },
    tags=["preprocessing"],
) as dag:
    _download_model = download_file.override(task_id="download_model")(
        local_path="/opt/airflow/models/{{params.model.split('/')[1]}}",
        bucket_name="{{params.model.split('/')[0]}}",
        object_name="{{'/'.join(params.model.split('/')[1:])}}",
        conn_id="minio",
    )

    _download_file = download_file.override(task_id="download_file")(
        local_path="/opt/airflow/raw",
        bucket_name="{{params.s3_file.split('/')[0]}}",
        object_name="{{params.s3_file.split('/')[1]}}",
        conn_id="minio",
    )

    _split_file = PythonOperator(
        task_id="split_file",
        python_callable=split_tif_file,
        op_kwargs={
            "file": "/opt/airflow/raw/{{params.s3_file.split('/')[1]}}",
            "output_dir": "/opt/airflow/split",
            "chunk_size": ("{{params.chunk_size}}", "{{params.chunk_size}}"),
            "overlap_factor": "{{params.overlap_factor}}",
        },
    )

    _create_project = create_project.override(task_id="create_project")()

    _upload_file = upload_file.override(task_id="upload_file", trigger_rule="all_done")(
        local_path="/opt/airflow/split/{{params.s3_file.split('/')[1].split('.')[0]}}",
        bucket_name="split",
        conn_id="minio",
    )

    _create_task = create_task.override(
        task_id="create_task", trigger_rule="all_done"
    )()

    _add_prediction = add_prediction.override(task_id="add_prediction")()

    (
        _download_model
        >> _download_file
        >> _split_file
        >> _create_project
        >> _upload_file
        >> _create_task
        >> _add_prediction
    )
