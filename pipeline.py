#!/usr/bin/env/python3

"""Pipeline customer-remarketing."""

from kfp.components import load_component_from_file, load_component_from_url
from kfp import dsl
from kfp import compiler
from kfp.gcp import use_gcp_secret


INGRESS_GATEWAY = "http://istio-ingressgateway.istio-system.svc.cluster.local"

download_op = load_component_from_url("https://raw.githubusercontent.com/kubeflow/pipelines/74c7773ca40decfd0d4ed40dc93a6af591bbc190/components/contrib/google-cloud/storage/download_blob/component.yaml")
ingestion_op = load_component_from_file("components/data_ingestion.yaml")  # pylint: disable=not-callable
prep_op = load_component_from_file("components/processing.yaml")  # pylint: disable=not-callable


@dsl.pipeline(
    name="Training pipeline", description=""
)
def customer_remarketing():
    """Thid method defines the pipeline tasks and operations"""

    download_data = (
        download_op(
            gcs_path='gs://my-bucket/path/model'
       ).set_display_name("Download data from GCS") 
    ).apply(use_gcp_secret('user-gcp-sa'))

    ingestion_task = (
        ingestion_op(
            input_data=download_data.outputs["Data"],
        ).set_display_name("Data Ingestion and Split")
    ).apply(use_gcp_secret('user-gcp-sa'))

    prep_task = (
        prep_op(
            input_data=ingestion_task.outputs["output_data"],
        ).after(ingestion_task).set_display_name("Data Preparation")
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        customer_remarketing, package_path="customer-remarketing.yaml"
    )