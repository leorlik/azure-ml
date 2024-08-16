from azure.ai.ml import load_component
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
import json
import pandas as pd
import mltable 
from azure.ai.ml import Input
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import Output


CREDENTIALS_PLACE = "../credentials/credentials.json"
WORKSPACE_NAME = "campus_recruitment_ws"
RESOURCE_GROUP_NAME = "rg_campus_recruitment_leorlik_azure"
COLUMNS = "\"gender ssc_p ssc_b hsc_p hsc_c degree_p degree_t workex etest_p specialisation mba_p\""
TARGET_COLUMNS = "status"
COMPUTE_INSTANCE_NAME = "c-r-compute-instance"

def get_credentials():
    
    with open(CREDENTIALS_PLACE) as f:
        data = json.load(f)
        f.close()

    subscription_id = data["subscription-id"]
    client_id = data["client-id"]
    client_secret = data["client-secret"]
    tenant_id = data["tenant-id"]

    return subscription_id, client_id, client_secret, tenant_id

# @pipeline()
# def pipeline_function(pipeline_job_input):


#     # drop_columns_component = ml_client.components.get(name = "drop_columns")
#     drop_columns_step = drop_columns(
#         input_data = pipeline_job_input,
#         columns_to_drop = "salary sl_no"
#     )

#     # encoding_component = ml_client.components.get(name = "label_encoding")
#     encoding_columns_step = label_encoding(
#         input_data = drop_columns_step.outputs.output_data
#     )
    
#     # normalize_component = ml_client.components.get(name = "normalize_data")
#     normalize_columns_step = normalize_data(
#         input_data = encoding_columns_step.outputs.output_data,
#         columns_to_normalize = "ssc_p hsc_p hsc_s degree_p etest_p mba_p",
#         type_of_norm = "l2"
#     )

#     # train_test_split_component = ml_client.components.get(name = "train_test_split")

#     train_test_split_step = train_test_split(
#         input_data = normalize_columns_step.outputs.output_data,
#     )

#     # training_component = ml_client.components.get(name = "train_gradient_boosting")
    
#     train_step = train_gradient_boosting(
#         input_training_data = train_test_split_step.outputs.output_train,
#         input_test_data = train_test_split_step.outputs.output_test,
#         target_column = TARGET_COLUMNS
#     )


#     return {
#         "pipeline_job_transformed_data": normalize_component.outputs.output_data,
#         "pipeline_job_trained_model": train_step.outputs.output_model,
#     }

def get_input(ml_client, name = "placement_csv", version = "1.0.0"):

    data_asset = ml_client.data.get(name=name, version=version)
    my_training_data_input = Input(type=AssetTypes.URI_FILE, path=data_asset.path)
    return my_training_data_input



def main():

    subscription_id, client_id, client_secret, tenant_id = get_credentials()

    credential = ClientSecretCredential(client_id=client_id, client_secret=client_secret, tenant_id=tenant_id)

    ml_client = MLClient(credential,  subscription_id= subscription_id,  resource_group_name=RESOURCE_GROUP_NAME, 
                        workspace_name=WORKSPACE_NAME)

    input_data = get_input(ml_client, name = "placement_csv", version = "1.0.0")

    drop_columns = ml_client.components.get(name="drop_columns")
    label_encoding = ml_client.components.get(name="label_encoding")
    normalize_data = ml_client.components.get(name="normalize_data")
    train_test_split = ml_client.components.get(name="train_test_split")
    train_gradient_boosting = ml_client.components.get(name="train_gradient_boosting")

    @pipeline(default_compute = COMPUTE_INSTANCE_NAME)
    def pipeline_function(pipeline_job_input):
    
    
        # drop_columns_component = ml_client.components.get(name = "drop_columns")
        drop_columns_step = drop_columns(
            input_data = pipeline_job_input,
            columns_to_drop = "\"salary sl_no\"",
            output_data_path = "."
        )
    
        # encoding_component = ml_client.components.get(name = "label_encoding")
        encoding_columns_step = label_encoding(
            input_data = drop_columns_step.outputs["output_data"],
            columns_to_encode = COLUMNS,
            output_data_path = "."
        )
        
        # normalize_component = ml_client.components.get(name = "normalize_data")
        normalize_columns_step = normalize_data(
            input_data = encoding_columns_step.outputs['output_data'],
            columns_to_normalize = "\"ssc_p hsc_p hsc_s degree_p etest_p mba_p\"",
            type_of_norm = "l2"
        )
    
        # train_test_split_component = ml_client.components.get(name = "train_test_split")
    
        train_test_split_step = train_test_split(
            input_data = normalize_columns_step.outputs['output_data'],
        )
    
        # training_component = ml_client.components.get(name = "train_gradient_boosting")
        
        train_step = train_gradient_boosting(
            input_training_data = train_test_split_step.outputs['output_train'],
            input_test_data = train_test_split_step.outputs['output_test'],
            target_column = TARGET_COLUMNS,
            output_model = ".",
            output_metrics = "."
        )
    
    
        return {
            "pipeline_job_transformed_data": normalize_columns_step.outputs['output_data'],
            "pipeline_job_trained_model": train_step.outputs['model_output'],
        }


    # Build and submit the pipeline
    pipeline_job = pipeline_function(get_input(ml_client))
    
    # Submit the pipeline job
    job = ml_client.jobs.create_or_update(pipeline_job)

    print(f"Pipeline job submitted: {job.name}")
    print(f"Pipeline job status: {job.status}")

    job_details = ml_client.jobs.get(job.name)

    print(f"Job URL: https://ml.azure.com/experiments/{job_details.experiment_name}/runs/{job.name}")

if __name__ == "__main__":
    main()

