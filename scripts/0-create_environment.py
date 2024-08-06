from azure.ai.ml.entities import Workspace
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data 
from azure.ai.ml.constants import AssetTypes
from azure.identity import ClientSecretCredential
from azure.mgmt.resource import ResourceManagementClient
import os
import json

CREDENTIALS_PLACE = "../credentials/credentials.json"


def get_credentials():
    
    with open(CREDENTIALS_PLACE) as f:
        data = json.load(f)
        f.close()

    subscription_id = data["subscription-id"]
    client_id = data["client-id"]
    client_secret = data["client-secret"]
    tenant_id = data["tenant-id"]

    return subscription_id, client_id, client_secret, tenant_id

def create_azure_rg(subscription_id, credential, group_name: str = "rg_campus_recruitment_leorlik_azure"):

    resource_client = ResourceManagementClient(credential, subscription_id)

    for resource_group in resource_client.resource_groups.list():
        print(resource_group.name)

    rg_result = resource_client.resource_groups.create_or_update(
            group_name, {"location": "eastus2"}
        )

    return group_name

def create_azure_ws(rg: str, ml_client, workspace_name: str = "campus_recruitment_ws"):

    ws = Workspace(
        name = workspace_name, 
        location = "eastus2",
        display_name = "Basic Workspace for Campus Recruitment",
        description = "Azure ML Workspace for the user showcase of Azure using the Campus Recruitment Dataset",
        resource_group = rg

    )

    ml_client.workspaces.begin_create(ws)

    return workspace_name

def create_uri_dataset(ml_client, path:str = "../data/Placement_Data_Full_Class.csv"):

    data_asset = Data(
        path = path,
        type = AssetTypes.URI_FILE,
        description = "Placement csv dataset for the azure project",
        name = "placement_csv",
        version = "1.0.0"
    )
    
    ml_client.data.create_or_update(data_asset)

def create_mltable_dataset(ml_client, path:str = "../data/data_MLTable"):

    data_asset = Data(
        path = path,
        type = AssetTypes.MLTABLE,
        description = "Placement dataset for the azure project, ml table",
        name = "placement_mltable",
        version = "1.0.0"
    )

    ml_client.data.create_or_update(data_asset)




def main():

    subscription_id, client_id, client_secret, tenant_id = get_credentials()

    credential = ClientSecretCredential(client_id=client_id, client_secret=client_secret, tenant_id=tenant_id)


    rg = create_azure_rg(subscription_id, credential)
    ml_client = MLClient(credential, resource_group = rg, subscription_id= subscription_id, resource_group_name=rg)
    ws = create_azure_ws(rg, credential, subscription_id )

    ml_client = MLClient(credential,  subscription_id= subscription_id,  resource_group_name=rg, workspace_name=ws.name)

    create_uri_dataset(ml_client)
    create_mltable_dataset(ml_client)

if __name__ == "__main__":
    main()

