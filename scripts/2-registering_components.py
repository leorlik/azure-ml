from azure.ai.ml import load_component
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient
import json

CREDENTIALS_PLACE = "../credentials/credentials.json"
WORKSPACE = "campus_recruitment_ws"
RESOURCE_GROUP = "rg_campus_recruitment_leorlik_azure"

def get_credentials():
    
    with open(CREDENTIALS_PLACE) as f:
        data = json.load(f)
        f.close()

    subscription_id = data["subscription-id"]
    client_id = data["client-id"]
    client_secret = data["client-secret"]
    tenant_id = data["tenant-id"]

    return subscription_id, client_id, client_secret, tenant_id

loaded_component_prep = load_component("label_encoding.yml")

subscription_id, client_id, client_secret, tenant_id = get_credentials()

credential = ClientSecretCredential(client_id=client_id, client_secret=client_secret, tenant_id=tenant_id)

ml_client = MLClient(credential,  subscription_id= subscription_id,  resource_group_name=RESOURCE_GROUP, workspace_name=WORKSPACE)

components = ["label_encoding.yml", "drop_columns.yml", "normalize_data.yml", 
              "train_test_split_component.yml", "train.yml"]

for c in components:
    loaded_component_prep = load_component(c)
    ml_client.components.create_or_update(loaded_component_prep)

