from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ComputeInstance
import json

CREDENTIALS_PLACE = "../credentials/credentials.json"
WORKSPACE_NAME = "campus_recruitment_ws"
RESOURCE_GROUP_NAME = "rg_campus_recruitment_leorlik_azure"
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

def main():

    subscription_id, client_id, client_secret, tenant_id = get_credentials()

    credential = ClientSecretCredential(client_id=client_id, client_secret=client_secret, tenant_id=tenant_id)

    ml_client = MLClient(credential,  subscription_id= subscription_id,  resource_group_name=RESOURCE_GROUP_NAME, 
                        workspace_name=WORKSPACE_NAME)
    
    ci_basic = ComputeInstance(
        name = COMPUTE_INSTANCE_NAME,
        size="STANDARD_DS3_v2"
    )

    ml_client.begin_create_or_update(ci_basic).result()
    
if __name__ == "__main__":
    main()

    
    