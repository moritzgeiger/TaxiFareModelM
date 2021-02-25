import os

from google.cloud import storage
from termcolor import colored

BUCKET_NAME = "wagon-ml-geiger-01"
STORAGE_LOCATION = "models/model.joblib"
myname = "moritzgeiger"
EXPERIMENT_NAME = f"[PT]TaxifareModel_{myname}"


BUCKET_NAME = "wagon-ml-geiger-01"

def storage_upload(model_directory, bucket=BUCKET_NAME, rm=False):
    client = storage.Client().bucket(bucket)

    storage_location = '{}/{}/{}/{}'.format(
        'models',
        'taxi_fare_model',
        model_directory,
        'model.joblib')
    blob = client.blob(storage_location)
    blob.upload_from_filename('model.joblib')
    print(colored("=> model.joblib uploaded to bucket {} inside {}".format(BUCKET_NAME, storage_location),
                  "green"))
    if rm:
        os.remove('model.joblib')