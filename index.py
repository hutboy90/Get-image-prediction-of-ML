
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START aiplatform_predict_image_classification_sample]
import base64
import os

from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict

PROJECT = 'YOUR_PROJECT_ID'
LOCATION = 'us-central1'
ENDPOINT_ID = 'YOUR_ENDPOINT_ID'
API_ENDPOINT = 'us-central1-aiplatform.googleapis.com'

print("----start init----")
aiplatform.init(
    # your Google Cloud Project ID or number
    # environment default used is not set
    project=PROJECT,

    # the Vertex AI region you will use
    # defaults to us-central1
    location=LOCATION,

    # Google Cloud Storage bucket in same region as location
    # used to stage artifacts
    staging_bucket='gs://...',

    # custom google.auth.credentials.Credentials
    # environment default creds used if not set
    # credentials="/Users/trungnc/workspace/python/vertextAI/60e16b7618ec.json",

    # customer managed encryption key resource name
    # will be applied to all Vertex AI resources if set
    # encryption_spec_key_name=my_encryption_key_name,

    # the name of the experiment to use to track
    # logged metrics and parameters
    # experiment='detect-box',

    # description of the experiment above
    # experiment_description='detect box'
)

print("----end init----")

def predict_image_classification_sample(
    filename: str,
    project: str = PROJECT,
    endpoint_id: str = ENDPOINT_ID,
    location: str = LOCATION,
    api_endpoint: str = API_ENDPOINT,
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    with open(filename, "rb") as f:
        file_content = f.read()

    # The format of each instance should conform to the deployed model's prediction input schema.
    encoded_content = base64.b64encode(file_content).decode("utf-8")
    instance = predict.instance.ImageClassificationPredictionInstance(
        content=encoded_content,
    ).to_value()
    instances = [instance]
    # See gs://google-cloud-aiplatform/schema/predict/params/image_classification_1.0.0.yaml for the format of the parameters.
    parameters = predict.params.ImageClassificationPredictionParams(
        confidence_threshold=0.5, max_predictions=5,
    ).to_value()
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    # See gs://google-cloud-aiplatform/schema/predict/prediction/image_classification_1.0.0.yaml for the format of the predictions.
    predictions = response.predictions
    for prediction in predictions:
      result = dict(prediction)
      print(" predict for image %s: {label: %s, confidences: %s}" % (filename, result['displayNames'], result['confidences']))


PATH_IMAGE = 'assets/images'
files = os.listdir(PATH_IMAGE)

for file in files:
  predict_image_classification_sample("%s/%s" % (PATH_IMAGE, file))