点击Single User
<img width="2145" alt="image" src="https://github.com/user-attachments/assets/ae07c4ce-6764-4813-9709-26dd82e00804">
点击创建用户profile
<img width="2069" alt="image" src="https://github.com/user-attachments/assets/c1b54001-eed5-4078-b8d1-e68c8f511812">
点击下一步
<img width="1321" alt="image" src="https://github.com/user-attachments/assets/27dfb7f9-5081-420f-9d86-4f75a0f2c22a">

<img width="1173" alt="image" src="https://github.com/user-attachments/assets/89ef9b0c-6bfa-4804-b9a7-002ef9f59b4e">

部署成功
<img width="2322" alt="image" src="https://github.com/user-attachments/assets/98f8e741-7075-4abf-ad5e-3730fc594567">

<img width="2302" alt="image" src="https://github.com/user-attachments/assets/9187e54d-a4bd-4c08-aa83-21bfe57e2975">
打开Studio，选择Jumpstart
<img width="2300" alt="image" src="https://github.com/user-attachments/assets/8e849e7c-a4a8-447a-bfeb-c165657d3f96">
huggingface目录下搜索Embedding模型
<img width="2240" alt="image" src="https://github.com/user-attachments/assets/cb429e25-8df0-4974-911e-419eaf79a57f">
找到模型部署即可
<img width="1494" alt="image" src="https://github.com/user-attachments/assets/19ccb344-2d3f-4b40-9a97-6724795940c6">

测试可以启动notebook进行测试测试代码如下：
```python
import sagemaker
from sagemaker import image_uris
import boto3
import os
import time
import json
import numpy as np

role = sagemaker.get_execution_role()  # execution role for the endpoint
sess = sagemaker.session.Session()  # sagemaker session for interacting with different AWS APIs
bucket = sess.default_bucket()  # bucket to house artifacts

region = sess._region_name
account_id = sess.account_id()
endpoint_name = "xxxx"
s3_client = boto3.client("s3")
sm_client = boto3.client("sagemaker")
smr_client = boto3.client("sagemaker-runtime")
def get_vector_by_sm_endpoint(questions, sm_client, endpoint_name):
    parameters = {
    }

    response_model = sm_client.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=json.dumps(
            {
                "inputs": questions,
                # "is_query": True,
                "instruction" :  "Represent this sentence for searching relevant passages:"
            }
        ),
        ContentType="application/json",
    )
    # 中文instruction => 为这个句子生成表示以用于检索相关文章：
    json_str = response_model['Body'].read().decode('utf8')
    json_obj = json.loads(json_str)
    embeddings = json_obj['embeddings']
    return embedding

prompts1 = ["How to compete with OCI","如何与OCI竞争","如何與OCI競爭"]

emb = get_vector_by_sm_endpoint(prompts1, smr_client, endpoint_name)


def cos_sim(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_v1 = np.linalg.norm(vector1)
    norm_v2 = np.linalg.norm(vector2)
    cos_sim = dot_product / (norm_v1 * norm_v2)
    return cos_sim

cos_sim(emb[1],emb[2])
```
