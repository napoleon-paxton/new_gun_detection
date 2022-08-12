import json
import urllib.parse
import boto3
from pathlib import *
import urllib3

http = urllib3.PoolManager()

print('Loading function')

s3 = boto3.client('s3')


def lambda_handler(event, context):
    # print("Received event: " + json.dumps(event, indent=2))

    # Get the object from the event and show its content type
    bucket = event['Records'][0]['s3']['bucket']['name']
    
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    
    # source = Path(bucket) / key
    source = key    
    print("Source =  " , source)
    response = http.request("GET", f"http://3.82.198.214:5000/predict?source={source}")

    print('Delete file from s3')
    
    s3.delete_object(Bucket=bucket, Key=key)

    return {
        'statusCode': 200,
        'body': response.data.decode('utf-8')
    }
    
    
    # try:
    #     response = s3.get_object(Bucket=bucket, Key=key)
    #     print("~~~~~~~~~~~~ response ~~~~~~~~~~~~\n" , response)
    #     print("CONTENT TYPE: " + response['ContentType'])
    #     return response['ContentType']
    # except Exception as e:
    #     print(e)
    #     print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(key, bucket))
    #     raise e
