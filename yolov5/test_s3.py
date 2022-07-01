# import boto3


# ACCESS_ID = 'AKIAWE266375TLWQ4W3L'
# ACCESS_KEY =  '1IvNqBVGdnEs/6L0az0tcUGmkSBUr+AHoe/REPxo'


# s3 = boto3.resource('s3',
# aws_access_key_id=ACCESS_ID,
#          aws_secret_access_key= ACCESS_KEY)


# s3_object = s3.Bucket('equitable-surveillance-s3').Object('accident_scene_Trim_Trim.mp4').get()

# print(s3_object)


from __future__ import print_function
import boto3
import os

os.environ['AWS_PROFILE'] = "MyProfile1"
os.environ['AWS_DEFAULT_REGION'] = "us-east-1"

# ec2 = boto3.client('ec2')

bucket = 'equitable-surveillance-s3'
key = 'accident_scene_Trim_Trim.mp4'

# s3 = boto3.resource('s3')
s3 = boto3.client('s3')
response = s3.get_object(Bucket=bucket, Key=key)

 # try:
    #     response = s3.get_object(Bucket=bucket, Key=key)
    #     print("~~~~~~~~~~~~ response ~~~~~~~~~~~~\n" , response)
    #     print("CONTENT TYPE: " + response['ContentType'])
    #     return response['ContentType']
    # except Exception as e:
    #     print(e)
    #     print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(key, bucket))
    #     raise e

# Retrieves all regions/endpoints that work with EC2
# response = ec2.describe_regions()
# print('Regions:', response['Regions'])