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
import json
from flask import Flask, request, jsonify, render_template, send_from_directory
import os

os.environ['AWS_PROFILE'] = "MyProfile1"
os.environ['AWS_DEFAULT_REGION'] = "us-east-1"

app = Flask(__name__)

@app.route('/predict')
def predict():
        
    bucket = 'equitable-surveillance-s3'
    key = 'accident_scene_Trim_Trim.mp4'

    # s3 = boto3.resource('s3')
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket, Key=key)

    lpath = '/home/ubuntu/capstone/gun_detection/yolov5/runs/detect/accident_scene_Trim_Trim8'

    s3path = 'accident_scene_Trim_Trim8'

    # s3bucket = 

    upload_folder_to_s3(bucket, lpath, s3path)
    # print('response body :' , response['Body'].read())

    print("CONTENT TYPE: " + response['ContentType'])
    # print(response)
    return jsonify({'predict': 'complete',      })


def upload_folder_to_s3(s3bucket, inputDir, s3Path):
    print("Uploading results to s3 initiated...")
    print("Local Source:",inputDir)
    os.system("ls -ltR " + str(inputDir))

    print("Dest  S3path:",s3Path)

    try:
        for path, subdirs, files in os.walk(inputDir):
            for file in files:
                dest_path = path.replace(inputDir,"")
                __s3file = os.path.normpath(s3Path + '/' + dest_path + '/' + file)
                __local_file = os.path.join(path, file)
                print("upload : ", __local_file, " to Target: ", __s3file, end="")
                s3bucket.upload_file(__local_file, __s3file)
                print(" ...Success")
    except Exception as e:
        print(" ... Failed!! Quitting Upload!!")
        print(e)
        raise e

app.run(host="0.0.0.0", port=5000, debug=True)


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