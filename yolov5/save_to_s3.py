import os
import boto3

inputDir = 'yolov5/runs/detect/accident_scene_Trim_Trim7'
# os.system("ls -ltR " + inputDir)
save_dir = 'yolov5/runs/detect/accident_scene_Trim_Trim9'

os.system("echo Hello from the other side!")
# os.system("aws s3 cp yolov5/runs/detect/accident_scene_Trim_Trim7 s3://equitable-surveillance-s3/outputs/accident_scene_Trim_Trim7 --recursive ")
os.system("aws s3 cp yolov5/runs/detect/{} s3://equitable-surveillance-s3/outputs/{} --recursive ".format(save_dir, save_dir))
