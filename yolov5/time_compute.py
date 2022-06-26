import cv2
import datetime
  
# create video capture object
data = cv2.VideoCapture('testdata/accident_scene_Trim_Trim.mp4')
  
# count the number of frames
frames = data.get(cv2.CAP_PROP_FRAME_COUNT)
fps = int(data.get(cv2.CAP_PROP_FPS))
  
# calculate dusration of the video
seconds = int(frames / fps)
video_time = str(datetime.timedelta(seconds=seconds))
print("duration in seconds:", seconds)
print("video time:", video_time)
print('frames : ', frames)
print('fps ', fps)