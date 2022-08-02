import argparse
import os
import sys
from pathlib import Path
import boto3
import urllib.request
from numpy import right_shift
from requests import head
import torch
import torch.backends.cudnn as cudnn
# import glob

import pandas as pd


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

print("~~~~~~~~ App Loaded ~~~~~~~~~~~~  : ")


from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

LOGGER.setLevel('INFO')

import json
from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import datetime
import compare_images as cimgs


os.environ['AWS_PROFILE'] = "MyProfile1"
os.environ['AWS_DEFAULT_REGION'] = "us-east-1"

app = Flask(__name__)

def list_folders(s3_client, bucket_name):
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix='', Delimiter='/')
    for content in response.get('CommonPrefixes', []):
        yield content.get('Prefix')

def merge_lists(list1, list2):

    print('merge_lists :')


    l_ended = r_ended = False

    merged_list = []
    pop_l = pop_r = True
    while True:
        if len(list1)>0:
            if pop_l:
                l = list1.pop()        
        else :
            if not l_ended:
                merged_list.append(l)
                l_ended = True
            l = None
            pop_r = True


        if len(list2)>0:
            if pop_r:
                r = list2.pop()
        else :
            if not r_ended:
                merged_list.append(r)
                r_ended = True
            r = None
            pop_l = True
        if l and r :
            if (l[1] <= r[1]) :
                merged_list.append(l)
                pop_r = False
                pop_l = True
            else :
                merged_list.append(r)
                pop_l = False
                pop_r = True

        elif l:
            merged_list.append(l)
        elif r:
            merged_list.append(r)
        else:
            break


    # print (merged_list)
    return merged_list


def get_offensive_lang_details(folder):
    # folder = '7-16-22-6-27'

    # retrieve language analytics

    S3_BUCKET_NAME = 'equitable-surveillance-processed-output-speech'
    
    s3 = boto3.resource('s3')
    ## Bucket to use
    bucket = s3.Bucket(S3_BUCKET_NAME)

    offensive_lang = []
    offensive_lang_summary = []
    
    # print('Prefix = ', folder)
    lang_objs = list(bucket.objects.filter(Prefix= folder ))
    # print(' .... .. ... .. len(lang_objs)  = ', len(lang_objs))
    for i in range(0, len(lang_objs)):
        # print('~~~~~~~~~i = ', i, lang_objs[i].key )

        lang_meta_data  = lang_objs[i].key.split('/')[1].split('_')
        # print(lang_meta_data)

        file_url = 'https://equitable-surveillance-processed-output-speech.s3.amazonaws.com/' + lang_objs[i].key
        # print(file_url)

        if lang_meta_data[0] == 'summary' :
            print( '[INFO: Summary file found ] ')
            for line in urllib.request.urlopen(file_url):
                line = line.decode("utf-8").replace('\n', ' ').split(".")
                offensive_lang_summary.append(line[0])
        else:
            for line in urllib.request.urlopen(file_url):
                # print('\n' , line)
                offensive_lang.append((lang_meta_data[0], lang_meta_data[1], line))


    print(offensive_lang_summary)
    # print(offensive_lang)

    return offensive_lang_summary, offensive_lang


@app.route('/test1')
def test_html():
    return render_template('test_1.html')

@app.route('/')
def index():
    s3_client = boto3.client('s3')
    S3_BUCKET_NAME = 'equitable-surveillance-processed-output'

    print("[INFO:] Connecting to cloud")

    folder_list_s3 = list_folders(s3_client, S3_BUCKET_NAME)

    folder_list = []
    for folder in folder_list_s3:
        folder_list.append(folder.split('/')[0])

    print('folder_list', folder_list)

    data_list = []
    for folder in folder_list:

        print('folder = ', folder)
        mnth , dt, yr, hr, mn = folder.split('-')
        if int(dt) < 10 :
            dt = "0{dt}".format(dt=dt)
        if int(mnth) < 10 :
            mnth = "0{mnth}".format(mnth=mnth)
        
        if len(mn)==3:
            mn = mn[0:1]

        s = "{dt}/{mnth}/{yr} {hr}:{mn}".format( dt=dt, mnth=mnth, yr=yr, hr=hr, mn=mn)
        s_dt = datetime.datetime.strptime(s, '%d/%m/%y %H:%M').strftime("%A,%d %B, %Y %H:%M") 
        data_list.append((s_dt, folder))

    return render_template('main.html', data_list = data_list)


@app.route('/handle_click', methods=['POST'])
def handle_click():
    print("handle_data invoked... .... .... . ")
    folder = request.form['folder']
    print ('folder :'   , folder)

    IMG_COUNTER = 0

    # print('folder = ', folder)
    mnth , dt, yr, hr, mn = folder.split('-')
    if int(dt) < 10 :
        dt = "0{dt}".format(dt=dt)
    if int(mnth) < 10 :
        mnth = "0{mnth}".format(mnth=mnth)
    
    if len(mn)==3:
        mn = mn[0:1]

    s = "{dt}/{mnth}/{yr} {hr}:{mn}".format( dt=dt, mnth=mnth, yr=yr, hr=hr, mn=mn)
    s_dt = datetime.datetime.strptime(s, '%d/%m/%y %H:%M').strftime("%A,%d %B, %Y %H:%M") 

    S3_BUCKET_NAME = 'equitable-surveillance-processed-output'
    s3 = boto3.resource('s3')
    ## Bucket to use
    bucket = s3.Bucket(S3_BUCKET_NAME)


    merged_output = []
    wd_imgs = []
    gun_objs = list(bucket.objects.filter(Prefix= '{}/crops/0/'.format(folder)   ))
    # print(' .... .. ... .. len(gun_objs)  = ', len(gun_objs))
    for i in range(0, len(gun_objs)):
        wd_time = int(gun_objs[i].key.split('_')[1])
        wd_min = int(int(wd_time)/60)
        wd_sec = wd_time%60
        wd_imgs.append((IMG_COUNTER, 'Weapon Detected', wd_min, wd_sec, 'https://equitable-surveillance-processed-output.s3.amazonaws.com/' + gun_objs[i].key) )       
        merged_output.append((IMG_COUNTER, 'Weapon Detected', wd_min, wd_sec, 'https://equitable-surveillance-processed-output.s3.amazonaws.com/' + gun_objs[i].key) )       
        IMG_COUNTER = IMG_COUNTER+1

    wd_detected = False
    if wd_imgs:
        wd_detected = True

    ppl_imgs = []
    # bucket = s3.Bucket('bucket-name')
    people_objs = list(bucket.objects.filter(Prefix= '{}/crops/person'.format(folder)   ))
    for i in range(0, len(people_objs)):
        # print(people_objs[i].key)
        ppl_time = int(people_objs[i].key.split('_')[1])
        ppl_min = int(int(ppl_time)/60)
        ppl_sec = ppl_time%60
        ppl_imgs.append((IMG_COUNTER, 'Person Detected', ppl_min, ppl_sec, 'https://equitable-surveillance-processed-output.s3.amazonaws.com/' + people_objs[i].key) )       
        merged_output.append((IMG_COUNTER, 'Person Detected', ppl_min, ppl_sec, 'https://equitable-surveillance-processed-output.s3.amazonaws.com/' + people_objs[i].key) )       
        IMG_COUNTER = IMG_COUNTER+1

    ppl_detected = False
    if ppl_imgs:
        ppl_detected = True


    lic_plate_bucket = s3.Bucket('equitable-surveillance-processed-output-1')
    lic_plate_objs = list(lic_plate_bucket.objects.filter(Prefix= folder) )
    lic_plates = []
    lp_summary = []
    for i in range(0, len(lic_plate_objs)):
        print('lic plate found ', lic_plate_objs[i].key )
        if '.jpg' in lic_plate_objs[i].key :
            lp_time = round (int(lic_plate_objs[i].key.split('_')[1])/1000, )
            lp_min = int(lp_time/60)
            lp_sec = lp_time%60
            lic_plates.append( (IMG_COUNTER, 'License Place Detected', lp_min, lp_sec, 'https://equitable-surveillance-processed-output-1.s3.amazonaws.com/' + lic_plate_objs[i].key)) 
            merged_output.append( (IMG_COUNTER, 'License Place Detected', lp_min, lp_sec, 'https://equitable-surveillance-processed-output-1.s3.amazonaws.com/' + lic_plate_objs[i].key)) 
            IMG_COUNTER = IMG_COUNTER+1
        elif 'summary_output.txt' in lic_plate_objs[i].key :
            print( '[INFO: Summary file for license plates found ] ')
            file_url = 'https://equitable-surveillance-processed-output-1.s3.amazonaws.com/' + lic_plate_objs[i].key
            for line in urllib.request.urlopen(file_url):
                line = line.decode("utf-8").replace('\n', ' ').split(".")
                lp_summary.append(line[0])

    if lp_summary:
        occ, lp_num = lp_summary[1].split(' ')
        # lp_summ = '{} occurances of license plate {} were found'.format(occ, lp_num )
        lp_summ = 'License plate {} was detected'.format(lp_num )
        print(lp_summ)
    else:
        lp_summ = ''

    merged_list_pd = pd.DataFrame(merged_output)
    merged_list_pd = merged_list_pd.sort_values([2, 3, 0])

    # retrieve language analytics
    offensive_lang_summary, offensive_lang = get_offensive_lang_details(folder)

    merged_list = list(merged_list_pd.itertuples(index=False, name=None))

    print('/n merged_list last record ' , merged_list[-1])

    return render_template('details.html',
                            name='Detected Gun', 
                            incident=s_dt, 
                            folder=folder, 
                            wd_imgs = wd_imgs, 
                            ppl_imgs = ppl_imgs, 
                            lic_plates=lic_plates, 
                            ppl_detected=ppl_detected, 
                            wd_detected=wd_detected,
                            offensive_lang_summary= offensive_lang_summary,
                            offensive_lang = offensive_lang,
                            merged_list=merged_list,
                            lp_summary=lp_summ)






@app.route('/predict')
def predict():


    print("------------  NEW PREDICT   ------------")
    
    conf = request.args.get('conf' , 0.6)

    weights= ROOT / request.args.get('weights' , 'best_model_including_kaggle_images.pt')
    # source= request.args.get('source' ,'7-3-22-12-25.mp4')
    source= request.args.get('source' ,'7-16-22-6-27.m4v')
    source = 'https://equitable-surveillance-s3.s3.amazonaws.com/' +  source    

    print('after appending URL to source ', source )

    # source = 'https://equitable-surveillance-s3.s3.us-east-1.amazonaws.com/' +  source
    # source = FILE.parents[1] / source


    data  = ROOT / 'data/coco128.yaml'
    imgsz = int(request.args.get('imgsz' , 640))
    # imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    imgsz = [imgsz, imgsz]
    conf_thres= request.args.get('conf' , 0.6)
    iou_thres=0.45
    max_det=1000
    device=''
    view_img=False
    save_txt=True 
    save_conf=True
    save_crop=True 
    nosave=False 
    classes=None 
    agnostic_nms=False
    augment=False
    visualize=False
    update=False
    project= ROOT / 'runs/detect'
    name= name = Path(source).stem #overwrite name to create a output folder based on the source name  
    exist_ok=False
    line_thickness=3
    hide_labels=False
    hide_conf=False
    half=False
    dnn=False

    print('Root -' , ROOT)
    print('weights = ', weights)
    print('imgsz = ', imgsz)
    if weights.is_file :
        print('model exists = ')
    print('source = ', source)


    save_dir = run(weights,  source,  data, imgsz, conf_thres,  iou_thres, max_det, device, view_img, save_txt, save_conf, save_crop, nosave, classes, agnostic_nms, augment, visualize, update, project, name, exist_ok, line_thickness, hide_labels, hide_conf, half, dnn)

    print('Save_dir ', save_dir) 
    print(type(save_dir) )
    print(save_dir.stem)

    # os.system("aws s3 cp yolov5/runs/detect/{} s3://equitable-surveillance-processed-output/{} --recursive ".format(save_dir.stem, save_dir.stem))

    print('Calling copy_high_conf_weapon_images_to_s3({})  \n'.format(save_dir.stem) )

    target_type = 0

    bst_file_list = copy_high_conf_weapon_images_to_s3(save_dir.stem, target_type )
    # print(bst_file_list)

    print('[Processing complete]... \n')

    print('[Cleaning up ]... \n')



    print('[INFO] run : rm -rf yolov5/runs/detect/{}'.format(save_dir.stem) )
    rm_ret_code = os.system("rm -rf yolov5/runs/detect/{}".format(save_dir.stem))
    print('Return code for rm -rf yolov5/runs/detect/{} is {}'.format(save_dir.stem, rm_ret_code))

    print("[INFO] run : aws s3 rm s3://equitable-surveillance-s3/{}.mp4".format(save_dir.stem))
    os.system("aws s3 rm s3://equitable-surveillance-s3/{}.mp4".format(save_dir.stem))


    print('[Begin extracting people.... ]')

    weights= ROOT / 'yolov5s.pt'
    save_dir = run(weights,  source,  data, imgsz, conf_thres,  iou_thres, max_det, device, view_img, save_txt, save_conf, save_crop, nosave, classes, agnostic_nms, augment, visualize, update, project, name, exist_ok, line_thickness, hide_labels, hide_conf, half, dnn)

    print('Save_dir ', save_dir) 
    print(type(save_dir) )
    print(save_dir.stem)

    print('Calling copy_high_conf_weapon_images_to_s3({})  \n'.format(save_dir.stem) )

    target_type = 'person'

    bst_file_list = copy_high_conf_weapon_images_to_s3(save_dir.stem, target_type)

    print('[Processing complete]... \n')

    print('[Cleaning up ]... \n')

    print("aws s3 cp /home/ubuntu/capstone/gun_detection/yolov5/runs/detect/{}/{}.mp4 s3://equitable-surveillance-processed-output/{}/crops/{}/{}.mp4 ".format( save_dir.stem, save_dir.stem, save_dir.stem, '0', save_dir.stem ))
    os.system("aws s3 cp /home/ubuntu/capstone/gun_detection/yolov5/runs/detect/{}/{}.mp4 s3://equitable-surveillance-processed-output/{}/crops/{}/{}.mp4 ".format( save_dir.stem, save_dir.stem, save_dir.stem, '0', save_dir.stem ))

    print('[INFO] run : rm -rf yolov5/runs/detect/{}'.format(save_dir.stem) )
    rm_ret_code = os.system("rm -rf yolov5/runs/detect/{}".format(save_dir.stem))
    print('Return code for rm -rf yolov5/runs/detect/{} is {}'.format(save_dir.stem, rm_ret_code))

    print("[INFO] run : aws s3 rm s3://equitable-surveillance-s3/{}.mp4".format(save_dir.stem))
    os.system("aws s3 rm s3://equitable-surveillance-s3/{}.m*".format(save_dir.stem))


    return jsonify({'predict_status': 'complete',  
                    'saved_to': save_dir.stem    
                    })



def copy_high_conf_weapon_images_to_s3(folder, target_type):

    # exp = Path.cwd() / 'runs' / 'detect' / folder 
    exp = ROOT / 'runs' / 'detect' / folder 
    
    labels = exp / 'labels'

    (exp / 'output').mkdir(parents=True, exist_ok=True)  # make dir
    out_dir = exp / 'output'
    (out_dir /  str(target_type)).mkdir(parents=True, exist_ok=True)  # make dir
    out_dir = out_dir / str(target_type)

    curr_time = prev_time = 0
    bst_conf = new_conf = 0 
    bst_file = ''
    bst_file_list = []
    first_file = True

    # Getting the list of directories
    labels_dir = os.listdir(labels)
    
    # Checking if the list is empty or not
    if len(labels_dir) == 0:
        print("Empty lables directory... No images to drop")
        return
    else:
        print("Found lables to process... ")


    for label_path in sorted(labels.rglob('*')):
        # print()
        curr_time = label_path.stem.split('_')[1]
        if curr_time == prev_time :
            with open(label_path) as f:
                li = f.readlines()
                new_conf = li[0].split(' ')[5].strip('\n')

            if float(bst_conf) <= float(new_conf) :
                bst_conf = new_conf
                bst_file = label_path
                prev_time = curr_time    
        else :
            if not first_file:
                
                bst_conf = 0.0
                print('[DEBUG] :', bst_file, type(bst_file))
                img_name =  bst_file.stem.rsplit('_', 1)[0] + '.jpg'
                # print(label_path.parent.parent / 'crops' / target_type / img_name )
                bst_file_list.append( {'time': curr_time, 'file_name': img_name})

                print('[INFO : cp {}/crops/{}/{} {}/{}]'.format(exp, target_type, img_name, out_dir, img_name), '\n'  )
                os.system('cp {}/crops/{}/{} {}/{}'.format(exp, target_type, img_name, out_dir, img_name) )

            prev_time = curr_time  
            first_file = False
        
    imgs_saved_to_s3, imgs_path = cimgs.delete_similar_images(out_dir)


    BASE_DIR = Path(__file__).resolve().parent
    final_images_path = BASE_DIR / 'runs/detect/{}/output/{}/'.format(folder, target_type)
    print('final_images_path == ', final_images_path)

    
    s3_folder = folder
    # s3_folder = str(folder.split('-')[-1])

    # if len(s3_folder) == 3 :
    #     s3_folder = folder[0: -1]
    # elif len(s3_folder) == 4 :
    #     s3_folder = folder[0: -2]

    print('\ns3_folder = ', s3_folder)

    print('[INFO current working dir just before glob] - ', Path.cwd())

# 
    # for img_path in sorted(glob.glob(final_images_path)):
    for img_path in sorted(os.listdir(final_images_path)):
        obj_name = Path(img_path).name
        img_path = final_images_path / img_path
        print('Image path = ', img_path)
        print("\naws s3 cp {} s3://equitable-surveillance-processed-output/{}/crops/{}/{}".format(img_path, s3_folder, target_type, obj_name ))
        os.system("aws s3 cp {} s3://equitable-surveillance-processed-output/{}/crops/{}/{}".format(img_path, s3_folder, target_type, obj_name ))


    print('\n ', folder)

    return bst_file_list



@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.40,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)
    print('source after run entry : ', source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_video = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)

    print('Is file : ', is_file)

    print('Is URL : ', is_url)

    if is_url and is_file:
        print('source = ', source)
        source = check_file(source)  # download

    # Directories
    name = Path(source).stem #overwrite name to create a output folder based on the source name
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # print("Line 140 -- Source :", source)

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size

    vid_path, vid_writer = [None] * bs, [None] * bs

    # Compute total frames
    if is_video:
        # inp_vid = cv2.VideoCapture('testdata/accident_scene_Trim_Trim.mp4')
        inp_vid = cv2.VideoCapture(source)
        total_frames = inp_vid.get(cv2.CAP_PROP_FRAME_COUNT)
        ip_fps = int(inp_vid.get(cv2.CAP_PROP_FPS))
    else :
        ip_fps = total_frames = 1

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        #frame number
        frame_number = s.split(' ')[2].split('/')[0][1:]

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            elapsed_time = int(ip_fps*(seen/total_frames))
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            p_stem = f'{seen+10000}_{elapsed_time}_' +  p.stem
            txt_path = str(save_dir / 'labels' /  p_stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # elapsed_time = int(ip_fps*(seen/total_frames))

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{seen+10000}_{elapsed_time}_{p.stem}.jpg', BGR=True)



            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)


        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    return save_dir




# def main(opt):
#     check_requirements(exclude=('tensorboard', 'thop'))
#     run(**vars(opt))


# if __name__ == "__main__":
#     opt = parse_opt()
#     main(opt)


# app.run(port=8080, debug=True)
app.run(host="0.0.0.0", port=5000, debug=True)