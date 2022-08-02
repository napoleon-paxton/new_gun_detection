import os
import sys
from pathlib import Path

def tree(directory):
    print(f'+ {directory}')
    for path in sorted(directory.rglob('*')):
        depth = len(path.relative_to(directory).parts)
        spacer = '    ' * depth
        print(f'{spacer}+ {path.name}')


def get_crops_list (exp_num):
    crops = Path.cwd() / 'runs' / 'detect' / exp_num /'crops'/'0'
    return crops

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def copy_high_conf_weapon_images_to_s3(folder):

    # exp = Path.cwd() / 'runs' / 'detect' / folder 
    exp = ROOT / 'runs' / 'detect' / folder 
    
    labels = exp / 'labels'

    print(labels)

    curr_time = prev_time = 0
    bst_conf = new_conf = 0 
    bst_file = ''
    bst_file_list = []
    first_file = True
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
                img_name =  bst_file.stem.rsplit('_', 1)[0] + '.jpg'
                print(label_path.parent.parent / 'crops' / '0' / img_name )
                bst_file_list.append( {'time': curr_time, 'file_name': img_name})

                os.system("aws s3 cp runs/detect/{}/crops/0/{} s3://equitable-surveillance-processed-output/{}/crops/0/{}".format(folder, img_name, folder, img_name))

            prev_time = curr_time  
            first_file = False
        
    # print(bst_file_list)
    return bst_file_list

exp = '7-5-22-10-30' 

bst_file_list = copy_high_conf_weapon_images_to_s3(exp)
print(bst_file_list)



# import os

# def check_s3_folder_exists(save_dir):
#     folder = os.system("aws s3 ls s3://equitable-surveillance-processed-output/{}".format(save_dir))
#     return folder

# save_dir = '7-4-22-6-276'

# if (check_s3_folder_exists(save_dir=save_dir)):
#     folder = True
# else:
#     folder = False

# print(folder)