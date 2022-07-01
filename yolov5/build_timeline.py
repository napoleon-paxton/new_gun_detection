import pathlib

def tree(directory):
    print(f'+ {directory}')
    for path in sorted(directory.rglob('*')):
        depth = len(path.relative_to(directory).parts)
        spacer = '    ' * depth
        print(f'{spacer}+ {path.name}')


def get_crops_list (exp_num):
    crops = pathlib.Path.cwd() / 'runs' / 'detect' / exp_num /'crops'/'0'
    return crops


def get_weapon_images(exp_num):

    exp = pathlib.Path.cwd() / 'runs' / 'detect' / exp_num 
    
    labels = exp / 'labels'
    # crops_list = get_crops_list (exp_num)

    curr_time = prev_time = 0
    bst_conf = new_conf = 0 
    bst_file = ''
    bst_file_list = []
    first_file = True
    for path in sorted(labels.rglob('*')):
        print()
        curr_time = path.stem.split('_')[1]
        if curr_time == prev_time :
            with open(path) as f:
                li = f.readlines()
                new_conf = li[0].split(' ')[5].strip('\n')

            if float(bst_conf) <= float(new_conf) :
                bst_conf = new_conf
                bst_file = path
                prev_time = curr_time    
        else :
            if not first_file:
                bst_file_list.append( {'time': curr_time, 'file_name': bst_file.name})
                bst_conf = 0.0
                img_name =  bst_file.stem.rsplit('_', 1)[0] + '.jpg'
                print(path.parent.parent / 'crops' / '0' / img_name )

            prev_time = curr_time  
            first_file = False
        
    print(bst_file_list)
    return bst_file_list

exp = 'exp43' 

bst_file_list = get_weapon_images(exp)
print(bst_file_list)