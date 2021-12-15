import os
import SimpleITK
import numpy as np

from PIL import Image
from matplotlib import cm
from pathlib import Path

def get_images_lists_from_path(my_path, idxslice=105, remove_first=2):
    """[summary]

    Args:
        my_path (str): Path of the major folder containing the subfolders (HGG in our case)
        idxslice (int): Time of the image to select, 
                        105 should be the max extension of the registration by the machine so is good
        remove_first (int): Initial files to remove, I had to do it for example because 
                            I had .DS_Store files

    Returns:
        [list]: [description]
    """
    dirs = []
    files =  []
    for dirname, _, filenames in os.walk(my_path):
        dirs.append(dirname)
        files.append(filenames)

    dirs = dirs[remove_first:]
    files = files[remove_first:]

    tot_files = []
    c=0
    for dir in dirs:
        for fil in files[c]:
            tot_files.append(str(dir) + '/' + str(fil))
        c+=1

    t2 = []
    t1ce = []
    t1 = []
    flair = []
    seg = []

    for file in tot_files:
        img = SimpleITK.ReadImage(file)
        arr = SimpleITK.GetArrayFromImage(img[:, :, idxslice])
        if 't2' in str(file):
            t2.append(arr)
        elif 't1ce' in str(file):
            t1ce.append(arr)
        elif 't1' in str(file):
            t1.append(arr)
        elif 'flair' in str(file):
            flair.append(arr)
        elif 'seg' in str(file):
            seg.append(arr)
    assert len(t2) == len(t1) == len(t1ce) == len(flair) == len(seg)
    return t2, t1ce, t1, flair, seg


def get_images_lists_from_more_paths(paths_list, idxslice=105, remove_first=2):
    """[summary]

    Args:
        paths_list (list): List of paths
        idxslice (int, optional): see get_images_lists_from_path.
        remove_first (int, optional): see get_images_lists_from_path.

    Returns:
        [list]: [description]
    """
    t2 = t1ce = t1 = flair = seg = []
    for path in paths_list:
        t2_add, t1ce_add, t1_add, flair_add, seg_add = get_images_lists_from_path(path, idxslice, remove_first)
        
        t2 = t2 + t2_add
        t1ce = t1ce + t1ce_add
        t1 = t1 + t1_add
        flair = flair + flair_add
        seg = seg + seg_add
    return t2, t1ce, t1, flair, seg


def export_images_list_jpeg(images_list, output_path=os.getcwd() + "/data_extracted/"):
    name = Path(output_path).name
    os.makedirs(output_path, exist_ok=True)
    for i in range(len(images_list)):
        img = Image.fromarray(np.uint8(cm.gist_earth(images_list[i]/np.max(images_list[i]+1))*255)[:, :, :3])
        img.save(output_path + '/' + name +'_'+ str(i) + '.jpeg')


def export_all_images_jpeg(img_list_of_lists, type_names, output_path=os.getcwd() + "/data_extracted/"):
    assert len(img_list_of_lists) == len(type_names)
    for i in range(len(img_list_of_lists)):
        export_images_list_jpeg(img_list_of_lists[i], output_path + type_names[i])



def save_images_from_path(pathlist, idxslice=105):
    """
    Args:
        my_path (str): Path of the major folder containing the subfolders (HGG in our case)
        idxslice (int): Time of the image to select, 
                        105 should be the max extension of the registration by the machine so is good

    """
    train_path = os.getcwd() + "/data/train/"
    label_path = os.getcwd() + "/data/labels/"

    if not os.path.isdir(train_path):
        os.mkdir(train_path)
    if not os.path.isdir(label_path):
        os.mkdir(label_path)

    for file in pathlist:
        img = SimpleITK.ReadImage(file)
        myarray = SimpleITK.GetArrayFromImage(img[:, :, idxslice])
        im = Image.fromarray(np.uint8(cm.gist_earth(myarray/np.max(myarray+1))*255)[:, :, :3])
        filename = file.split("\\")[-1].split(".")[0] + ".jpg"

        if 'seg' in str(file):
            im.save(label_path + filename)
        else:
            im.save(train_path + filename)

    return
