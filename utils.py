from PIL import Image
import os
import shutil
import pandas as pd
import threading
import numpy as np
from keras.preprocessing.image import img_to_array, load_img


def load_image(label, path, q=None):
    """
    label in here is a real number. 
    load image and put in a Queue.
    """
    img = load_img(path, grayscale=True)
    arr = img_to_array(img)
    arr = np.reshape(arr, (1,)+arr.shape)
    data = {'y': label, 'data':arr}
    if q:
        q.put(data)
        q.task_done()
    return arr


def ImgProcess(file, outdir, size):
    img = Image.open(file['Files'])
    oldsize = img.size
    new_img = Image.new('RGB', (oldsize[1], oldsize[1]), (255,255,255))
    new_img.paste(img)
    new_img.thumbnail(size, Image.ANTIALIAS)
    new_img.save(os.path.join(outdir, file['Files'].split('/')[-1]))


def CropAndResize(indir, output_size, label):
    """
    padding image to become squared image then size to the desire size.
    """
    # if label:
    label = pd.read_csv(label)
    outlable = pd.DataFrame(label)
    outdir = indir+'_'+str.replace(str(output_size), ' ', '')
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.mkdir(outdir)
    for index, row in label.iterrows():
        label['files'][index] = str(os.path.join(outdir, row['files'].split('/')[-1]))
        t = threading.Thread(target=ImgProcess, args=(row,outdir,output_size,))
        t.start()
    size = str(output_size)
    size = str.replace(size, ' ', '')
    outlable.to_csv(os.path.join(outdir, indir.split('/')[-1]+
        size)+'.csv')
    print "DONE!"
    # files_name = []
    
    # for f in os.listdir(in_dir):
    #     if f.endswith('.jpg'):


def LoadandPreprocessImg(path):
    path = getRealPath(path)
    img = load_img(path, grayscale=True)
    #oldsize = img.size
    #new_img = Image.new('RGB', (oldsize[1], oldsize[1]), (255,255,255))
    #new_img.convert("L")
    #new_img.paste(img)
    img.thumbnail((100, 100), Image.ANTIALIAS)
    arr = img_to_array(img)
    arr /= 255.0
    print arr.shape
    arr = np.resize(arr, (1, 40, 100, 1))
    return arr

def getRealPath(path):
    return os.path.join(os.getcwd(), 'CarDataset/cars_train/'+os.path.normpath(path))
