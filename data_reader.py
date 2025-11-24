# https://github.com/Western-OC2-Lab/Intrusion-Detection-System-Using-CNN-and-Transfer-Learning/blob/main/1-Data_pre-processing_CAN.ipynb

import numpy as np
import pandas as pd
import os
import cv2
import math
import random
import matplotlib.pyplot as plt
import shutil
from sklearn.preprocessing import QuantileTransformer
from PIL import Image
import warnings
from enum import Enum
warnings.filterwarnings("ignore")

# =====================================================
# =========       Constants and options       =========
# =====================================================

class Mode(Enum):
    BINARY = "binary"
    DECIMAL = "decimal"
    HEXADECIMAL = "hexadecimal"

DEFAULT_MODE = Mode.HEXADECIMAL 

# for multiclass classification
USE_MULTICLASS = True

BINCLASS_DICT = {
    "BENIGN" : 0,
    "ATTACK" : 1,
}
MULTICLASS_DICT = {
    "BENIGN" : 0,
    "DOS" : 1,
    "GAS" : 2,
    "RPM" : 3,
    "SPEED" : 4,
    "STEERING_WHEEL" : 5
}

# =====================================================
# =========            Functions              =========
# =====================================================

def load_data_to_DF(data_dir, mode=DEFAULT_MODE):
    folder = os.path.join(data_dir, mode.value)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    data = load_all_csvs(folder)
    return data

def load_all_csvs(folder_path):
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")

    df_list = []
    for f in all_files:
        df = pd.read_csv(f, low_memory=False)
        df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)
    df.columns = [c.strip() for c in df.columns] # normalize column names
    # ensure numeric conversion for data columns
    for col in df.columns:
        if col.startswith("DATA_") or col in ["ID", "DLC"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.fillna(0) # vyplnit pripadne prazdne hodnoty nulou
    return df

# =====================================================
# =========               Main                =========
# =====================================================

# =======   READING DATA

MODE = Mode.DECIMAL
DATA_PATH_NO_SPLIT = r"C:\Users\fisar\Desktop\Diplomka\other_models_keras\data\CICIoV2024"
#Read dataset
df = load_data_to_DF(DATA_PATH_NO_SPLIT, MODE)
# print(df)

# =======   DATA TRANSFORMATION

# Transform all features into the scale of [0,1]
numeric_features = df.dtypes[df.dtypes != 'object'].index
scaler = QuantileTransformer() 
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Multiply the feature values by 255 to transform them into the scale of [0,255]
df[numeric_features] = df[numeric_features].apply(
    lambda x: (x*255))

# print(df.describe())


# =======   GENERATE IMAGES

df_0 = df[df['specific_class'] == 'BENIGN'].drop(['label', 'category', 'specific_class'], axis=1) # BENIGN
df_1 = df[df['specific_class'] == 'DoS'].drop(['label', 'category', 'specific_class'], axis=1) # DOS
df_2 = df[df['specific_class'] == 'GAS'].drop(['label', 'category', 'specific_class'], axis=1) # GAS
df_3 = df[df['specific_class'] == 'RPM'].drop(['label', 'category', 'specific_class'], axis=1) # RPM
df_4 = df[df['specific_class'] == 'SPEED'].drop(['label', 'category', 'specific_class'], axis=1) # SPEED
df_5 = df[df['specific_class'] == 'STEERING_WHEEL'].drop(['label', 'category', 'specific_class'], axis=1) # STEERING_WHEEL

# print(df_0)

def generate_image(df,dataset_number):
    count=0
    ims = []

    image_path = f"train/{dataset_number}/"
    os.makedirs(image_path)

    for i in range(0, len(df)):  
        count=count+1
        if count<=27: 
            im=df.iloc[i].values
            ims=np.append(ims,im)
        else:
            ims=np.array(ims).reshape(9,9,3)
            array = np.array(ims, dtype=np.uint8)
            new_image = Image.fromarray(array)
            new_image.save(image_path+str(i)+'.png')
            count=0
            ims = []

# generate_image(df_0,0)
# generate_image(df_1,1)
# generate_image(df_2,2)
# generate_image(df_3,3)
# generate_image(df_4,4)
# generate_image(df_5,5)


# =======   SPLITTING THE TRAINING AND TESTING SET

# Create folders to store images
train_dir='./train/'
val_dir='./test/'
all_imgs=[]
for subdir in os.listdir(train_dir):
    for filename in os.listdir(os.path.join(train_dir,subdir)):
        filepath=os.path.join(train_dir,subdir,filename)
        all_imgs.append(filepath)
print(f"total number of images: {len(all_imgs)}") # 47624

#split a test set from the dataset, train/test size = 80%/20%
numbers = len(all_imgs)//5 	#size of test set (20%)

def my_move_file(src_file,dst_file):
    if not os.path.isfile(src_file):
        print ("%s not exist!"%(src_file))
    else:
        fpath,fname=os.path.split(dst_file)    
        if not os.path.exists(fpath):
            os.makedirs(fpath)               
        shutil.move(src_file,dst_file)          
        #print ("move %s -> %s"%(srcfile,dstfile))

def create_test_set():
    # Create the test set
    val_imgs=random.sample(all_imgs,numbers)
    for img in val_imgs:
        dest_path=img.replace(train_dir,val_dir)
        my_move_file(img,dest_path)
    print('Finish creating test set')

def resize_to_224():

    #resize the images 224*224 for better CNN training
    def get_224(folder,dstdir):
        imgfilepaths=[]
        for root,dirs,imgs in os.walk(folder):
            for thisimg in imgs:
                thisimg_path=os.path.join(root,thisimg)
                imgfilepaths.append(thisimg_path)
        for thisimg_path in imgfilepaths:
            dir_name,filename=os.path.split(thisimg_path)
            dir_name=dir_name.replace(folder,dstdir)
            new_file_path=os.path.join(dir_name,filename)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            img=cv2.imread(thisimg_path)
            img=cv2.resize(img,(224,224))
            cv2.imwrite(new_file_path,img)
        print('Finish resizing'.format(folder=folder))

    DATA_DIR_224='./train_224/'
    get_224(folder='./train/',dstdir=DATA_DIR_224)    

    DATA_DIR2_224='./test_224/'
    get_224(folder='./test/',dstdir=DATA_DIR2_224)

# fix
# create_test_set()
# resize_to_224()

# Read the images for each category, the file name may vary (27.png, 83.png...)
img1 = Image.open('./train_224/0/27.png')
img2 = Image.open('./train_224/1/83.png')
img3 = Image.open('./train_224/2/27.png')
img4 = Image.open('./train_224/3/27.png')
img5 = Image.open('./train_224/4/27.png')

plt.figure(figsize=(10, 10)) 
plt.subplot(1,5,1)
plt.imshow(img1)
plt.title("Normal")
plt.subplot(1,5,2)
plt.imshow(img2)
plt.title("RPM Spoofing")
plt.subplot(1,5,3)
plt.imshow(img3)
plt.title("Gear Spoofing")
plt.subplot(1,5,4)
plt.imshow(img4)
plt.title("DoS Attack")
plt.subplot(1,5,5)
plt.imshow(img5)
plt.title("Fuzzy Attack")
plt.show()  # display it