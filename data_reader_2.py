# https://ocslab.hksecurity.net/Datasets/car-hacking-dataset
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
import glob


# =====================================================
# =========       Constants and options       =========
# =====================================================

MULTICLASS_DICT = {
    "BENIGN" : 0,
    "DOS" : 1,
    "FUZZY" : 2,
    "SPOOFING_GEAR" : 3,
    "SPOOFING_RPM" : 4,
}   

# =====================================================
# =========            Functions              =========
# =====================================================

def load_first_png(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.png")))
    if not files:
        raise FileNotFoundError(f"!!! Zadny .png soubor ve slozce {folder}")
    return Image.open(files[0])

def load_data_to_DF(data_dir):
    folder = os.path.join(data_dir)
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
        rows = []
        with open(f, "r", encoding="utf-8", errors="replace") as src:
            for line in src:
                parts = line.strip().split(",")
                if len(parts) < 5:
                    continue

                # Car-Hacking CSV row:
                # timestamp,id,dlc,data_0,...,data_(dlc-1),R
                try:
                    dlc = int(parts[2])
                except ValueError:
                    continue

                payload = parts[3:-1]
                payload = payload[:dlc]
                if len(payload) < dlc:
                    continue

                padded_payload = payload + ["00"] * (8 - len(payload))

                rows.append(
                    {
                        "DATA_0": padded_payload[0],
                        "DATA_1": padded_payload[1],
                        "DATA_2": padded_payload[2],
                        "DATA_3": padded_payload[3],
                        "DATA_4": padded_payload[4],
                        "DATA_5": padded_payload[5],
                        "DATA_6": padded_payload[6],
                        "DATA_7": padded_payload[7],
                    }
                )

        df_file = pd.DataFrame(rows)
        df_list.append(df_file)

        

    if not df_list:
        raise ValueError(f"No valid rows parsed from CSV files in {folder_path}")

    df = pd.concat(df_list, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]  # normalize column names

    # same idea as data_reader.py: convert data columns to numeric
    for col in df.columns:
        if col.startswith("DATA_"):
            # Car-Hacking values are hexadecimal strings
            df[col] = df[col].apply(
                lambda x: int(str(x), 16) if isinstance(x, str) and str(x) != "" else x
            )

    df = df.fillna(0)
    return df

# =====================================================
# =========               Main                =========
# =====================================================

# =======   READING DATA

DATA_PATH_NO_SPLIT = r"C:\Users\fisar\Desktop\Diplomka\other_models_keras\data\Car-Hacking Dataset"
# read dataset
df = load_data_to_DF(DATA_PATH_NO_SPLIT)
# print(df)

# =======   DATA TRANSFORMATION

# transform all features into the scale of [0,1]
numeric_features = df.dtypes[df.dtypes != 'object'].index
scaler = QuantileTransformer() 
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# multiply the feature values by 255 to transform them into the scale of [0,255]
df[numeric_features] = df[numeric_features].apply(
    lambda x: (x*255))

# print(df.describe())

# =======   GENERATE IMAGES

def load_single_car_hacking_csv(file_path):
    rows = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as src:
        for line in src:
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue

            try:
                dlc = int(parts[2])
            except ValueError:
                continue

            payload = parts[3:-1][:dlc]
            if len(payload) < dlc:
                continue

            padded_payload = payload + ["00"] * (8 - len(payload))
            rows.append([int(x, 16) for x in padded_payload])

    return pd.DataFrame(rows, columns=[f"DATA_{i}" for i in range(8)])


class_files = [
    ("BENIGN", "normal_run_data.csv"),
    ("DOS", "DoS_dataset.csv"),
    ("FUZZY", "Fuzzy_dataset.csv"),
    ("SPOOFING_GEAR", "gear_dataset.csv"),
    ("SPOOFING_RPM", "RPM_dataset.csv"),
]

# print(df_0)

def generate_image(df,dataset_number):
    count=0
    ims = []

    image_path = f"train_car_hack/{dataset_number}/"
    os.makedirs(image_path, exist_ok=True)

    for i in range(0, len(df)):
        count=count+1
        if count<=27:
            im=df.iloc[i].values
            ims=np.append(ims,im)
        else:
            # DATA_0..DATA_7 gives 216 values for 27 packets; pad to 243 (9*9*3)
            ims=np.array(ims)
            if ims.size < 243:
                ims = np.pad(ims, (0, 243 - ims.size), mode="constant")
            else:
                ims = ims[:243]
            ims=ims.reshape(9,9,3)
            array = np.array(ims, dtype=np.uint8)
            new_image = Image.fromarray(array)
            new_image.save(image_path+str(i)+'.png')
            count=0
            ims = []

for idx, (_class_name, filename) in enumerate(class_files):
    class_path = os.path.join(DATA_PATH_NO_SPLIT, filename)
    class_df = load_single_car_hacking_csv(class_path)
    generate_image(class_df, idx)


# =======   SPLITTING THE TRAINING AND TESTING SET

# Create folders to store images
train_dir = './train_car_hack/'
val_dir = './val_car_hack/'
test_dit = './test_car_hack/'
all_imgs=[]

for subdir in os.listdir(train_dir):
    for filename in os.listdir(os.path.join(train_dir,subdir)):
        filepath=os.path.join(train_dir,subdir,filename)
        all_imgs.append(filepath)
print(f"total number of images: {len(all_imgs)}") # 47624

# split a test set from the dataset, train/val/test size = 60/20/20
TEST_PERC = 20 # 20% Test
VAL_PERC = 20 # 20% Val
IMAGE_COUNT_TEST = len(all_imgs)//int(100/TEST_PERC) 
IMAGE_COUNT_VAL = len(all_imgs)//int(100/VAL_PERC)  

def create_test_and_val_set():
    def my_move_file(src_file,dst_file):
        if not os.path.isfile(src_file):
            print ("%s not exist!"%(src_file))
        else:
            fpath,fname=os.path.split(dst_file)    
            if not os.path.exists(fpath):
                os.makedirs(fpath)               
            shutil.move(src_file,dst_file)          
            #print ("move %s -> %s"%(srcfile,dstfile))
    # Create the test set
    all_imgs_shuffled = all_imgs.copy()
    random.shuffle(all_imgs_shuffled)
    
    val_imgs = all_imgs_shuffled[:IMAGE_COUNT_VAL]
    # val_imgs = random.sample(all_imgs, IMAGE_COUNT_VAL)
    for img in val_imgs:
        dest_path=img.replace(train_dir,val_dir)
        my_move_file(img, dest_path)
    print('Finish creating validation set')

    test_imgs = all_imgs_shuffled[IMAGE_COUNT_VAL:(IMAGE_COUNT_VAL + IMAGE_COUNT_TEST)]
    # test_img = random.sample(all_imgs, IMAGE_COUNT_TEST)
    for img in test_imgs:
        dest_path=img.replace(train_dir,test_dit)
        my_move_file(img, dest_path)
    
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

    DATA_DIR_224='./train_car_hack_224/'
    get_224(folder='./train_car_hack/',dstdir=DATA_DIR_224)    

    DATA_DIR2_224='./test_car_hack_224/'
    get_224(folder='./test_car_hack/',dstdir=DATA_DIR2_224)

    DATA_DIR2_224='./val_car_hack_224/'
    get_224(folder='./val_car_hack/',dstdir=DATA_DIR2_224)

# fix
create_test_and_val_set()
resize_to_224()

# Read the images for each category, the file name may vary (27.png, 83.png...)
img1 = load_first_png('./train_car_hack_224/0')
img2 = load_first_png('./test_car_hack_224/1')
img3 = load_first_png('./val_car_hack_224/2')


plt.figure(figsize=(10, 10)) 
plt.subplot(1,5,1)
plt.imshow(img1)
plt.title("BENIGN")
plt.subplot(1,5,2)
plt.imshow(img2)
plt.title("DOS")
plt.subplot(1,5,3)
plt.imshow(img3)
plt.title("FUZZY")
