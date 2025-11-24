import os
import tensorflow as tf
from keras.layers import Dense,Flatten,GlobalAveragePooling2D,Input,Conv2D,MaxPooling2D,Dropout
import matplotlib.pyplot as plt
import keras
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Models
from keras.models import Model,load_model,Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
from keras.applications.xception import  Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet
import keras.callbacks as kcallbacks
import keras
from keras.utils import load_img
from keras.utils import img_to_array
import math
import random
import datetime
from keras.utils import plot_model

# Training and metrics
import sklearn.metrics as metrics
from keras import layers
import numpy as np
from PIL import Image
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

# Misc
import warnings
warnings.filterwarnings("ignore")

# =====================================================
# =========             Constants             =========
# =====================================================
NUM_CLASSES = 6
TARGET_SIZE = (224, 224)
INPUT_SIZE = (224, 224, 3)
BATCHSIZE = 4 # could try 128 or 32 
# BATCHSIZE only 4 because of low RAM on my PC and big image size

def gpu_available():
    gpus = tf.config.list_physical_devices('GPU')
    print("Detected GPUs:", gpus)
    if gpus: print("\t✔ TensorFlow can see the GPU.")
    else: print("\t✘ No GPU detected by TensorFlow.")


# =============================================================================
# =========             Generate Training and Test Images             =========
# =============================================================================
train_dataset = tf.keras.utils.image_dataset_from_directory(
    './train_224/',
    image_size=TARGET_SIZE,
    batch_size=BATCHSIZE,
    label_mode='categorical' # stejné jako class_mode='categorical'
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    './test_224/',
    image_size=TARGET_SIZE,
    batch_size=BATCHSIZE,
    label_mode='categorical'
)


# Normalizace
normalization = layers.Rescaling(1./255)

train_dataset = train_dataset.map(lambda x, y: (normalization(x), y))
validation_dataset = validation_dataset.map(lambda x, y: (normalization(x), y))

# =============================================================================
# =========            Define the image plotting functions            =========
# =============================================================================

#plot the figures
class LossHistory(keras.callbacks.Callback):
    def __init__(
        self,
        model_name="model",

        # oddělené limity pro LOSS graf
        y_min_loss=0,
        y_max_loss=None,

        # oddělené limity pro ACC graf
        y_min_acc=0,
        y_max_acc=None,
    ):
        super().__init__()
        self.model_name = model_name

        self.y_min_loss = y_min_loss
        self.y_max_loss = y_max_loss

        self.y_min_acc = y_min_acc
        self.y_max_acc = y_max_acc

    def on_train_begin(self, logs=None):
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))

    def save_plots(self):
        import datetime, os
        import matplotlib.pyplot as plt

        # ====== 1) vytvoření složky result/YYYY_MM_DD_HH_MM_modelname ======
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        save_dir = os.path.join("result", f"{timestamp}_{self.model_name}")
        os.makedirs(save_dir, exist_ok=True)

        epochs = range(len(self.losses))

        # Pokud není y_max zadán, nastavíme jej automaticky podle dat
        if self.y_max_loss is None:
            self.y_max_loss = max(self.losses + self.val_losses)

        if self.y_max_acc is None:
            self.y_max_acc = max(self.acc + self.val_acc)

        # ====== 2) graf ztrátovosti (LOSS) ======
        plt.figure()
        plt.plot(epochs, self.losses, label="Train Loss")
        plt.plot(epochs, self.val_losses, label="Val Loss")
        plt.ylim(self.y_min_loss, self.y_max_loss)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_dir, "graf_loss.pdf"), format="pdf")
        plt.close()

        # ====== 3) graf přesnosti (ACCURACY) ======
        plt.figure()
        plt.plot(epochs, self.acc, label="Train Accuracy")
        plt.plot(epochs, self.val_acc, label="Val Accuracy")
        plt.ylim(self.y_min_acc, self.y_max_acc)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training & Validation Accuracy")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_dir, "graf_acc.pdf"), format="pdf")
        plt.close()

        print(f"\n📁 Grafy uloženy do: {save_dir}\n")

# ================================================
# =========            Models            =========
# ================================================

# ---------     CNN 1
def cnn_by_own(input_shape, num_class, epochs, savepath='./model_own.h5', history=None):
    if history is None:
        raise ValueError("Missing history callback: LossHistory()")
    
    model = Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3,3), activation="relu", padding="same"),
        layers.Conv2D(128, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.Conv2D(256, (3,3), activation="relu", padding="same"),
        layers.Conv2D(256, (3,3), activation="relu", padding="same"),
        layers.Conv2D(256, (3,3), activation="relu", padding="same"),

        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_class, activation="softmax")
    ])

    # Optimizer s learning rate
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    earlyStopping = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True
    )

    saveBestModel = keras.callbacks.ModelCheckpoint(
        savepath,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[earlyStopping, saveBestModel, history]
    )

    return model

# ---------     Xception

def xception(input_shape, num_class, epochs, savepath='./xception.h5', history=None):
    if history is None:
        raise ValueError("Missing history callback: LossHistory()")
    
    # Načíst předtrénovaný Xception bez fully-connected vrstvy
    base_model = Xception(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )

    # Freeze prvních 131 vrstev (lze upravit)
    for layer in base_model.layers[:131]:
        layer.trainable = False
    for layer in base_model.layers[131:]:
        layer.trainable = True

    # Postupné dobudování top části
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_class, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    # Optimizer s learning rate
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    # Callbacks (jako u prvního modelu)
    earlyStopping = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True
    )

    saveBestModel = keras.callbacks.ModelCheckpoint(
        filepath=savepath,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    # Trénování — stejná API jako u prvního modelu
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[earlyStopping, saveBestModel, history]
    )

    return model

# ---------     VGG16

def vgg16(input_shape, num_class, epochs, savepath='./VGG16.h5', history=None):
    if history is None:
        raise ValueError("Missing history callback: LossHistory()")
    
    # Načíst předtrénovaný VGG16 bez fully-connected vrstev
    base_model = VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )

    # Freeze prvních 15 vrstev
    for layer in base_model.layers[:15]:
        layer.trainable = False

    for layer in base_model.layers[15:]:
        layer.trainable = True

    # Přidat vlastní klasifikační část
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_class, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs, name="VGG16_custom")

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    # Callbacks
    earlyStopping = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=2,
        restore_best_weights=True,
        verbose=1
    )

    saveBestModel = keras.callbacks.ModelCheckpoint(
        filepath=savepath,
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True
    )

    # Trénování – stejná API jako tvé ostatní modely
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[earlyStopping, saveBestModel, history]
    )

    return model
# ===============================================
# =========            Train            =========
# ===============================================

Y_MIN_LOSS = 0
Y_MAX_LOSS = 0.4
Y_MIN_ACC = 0.8
Y_MAX_ACC = 1.2

EPOCHS = 20


history_this = LossHistory()
gpu_available()
history_this = LossHistory(model_name="vgg16", y_min_loss=Y_MIN_LOSS, y_max_loss=Y_MAX_LOSS, y_min_acc=Y_MIN_ACC, y_max_acc=Y_MAX_ACC)
model = vgg16(INPUT_SIZE, num_class=NUM_CLASSES, epochs=EPOCHS, history=history_this)
history_this.save_plots()

# gpu_available()
# model = vgg16(INPUT_SIZE, num_class=NUM_CLASSES, epochs=20)
# history_this.plot()
# plt.show()