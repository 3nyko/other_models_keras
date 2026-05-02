import os
import tensorflow as tf
from keras.layers import Dense,Flatten,GlobalAveragePooling2D,Input,Conv2D,MaxPooling2D,Dropout
import matplotlib.pyplot as plt
import keras
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
import time

# Misc
import warnings
warnings.filterwarnings("ignore")

# =====================================================
# =========             Constants             =========
# =====================================================
TARGET_SIZE = (224, 224)
INPUT_SIZE = (224, 224, 3)
BATCHSIZE = 2  # could try 128 or 32 
# BATCHSIZE only 4 because of low RAM on my PC and big image size

# Dataset switch:
# - "CICIoV2024"
# - "Car_Hacking"
DATASET_CHOICE = "Car_Hacking"

DATASET_PATHS = {
    "CICIoV2024": {
        "train": "./train_224/",
        "val": "./val_224/",
        "test": "./test_224/",
        "results_subdir": "CICIoV2024",
        "num_classes": 6,
    },
    "Car_Hacking": {
        "train": "./train_car_hack_224/",
        "val": "./val_car_hack_224/",
        "test": "./test_car_hack_224/",
        "results_subdir": "Car_Hacking",
        "num_classes": 5,
    },
}


def get_dataset_config(dataset_choice):
    if dataset_choice not in DATASET_PATHS:
        raise ValueError(
            f"Unknown DATASET_CHOICE '{dataset_choice}'. "
            f"Allowed: {list(DATASET_PATHS.keys())}"
        )
    return DATASET_PATHS[dataset_choice]


DATASET_CONFIG = get_dataset_config(DATASET_CHOICE)
NUM_CLASSES = DATASET_CONFIG["num_classes"]

def gpu_available():
    gpus = tf.config.list_physical_devices('GPU')
    print("Detected GPUs:", gpus)
    if gpus: print("\t✔ TensorFlow can see the GPU.")
    else: print("\t✘ No GPU detected by TensorFlow.")


# =============================================================================
# =========             Generate Training and Test Images             =========
# =============================================================================
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_CONFIG["train"],
    image_size=TARGET_SIZE,
    batch_size=BATCHSIZE,
    label_mode='categorical'
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_CONFIG["val"],
    image_size=TARGET_SIZE,
    batch_size=BATCHSIZE,
    label_mode='categorical'
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_CONFIG["test"],
    image_size=TARGET_SIZE,
    batch_size=BATCHSIZE,
    label_mode='categorical'
)

# normalization
normalization = layers.Rescaling(1./255)

train_dataset = train_dataset.map(lambda x, y: (normalization(x), y))
validation_dataset = validation_dataset.map(lambda x, y: (normalization(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization(x), y))

# =============================================================================
# =========            Define the image plotting functions            =========
# =============================================================================

# plot the figures
class LossHistory(keras.callbacks.Callback):
    def __init__(
        self,
        model_name="model",

        # separate limits for LOSS graph
        y_min_loss=0,
        y_max_loss=None,

        # separate limits for ACC graph
        y_min_acc=0,
        y_max_acc=None,
    ):
        super().__init__()
        self.model_name = model_name

        self.y_min_loss = y_min_loss
        self.y_max_loss = y_max_loss

        self.y_min_acc = y_min_acc
        self.y_max_acc = y_max_acc
        self.save_dir = None

    def on_train_begin(self, logs=None):
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))

    def on_train_end(self, logs=None):
        self.end_time = time.time()

    @property
    def training_time(self):
        # training time in seconds
        return self.end_time - self.start_time
    @property
    def epochs_trained(self):
        # actual number of epochs trained (after EarlyStopping)
        return len(self.losses)
    def get_best_val_metrics(self):
        """Vrátí (best_val_acc, best_val_loss, best_epoch) podle nejlepší val_acc."""
        if not self.val_acc:
            return None, None, None
        best_epoch = int(np.argmax(self.val_acc))
        best_val_acc = float(self.val_acc[best_epoch])
        best_val_loss = float(self.val_losses[best_epoch])
        return best_val_acc, best_val_loss, best_epoch

    def save_plots(self):
        import datetime, os
        import matplotlib.pyplot as plt

        # ====== 1) folder: results/<results_subdir>/YYYY_MM_DD_HH_MM_modelname ======
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        save_dir = os.path.join(
            "results",
            DATASET_CONFIG["results_subdir"],
            f"{timestamp}_{self.model_name}",
        )
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

        epochs = range(len(self.losses))

        # if y_max is not specified, we set it automatically based on the data.
        if self.y_max_loss is None:
            self.y_max_loss = max(self.losses + self.val_losses)

        if self.y_max_acc is None:
            self.y_max_acc = max(self.acc + self.val_acc)

        # ====== 2) loss graph (LOSS) ======
        plt.figure()
        plt.plot(epochs, self.losses, label="Train Loss")
        plt.plot(epochs, self.val_losses, label="Val Loss")
        plt.ylim(self.y_min_loss, self.y_max_loss)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{self.model_name} – Training & Validation Loss")
        plt.grid(True)
        plt.legend()
        loss_filename = f"graf_loss_{self.model_name}.pdf"
        plt.savefig(os.path.join(save_dir, loss_filename), format="pdf")
        plt.close()

        # ====== 3) accuracy graph (ACCURACY) ======
        plt.figure()
        plt.plot(epochs, self.acc, label="Train Accuracy")
        plt.plot(epochs, self.val_acc, label="Val Accuracy")
        plt.ylim(self.y_min_acc, self.y_max_acc)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{self.model_name} – Training & Validation Accuracy")
        plt.grid(True)
        plt.legend()
        acc_filename = f"graf_acc_{self.model_name}.pdf"
        plt.savefig(os.path.join(save_dir, acc_filename), format="pdf")
        plt.close()

        # ====== 4) full-range accuracy graph in percent (0-100) ======
        plt.figure()
        train_acc_percent = np.array(self.acc) * 100.0
        val_acc_percent = np.array(self.val_acc) * 100.0
        plt.plot(epochs, train_acc_percent, label="Train Accuracy (%)")
        plt.plot(epochs, val_acc_percent, label="Val Accuracy (%)")
        plt.ylim(0, 100)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title(f"{self.model_name} – Training & Validation Accuracy (0-100%)")
        plt.grid(True)
        plt.legend()
        acc_full_filename = f"graf_acc_full_0_100_{self.model_name}.pdf"
        plt.savefig(os.path.join(save_dir, acc_full_filename), format="pdf")
        plt.close()

        print(f"\n Grafy uloženy do: {save_dir}\n")

    def save_training_summary(self, model, test_loss, test_acc):
        if self.save_dir is None:
            raise ValueError("save_plots() must be called before save_training_summary().")

        best_val_acc, best_val_loss, best_epoch = self.get_best_val_metrics()
        summary_path = os.path.join(self.save_dir, f"summary_{self.model_name}.txt")

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"model_name: {self.model_name}\n")
            f.write(f"epochs_trained: {self.epochs_trained}\n")
            f.write(f"model_parameters: {model.count_params()}\n")
            f.write(f"training_time_seconds: {self.training_time:.2f}\n")
            if best_epoch is not None:
                f.write(f"best_val_epoch: {best_epoch + 1}\n")
                f.write(f"best_val_accuracy: {best_val_acc:.6f}\n")
                f.write(f"best_val_loss: {best_val_loss:.6f}\n")
            else:
                f.write("best_val_epoch: N/A\n")
                f.write("best_val_accuracy: N/A\n")
                f.write("best_val_loss: N/A\n")
            f.write(f"test_accuracy: {test_acc:.6f}\n")
            f.write(f"test_loss: {test_loss:.6f}\n")

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

    # optimizer with learning rate
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
    
    # load pre-trained Xception without fully-connected layer
    base_model = Xception(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )

    # freeze the first 131 layers (can be adjusted)
    for layer in base_model.layers[:131]:
        layer.trainable = False
    for layer in base_model.layers[131:]:
        layer.trainable = True

    # gradual completion of the top section
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_class, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    # optimizer with learning rate
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

    # Callbacks (as with the first model)
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

    # Training — same API as in the first model
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
    
    # load pre-trained VGG16 without fully-connected layer
    base_model = VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )

    # freeze the first 15 layers
    for layer in base_model.layers[:15]:
        layer.trainable = False

    for layer in base_model.layers[15:]:
        layer.trainable = True

    # add your own classification section
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_class, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs, name="VGG16_custom")

    # optimizer
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

    # callbacks
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

    # Training — same API as in the first model
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[earlyStopping, saveBestModel, history]
    )

    return model

# ---------     VGG19
def vgg19(input_shape, num_class, epochs, savepath="./VGG19.h5", history=None):
    if history is None:
        raise ValueError("Missing history callback: LossHistory()")
    
    # load pre-trained VGG19 without fully-connected layer
    base_model = VGG19(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )

    # freeze the first 19 layers
    for layer in base_model.layers[:19]:
        layer.trainable = False

    for layer in base_model.layers[19:]:
        layer.trainable = True

    # add your own classification section
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_class, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs, name="VGG19_custom")

    # compilation
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

    # callbacks
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

    # training
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[earlyStopping, saveBestModel, history]
    )

    return model

# ---------     ResNet
def resnet(input_shape, num_class, epochs, savepath="./resnet.h5", history=None):
    if history is None:
        raise ValueError("Missing history callback: LossHistory()")

    # load pre-trained ResNet50 without fully-connected layer
    base_model = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )

    # freeze the first ~120 layers
    for layer in base_model.layers[:120]:
        layer.trainable = False

    # other trainable layers
    for layer in base_model.layers[120:]:
        layer.trainable = True

    # add your own classification section
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_class, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs, name="ResNet50_custom")

    # compilation
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

    # callbacks
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

    # training
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[earlyStopping, saveBestModel, history]
    )

    return model

# ---------     Inception
def inception(input_shape, num_class, epochs, savepath="./inception.h5", history=None):
    if history is None:
        raise ValueError("Missing history callback: LossHistory()")

    # load pre-trained InceptionV3 without fully-connected layer
    base_model = InceptionV3(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )

    # freeze the first 35 layers
    for layer in base_model.layers[:35]:
        layer.trainable = False

    for layer in base_model.layers[35:]:
        layer.trainable = True

    # add your own classification section
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_class, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs, name="InceptionV3_custom")

    # compilation
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

    # callbacks
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

    # training
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[earlyStopping, saveBestModel, history]
    )

    return model

# ---------     InceptionResnet
def inceptionresnet(input_shape, num_class, epochs, savepath="./inceptionresnet.h5", history=None):
    if history is None:
        raise ValueError("Missing history callback: LossHistory()")

    # load pre-trained InceptionResNetV2 without fully-connected layer
    base_model = InceptionResNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )

    # freeze the first 500 layers
    for layer in base_model.layers[:500]:
        layer.trainable = False

    for layer in base_model.layers[500:]:
        layer.trainable = True

    # add your own classification section
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_class, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs, name="InceptionResNetV2_custom")

    # compilation
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

    # callbacks
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

    # training
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[earlyStopping, saveBestModel, history]
    )

    return model

# ---------     LSTM
def lstm(input_shape, num_class, epochs, savepath="./lstm.h5", history=None):
    if history is None:
        raise ValueError("Missing history callback: LossHistory()")

    model = Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        # Convert feature map to sequence (timesteps, features) for LSTM.
        layers.Reshape((28, 28 * 128)),
        layers.LSTM(128),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_class, activation="softmax")
    ])

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
        filepath=savepath,
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

MODEL_REGISTRY = {
    "cnn_by_own": cnn_by_own,
    "xception": xception,
    "vgg16": vgg16,
    "vgg19": vgg19,
    "resnet": resnet,
    "inception": inception,
    "inceptionresnet": inceptionresnet,
    "lstm": lstm
}

Y_MIN_LOSS = 0
Y_MAX_LOSS = 0.4
Y_MIN_ACC = 0.8
Y_MAX_ACC = 1.2

EPOCHS = 20

def train_model_main(model_name):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' není v MODELS. "
                         f"Dostupné: {list(MODEL_REGISTRY.keys())}")

    model_fn = MODEL_REGISTRY[model_name]

    # ===============================================
    # =========            Train            =========
    # ===============================================

    history_this = LossHistory()
    history_this = LossHistory(model_name=model_name, y_min_loss=Y_MIN_LOSS, y_max_loss=Y_MAX_LOSS, y_min_acc=Y_MIN_ACC, y_max_acc=Y_MAX_ACC)
    model = model_fn(INPUT_SIZE, num_class=NUM_CLASSES, epochs=EPOCHS, history=history_this)
    history_this.save_plots()

    # === Validation metrics (best epoch according to val_accuracy) ===
    best_val_acc, best_val_loss, best_epoch = history_this.get_best_val_metrics()

    # === Test metrics ===
    test_loss, test_acc = model.evaluate(test_dataset, verbose=0)

    # === Summary ===
    print("\n================== SHRNU TRENOVANI ==================")
    print(f"model: {history_this.model_name}")
    print(f"počet parametrů: {model.count_params():,}")
    print(f"doba trénování: {history_this.training_time:.2f} s")
    print(f"počet epoch: {history_this.epochs_trained}")
    if best_epoch is not None:
        print(f"val acc (best): {best_val_acc:.4f} (epoch {best_epoch+1})")
        print(f"val loss (best): {best_val_loss:.4f} (epoch {best_epoch+1})")
    else:
        print("val acc: N/A")
        print("val loss: N/A")
    print(f"test acc: {test_acc:.4f}")
    print(f"test loss: {test_loss:.4f}")
    print("=====================================================\n")

    # Save text summary next to graphs.
    history_this.save_training_summary(model=model, test_loss=test_loss, test_acc=test_acc)

def main():
    print(f"Dataset: {DATASET_CHOICE}")
    print(f"Train path: {DATASET_CONFIG['train']}")
    print(f"Val path: {DATASET_CONFIG['val']}")
    print(f"Test path: {DATASET_CONFIG['test']}")
    print(
        f"Plots & summary directory base: results/{DATASET_CONFIG['results_subdir']}/"
    )

    gpu_available()

    # train_model_main("cnn_by_own")
    #train_model_main("xception")
    # train_model_main("vgg16")
    train_model_main("vgg19")
    # train_model_main("resnet")
    # train_model_main("inception")
    # train_model_main("inceptionresnet")
    # train_model_main("lstm")


if __name__ == "__main__":
    main()