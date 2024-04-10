import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint  # type: ignore
import albumentations as A
from keras.losses import SparseCategoricalCrossentropy
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D  # type: ignore
from tensorflow.keras.layers import Dropout, Conv2DTranspose  # type: ignore
from tensorflow.keras.layers import concatenate  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras import models as K  # type: ignore
import matplotlib.pyplot as plt


def loading_data(path):
    """
    Loading CSV file from folder
    """
    df = pd.read_csv(path)
    return df


def prepare_data(df, input_shape, transform):
    """
    Preparation of data,
    resizing, conversion RGB,
    transformation image to array,
    array type float to type int,
    color stabilization with Image.NEAREST,
    transformation colors to class,
    use of datagenerator albumentation
    """
    images_tab = []
    masks_tab = []
    for i in range(len(df)):
        image_path = df['Image_Path'][i]
        mask_path = df['Target_Path'][i]
        image = img_to_array(Image.open(image_path).convert('RGB')
                             .resize(input_shape[:2]))
        image = image.astype(np.uint8)
        images_tab.append(image)

        mask = img_to_array(Image.open(mask_path).resize(input_shape[:2],
                                                         Image.NEAREST))
        mask = mask.astype(np.uint8)
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if (mask[x, y] == [250, 170, 30]).all():
                    mask[x, y] = 0
                elif (mask[x, y] == [0, 0, 142]).all():
                    mask[x, y] = 1
                elif (mask[x, y] == [102, 102, 156]).all():
                    mask[x, y] = 2
                elif (mask[x, y] == [220, 20, 60]).all():
                    mask[x, y] = 3
                elif (mask[x, y] == [153, 153, 153]).all():
                    mask[x, y] = 4
                elif (mask[x, y] == [244, 35, 232]).all():
                    mask[x, y] = 5
                elif (mask[x, y] == [70, 70, 70]).all():
                    mask[x, y] = 6
                elif (mask[x, y] == [70, 130, 180]).all():
                    mask[x, y] = 7
        mask = mask[:, :, 0:1]
        masks_tab.append(mask)

        transformed = transform(image=image, mask=mask)
        transformed_img = transformed['image']
        images_tab.append(transformed_img)

        transformed_img = transformed['mask']
        for x in range(transformed_img.shape[0]):
            for y in range(transformed_img.shape[1]):
                if (transformed_img[x, y] == [250, 170, 30]).all():
                    transformed_img[x, y] = 0
                elif (transformed_img[x, y] == [0, 0, 142]).all():
                    transformed_img[x, y] = 1
                elif (transformed_img[x, y] == [102, 102, 156]).all():
                    transformed_img[x, y] = 2
                elif (transformed_img[x, y] == [220, 20, 60]).all():
                    transformed_img[x, y] = 3
                elif (transformed_img[x, y] == [153, 153, 153]).all():
                    transformed_img[x, y] = 4
                elif (transformed_img[x, y] == [244, 35, 232]).all():
                    transformed_img[x, y] = 5
                elif (transformed_img[x, y] == [70, 70, 70]).all():
                    transformed_img[x, y] = 6
                elif (transformed_img[x, y] == [70, 130, 180]).all():
                    transformed_img[x, y] = 7
        transformed_img = transformed_img[:, :, 0:1]
        masks_tab.append(transformed_img)

    images_tab = np.array(images_tab)
    masks_tab = np.array(masks_tab)
    return images_tab, masks_tab


def unet_model(input_shape, num_classes):
    """
    U-net model with encoder and decoder
    """
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    # Decoder
    up6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(drop5)
    up6 = concatenate([up6, drop4], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv3], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = concatenate([up8, conv2], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = concatenate([up9, conv1], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model


def dice_coef(y_true, y_pred, smooth=1e-6):
    y_pred = tf.argmax(y_pred, axis=-1)

    y_true = K.cast(y_true, dtype=K.floatx())
    y_pred = K.cast(y_pred, dtype=K.floatx())

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f)
                                           + K.sum(y_pred_f) + smooth)


def iou(y_true, y_pred, smooth=1e-6):
    y_true = K.cast(y_true, dtype=K.floatx())
    y_pred = K.cast(y_pred, dtype=K.floatx())

    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    return K.mean((intersection + smooth) / (union + smooth), axis=0)


def save_metrics(metric_name, metric_train, metric_val):
    _epochs = list(range(1, len(metric_train) + 1))
    plt.plot(_epochs, metric_train, 'g', label=f'{metric_name} Train')
    plt.plot(_epochs, metric_val, 'b', label=f'{metric_name} Val')
    plt.title(f'{metric_name} Train & Val')
    plt.xlabel('Epochs')
    plt.ylabel(f'{metric_name}')
    plt.legend()
    plt.savefig(f'./Data/metrics/{metric_name}.png')
    plt.close()


def model_training(X_train, y_train, X_val, y_val, input_shape, num_classes):
    """
    Model training with earlystopping each improvement of loss function
    loss function sparse is better with categories
    """
    tf.keras.utils.get_custom_objects()['dice_coef'] = dice_coef
    tf.keras.utils.get_custom_objects()['iou'] = iou

    model = unet_model(input_shape, num_classes)
    model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(),
                  metrics=['accuracy', dice_coef, iou])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1,
                                   restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('./models/model_X.keras',
                                       monitor='val_loss', save_best_only=True,
                                       verbose=1)

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        batch_size=32,
                        epochs=100,
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=1)
    return history


def main():
    df_train = loading_data("./Data/df_train.csv")
    df_val = loading_data("./Data/df_val.csv")

    transform = A.Compose([A.HorizontalFlip(p=1.0)])
    X_train, y_train = prepare_data(df_train, (256, 256, 3), transform)
    X_val, y_val = prepare_data(df_val, (256, 256, 3), transform)
    num_class = 8

    history = model_training(
        X_train,
        y_train,
        X_val,
        y_val,
        (256, 256, 3),
        num_class
        )

    save_metrics('Loss', history.history['loss'],
                 history.history['val_loss'])
    save_metrics('Accuracy', history.history['accuracy'],
                 history.history['val_accuracy'])
    save_metrics('Dice Coef', history.history['dice_coef'],
                 history.history['val_dice_coef'])
    save_metrics('Iou', history.history['iou'],
                 history.history['val_iou'])


if __name__ == "__main__":
    main()
