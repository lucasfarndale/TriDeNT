import tensorflow as tf
import matplotlib.pyplot as plt
import os

def create_projector(input_shape=(2048,), output_size=8192):
    projector = tf.keras.models.Sequential()
    projector.add(tf.keras.Input(shape=input_shape))
    projector.add(tf.keras.layers.Dense(output_size))
    projector.add(tf.keras.layers.BatchNormalization())
    projector.add(tf.keras.layers.Activation('relu'))

    projector.add(tf.keras.layers.Dense(output_size))
    projector.add(tf.keras.layers.BatchNormalization())
    projector.add(tf.keras.layers.Activation('relu'))

    projector.add(tf.keras.layers.Dense(output_size))
    return projector

def create_resnet(input_shape):
    return tf.keras.applications.resnet50.ResNet50(include_top=False, weights=None, pooling='avg', input_shape=input_shape)

def create_adam_opt(lr_decayed_fn, clipnorm=1.):
    return tf.keras.optimizers.Adam(learning_rate=lr_decayed_fn, clipnorm=clipnorm)

def plot_samples_from_ds(samples):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(samples.numpy()[n].astype(int))
        plt.axis("off")
    plt.show()
        
def load_vicreg_weights(folder_path, enc_list, proj_list):
    for i, enc in enumerate(enc_list):
        enc.load_weights(os.path.join(folder_path,f'encoder_weights_{i}'))
    for i, proj in enumerate(proj_list):
        proj.load_weights(os.path.join(folder_path,f'projector_weights_{i}'))
        
def save_vicreg_weights(folder_path, enc_list, proj_list):
    for i, enc in enumerate(enc_list):
        enc.save_weights(os.path.join(folder_path,f'encoder_weights_{i}'))
    for i, proj in enumerate(proj_list):
        proj.save_weights(os.path.join(folder_path,f'projector_weights_{i}'))
