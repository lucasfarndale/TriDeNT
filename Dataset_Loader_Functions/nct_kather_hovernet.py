import tensorflow as tf
import os
from VICReg.augmentations import *
from VICReg.dataset_utils import preprocess_ds, create_siamese_dataset, load_datasets

augment_im = lambda x: custom_augment_image(x, input_shape=(224,224,3), output_shape=(224,224,3))

def load_nct_dataset(train_path, valid_path=None, return_task_ds=True, task_ds_list=[('original_images','tissue'),('original_images','cell'),('masked_images','tissue'),('masked_images','cell')], branches=['original_images', 'original_images'], augment_func=augment_im, preprocess=True, batch_size=128, shuffle_no=2**10, seed=42, rei=True, drop_remainder=True, im_resize=224):
    AUTO = tf.data.AUTOTUNE
    def decode_branches(x, key):
        if key == 'original_images':
            return x[key]
        elif key == 'masked_images':
            return tf.stack([x['masked_images']for _ in range(3)],-1)
        else:
            return key
        
    def hist(x):
        return tf.histogram_fixed_width(x, [0,5], nbins=6)

    @tf.function
    def calc_mode(x):
        if tf.reduce_sum(hist(x))==0:
            return tf.constant(0)
        else:
            unique, _, count = tf.unique_with_counts(tf.reshape(x, (-1,)))
            max_count = tf.math.argmax(count)
            mode = unique[max_count]
            return mode

    def decode_labels(x, key):
        if key=='tissue':
            return x['tissue_types']
        elif key=='cell':
            return calc_mode(tf.cast(x['cell_types'], tf.int32))
        else:
            return key
    
    feature_description_crc = {'original_images': tf.uint8,
                               'masked_images': tf.uint8,
                               'cell_types': tf.uint8,
                               'tissue_types': tf.uint8,
                              }
    (image_train_ds,) = load_datasets(train_path, output_dict=feature_description_crc)
    if valid_path is not None:
        (image_valid_ds,) = load_datasets(valid_path, output_dict=feature_description_crc)
    if branches is not None:
        if valid_path is not None:
            patch_valid_ds = image_valid_ds.map(lambda x: tuple(augment_func(decode_branches(x, key)) for key in branches), num_parallel_calls=AUTO)
        patch_train_ds = image_train_ds.map(lambda x: tuple(augment_func(decode_branches(x, key)) for key in branches), num_parallel_calls=AUTO)
    if preprocess:
        patch_train_ds = preprocess_ds(patch_train_ds, batch_size=batch_size, shuffle_no=shuffle_no, seed=seed, pre=AUTO, rei=rei, drop_remainder=drop_remainder)
        if valid_path is not None:
            patch_valid_ds = patch_valid_ds.batch(batch_size).prefetch(AUTO)
   
    task_ds_output_dict = {}
    if return_task_ds:
        for task in task_ds_list:
            task_train_ds = image_train_ds.map(lambda x: (augment_func(decode_branches(x, task[0])), decode_labels(x, task[1])), num_parallel_calls=AUTO)
            task_test_ds = image_valid_ds.map(lambda x: (tf.cast(decode_branches(x, task[0]), tf.float32), decode_labels(x, task[1])), num_parallel_calls=AUTO)

            if preprocess:
                task_train_ds = preprocess_ds(task_train_ds, batch_size=batch_size, shuffle_no=shuffle_no, seed=seed, pre=AUTO, rei=rei, drop_remainder=drop_remainder)
                task_test_ds = task_test_ds.batch(batch_size).prefetch(AUTO)
                task_train_ds = task_train_ds.map(lambda x, y: (x, tf.reshape(y, shape=(-1,1))), num_parallel_calls=AUTO)
                task_test_ds = task_test_ds.map(lambda x, y: (x, tf.reshape(y, shape=(-1,1))), num_parallel_calls=AUTO)
                task_ds_output_dict[f'{task[0]}-{task[1]}']=(task_train_ds, task_test_ds)
    return patch_train_ds, patch_valid_ds, task_ds_output_dict


def resize_nct_dataset(train_path, resize_shape, **kwargs):
    augment_im = lambda x: custom_augment_image(x, input_shape=resize_shape, output_shape=resize_shape)
    af = lambda x: augment_im(tf.image.resize(tf.image.resize_with_crop_or_pad(x, 224, 224), (resize_shape[0], resize_shape[1])))
    return load_nct_dataset(train_path, augment_func=af, **kwargs)
