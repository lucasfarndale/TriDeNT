import tensorflow as tf
import os
from VICReg.augmentations import *
from VICReg.dataset_utils import preprocess_ds, create_siamese_dataset, load_datasets

IM_SIZE = 256
NO_GENES = 5020
DATASET_SIZE = 148120

augment_im = lambda x: custom_augment_image(x, input_shape=(IM_SIZE,IM_SIZE,3), output_shape=(IM_SIZE,IM_SIZE,3))
augment_genes = lambda x: custom_augment_genes(x, input_shape=(NO_GENES,), output_shape=(NO_GENES,))

feature_dict = {'he': tf.uint8,
                'spots': tf.uint8,
                'xpos': tf.uint8,
                'ypos': tf.uint8,
                'counts': tf.uint8,
                'counts_with_ambiguous': tf.uint8,
                'annotations': tf.uint8,
                'x_y_idx': tf.string,
                'sample': tf.string,
               }
               
def load_als_st_dataset(train_path, valid_frac=None, return_task_ds=True, task_ds_list=['he', 'counts'], branches=['he', 'he', 'counts'], augment_func_im=augment_im, augment_func_st=augment_genes, preprocess=True, batch_size=256, shuffle_no=2**10, seed=42, rei=True, drop_remainder=True):
    AUTO = tf.data.AUTOTUNE
    
    augment_dict = {'he': augment_func_im, 'counts': augment_func_st}

    (image_ds,) = load_datasets(train_path, output_dict=feature_dict)
    if valid_frac is not None:
        image_train_ds = image_ds.skip(np.round(valid_frac*DATASET_SIZE, 0))
        image_valid_ds = image_ds.take(np.round(valid_frac*DATASET_SIZE, 0))
    else:
        image_train_ds = image_ds

    if branches is not None:
        if valid_frac is not None:
            patch_valid_ds = image_valid_ds.map(lambda x: tuple(augment_dict[key](x[key]) for key in branches), num_parallel_calls=AUTO)
        patch_train_ds = image_train_ds.map(lambda x: tuple(augment_dict[key](x[key]) for key in branches), num_parallel_calls=AUTO)
    if preprocess:
        patch_train_ds = preprocess_ds(patch_train_ds, batch_size=batch_size, shuffle_no=shuffle_no, seed=seed, pre=AUTO, rei=rei, drop_remainder=drop_remainder)
        if valid_frac is not None:
            patch_valid_ds = patch_valid_ds.batch(batch_size).prefetch(AUTO)
   
    task_ds_output_dict = {}
    if return_task_ds:
        for task in task_ds_list:
            task_train_ds = image_train_ds.map(lambda x: (augment_dict[task](x[task]), x['annotations']), num_parallel_calls=AUTO)
            task_test_ds = image_valid_ds.map(lambda x: (tf.cast(x[task], tf.float32), x['annotations']), num_parallel_calls=AUTO)
            task_train_ds = task_train_ds.filter(lambda x, y: tf.shape(y)[0]==1)
            task_test_ds = task_test_ds.filter(lambda x, y: tf.shape(y)[0]==1)
            task_train_ds = task_train_ds.map(lambda x, y: (x, tf.squeeze(y)), num_parallel_calls=AUTO)
            task_test_ds = task_test_ds.map(lambda x, y: (x, tf.squeeze(y)), num_parallel_calls=AUTO)
            if preprocess:
                task_train_ds = preprocess_ds(task_train_ds, batch_size=batch_size, shuffle_no=shuffle_no, seed=seed, pre=AUTO, rei=rei, drop_remainder=drop_remainder)
                task_test_ds = task_test_ds.batch(batch_size).prefetch(AUTO)
                if task=='he':
                    shape = (-1, IM_SIZE, IM_SIZE, 3)
                else:
                    shape = (-1, NO_GENES)
                task_train_ds = task_train_ds.map(lambda x, y: (tf.reshape(x, shape), tf.reshape(y, shape=(-1,13))), num_parallel_calls=AUTO)
                task_test_ds = task_test_ds.map(lambda x, y: (tf.reshape(x, shape), tf.reshape(y, shape=(-1,13))), num_parallel_calls=AUTO)
                task_ds_output_dict[f'{task}']=(task_train_ds, task_test_ds)
    return patch_train_ds, patch_valid_ds, task_ds_output_dict
