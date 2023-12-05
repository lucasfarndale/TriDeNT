import tensorflow as tf
import os

def load_datasets(ds_path_list, output_dict, shuffle=False, num_parallel_calls=tf.data.AUTOTUNE):
    feature_description = {key: tf.io.FixedLenFeature([], tf.string) for key in output_dict.keys()}
    
    @tf.function
    def _parse_record(x):
        # Parse the input tf.train.Example proto using the dictionary above.
        return tf.io.parse_single_example(x, feature_description)
    
    
    @tf.function
    def _parse_tensor(x):
        output = {key:tf.io.parse_tensor(x[key],out_type=value) for key, value in output_dict.items()}
        return output
    
    if not isinstance(ds_path_list, list):
        ds_dir_list = [ds_path_list]
    else:
        ds_dir_list = ds_path_list
    output_list = []
    for ds_dir in ds_dir_list:
        ds_list = [os.path.join(root, filename) for root,_,files in os.walk(ds_dir) for filename in files]
        dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files(ds_list, shuffle=shuffle))
        dataset = dataset.map(_parse_record, num_parallel_calls=num_parallel_calls).map(_parse_tensor, num_parallel_calls=num_parallel_calls)
        output_list.append(dataset)
    return (*output_list,)

def preprocess_ds(ds, batch_size=1, shuffle_no=1, seed=0, pre=tf.data.AUTOTUNE, rei=True, drop_remainder=False):
    return ds.shuffle(shuffle_no, seed=seed, reshuffle_each_iteration=rei).batch(batch_size, drop_remainder=drop_remainder).prefetch(pre)
