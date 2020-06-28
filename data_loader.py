import numpy as np
import tensorflow as tf
import os

SAMPLING_STRIDE = 10
PATCH_SIZE = 128

def load_npz(npz_file):
    npz = np.load(npz_file)
    assert(npz["mix"].shape == npz["vocal"].shape)
    return np.expand_dims(npz['mix'], -1), np.expand_dims(npz['vocal'], -1)

def read_list(data_dir, data_list):
    fp = open(data_list, 'r')
    line = fp.readline()

    filenames = []
    while line:
        line = line.replace('\n', '')
        filenames.append(os.path.join(data_dir, line))
        line = fp.readline()  
    fp.close()
    return filenames

def _parse_function(filenames):
    filenames = filenames.numpy().decode("utf-8")
    mix, target = load_npz(filenames)
    data = []
    starts = np.random.randint(0, mix.shape[1] - PATCH_SIZE, (mix.shape[1] - PATCH_SIZE) // SAMPLING_STRIDE)
    #print("starts print")
    #print(starts)
    for start in starts:
        end = start + PATCH_SIZE
        data_all = np.concatenate((mix[1:, start:end, :], target[1:, start:end, :]), axis=2)
        data.append(data_all)
    return data

def tf_parse_function(filenames):
    data = tf.py_function(_parse_function, 
                                inp=[filenames], 
                                Tout=tf.float32)
    return data

class DataLoader(object):
    def __init__(self, data_dir, data_dir2 ,train_list=None, train_list2=None, val_list=None):
        self.train_list = train_list
        self.val_list = val_list

        if train_list:
            self.train = read_list(data_dir, train_list)
            self.train_size = len(self.train)

        if train_list2:
            train2 = read_list(data_dir2, train_list2)
            self.train.extend(train2)
            self.train_size = len(self.train)

        if val_list:
            self.val = read_list(data_dir, val_list)
            self.val_size = len(self.val)
    
    def create_tf_dataset(self, flags):
        train_dataset, val_dataset = None, None
        if self.train_list:
            '''Prepare for training dataset'''
            train_dataset = tf.data.Dataset.from_tensor_slices((self.train))
            train_dataset = train_dataset.map(lambda x: tf_parse_function(x), num_parallel_calls=8) #shape=(512, 128, 2)
            #for x in train_dataset: 
            #    print(x.shape)
            # Preprocessing
            train_dataset = train_dataset.shuffle(buffer_size=self.train_size)
            train_dataset = train_dataset.batch(flags.batch_size, drop_remainder=True)
        
        if self.val_list:
            '''Prepare for validation dataset'''
            val_dataset = tf.data.Dataset.from_tensor_slices((self.val))
            val_dataset = val_dataset.map(lambda x: tf_parse_function(x), num_parallel_calls=8)
            val_dataset = val_dataset.batch(1)
        
        return train_dataset, val_dataset