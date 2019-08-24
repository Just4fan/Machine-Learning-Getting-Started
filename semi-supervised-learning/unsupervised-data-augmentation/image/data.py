import numpy as np
import tensorflow as tf
from autoaugment import policies as found_policies
from autoaugment import augmentation_transforms
from tensorflow.python.keras.datasets import cifar10

path = 'datasets/'


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def write_data():
    sup_x = np.load('datasets/sup_x.npy')
    sup_y = np.load('datasets/sup_y.npy')
    unsup_x = np.load('datasets/unsup_x.npy')
    aug_x = np.load('datasets/aug_x.npy')
    sup_y = sup_y.reshape(sup_y.shape[0])
    sup_x = sup_x.reshape(sup_x.shape[0], sup_x.shape[1] * sup_x.shape[2] * sup_x.shape[3])
    unsup_x = unsup_x.reshape(unsup_x.shape[0], unsup_x.shape[1] * unsup_x.shape[2] * unsup_x.shape[3])
    aug_x = aug_x.reshape(aug_x.shape[0], aug_x.shape[1] * aug_x.shape[2] * aug_x.shape[3])
    sup_filename = 'sup.tfrecord'
    writer = tf.python_io.TFRecordWriter(path + sup_filename)
    for (x, y) in zip(sup_x, sup_y):
        example = tf.train.Example(
            features=tf.train.Features(feature={
                "image": _float_list_feature(x),
                "label": _int64_feature(y)
            })
        )
        writer.write(example.SerializeToString())
    unsup_filename = 'unsup.tfrecord'
    writer = tf.python_io.TFRecordWriter(path + unsup_filename)
    for (x, y) in zip(unsup_x, aug_x):
        example = tf.train.Example(
            features=tf.train.Features(feature={
                "unsup": _float_list_feature(x),
                "aug": _float_list_feature(y)
            })
        )
        writer.write(example.SerializeToString())


def data_augmentation(unsup):
    augs = []
    unsup = unsup / 255.0
    mean, std = augmentation_transforms.get_mean_and_std()
    unsup = (unsup - mean) / std
    aug_policies = found_policies.cifar10_policies()
    for image in unsup:
        chosen_policy = aug_policies[np.random.choice(
            len(aug_policies))]
        aug = augmentation_transforms.apply_policy(
            chosen_policy, image)
        aug = augmentation_transforms.cutout_numpy(aug)
        augs.append(aug)
    return np.array(augs), unsup


'''
生成增强样本 aug_copy: 单个样本增强的数量
'''


def generate_data(aug_copy):
    (unsup_x, _), (_, _) = cifar10.load_data()
    for i in range(aug_copy):
        aug, unsup = data_augmentation(unsup_x)
        unsup = unsup.reshape(unsup.shape[0], unsup.shape[1] * unsup.shape[2] * unsup.shape[3])
        aug = aug.reshape(aug.shape[0], aug.shape[1] * aug.shape[2] * aug.shape[3])
        unsup_filename = 'unsup_{0}.tfrecord'.format(i)
        writer = tf.python_io.TFRecordWriter(path + unsup_filename)
        for (x, y) in zip(unsup, aug):
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    "unsup": _float_list_feature(x),
                    "aug": _float_list_feature(y)
                })
            )
            writer.write(example.SerializeToString())


def load_data(features, filename, batch_size, buffer_size):
    def process(example):
        for key in example.keys():
            val = example[key]
            if val.dtype == tf.int64:
                val = tf.to_int32(val)
            if tf.keras.backend.is_sparse(val):
                val = tf.keras.backend.to_dense(val)
            example[key] = val

        return example

    def parser(example_proto):
        example = tf.parse_single_example(
            serialized=example_proto,
            features=features
        )
        for key in example.keys():
            if key == 'image' or key == 'unsup' or key == 'aug':
                example[key] = tf.reshape(example[key], [32, 32, 3])
            else:
                example[key] = tf.reshape(example[key], ())
        process(example)

        return example

    dataset = tf.data.TFRecordDataset([filename])
    dataset = dataset.map(parser)

    return dataset.shuffle(buffer_size).repeat().batch(batch_size, drop_remainder=True)


def input_fn(batch_size, ratio, is_train=True, aug_copy=10):
    # write_data()
    sup_dataset = load_data(
        features={
            "image": tf.FixedLenFeature([32 * 32 * 3], tf.float32),
            "label": tf.FixedLenFeature([1], tf.int64)
        },
        filename=path + "sup.tfrecord",
        batch_size=batch_size,
        buffer_size=4000
    )
    if not is_train:
        return sup_dataset

    unsup_dataset = load_data(
        features={
            "unsup": tf.FixedLenFeature([32 * 32 * 3], tf.float32),
            "aug": tf.FixedLenFeature([32 * 32 * 3], tf.float32)
        },
        filename=[path + "unsup_{0}.tfrecord".format(i) for i in range(aug_copy)],
        batch_size=ratio * batch_size,
        buffer_size=100000
    )

    return tf.data.Dataset.zip((sup_dataset, unsup_dataset))

# if __name__ == '__main__':
#     generate_data(10)
