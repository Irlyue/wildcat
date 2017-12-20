import os
import utils
import tensorflow as tf
import numpy as np

from collections import defaultdict


logger = utils.get_logger('root')
all_classes = """person
bird cat cow dog horse sheep
aeroplane bicycle boat bus car motorbike train
bottle chair diningtable pottedplant sofa tvmonitor
""".split()


def read_image_and_label_list(type_, data_dir, list_dir):
    """
    :param type_: str,
    :param data_dir:
    :param list_dir:
    :return:
        image_list: list, a list of full path with each item for one image
        label_list: list, a list of list containing class labels, indicating the specific classes in an image
    """
    logger.info('Reading list file for images...')
    image_list = []
    labels = []
    if type_ == 'test':
        with open(os.path.join(list_dir, type_ + '.txt')) as f:
            for line in f:
                full_path = os.path.join(data_dir, line + '.jpg')
                image_list.append(full_path)
    elif type_ == 'train' or type_ == 'val' or type_ == 'trainval':
        all_labels = defaultdict(list)
        # iterate list file of all classes
        for i, cls_name in enumerate(all_classes):
            with open(os.path.join(list_dir, cls_name + '_' + type_ + '.txt')) as fin:
                for line in fin:
                    file_name, label = line.split()
                    if int(label) == 1:
                        all_labels[file_name].append(i)
        labels = np.zeros((len(all_labels), len(all_classes)))
        for i, key in enumerate(all_labels):
            image_list.append(os.path.join(data_dir, key + '.jpg'))
            labels[i, all_labels[key]] = 1
    logger.info('Done list reading(%s), %d images in total!' % (type_, len(image_list)))
    return image_list, labels


def read_image_from_disk(queue):
    img = tf.read_file(queue[0])
    img = tf.image.decode_jpeg(img, channels=3)
    if len(queue) > 1:
        return img, queue[1]
    return img


class ImageReader:
    def __init__(self, data_dir, list_dir, type_, prep_func=None, shuffle=False):
        """
        Image reader for VOC2012 data set.

        :param data_dir: str, directory for the JPEG images.
        :param list_dir: str, path to directory containing the list file(e.g. train.txt).
        :param type_: str, one of 'train', 'val', 'test'.
        :param prep_func: (Tensor)-> Tensor, pre-process function
        :param shuffle: boolean,
        """
        image_list, labels = read_image_and_label_list(type_, data_dir, list_dir)
        images = tf.convert_to_tensor(image_list, dtype=tf.string)
        if type_ == 'test':
            queue = tf.train.slice_input_producer([images], shuffle=False)
        elif type_ == 'train' or type_ == 'val' or type_ == 'trainval':
            queue = tf.train.slice_input_producer([images, labels], shuffle=shuffle)
        image, label = read_image_from_disk(queue)
        if prep_func:
            image = prep_func(image)
        self.type_ = type_
        self.image = image
        self.label = label

    def data_batch(self, batch_size):
        """
        Generate the image batch and label batch.
        WARNING. If the images are not re-sized into a fixed size, call to this function may cause trouble.

        :param batch_size: int,
        :return:
            batch_images: Tensor, containing the image batch with shape(batch_size, h, w, 3)
            batch_labels: Tensor, containing the label batch with shape(batch_size, n_classes)
        """
        if self.type_ == 'test':
            return tf.train.batch([self.image], batch_size=batch_size)
        elif self.type_ == 'train' or self.type_ == 'val':
            return tf.train.batch([self.image, self.label], batch_size=batch_size)
        else:
            logger.error('Data type `%s` undefined!' % self.type_)
            raise NotImplemented


if __name__ == '__main__':
    print(len(all_classes))
    print(all_classes)
