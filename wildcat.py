import os
import sys
import utils
import tensorflow as tf
import tensorflow.contrib.slim as slim


config = utils.load_config()
sys.path.append(os.path.expanduser(config['slim_path']))
resnet = utils.load_module('nets.resnet_v1')
logger = utils.get_logger('root')


class WildCat:
    def __init__(self, images, labels=None, n_classes=None, training=False, transfer_conv_size=(3, 3),
                 n_maps_per_class=5, alpha=1.0, learning_rate=1e-3, k=1):
        self.images = images
        self.labels = labels
        self.n_classes = n_classes
        self.training = training
        self.transfer_conv_size = transfer_conv_size
        self.n_maps_per_class = n_maps_per_class
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.k = k
        self.build_graph()

    def train_one_step(self, verbose=True):
        sess = tf.get_default_session()
        loss, _ = sess.run([self.loss, self.train_op])
        if verbose:
            return loss

    def train_from_scratch(self, config=None):
        n_epochs_for_train = config['n_epochs_for_train']
        log_every = config['log_every']
        with tf.Session() as sess:
            self.initialize()
            for step in range(n_epochs_for_train):
                loss_val = self.train_one_step()
                if step % log_every:
                    logger.info('Step %d, loss %.2f' % (step, loss_val))

    def initialize(self):
        pass

    def build_graph(self):
        logger.info('Building graph...')
        global_step = tf.train.get_or_create_global_step()
        with slim.arg_scope(resnet.resnet_arg_scope()):
            conv5, endpoints = resnet.resnet_v1_50(self.images,
                                                   is_training=self.training,
                                                   num_classes=None,
                                                   global_pool=False)
        multi_map = slim.conv2d(conv5,
                                num_outputs=self.n_maps_per_class*self.n_classes,
                                kernel_size=self.transfer_conv_size,
                                scope='multi_map_transfer')
        class_pool = utils.class_wise_pooling(multi_map, self.n_maps_per_class)
        spatial_pool = utils.spatial_pooling(class_pool, self.k, alpha=self.alpha)
        self.logits = spatial_pool
        self.probs = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.labels, name='probs')
        self.loss = tf.reduce_mean(self.probs, name='data_loss')
        solver = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = solver.minimize(self.loss, global_step=global_step)
        # record the intermediate nodes
        endpoints['multi_map'] = multi_map
        endpoints['class_pool'] = class_pool
        endpoints['spatial_pool'] = spatial_pool
        endpoints['logits'] = self.logits
        endpoints['probs'] = self.probs
        self.endpoints = endpoints
        logger.info('Done graph building!')

    def load_pretrained_resnet(self):
        pass

    def __repr__(self):
        ret = ''
        for i, (key, value) in enumerate(self.endpoints.items()):
            ret += '{} {}{}\n'.format(i, key, value.shape)
        return ret


if __name__ == '__main__':
    with tf.Graph().as_default() as g:
        n_classes = 10
        images = tf.placeholder(tf.float32, shape=(32, 64, 64, 3))
        labels = tf.placeholder(tf.float32, shape=(32, n_classes))
        model = WildCat(images, labels, n_classes=n_classes)
        writer = tf.summary.FileWriter('/tmp/wildcat', graph=g)
        logger.info(model)
