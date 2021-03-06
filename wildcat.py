import os
import sys
import json
import utils
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


config = utils.load_config()
sys.path.append(os.path.expanduser(config['slim_path']))
resnet = utils.load_module('nets.resnet_v1')
logger = utils.get_default_logger()


class WildCat:
    def __init__(self, images, labels=None, n_classes=None, training=False, transfer_conv_size=(3, 3),
                 n_maps_per_class=5, alpha=1.0, k=1, reg=None):
        self.images = images
        self.labels = labels
        self.n_classes = n_classes
        self.training = training
        self.transfer_conv_size = transfer_conv_size
        self.n_maps_per_class = n_maps_per_class
        self.alpha = alpha
        self.k = k
        self.reg = reg
        # store a snap version of __dict__ for string representation
        self.params = self.__dict__.copy()
        self.build_graph()

    def set_default_session(self, sess):
        self.sess = sess

    def get_default_session(self):
        return self.sess

    def _build_train_op(self):
        # pre-set summary
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # generate training op
        global_step = tf.train.get_or_create_global_step()
        solver = tf.train.AdamOptimizer(config['learning_rate'])
        data_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.labels, logits=self.logits)
        if self.reg:
            reg_loss = self.reg * tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')
        else:
            reg_loss = 0.0
        total_loss = tf.add(data_loss, reg_loss, name='total_loss')
        train_op = slim.learning.create_train_op(total_loss,
                                                 solver,
                                                 global_step=global_step)
        # add loss summary
        summaries.add(tf.summary.scalar('loss/data_loss', data_loss))
        summaries.add(tf.summary.scalar('loss/total_loss', total_loss))

        with tf.name_scope('accuracy'):
            zeros_const = tf.constant(0, dtype=tf.int32, shape=(config['batch_size'], self.n_classes))
            ones_const = tf.constant(1, dtype=tf.int32, shape=(config['batch_size'], self.n_classes))
            output = tf.where(self.logits < 0, zeros_const, ones_const)
            correct = tf.cast(tf.equal(tf.cast(self.labels, dtype=tf.int32), output), dtype=tf.float32)
            accuracy = tf.reduce_mean(correct, name='accuracy')
            accuracy = tf.Print(accuracy, data=[self.logits, tf.reduce_sum(correct)], message='logits')
        summaries.add(tf.summary.scalar('accuracy', accuracy))

        summary_op = tf.summary.merge(list(summaries), name='summary_op')
        return train_op, summary_op

    def train_from_scratch(self, config):
        logger.info('Training from scratch....')
        logger.info(self)
        logger.info('config=\n' + json.dumps(config, indent=2))
        n_steps_per_epoch = int(np.ceil(config['n_examples_for_train'] // config['batch_size']))
        n_steps_for_train = config['n_epochs_for_train'] * n_steps_per_epoch

        # since we're training from scratch,
        # so always remove the existing directory first.
        utils.delete_if_exists(config['train_dir'])

        if not os.path.exists(config['ckpt_path']):
            logger.info('Checkpoint file not exists yet, extracting from %s...' % config['ckpt_tar_path'])
            utils.extract_to(config['ckpt_tar_path'], tempfile.gettempdir())

        train_op, summary_op = self._build_train_op()
        last_loss = slim.learning.train(train_op,
                                        logdir=config['train_dir'],
                                        master=config['master'],
                                        summary_op=summary_op,
                                        init_fn=self._get_init_fn(config),
                                        log_every_n_steps=config['log_every'],
                                        save_summaries_secs=config['save_summaries_secs'],
                                        number_of_steps=n_steps_for_train)
        logger.info('Last loss: %3f' % last_loss)

    def train_from_last_run(self, config):
        logger.info('Training from last run...')

        if not tf.train.latest_checkpoint(config['train_dir']):
            logger.warning('No checkpoint file found in %s' % config['train_dir'])

        n_steps_per_epoch = int(np.ceil(config['n_examples_for_train'] // config['batch_size']))
        n_steps_for_train = config['n_epochs_for_train'] * n_steps_per_epoch
        train_op, summary_op = self._build_train_op()
        last_loss = slim.learning.train(train_op,
                                        logdir=config['train_dir'],
                                        master=config['master'],
                                        summary_op=summary_op,
                                        log_every_n_steps=config['log_every'],
                                        save_summaries_secs=config['save_summaries_secs'],
                                        number_of_steps=n_steps_for_train)
        logger.info('Last loss: %3f' % last_loss)

    def _get_init_fn(self, config):
        if config['ckpt_path'] is None:
            return None

        if tf.train.latest_checkpoint(config['train_dir']):
            logger.warning('Ignoring --checkpoint_path because a checkpoint already exists in %s' % config['train_dir'])
            return None

        # TODO
        # need more elegant way to get all the variables to recover
        variables_to_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='resnet')
        return slim.assign_from_checkpoint_fn(config['ckpt_path'],
                                              variables_to_restore,
                                              ignore_missing_vars=False)

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
                                scope='multi_map_transfer',
                                activation_fn=None)
        class_pool = utils.class_wise_pooling(multi_map, self.n_maps_per_class, scope='class_pool')
        spatial_pool = utils.spatial_pooling(class_pool, self.k, alpha=self.alpha, scope='spatial_pool')
        self.logits = spatial_pool
        self.probs = tf.nn.sigmoid(self.logits, name='probs')
        # record the intermediate nodes
        endpoints['multi_map'] = multi_map
        endpoints['class_pool'] = class_pool
        endpoints['spatial_pool'] = spatial_pool
        endpoints['logits'] = self.logits
        endpoints['probs'] = self.probs
        self.endpoints = endpoints
        logger.info('Done graph building!')

    def __repr__(self):
        ret = '\n'
        for i, (key, value) in enumerate(self.endpoints.items()):
            ret += '{:>2} {:<50}{}\n'.format(i, key, value.shape)
        for key, value in self.params.items():
            ret += '{:<20}= {}\n'.format(key, value)
        return ret


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default() as g:
        n_classes = 10
        np.random.seed(0)
        xx = np.random.rand(10, 64, 64, 3)
        yy = np.random.randint(2, size=(10, n_classes))
        x = tf.constant(xx, dtype=tf.float32)
        y = tf.constant(yy, dtype=tf.float32)
        logger.info('yy: %s' % yy)
        # images, labels = tf.train.batch([x, y], batch_size=config['batch_size'], enqueue_many=True)
        model = WildCat(x, y, n_classes=n_classes)
        model.train_from_scratch(config)
