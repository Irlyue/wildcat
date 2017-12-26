import utils
import argparse
import inputs.voc2012 as voc2012
from wildcat import WildCat
from inputs.preprocessing import default_prep


parser = argparse.ArgumentParser()
parser.add_argument('--type', default='recover', type=str,
                    help='Must be one of ["recover", "rerun"], whether to recover from last run.')
parser.add_argument('--dataset', default='voc2012', type=str,
                    help='Specify the data set.')


def main():
    config = utils.load_config()
    input_image_size = config['input_image_size']
    if type(input_image_size) != list:
        input_image_size = (input_image_size, input_image_size)
    data = voc2012.ImageReader(config['data_dir'], config['list_dir'],
                               type_=config['data_type'],
                               prep_func=lambda x: default_prep(x, input_image_size),
                               shuffle=False)
    images, labels = data.data_batch(config['batch_size'])
    model = WildCat(images, labels,
                    n_classes=config['n_classes'],
                    training=True,
                    n_maps_per_class=config['n_feature_maps_per_class'],
                    alpha=config['alpha'],
                    k=config['k'],
                    reg=config['reg'])
    assert FLAGS.type in ['rerun', 'recover'], "--type parameter not allowed"
    if FLAGS.type == 'rerun':
        model.train_from_scratch(config)
    elif FLAGS.type == 'recover':
        model.train_from_last_run(config)


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    main()

