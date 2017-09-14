import sys
import click
import tensorflow as tf
from keras import backend as K
import numpy
import os
import os.path

import dataset
import lib.log
import lib.model
import lib.test


def echo(*args):
    click.secho(' '.join(str(arg) for arg in args), fg='green', err=True)


@click.group()
def main():
    pass


@main.command()
@click.option('--name', help='model name')
@click.option('--epochs', default=100, type=int)
@click.option('--batch_size', default=100, type=int)
@click.option('--labels', '-l', default="100")
@click.option('--unlabels', '-u', default=None)
@click.option('--noise-type', default='uniform', type=click.Choice(['normal', 'uniform']))
@click.option('--resume', help='when resume learning from the snapshot')
def train(name, epochs, batch_size, labels, unlabels, noise_type, resume):

    # paths
    log_path = "logs/{}.json".format(name)
    out_path = "snapshots/" + name + ".{epoch:06d}.h5"
    echo('log path', log_path)
    echo('out path', out_path)
    result_dir = "result/{}".format(name)
    echo('result images', result_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    lib.log.info(log_path, {'_commandline': {
        'name': name,
        'epochs': epochs,
        'batch_size': batch_size,
        'labels': labels,
        'unlabels': unlabels,
        'noise_type': noise_type,
        'resume': resume
    }})

    # init
    echo('train', (name, resume))
    session = tf.Session('')
    K.set_session(session)
    K.set_learning_phase(1)

    # dataset
    echo('dataset loading...')
    train_l, train_u, test = dataset.batch_generator(labels, unlabels, batch_size=batch_size)
    gen_train_l, num_train_l = train_l
    gen_train_u, num_train_u = train_u
    data_test, num_test = test

    TRUE = numpy.ones((batch_size,)) - .1
    FALSE = numpy.zeros((batch_size,))

    # model building
    echo('model building...')
    clf, dis, gen, dg = lib.model.build()
    echo('Models')
    echo('clf')
    clf.summary()
    echo('dis')
    dis.summary()
    echo('gen')
    gen.summary()
    echo('dg')
    dg.summary()

    def make_noise():
        if noise_type == 'normal':
            return numpy.random.randn(batch_size, lib.model.Z_DIM)
        else:
            return numpy.random.normal(0, 1, size=[batch_size, lib.model.Z_DIM])

    # training
    echo('start learning...')
    numpy.random.seed(42)
    for _epoch in range(epochs):

        INTERVAL = 20
        clf_loss = 0
        clf_acc = 0
        dis_true_loss = 0
        dis_false_loss = 0
        gen_loss = 0
        last_log = None

        m = max(num_train_l, num_train_u) // batch_size
        INTERVAL = min(INTERVAL, m)
        for i in range(m):

            # classifier
            X, Y = next(gen_train_l)
            loss, acc = clf.train_on_batch(X, Y)
            clf_loss += loss
            clf_acc += acc

            # discriminator for true
            X = next(gen_train_u)
            dis_true_loss += dis.train_on_batch(X, TRUE)

            # discriminator for fake
            z = make_noise()
            x_fake = gen.predict_on_batch(z)
            dis_false_loss += dis.train_on_batch(x_fake, FALSE)

            # generator to mislead
            dis.trainable = False
            z = make_noise()
            gen_loss += dg.train_on_batch(z, TRUE)
            dis.trainable = True

            if i % INTERVAL == INTERVAL - 1:
                if i > INTERVAL:
                    sys.stdout.write('\r')

                clf_loss /= INTERVAL
                clf_acc /= INTERVAL
                dis_true_loss /= INTERVAL
                dis_false_loss /= INTERVAL
                gen_loss /= INTERVAL
                last_log = (clf_loss, clf_acc, dis_true_loss, dis_false_loss, gen_loss)

                sys.stdout.write(
                    "Epoch {} | train: clf={:.4f} (acc={:.4f}) dis_true={:.4f} dis_false={:.4f} gen={:.4f} ".format(
                        _epoch + 1,
                        clf_loss,
                        clf_acc,
                        dis_true_loss,
                        dis_false_loss,
                        gen_loss))
                clf_loss = 0
                clf_acc = 0
                dis_true_loss = 0
                dis_false_loss = 0
                gen_loss = 0

        val_clf_loss, val_clf_acc = clf.test_on_batch(data_test[0], data_test[1])
        print("| val: clf={:.4f} (acc={:.4f})".format(val_clf_loss, val_clf_acc))
        lib.test.image_save(x_fake, '{base}/{epoch:03d}.{{i:03d}}.png'.format(base=result_dir, epoch=_epoch))
        lib.log.write(log_path, {
            'epoch': _epoch + 1,
            'clf_loss': float(last_log[0]),
            'clf_acc': float(last_log[1]),
            'dis_true_loss': float(last_log[2]),
            'dis_false_loss': float(last_log[3]),
            'gen_loss': float(last_log[4]),
            'val_clf_loss': float(val_clf_loss),
            'val_clf_acc': float(val_clf_acc),
        })


@main.command()
@click.argument('snapshot')
def test(snapshot):

    # init
    echo('test', (snapshot,))
    session = tf.Session('')
    K.set_session(session)
    K.set_learning_phase(0)

    # model loading
    model = lib.model.build()
    model.load_weights(snapshot)


if __name__ == '__main__':
    main()
