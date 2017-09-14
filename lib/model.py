from keras import backend as K
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten, Lambda,
                          Reshape, UpSampling2D)
from keras.models import Sequential
from keras.optimizers import Adam

X_SHAPE = (28, 28)
Z_DIM = 128


def build_conv():
    model = Sequential(name='conv')
    model.add(Reshape(X_SHAPE + (1,), input_shape=X_SHAPE))
    model.add(Conv2D(64, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Dropout(.3))
    model.add(Conv2D(128, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Dropout(.3))
    model.add(Flatten())
    model.add(Dense(10))
    return model


def build_classifier(conv):
    model = Sequential(name='classifier')
    model.add(conv)
    model.add(Activation('softmax'))
    return model


def build_discriminator(conv):
    model = Sequential(name='discriminator')
    model.add(conv)

    def predict(y):
        p = 1.0 - (1.0 / (K.sum(K.exp(y), axis=-1, keepdims=True) + 1.0))
        return p

    model.add(Lambda(predict))
    return model


def build_gen():
    model = Sequential(name='gen')
    model.add(Dense(128 * 7 * 7, activation='relu', input_shape=(Z_DIM,)))
    model.add(Reshape((7, 7, 128)))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(1, (5, 5), activation='sigmoid', padding='same'))
    model.add(Reshape(X_SHAPE))
    return model


def build_gan(gen, dis):
    model = Sequential(name='gan')
    model.add(gen)
    model.add(dis)
    return model


def build():

    conv = build_conv()
    clf = build_classifier(conv)
    dis = build_discriminator(conv)
    gen = build_gen()

    opt = Adam(clipvalue=1.0, lr=0.0002, beta_1=0.5)
    opt_weak = Adam(clipvalue=0.1, lr=0.0001)

    clf_loss = 'sparse_categorical_crossentropy'
    metrics = ['acc']
    clf.compile(loss=clf_loss, metrics=metrics, optimizer=opt_weak)

    dis_loss = 'binary_crossentropy'
    dis.compile(loss=dis_loss, optimizer=opt)

    dis.trainable = False
    gan = build_gan(gen, dis)
    gan_loss = 'binary_crossentropy'
    gan.compile(loss=gan_loss, optimizer=opt)

    return clf, dis, gen, gan


# print('model building...')
# clf, dis, gen, dg = build()
# print('Models')
# print('clf')
# clf.summary()
# print('dis')
# dis.summary()
# print('gen')
# gen.summary()
# print('dg')
# dg.summary()
