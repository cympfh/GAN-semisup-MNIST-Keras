from keras.datasets import mnist
import numpy


def iota(xs, start, n):
    m = len(xs)
    return [xs[i % m] for i in range(start, start + n)]


def generator_generator(batch_size, data, indices):

    if type(data) == tuple:
        gens = [generator_generator(batch_size, item, indices) for item in data]
        for xs in zip(*gens):
            yield xs
        return

    for cx in range(20000000000000):
        batch_indices = iota(indices, cx * batch_size, batch_size)
        yield numpy.array([data[i] for i in batch_indices])


def batch_generator(labels, unlabels, batch_size=50, aug=False):

    def p(a: str) -> [int]:
        if a is None:
            return [1e20] * 10
        if ',' in a:
            return list(map(int, a.split(',')))
        n = int(a)
        m = n // 10
        return [m] * 10

    # load
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('f') / 255.0
    x_test = x_test.astype('f') / 255.0

    # train labeling
    l_indices = []
    u_indices = []
    l_count = [0] * 10
    u_count = [0] * 10
    l_sup = p(labels)
    u_sup = p(unlabels)

    for i in range(len(x_train)):
        klass = int(y_train[i])
        if l_count[klass] < l_sup[klass]:
            l_indices.append(i)
            l_count[klass] += 1
        elif u_count[klass] < u_sup[klass]:
            u_indices.append(i)
            u_count[klass] += 1

    assert len(l_indices) > 0, 'No items are labeeld. see --labels'
    assert len(u_indices) > 0, 'No items are unlabeled. see --unlabels'
    print("{} items are labeled".format(len(l_indices)))
    print("rest {} item are unlabeled".format(len(u_indices)))

    gen_train_l = generator_generator(batch_size, (x_train, y_train), l_indices)
    gen_train_u = generator_generator(batch_size, x_train, u_indices)
    test = (x_test, y_test)

    return (gen_train_l, len(l_indices)), (gen_train_u, len(u_indices)), (test, len(x_test))
