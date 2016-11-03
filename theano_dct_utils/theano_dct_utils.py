
import numpy as np
import theano
import theano.tensor as T

def numpy_dct_matrix(N):
    '''
    Computes the NxN DCT matrix. This function is adapted from dctmtx in Matlab
    1D DCT can be performed as follows:
    x = np.random.random(10, 1)
    D = theano_dct_matrix(10) # Shape = (10, 10)
    dct_coefs = D.dot(x)
    :param N: size of the DCT Matrix
    :return: Matrix containing DCT matrix.
    '''
    x = np.linspace(0, N - 1, N)
    y = np.linspace(0, N - 1, N)
    cc, rr = np.meshgrid(x, y)
    c = np.sqrt(2.0 / float(N)) * np.cos(np.pi * (2.0 * cc + 1.0) * (rr / (2.0 * float(N))))
    c[0, :] = c[0,:] / np.sqrt(2.0)
    return c


def theano_dct_matrix(N):
    '''
    Computes the NxN DCT matrix. This function is adapted from dctmtx in Matlab
    1D DCT can be performed as follows:
    x = <Theano Vector>
    D = theano_dct_matrix(10)
    dct_coefs = D.dot(x)
    :param N: size of the DCT Matrix
    :return: Theano Matrix containing DCT matrix.
    '''
    rr, cc = T.mgrid[0:N, 0:N]
    N = T.cast(N, 'float32')
    c = T.sqrt(2.0 / N) * T.cos(np.pi * (2.0 * cc + 1.0) * (rr / (2.0 * N)))
    c = T.set_subtensor(c[0, :], c[0, :] / T.sqrt(2.0))
    return c


def dct1d(x):
    '''
    Perfoms 2D DCT on Matrix X
    :param x: Theano Matrix containing values on which to perform 2D DCT.
    :return: Theano Matrix containing DCT coefficients
     '''
    n = x.shape[-1]
    D = theano_dct_matrix(n)

    def __dct(x):
        return D.dot(x)

    return __dct(x)


def idct1d(x):
    '''
    Perfoms 1D IDCT on Vector X
    :param x: Theano Vector containing values on which to perform 1D DCT.
    :return: Theano Vector containing DCT coefficients
     '''
    n = x.shape[-1]
    D = theano_dct_matrix(n)

    def __idct(x):
        return (D.T).dot(x)

    return __idct(x)


def dct2d(x):
    '''
    Perfoms 2D DCT on Matrix X
    :param x: Theano Matrix containing values on which to perform 2D DCT.
    :return: Theano Matrix containing DCT coefficients
     '''
    n = x.shape[-1]
    D = theano_dct_matrix(n)

    def __dct(x):
        return D.dot(x).dot(D.T)

    return __dct(x)


def idct2d(x):
    '''
    Perfoms 2D IDCT on Matrix X
    :param x: Theano Matrix containing DCT coefficients on which to perform 2D IDCT.
    :return: Theano Matrix containing values after IDCT
     '''
    n = x.shape[-1]
    D = theano_dct_matrix(n)

    def __idct(x):
        return (D.T).dot(x).dot(D)

    return __idct(x)


def dct3d(x):

    n = x.shape[-1]
    D = theano_dct_matrix(n)

    def __dct(x):

        x = x.dot(D.T)
        x = x.transpose(0, 2, 1).dot(D.T)
        x = x.transpose(2, 1, 0).dot(D.T)
        x = x.transpose(2, 1, 0).transpose(0, 2, 1)
        return x

    return __dct(x)


def idct3d(x):

    n = x.shape[-1]
    D = theano_dct_matrix(n)

    def __idct(x):
        x = x.dot(D)
        x = x.transpose(0, 2, 1).dot(D)
        x = x.transpose(2, 1, 0).dot(D)
        x = x.transpose(2, 1, 0).transpose(0, 2, 1)
        return x

    return __idct(x)


def batch_dct2d(x):
    '''
    Performs 2D DCT on the tensor x. The DCT is performed on the last 2 axes
    :param x: 3D or 4D Tneano tensor
    :return: 3D or 4D Theano tensor containing DCT coefficients in the last 2 axes
    '''
    ndim = x.ndim
    n = x.shape[-1]
    D = theano_dct_matrix(n)

    def __dct(x):
        return D.dot(x).dot(D.T)


    if ndim == 3:
        def __batch_wrapper(x):
            return T.set_subtensor(x, __dct(x))

        x, updates = theano.scan(fn=__batch_wrapper, outputs_info=None, sequences=[x], n_steps=x.shape[0])
    elif ndim == 4:
        def __batch_wrapper(x):
            return T.set_subtensor(x[0], __dct(x[0]))

        x, updates = theano.scan(fn=__batch_wrapper, outputs_info=None, sequences=[x], n_steps=x.shape[0])

    return x


def batch_idct2d(x):
    '''
    Performs 2D IDCT on the tensor x. The IDCT is performed on the last 2 axes
    :param x: 3D or 4D Tneano tensor containing DCT coefficients in the last 2 axes
    :return: 3D or 4D Theano tensor containing original values
    '''

    ndim = x.ndim
    n = x.shape[-1]
    D = theano_dct_matrix(n)

    def __dct(x):
        return (D.T).dot(x).dot(D)

    if ndim == 3:
        def __batch_wrapper(x):
            return T.set_subtensor(x, __dct(x))

        x, updates = theano.scan(fn=__batch_wrapper, outputs_info=None, sequences=[x], n_steps=x.shape[0])
    elif ndim == 4:
        def __batch_wrapper(x):
            return T.set_subtensor(x[0], __dct(x[0]))

        x, updates = theano.scan(fn=__batch_wrapper, outputs_info=None, sequences=[x], n_steps=x.shape[0])

    return x


def batch_dct3d(x):
    '''
    Performs 3D DCT on the tensor x. The DCT is performed on the last 3 axes
    :param x: 4D or 5D Tneano tensor
    :return: 4D or 5D Theano tensor containing DCT coefficients in the last 3 axes
    '''
    ndim = x.ndim
    n = x.shape[-1]
    D = theano_dct_matrix(n)

    def __dct(x):
        x = x.dot(D.T)
        x = x.transpose(0, 2, 1).dot(D.T)
        x = x.transpose(2, 1, 0).dot(D.T)
        x = x.transpose(2, 1, 0).transpose(0, 2, 1)
        return x


    if ndim == 4:
        def __batch_wrapper(x):
            return T.set_subtensor(x, __dct(x))

        x, updates = theano.scan(fn=__batch_wrapper, outputs_info=None, sequences=[x], n_steps=x.shape[0])
    elif ndim == 5:
        def __batch_wrapper(x):
            return T.set_subtensor(x[0], __dct(x[0]))

        x, updates = theano.scan(fn=__batch_wrapper, outputs_info=None, sequences=[x], n_steps=x.shape[0])

    return x


def batch_idct3d(x):
    '''
    Performs 3D IDCT on the tensor x. The IDCT is performed on the last 3 axes
    :param x: 4D or 5D Tneano tensor containing DCT coefficients in the last 3 axes
    :return: 4D or 5D Theano tensor containing original values
    '''

    ndim = x.ndim
    n = x.shape[-1]
    D = theano_dct_matrix(n)

    def __dct(x):
        x = x.dot(D)
        x = x.transpose(0, 2, 1).dot(D)
        x = x.transpose(2, 1, 0).dot(D)
        x = x.transpose(2, 1, 0).transpose(0, 2, 1)
        return x

    if ndim == 4:
        def __batch_wrapper(x):
            return T.set_subtensor(x, __dct(x))

        x, updates = theano.scan(fn=__batch_wrapper, outputs_info=None, sequences=[x], n_steps=x.shape[0])
    elif ndim == 5:
        def __batch_wrapper(x):
            return T.set_subtensor(x[0], __dct(x[0]))

        x, updates = theano.scan(fn=__batch_wrapper, outputs_info=None, sequences=[x], n_steps=x.shape[0])

    return x
