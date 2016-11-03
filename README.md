# theano_dct_utils
Small package for computing Discrete Cosine Transform (DCT) coefficients in Theano.

All functions are orthonormalized and tested against scipy.fftpack implementations. There are accurate to approx 1e-6 of the scipy imeplemntations. 

##Install 
```
python setup.py build
python setup.py install --user
```
#API

* numpy_dct_matrix(M)

    Computes the NxN DCT matrix. This function is adapted from dctmtx in Matlab
    1D DCT can be performed as follows:
    ```python
    x = <Theano Vector>
    D = theano_dct_matrix(10)
    dct_coefs = D.dot(x)
    ```
    param N: size of the DCT Matrix
    return: Theano Matrix containing DCT matrix.

* theano_dct_matrix(N):
    
    Computes the NxN DCT matrix. This function is adapted from dctmtx in Matlab
    1D DCT can be performed as follows:
    ```python
    x = <Theano Vector>
    D = theano_dct_matrix(10)
    dct_coefs = D.dot(x)
    ```
    param N: size of the DCT Matrix
    return: Theano Matrix containing DCT matrix.
    

* dct1d(x):
    Perfoms 2D DCT on Matrix X
    param x: Theano Matrix containing values on which to perform 2D DCT.
    return: Theano Matrix containing DCT coefficients


* idct1d(x):
    Perfoms 1D IDCT on Vector X
    param x: Theano Vector containing values on which to perform 1D DCT.
    return: Theano Vector containing DCT coefficients


* dct2d(x):
    Perfoms 2D DCT on Matrix X
    param x: Theano Matrix containing values on which to perform 2D DCT.
    return: Theano Matrix containing DCT coefficients


* idct2d(x):
    Perfoms 2D IDCT on Matrix X
    param x: Theano Matrix containing DCT coefficients on which to perform 2D IDCT.
    return: Theano Matrix containing values after IDCT

* dct3d(x):
    Perfoms 3D DCT on tensor3 X
    param x: Theano tensor3 containing values on which to perform 3D DCT.
    return: Theano tensor3 containing DCT coefficients


* idct3d(x):
    Perfoms 3D IDCT on tensor3 X
    param x: Theano tensor3 containing DCT coefficients on which to perform 3D IDCT.
    return: Theano tensor3 containing values after IDCT



* batch_dct2d(x):
    Performs 2D DCT on the tensor x. The DCT is performed on the last 2 axes
    param x: 3D or 4D Tneano tensor
    return: 3D or 4D Theano tensor containing DCT coefficients in the last 2 axes


* batch_idct2d(x):
    Performs 2D IDCT on the tensor x. The IDCT is performed on the last 2 axes
    param x: 3D or 4D Tneano tensor containing DCT coefficients in the last 2 axes
    return: 3D or 4D Theano tensor containing original values


* batch_dct3d(x):
    Performs 3D DCT on the tensor x. The DCT is performed on the last 3 axes
    param x: 4D or 5D Tneano tensor
    return: 4D or 5D Theano tensor containing DCT coefficients in the last 3 axes


* batch_idct3d(x):
    Performs 3D IDCT on the tensor x. The IDCT is performed on the last 3 axes
    param x: 4D or 5D Tneano tensor containing DCT coefficients in the last 3 axes
    return: 4D or 5D Theano tensor containing original values
