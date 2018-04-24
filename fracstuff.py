import numpy as np
import pandas as pd
from scipy.ndimage.filters import convolve, maximum_filter, minimum_filter
from scipy.stats import linregress

######## Utils ########

def rgb2gray(rgb):
    """
    Transform rbg image to grayscale
    """
    return np.dot(rgb[...,:3], [1/3, 1/3, 1/3])

######## Capacitarian (емкостная) dimension ########

def fractal_dimension(immat, ws=range(1, 10)) -> float:
    """
    Calculate classical fractal dimension of binary Image
    :param immat: binary image
    :param ws:    size of subsets of Image
    :rparam:      return fractal dimension
    """
    # Задаем размер квадрата, которым будем ходить по изображению
    # К-во квадратов, куда попал черный пиксель
    ns = []
    for w in ws:
        ns.append(np.sum(maximum_filter(immat, (w, w), mode='constant')[::w, ::w]))

    x = np.log(1/np.array(ws))
    y = np.log(ns)

    return linregress(x, y).slope

######## Renyi spectre ########

def reni_entropy(p : float, q : int):
    """
    Calculate renyi entropy
    :param p: vector of probabilities
    :param q: integer number that allows this entropy to be universal
    """
    if q != 1:
        entropy = (1 / (1 - q) * np.log(np.sum(np.power(p, q))))
    else:
        entropy = -np.sum(p * np.log(q))
    return entropy

def get_reni_dim(immat, q : int):
    """
    Calculate renyi dimension.

    does not work for q = 1
    :param immat: numpy array, grayscale representation of matrix
    :q: some int number
    """

    ws = range(1, 20)
    ns =[]

    for w in ws:
        conv = convolve(immat, np.ones((w, w)), mode='constant')[::w, ::w]
        ns.append(reni_entropy(conv / np.sum(conv), q))

    x = -np.log(ws)
    y = ns

    return linregress(x, y).slope

def get_reni_spectre(immat, qs):
    """
    Calculate renyi spectre

    `immat` should be preprocessed with `rgb2gray`
    """
    if immat.ndim != 2:
        raise Exception("Image matrix should have 2 dimensions, preprocess image " /
                        "with `rgb2gray`")

    return qs, list(map(lambda x: get_reni_dim(immat, x), qs))


######## Fractal signature ########

def fractal_volume(imar, d_=10) -> list:
    """
    Calculate fractal volume of image.

    :param imar: grayscale image
    :param d_:   number of volumes to compute
    """
    u = imar.copy()
    b = imar.copy()

    footprint=np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
    ds = range(1, d_)
    vols = []

    for d in ds:
        fst_u = u + 1
        fst_b = b - 1

        scnd_u = maximum_filter(u, mode='constant', footprint=footprint, cval=0)
        scnd_b = minimum_filter(b, mode='constant', footprint=footprint, cval=255)

        u = np.maximum(fst_u, scnd_u)
        b = np.minimum(fst_b, scnd_b)

        vols.append(np.sum(u - b))

    return vols

def __fractal_signature(imar, d_=10) -> float:
    """
    Calculate fractal signature of subset of image.

    :param imar: subset of original image. image should be array of ints.
    :param d_:   number of volumes to Calculate. Should be greater than 2
    """
    vols = fractal_volume(imar, d_)
    return (vols[-1] - vols[-2]) / 2

def fractal_signature(imar, epsilons=range(4, 30, 2), d=10) -> list:
    """
    Calculate fractal signature over subset of image.

    :param imar:     grayscale image
    :param epsilons: itrable for sizes of image subsets
    :param d:        number of volumes to Calculate. Should be greater than 2
    :rparam:         list of local signatures
    """
    local_signatures = []

    for eps in epsilons:
        ads = 0
        for start1, end1 in zip(range(0, imar.shape[0]-eps, eps), range(eps, imar.shape[0], eps)):
            for start2, end2 in zip(range(0, imar.shape[1]-eps, eps), range(eps, imar.shape[1], eps)):
                ads += __fractal_signature(imar[start1:end1, start2:end2], d)

        local_signatures.append(ads)

    return local_signatures

######## Multifractal signature ########
def get_multifractal_dimension_image(immat, rs=range(2, 10)):
    """
    Calculate multifractal dimension

    This is rather long operations, take your time

    :param immat: grayscale image
    :param rs:    radiuses of image subsets
    :rparam:      matrix of size of original image where values are multifractal
                  dims
    """
    dxs = []
    for r in rs:
        dx = convolve(immat, np.ones((r*2, r*2)), mode='constant')
        dxs.append(dx.ravel())

    frac_dims = []
    for l in np.array(dxs).T:
        frac_dims.append(linregress(np.log(rs), np.log(l)).slope)

    frac_im = np.array(frac_dims).reshape(immat.shape)
    return frac_im

def multifractal_spectre(immat, rs=range(2, 10)) -> list:
    """
    Calculate multifractal spectre

    :param immat: grayscale image
    :param rs:    radiuses of image subsets for multifractal dimension
    :rparam:      list of fractal dimensions
    """
    frac_im = get_multifractal_dimension_image(immat, rs)

    frac_spec = []
    alphas = np.linspace(np.min(frac_dims), np.max(frac_dims), num=15)
    for a0, a1 in zip(alphas[:-1], alphas[1:]):
        frac_im = (frac_im > a0) & (frac_im < a1)
        frac_spec.append(fractal_dimension(frac_im))

    return frac_spec
