import imageio
import tempfile
import os
import numpy as np

def channelwise_enhance_contrast(image_matrix, bins):
    image_flattened = image_matrix.flatten()
    image_hist = np.zeros(bins)

    # frequency count of each pixel
    for pix in image_matrix:
        image_hist[pix] += 1

    # cummulative sum
    cum_sum = np.cumsum(image_hist)
    norm = (cum_sum - cum_sum.min()) * 255
    # normalization of the pixel values
    n_ = cum_sum.max() - cum_sum.min()
    uniform_norm = norm / n_
    uniform_norm = uniform_norm.astype('int')

    # flat histogram
    image_eq = uniform_norm[image_flattened]
    # reshaping the flattened matrix to its original shape
    image_eq = np.reshape(a=image_eq, newshape=image_matrix.shape)

    return image_eq

def enhance_contrast(image_matrix, bins=256):
    if len(image_matrix.shape) == 2 :
        return channelwise_enhance_contrast( image_matrix, bins )
    if len(image_matrix.shape) == 4 :
        image_matrix = image_matrix[:3]
    image_matrix[:,:,0] = channelwise_enhance_contrast( image_matrix[:,:,0], bins )
    image_matrix[:,:,1] = channelwise_enhance_contrast( image_matrix[:,:,1], bins )
    image_matrix[:,:,2] = channelwise_enhance_contrast( image_matrix[:,:,2], bins )
    return image_matrix


def implementation( image_widget ):
    img_path = image_widget.get_snapshot_file()
    img = imageio.imread( img_path )
    enh = enhance_contrast( img )
    tmp_file = os.path.join( tempfile.gettempdir(), 'rgb2gray_cache.png' )
    imageio.imwrite( tmp_file, np.asarray( enh, dtype='uint8' ) )
    image_widget.update_content_file( tmp_file )

def interface():
    def detailed_implementation( image_widget ):
        def fun():
            return implementation( image_widget )
        return fun

    return 'ContrastEnhance', detailed_implementation

