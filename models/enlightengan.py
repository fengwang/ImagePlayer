import os
import cv2
import numpy as np
from onnxruntime import InferenceSession
import tempfile
import sys
import imageio
import string
import random

model = None
# load model from local dir
def get_model( model_path ):
    global model
    if model is None:
        model = InferenceSession( model_path )
    return model

# pad image to meet the dimension constrain
def pad(img):
    h, w, _ = img.shape
    block_size = 16
    min_height = (h // block_size + 1) * block_size
    min_width = (w // block_size + 1) * block_size
    img = np.pad(img, ((0, min_height - h), (0, min_width - w), (0, 0)), mode='constant', constant_values=0)
    return img, (h, w)

# channel-first ....
def preprocess( img ):
    return np.expand_dims(np.transpose(img, (2, 0, 1)).astype(np.float32) / 255., 0)

# inference
def predict( image_path, model_path ):
    #img = cv2.imread( image_path )
    img = imageio.imread( image_path )
    padded, (h, w) = pad(img)
    model = get_model( model_path )
    image_numpy, = model.run(['output'], {'input': preprocess(padded)})
    image_numpy = (np.transpose(image_numpy[0], (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    return image_numpy.astype('uint8')[:h, :w, :]

# process current UI image
def implementation( image_widget ):
    local_model_path = image_widget.download_remote_model( 'enlightenGAN', 'https://github.com/fengwang/ImagePlayer/releases/download/enlightengan/enlightenGAN.onnx' )

    img_path = image_widget.get_snapshot_file()
    random_file_prefix = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 13))
    tmp_file = os.path.join( tempfile.gettempdir(), f'{random_file_prefix}_enlightenGAN_cache.png' )

    # fix rgba issue
    img = imageio.imread( img_path )
    if len(img.shape) == 3  and img.shape[2] == 4:
        img = img[:,:,:3]
    if len(img.shape) == 2: # fix for gray images
        img = img.reshape( img.shape + (1,) )
        img = np.concatenate( [img, img, img], axis=-1 )

    imageio.imwrite( tmp_file, np.asarray(img, dtype='uint8') )
    prediction = predict( tmp_file, local_model_path )
    imageio.imwrite( tmp_file, np.asarray(prediction, dtype='uint8') )
    image_widget.update_content_file( tmp_file )

# plugin interface for the UI to load
def interface():
    def detailed_implementation( image_widget ):
        def fun():
            return implementation( image_widget )
        return fun

    return 'EnlightenGAN', detailed_implementation

