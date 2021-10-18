import imageio
import tempfile
import os
import numpy as np

def implementation( image_widget ):
    img_path = image_widget.get_snapshot_file()
    img = imageio.imread( img_path )
    if len(img.shape) == 2 :
        return
    gray = 0.2125*img[:,:,0] + 0.7154*img[:,:,1]+ 0.0721*img[:,:,2]
    tmp_file = os.path.join( tempfile.gettempdir(), 'rgb2gray_cache.png' )
    imageio.imwrite( tmp_file, np.asarray( gray, dtype='uint8' ) )
    image_widget.update_content_file( tmp_file )

def interface():
    def rgb2gray( image_widget ):
        def fun():
            return implementation( image_widget )
        return fun

    return 'RGB2Gray', rgb2gray

