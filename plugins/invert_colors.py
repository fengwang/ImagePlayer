import imageio
import tempfile
import os
import numpy as np

def implementation( image_widget ):
    img_path = image_widget.get_snapshot_file()
    img = 255 - imageio.imread( img_path )
    tmp_file = os.path.join( tempfile.gettempdir(), 'invertcolors_cache.png' )
    imageio.imwrite( tmp_file, np.asarray( img, dtype='uint8' ) )
    image_widget.update_content_file( tmp_file )


def interface():
    def rgb2gray( image_widget ):
        def fun():
            return implementation( image_widget )
        return fun

    return 'InvertColors', rgb2gray

