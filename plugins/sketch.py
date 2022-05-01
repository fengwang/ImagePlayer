import imageio
import tempfile
import os
import numpy as np
import cv2

def implementation( image_widget ):
    img_path = image_widget.get_snapshot_file()
    image = cv2.imread( img_path ) # load iamge
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to gray image
    invert = cv2.bitwise_not(grey_image)
    blur = cv2.GaussianBlur(invert, (21, 21), 0)
    invertedblur = cv2.bitwise_not(blur)
    sketch = cv2.divide(grey_image, invertedblur, scale=256.0)
    tmp_file = os.path.join( tempfile.gettempdir(), 'sketch_cache.png' )
    cv2.imwrite( tmp_file, sketch )
    image_widget.update_content_file( tmp_file )


def interface():
    def sketch( image_widget ):
        def fun():
            return implementation( image_widget )
        return fun

    return 'Sketch', sketch

