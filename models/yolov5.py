import torch

import tempfile
import os
import imageio

model = None
def implementation( image_widget ):
    global model
    if model is None:
        #model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom
        model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')  # or yolov5m, yolov5l, yolov5x, custom

    img_path = image_widget.get_snapshot_file()
    results = model( img_path )
    results.save( tempfile.gettempdir() )
    new_img_path = list( img_path )
    new_img_path[-3:] = 'jpg'
    new_img_path = ''.join( new_img_path )
    image_widget.update_content_file( new_img_path )

def interface():
    def detailed_implementation( image_widget ):
        def fun():
            return implementation( image_widget )
        return fun

    return 'Yolov5x6', detailed_implementation



