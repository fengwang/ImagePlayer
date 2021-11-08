import os
import string
import sys
import random
import pathlib
import importlib
import tempfile

photo2cartoon_model = None
def implementation( image_widget ):

    local_photo2cartoon_model_path = image_widget.download_remote_model( 'Photo2Cartoon', 'https://github.com/fengwang/ImagePlayer/releases/download/photo2cartoon/photo2cartoon_weights.onnx' )
    local_seg_model_path = image_widget.download_remote_model( 'Photo2Cartoon', 'https://github.com/fengwang/ImagePlayer/releases/download/photo2cartoon/seg_model_384.pb' )
    input_image_path = image_widget.get_snapshot_file()
    random_file_prefix = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 13))
    output_image_path = os.path.join( tempfile.gettempdir(), f'{random_file_prefix}_photo2cartoon_cache.png' )

    global photo2cartoon_model
    if photo2cartoon_model is None: # load module dynamically
        current_directory = os.path.dirname(os.path.realpath(__file__))
        implementation_file_folder = os.path.join( current_directory, 'photo2cartoon' ) # <-
        sys.path.append( implementation_file_folder )
        implementation_file_path = os.path.join( implementation_file_folder, 'photo2cartoon_implementation.py' ) # <-
        module_name = pathlib.Path( implementation_file_path ).stem
        module = importlib.import_module(module_name)
        photo2cartoon_model = module.photo2cartoon_implementation()

    if photo2cartoon_model( input_image_path, output_image_path, local_photo2cartoon_model_path, local_seg_model_path ):
        image_widget.update_content_file( output_image_path )


def interface():
    def detailed_implementation( image_widget ):
        def fun():
            return implementation( image_widget )
        return fun

    return 'Photo2Cartoon', detailed_implementation

