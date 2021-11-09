import os
import string
import sys
import random
import pathlib
import importlib
import tempfile

facepaint_model = None
def implementation( image_widget ):

    local_facepaint_model_path = image_widget.download_remote_model( 'FacePaint', 'https://github.com/fengwang/ImagePlayer/releases/download/models/face_paint_512_v2_0.pt' )
    input_image_path = image_widget.get_snapshot_file()
    random_file_prefix = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 13))
    output_image_path = os.path.join( tempfile.gettempdir(), f'{random_file_prefix}_facepaint_cache.png' )

    global facepaint_model
    if facepaint_model is None: # load module dynamically
        current_directory = os.path.dirname(os.path.realpath(__file__))
        implementation_file_folder = os.path.join( current_directory, 'facepaint' ) # <-
        sys.path.append( implementation_file_folder )
        implementation_file_path = os.path.join( implementation_file_folder, 'facepaint_implementation.py' ) # <-
        module_name = pathlib.Path( implementation_file_path ).stem
        module = importlib.import_module(module_name)
        facepaint_model = module.facepaint_implementation()

    if facepaint_model( input_image_path, output_image_path, local_facepaint_model_path ):
        image_widget.update_content_file( output_image_path )


def interface():
    def detailed_implementation( image_widget ):
        def fun():
            return implementation( image_widget )
        return fun

    return 'FacePaint', detailed_implementation

