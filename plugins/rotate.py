import imageio
import tempfile
import os
import numpy as np
from scipy.ndimage import rotate

from PySide6.QtWidgets import (QLineEdit, QPushButton, QApplication, QVBoxLayout, QHBoxLayout, QDialog, QSlider, QLabel)
from PySide6.QtCore import Qt

rotation_angle = 0

class Form(QDialog):

    def __init__(self, parent=None):
        super(Form, self).__init__(parent)
        self.hbox = QHBoxLayout()

        self.sld = QSlider( Qt.Horizontal, self )
        self.sld.setRange( -90, 90 )
        self.sld.setFocusPolicy( Qt.NoFocus )
        self.sld.setPageStep( 1 )
        self.sld.setMinimumWidth( 180 )
        self.sld.valueChanged.connect(self.set_angle)


        self.button = QPushButton("Apply")
        self.button.clicked.connect(self.close)

        self.label = QLabel( '0', self )
        self.label.setAlignment( Qt.AlignCenter | Qt.AlignVCenter )
        self.label.setMinimumWidth( 80 )


        self.hbox.addWidget(self.sld)
        self.hbox.addSpacing( 8 )
        self.hbox.addWidget( self.label )
        self.hbox.addSpacing( 15 )
        self.hbox.addWidget(self.button)
        self.setLayout(self.hbox)
        self.setWindowTitle( "Setting Rotation angles (in degrees)" )

    # Greets the user
    def set_angle(self, value):
        global rotation_angle
        rotation_angle = value
        self.label.setText( str(value) )


def implementation( image_widget ):
    global rotation_angle
    img_path = image_widget.get_snapshot_file()
    img = imageio.imread( img_path )
    img = rotate( img, rotation_angle, reshape=True, prefilter=False, cval=255 )
    tmp_file = os.path.join( tempfile.gettempdir(), 'rotate_cache.ppm' )
    imageio.imwrite( tmp_file, np.asarray( img, dtype='uint8' ) )
    image_widget.update_content_file( tmp_file )

def interface():
    def detailed_implementation( image_widget ):
        def fun():
            # here a gui
            form = Form( image_widget )
            #form.show()
            form.exec_()
            return implementation( image_widget )
        return fun

    return 'Rotate', detailed_implementation

