import imageio
import tempfile
import os

import numpy as np
import imageio
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from PIL import Image


from PySide6.QtWidgets import (QLineEdit, QPushButton, QApplication, QVBoxLayout, QHBoxLayout, QDialog, QSlider, QLabel)
from PySide6.QtCore import Qt

color = 64
quality = 50

class Form(QDialog):

    def __init__(self, parent=None):
        super(Form, self).__init__(parent)
        #self.hbox = QHBoxLayout()
        self.vbox = QVBoxLayout()

        self.quality_sld = QSlider( Qt.Horizontal, self )
        self.quality_sld.setRange( 10, 100 )
        self.quality_sld.setFocusPolicy( Qt.NoFocus )
        self.quality_sld.setPageStep( 1 )
        self.quality_sld.setMinimumWidth( 100 )
        self.quality_sld.valueChanged.connect(self.set_quality)

        self.color_sld = QSlider( Qt.Horizontal, self )
        self.color_sld.setRange( 16, 1024 )
        self.color_sld.setFocusPolicy( Qt.NoFocus )
        self.color_sld.setPageStep( 8 )
        self.color_sld.setMinimumWidth( 100 )
        self.color_sld.valueChanged.connect(self.set_color)

        self.button = QPushButton("Apply")
        self.button.clicked.connect(self.close)

        self.color_label = QLabel( 'Image colors: 64', self )
        self.color_label.setAlignment( Qt.AlignCenter | Qt.AlignVCenter )
        self.color_label.setMinimumWidth( 100 )

        self.quality_label = QLabel( 'Image quality: 50', self )
        self.quality_label.setAlignment( Qt.AlignCenter | Qt.AlignVCenter )
        self.quality_label.setMinimumWidth( 100 )


        self.vbox.addWidget(self.color_sld)
        self.vbox.addSpacing( 8 )
        self.vbox.addWidget( self.color_label )
        self.vbox.addSpacing( 15 )

        self.vbox.addWidget(self.quality_sld)
        self.vbox.addSpacing( 8 )
        self.vbox.addWidget( self.quality_label )
        self.vbox.addSpacing( 15 )

        self.vbox.addWidget(self.button)
        self.setLayout(self.vbox)
        self.setWindowTitle( "Setting reduced image colors and quality" )

    def set_color(self, value):
        global color
        color = value
        self.color_label.setText( "Image color: " + str(value) )

    def set_quality(self, value):
        global quality
        quality = value
        self.quality_label.setText( "Image quality: " + str(value) )

def recreate_image(codebook, labels, w, h):
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

def kms_compress_image( image_path, n_colors ):
    image_array = np.asarray( imageio.imread(image_path), dtype='float64' ) / 255.0
    if len(image_array.shape) == 2: #fix gray image
        image_array = image_array.reshape( image_array.shape + (1,) )

    w, h, d = image_array.shape
    image_array = np.reshape(image_array, (w * h, d))

    sample_pixels = min(w*h, 4096 )
    image_array_sample = shuffle(image_array, random_state=0)[:sample_pixels]
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
    labels = kmeans.predict(image_array)
    new_image = recreate_image( kmeans.cluster_centers_, labels, w, h )
    return new_image

def compress_image(input_file_path, output_file_path, quality=50, n_colors=64):
    new_image = kms_compress_image(input_file_path, n_colors)
    new_image = np.asarray( new_image*255.0, dtype='uint8' )
    picture = Image.fromarray( new_image )
    picture.save(output_file_path, "JPEG", optimize = True, quality = quality)


def implementation( image_widget ):
    global color
    global quality
    img_path = image_widget.get_snapshot_file()
    tmp_file = os.path.join( tempfile.gettempdir(), 'reduce_color_cache.jpeg' )
    compress_image( img_path, tmp_file, quality=quality, n_colors=color )
    image_widget.update_content_file( tmp_file )

def interface():
    def detailed_implementation( image_widget ):
        def fun():
            form = Form( image_widget )
            form.exec_() # after closing the gui, the image is getting compressed
            return implementation( image_widget )
        return fun

    return 'ReduceColors', detailed_implementation






