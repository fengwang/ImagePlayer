#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''

    ImagePlayer: playing with deep image models.

    Copyright (C) 2021  Feng Wang

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

'''

import os
os.environ["QT_LOGGING_RULES"]= '*.debug=false;qt.qpa.*=false'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from PySide6.QtPrintSupport import QPrintDialog, QPrinter
from PySide6.QtWidgets import (QApplication, QDialog, QFileDialog, QLabel, QMainWindow, QMenuBar, QMessageBox, QScrollArea, QScrollBar, QSizePolicy, QStatusBar)
from PySide6.QtGui import (QAction, QClipboard, QColorSpace, QGuiApplication, QImage, QImageReader, QImageWriter, QKeySequence, QPalette, QPainter, QPixmap, QScreen, QPainter, QIcon)
from PySide6.QtCore import QDir, QMimeData, QStandardPaths, Qt, Slot, QSize, QPoint
from argparse import ArgumentParser, RawTextHelpFormatter
from PySide6.QtWidgets import (QApplication)
from qt_material import apply_stylesheet

import tempfile
import glob
import importlib, pathlib, sys
import imageio
import numpy as np
import shutil
import random
import string
import sys
from requests import get

ABOUT = "Deep Image Models At Ease."

class ImagePlayer(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scale_factor = 1.0
        self._first_file_dialog = True
        self._image_label = QLabel()
        self._image_label.setBackgroundRole(QPalette.Base)
        self._image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._image_label.setScaledContents(True)

        self._scroll_area = QScrollArea()
        self._scroll_area.setBackgroundRole(QPalette.Dark)
        self._scroll_area.setWidget(self._image_label)
        self._scroll_area.setVisible(False)
        self.setCentralWidget(self._scroll_area)

        self._create_actions()

        self.resize(QGuiApplication.primaryScreen().availableSize() ) # PySide6.QtCore.QSize(1920, 1200)

        # data zone
        self.tmp_image_counter = 0
        self.current_cached_image_path = None
        self.current_image_presented = None
        self.random_file_prefix = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 13))

        dir_path = os.path.dirname(os.path.realpath(__file__))
        self._plugin_list = []
        self.load_plugins(os.path.join(dir_path, 'plugins'))

        self._model_list = []
        self.load_models(os.path.join(dir_path, 'models'))

        # drop image file to open, accept
        self.setAcceptDrops(True)

        icon_path = self.load_plugins(os.path.join(dir_path, 'src', 'logo.png'))
        self.setWindowIcon(QIcon(icon_path))

        # home model path
        self.user_model_path = os.path.join( os.path.expanduser('~'), '.deepoffice', 'imageplayer', 'model' )
        if not os.path.exists(self.user_model_path):
            os.makedirs( self.user_model_path )

    # download model files from github release
    def download_remote_model(self, model_name, model_url):
        model_path = os.path.join( self.user_model_path, model_name )

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        file_name = model_url.rsplit('/', 1)[1]
        local_model_path = os.path.join( model_path, file_name )
        if not os.path.isfile(local_model_path):
            print( f'downloading model file {local_model_path} from {model_url}' )
            with open(local_model_path, "wb") as file:
                response = get(model_url)
                file.write(response.content)
            print( f'downloaded model file {local_model_path} from {model_url}' )

        return local_model_path


    # interface to plugins 1
    def get_snapshot_file( self, index=None ):
        return self.get_new_snapshot_file( self.tmp_image_counter-1 )

    # interface to plugins 2: a single file
    def update_content_file(self, fileName, rescaling_flag=False, in_a_new_window=True):
        if not in_a_new_window:
            self.load_file( fileName, rescaling_flag )
        else:
            image_player = ImagePlayer()
            image_player.show()
            image_player.update_content_file( fileName, rescaling_flag=rescaling_flag, in_a_new_window=False );

    # interface to plugins 3: many files
    def update_content_files(self, fileNames, rescaling_flag=False, in_a_new_window=True):
        for fileName in fileNames:
            self.update_content_file( fileName, rescaling_flag, in_a_new_window )


    def get_new_snapshot_file( self, index=None ):
        if index is None:
            index = self.tmp_image_counter
        tmp_dir = tempfile.gettempdir()
        tmp_png_file = os.path.join( tmp_dir, f'{self.random_file_prefix}_{index}.png' )
        return tmp_png_file


    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.accept()
        else:
            e.ignore()
            print( f'ignoring {e.mimeData().text()}' )

    def dropEvent(self, e):
        if e.mimeData().hasUrls():
            for url in  e.mimeData().urls():
                self.load_file( url.path() )
                break


    def wheelEvent(self, event):
        numDegrees = event.angleDelta() / 8.0

        if not numDegrees.isNull():
            x = numDegrees.x() / 150.0
            y = numDegrees.y() / 150.0

            new_scale = x + 1.0
            if abs(y) > abs(x):
                new_scale = y + 1.0

            new_scale = min( 2.0, max( 0.5, new_scale) )
            self._scale_image(new_scale)

        event.accept()


    def load_file(self, fileName, rescaling_flag=True):
        reader = QImageReader(fileName)
        reader.setAutoTransform(True)
        new_image = reader.read()
        native_filename = QDir.toNativeSeparators(fileName)
        if new_image.isNull():
            error = reader.errorString()
            #QMessageBox.information(self, QGuiApplication.applicationDisplayName(), f"Cannot load {native_filename}: {error}")
            error_message = f'cannot open file {native_filename} -->  Error: {error}'
            self.statusBar().showMessage(error_message)
            return False
        self.current_image_presented = fileName
        self._set_image(new_image)
        self.setWindowFilePath(fileName)

        w = max( self._image.width() * 1.15, 300 )
        h = max( self._image.height() * 1.15, 300 )
        d = self._image.depth()
        max_size = QGuiApplication.primaryScreen().availableSize()
        self._current_size = QSize(min(w, max_size.width()), min( h, max_size.height()) )
        self.resize( self._current_size )
        if rescaling_flag:
            self._scale_image( self._scale_factor )
        message = f'Opened "{native_filename}": {self._image.width()}x{self._image.height()}'
        self.statusBar().showMessage(message)

        self._save_tmp_file()
        return True

    def _set_image(self, new_image):
        self._image = new_image
        if self._image.colorSpace().isValid():
            self._image.convertToColorSpace(QColorSpace.SRgb)
        self._image_label.setPixmap(QPixmap.fromImage(self._image))
        self._scroll_area.setAlignment(Qt.AlignCenter)
        self._scale_factor = 1.0

        self._scroll_area.setVisible(True)
        self._print_act.setEnabled(True)
        self._update_actions()

        if not self._fit_to_window_act.isChecked():
            self._image_label.adjustSize()

    def _save_file(self, fileName):

        _, current_extension = os.path.splitext( self.current_image_presented )
        _, new_extension = os.path.splitext( fileName )
        if (current_extension == new_extension ):
            shutil.copy( self.current_image_presented, fileName )
            return True


        writer = QImageWriter(fileName)

        native_filename = QDir.toNativeSeparators(fileName)
        if not writer.write(self._image):
            error = writer.errorString()
            message = f"Cannot write {native_filename}: {error}"
            QMessageBox.information(self, QGuiApplication.applicationDisplayName(), message)
            self.statusBar().showMessage( message );
            return False
        return True

    def _save_tmp_file( self ):
        tmp_png_file = self.get_new_snapshot_file()
        if not self._save_file(tmp_png_file):
            print( f'Failed saving tmp file to {tmp_png_file}' )
            return None
        print( f'saving tmp file: {tmp_png_file}' )
        self.tmp_image_counter += 1
        self.current_cached_image_path = tmp_png_file
        return tmp_png_file

    @Slot()
    def _undo( self ):
        prev_cache_file = self.get_new_snapshot_file( self.tmp_image_counter-2 )
        if os.path.isfile( prev_cache_file ):
            self.load_file( prev_cache_file )
            self.tmp_image_counter -=  2
            print( f'Undo: updating image counter to {self.tmp_image_counter}')
        else:
            print( f'cannot load {prev_cache_file=}' )

    @Slot()
    def _redo( self ):
        next_cache_file = self.get_new_snapshot_file( self.tmp_image_counter )
        if os.path.isfile( next_cache_file ):
            self.load_file( next_cache_file )
            print( f'Redo: updating image counter to {self.tmp_image_counter}')
        else:
            print( f'cannot load {next_cache_file=}' )

    @Slot()
    def _clean_tmp_file( self ):
        for idx in range( self.tmp_image_counter ):
            tmp_dir = tempfile.gettempdir()
            tmp_png_file = os.path.join( tmp_dir, f'{self.random_file_prefix}_{idx}.png' )
            if os.path.isfile( tmp_png_file ):
                os.remove( tmp_png_file )
            print( f'removing file {tmp_png_file}' )
        self.tmp_image_counter = 0
        self.current_cached_image_path = None


    @Slot()
    def _open(self):
        dialog = QFileDialog(self, "Open File")
        self._initialize_image_filedialog(dialog, QFileDialog.AcceptOpen)
        while (dialog.exec() == QDialog.Accepted
               and not self.load_file(dialog.selectedFiles()[0])):
            pass

    @Slot()
    def _save_as(self):
        dialog = QFileDialog(self, "Save File As")
        self._initialize_image_filedialog(dialog, QFileDialog.AcceptSave)
        while (dialog.exec() == QDialog.Accepted and not self._save_file(dialog.selectedFiles()[0])):
            pass

    @Slot()
    def _print_(self):
        printer = QPrinter()
        dialog = QPrintDialog(printer, self)
        if dialog.exec() == QDialog.Accepted:
            painter = QPainter(printer)
            pixmap = self._image_label.pixmap()
            rect = painter.viewport()
            size = pixmap.size()
            size.scale(rect.size(), Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(pixmap.rect())
            painter.drawPixmap(0, 0, pixmap)
            painter.end()

    @Slot()
    def _copy(self):
        QGuiApplication.clipboard().setImage(self._image)

    @Slot()
    def _paste(self):
        new_image = QGuiApplication.clipboard().image()
        if new_image.isNull():
            self.statusBar().showMessage("No image in clipboard")
        else:
            self._set_image(new_image)
            self.setWindowFilePath('')
            w = new_image.width()
            h = new_image.height()
            d = new_image.depth()
            message = f"Obtained image from clipboard, {w}x{h}, Depth: {d}"
            self.statusBar().showMessage(message)

    @Slot()
    def _zoom_in(self):
        if self.current_image_presented is not None:
            self._scale_image(1.25)

    @Slot()
    def _zoom_out(self):
        if self.current_image_presented is not None:
            self._scale_image(0.8)

    @Slot()
    def _normal_size(self):
        if self.current_image_presented is not None:
            self._image_label.adjustSize()
            self._scale_factor = 1.0

    @Slot()
    def _fit_to_window(self):
        if self.current_image_presented is not None:
            fit_to_window = self._fit_to_window_act.isChecked()
            self._scroll_area.setWidgetResizable(fit_to_window)
            if not fit_to_window:
                self._normal_size()
            self._update_actions()

    @Slot()
    def _about(self):
        QMessageBox.about(self, "About Image Viewer", ABOUT)

    def _create_actions(self):
        file_menu = self.menuBar().addMenu("&File")

        self._open_act = file_menu.addAction("&Open...")
        self._open_act.triggered.connect(self._open)
        self._open_act.setShortcut(QKeySequence.Open)

        self._save_as_act = file_menu.addAction("&Save As...")
        self._save_as_act.triggered.connect(self._save_as)
        self._save_as_act.setEnabled(False)

        self._print_act = file_menu.addAction("&Print...")
        self._print_act.triggered.connect(self._print_)
        self._print_act.setShortcut(QKeySequence.Print)
        self._print_act.setEnabled(False)

        file_menu.addSeparator()

        self._exit_act = file_menu.addAction("E&xit")
        self._exit_act.triggered.connect(self.close)
        self._exit_act.triggered.connect(self._clean_tmp_file)
        self._exit_act.setShortcut("Ctrl+Q")

        edit_menu = self.menuBar().addMenu("&Edit")

        self._copy_act = edit_menu.addAction("Undo")
        self._copy_act.triggered.connect(self._undo)

        self._copy_act = edit_menu.addAction("Redo")
        self._copy_act.triggered.connect(self._redo)

        self._copy_act = edit_menu.addAction("&Copy")
        self._copy_act.triggered.connect(self._copy)
        self._copy_act.setShortcut(QKeySequence.Copy)
        self._copy_act.setEnabled(False)

        self._paste_act = edit_menu.addAction("&Paste")
        self._paste_act.triggered.connect(self._paste)
        self._paste_act.setShortcut(QKeySequence.Paste)

        view_menu = self.menuBar().addMenu("&View")

        self._zoom_in_act = view_menu.addAction("Zoom &In (25%)")
        self._zoom_in_act.setShortcut(QKeySequence.ZoomIn)
        self._zoom_in_act.triggered.connect(self._zoom_in)
        self._zoom_in_act.setEnabled(False)

        self._zoom_out_act = view_menu.addAction("Zoom &Out (25%)")
        self._zoom_out_act.triggered.connect(self._zoom_out)
        self._zoom_out_act.setShortcut(QKeySequence.ZoomOut)
        self._zoom_out_act.setEnabled(False)

        self._normal_size_act = view_menu.addAction("&Normal Size")
        self._normal_size_act.triggered.connect(self._normal_size)
        self._normal_size_act.setShortcut("Ctrl+S")
        self._normal_size_act.setEnabled(False)

        view_menu.addSeparator()

        self._fit_to_window_act = view_menu.addAction("&Fit to Window")
        self._fit_to_window_act.triggered.connect(self._fit_to_window)
        self._fit_to_window_act.setEnabled(False)
        self._fit_to_window_act.setCheckable(True)
        self._fit_to_window_act.setShortcut("Ctrl+F")

        self.plugin_menu = self.menuBar().addMenu("&Plugins")
        self.model_menu = self.menuBar().addMenu("&Models")

        help_menu = self.menuBar().addMenu("&Help")

        about_act = help_menu.addAction("&About")
        about_act.triggered.connect(self._about)
        about_qt_act = help_menu.addAction("About &Qt")
        about_qt_act.triggered.connect(QApplication.aboutQt)

    def load_plugin( self, plugin_path ):
        module_name = pathlib.Path(plugin_path).stem
        module = importlib.import_module(module_name)
        plugin_name, event = module.interface()
        _act = self.plugin_menu.addAction( plugin_name )
        _act.triggered.connect( event(self) )
        self._plugin_list.append( plugin_name )


    def load_plugins( self, folder ):
        sys.path.append( folder )
        plugin_paths = glob.glob( f'{folder}/*.py' ) +  glob.glob( f'{str(pathlib.Path.home())}/.deepoffice/plugins/*.py' )
        for plugin_path in plugin_paths:
            self.load_plugin( plugin_path )


    def load_model( self, model_path ):
        module_name = pathlib.Path(model_path).stem
        module = importlib.import_module(module_name)
        model_name, event = module.interface()
        _act = self.model_menu.addAction( model_name )
        _act.triggered.connect( event(self) )
        self._model_list.append( model_name )

    def load_models( self, folder ):
        sys.path.append( folder )
        model_paths = glob.glob( f'{folder}/*.py' ) +  glob.glob( f'{str(pathlib.Path.home())}/.deepoffice/models/*.py' )
        for model_path in model_paths:
            self.load_model( model_path )

    def _update_actions(self):
        has_image = not self._image.isNull()
        self._save_as_act.setEnabled(has_image)
        self._copy_act.setEnabled(has_image)
        enable_zoom = not self._fit_to_window_act.isChecked()
        self._zoom_in_act.setEnabled(enable_zoom)
        self._zoom_out_act.setEnabled(enable_zoom)
        self._normal_size_act.setEnabled(enable_zoom)

    def _scale_image(self, factor):
        if self.current_image_presented is not None:
            self._scale_factor *= factor
            new_size = self._scale_factor * self._image_label.pixmap().size()
            self._image_label.resize(new_size)

            self._adjust_scrollbar(self._scroll_area.horizontalScrollBar(), factor)
            self._adjust_scrollbar(self._scroll_area.verticalScrollBar(), factor)

            self._zoom_in_act.setEnabled(self._scale_factor < 3.0)
            self._zoom_out_act.setEnabled(self._scale_factor > 0.333)

            self._current_size *= factor

    def _adjust_scrollbar(self, scrollBar, factor):
        pos = int(factor * scrollBar.value() + ((factor - 1) * scrollBar.pageStep() / 2))
        print( f'adjusting scrollbar to {pos=}' )
        scrollBar.setValue(pos)

    def _initialize_image_filedialog(self, dialog, acceptMode):
        if self._first_file_dialog:
            self._first_file_dialog = False
            locations = QStandardPaths.standardLocations(QStandardPaths.PicturesLocation)
            directory = locations[-1] if locations else QDir.currentPath()
            dialog.setDirectory(directory)

        mime_types = [m.data().decode('utf-8') for m in QImageWriter.supportedMimeTypes()]
        mime_types.sort()

        dialog.setMimeTypeFilters(mime_types)
        dialog.setAcceptMode(acceptMode)
        if acceptMode == QFileDialog.AcceptSave:
            dialog.setDefaultSuffix("png")

if __name__ == '__main__':
    arg_parser = ArgumentParser(description="Image Viewer", formatter_class=RawTextHelpFormatter)
    arg_parser.add_argument('file', type=str, nargs='?', help='Image file')
    args = arg_parser.parse_args()

    app = QApplication(sys.argv)
    image_player = ImagePlayer()
    extra = { 'danger': '#dc3545', 'warning': '#ffc107', 'success': '#17a2b8', 'font-family': 'Roboto', }
    apply_stylesheet(app, 'light_cyan_500.xml', invert_secondary=True, extra=extra)

    if args.file and not image_player.load_file(args.file):
        sys.exit(-1)

    image_player.show()
    sys.exit(app.exec())


