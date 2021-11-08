from tensorflow.python.platform import gfile
import argparse
import cv2
import face_alignment
import math
import numpy as np
import onnxruntime
import os
import tensorflow as tf

class FaceDetect:
    def __init__(self, device, detector):
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device, face_detector=detector)

    def align(self, image):
        landmarks = self.__get_max_face_landmarks(image)

        if landmarks is None:
            return None

        else:
            return self.__rotate(image, landmarks)

    def __get_max_face_landmarks(self, image):
        preds = self.fa.get_landmarks(image)
        if preds is None:
            return None

        elif len(preds) == 1:
            return preds[0]

        else:
            # find max face
            areas = []
            for pred in preds:
                landmarks_top = np.min(pred[:, 1])
                landmarks_bottom = np.max(pred[:, 1])
                landmarks_left = np.min(pred[:, 0])
                landmarks_right = np.max(pred[:, 0])
                areas.append((landmarks_bottom - landmarks_top) * (landmarks_right - landmarks_left))
            max_face_index = np.argmax(areas)
            return preds[max_face_index]

    @staticmethod
    def __rotate(image, landmarks):
        # rotation angle
        left_eye_corner = landmarks[36]
        right_eye_corner = landmarks[45]
        radian = np.arctan((left_eye_corner[1] - right_eye_corner[1]) / (left_eye_corner[0] - right_eye_corner[0]))

        # image size after rotating
        height, width, _ = image.shape
        cos = math.cos(radian)
        sin = math.sin(radian)
        new_w = int(width * abs(cos) + height * abs(sin))
        new_h = int(width * abs(sin) + height * abs(cos))

        # translation
        Tx = new_w // 2 - width // 2
        Ty = new_h // 2 - height // 2

        # affine matrix
        M = np.array([[cos, sin, (1 - cos) * width / 2. - sin * height / 2. + Tx],
                      [-sin, cos, sin * width / 2. + (1 - cos) * height / 2. + Ty]])

        image_rotate = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(255, 255, 255))

        landmarks = np.concatenate([landmarks, np.ones((landmarks.shape[0], 1))], axis=1)
        landmarks_rotate = np.dot(M, landmarks.T).T
        return image_rotate, landmarks_rotate


class FaceSeg:
    def __init__(self, model_path):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self._graph = tf.Graph()
        self._sess = tf.compat.v1.Session(config=config, graph=self._graph)

        self.pb_file_path = model_path
        self._restore_from_pb()
        self.input_op = self._sess.graph.get_tensor_by_name('input_1:0')
        self.output_op = self._sess.graph.get_tensor_by_name('sigmoid/Sigmoid:0')

    def _restore_from_pb(self):
        with self._sess.as_default():
            with self._graph.as_default():
                with tf.io.gfile.GFile(self.pb_file_path, 'rb') as f:
                    graph_def = tf.compat.v1.GraphDef()
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, name='')

    def input_transform(self, image):
        image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_AREA)
        image_input = (image / 255.)[np.newaxis, :, :, :]
        return image_input

    def output_transform(self, output, shape):
        output = cv2.resize(output, (shape[1], shape[0]))
        image_output = (output * 255).astype(np.uint8)
        return image_output

    def get_mask(self, image):
        image_input = self.input_transform(image)
        output = self._sess.run(self.output_op, feed_dict={self.input_op: image_input})[0]
        return self.output_transform(output, shape=image.shape[:2])


class Preprocess:
    def __init__(self, seg_model_path, device='cpu', detector='dlib'):
        self.detect = FaceDetect(device, detector)
        self.segment = FaceSeg(seg_model_path)

    def process(self, image):
        face_info = self.detect.align(image)
        if face_info is None:
            return None
        image_align, landmarks_align = face_info

        face = self.__crop(image_align, landmarks_align)
        mask = self.segment.get_mask(face)
        return np.dstack((face, mask))

    @staticmethod
    def __crop(image, landmarks):
        landmarks_top = np.min(landmarks[:, 1])
        landmarks_bottom = np.max(landmarks[:, 1])
        landmarks_left = np.min(landmarks[:, 0])
        landmarks_right = np.max(landmarks[:, 0])

        # expand bbox
        top = int(landmarks_top - 0.8 * (landmarks_bottom - landmarks_top))
        bottom = int(landmarks_bottom + 0.3 * (landmarks_bottom - landmarks_top))
        left = int(landmarks_left - 0.3 * (landmarks_right - landmarks_left))
        right = int(landmarks_right + 0.3 * (landmarks_right - landmarks_left))

        if bottom - top > right - left:
            left -= ((bottom - top) - (right - left)) // 2
            right = left + (bottom - top)
        else:
            top -= ((right - left) - (bottom - top)) // 2
            bottom = top + (right - left)

        image_crop = np.ones((bottom - top + 1, right - left + 1, 3), np.uint8) * 255

        h, w = image.shape[:2]
        left_white = max(0, -left)
        left = max(0, left)
        right = min(right, w-1)
        right_white = left_white + (right-left)
        top_white = max(0, -top)
        top = max(0, top)
        bottom = min(bottom, h-1)
        bottom_white = top_white + (bottom - top)

        image_crop[top_white:bottom_white+1, left_white:right_white+1] = image[top:bottom+1, left:right+1].copy()
        return image_crop


class Photo2Cartoon:
    def __init__(self):
        #self.pre = Preprocess( seg_model_path )
        #self.session = onnxruntime.InferenceSession( photo2catoon_model_path )
        self.pre =  None
        self.session = None

    def inference(self, img):
        # face alignment and segmentation
        face_rgba = self.pre.process(img)
        if face_rgba is None:
            return None

        face_rgba = cv2.resize(face_rgba, (256, 256), interpolation=cv2.INTER_AREA)
        face = face_rgba[:, :, :3].copy()
        mask = face_rgba[:, :, 3][:, :, np.newaxis].copy() / 255.
        face = (face*mask + (1-mask)*255) / 127.5 - 1

        face = np.transpose(face[np.newaxis, :, :, :], (0, 3, 1, 2)).astype(np.float32)

        # inference
        cartoon = self.session.run(['output'], input_feed={'input':face})

        # post-process
        cartoon = np.transpose(cartoon[0][0], (1, 2, 0))
        cartoon = (cartoon + 1) * 127.5
        cartoon = (cartoon * mask + 255 * (1 - mask)).astype(np.uint8)
        cartoon = cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)
        return cartoon

    def __call__(self, input_image_path, output_image_path, photo2catoon_model_path, seg_model_path ):
        if self.pre is None:
            self.pre = Preprocess( seg_model_path )
        if self.session is None:
            self.session = onnxruntime.InferenceSession( photo2catoon_model_path )

        img = cv2.cvtColor(cv2.imread(input_image_path), cv2.COLOR_BGR2RGB)
        cartoon = self.inference(img)
        if cartoon is not None:
            cv2.imwrite(output_image_path, cartoon)
            return True
        return False


def photo2cartoon_implementation():
    return Photo2Cartoon()

