import tensorflow as tf
import tensorflow.keras as keras
from os.path import dirname, abspath
import PIL
from PIL import Image
import numpy as np
import tempfile
import os
import sys
import random
import string

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)



def calc_loss(img, model):
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]
    losses = []
    for act in layer_activations:
        loss = tf.math.reduce_mean(act)
        losses.append(loss)
    return tf.reduce_sum(losses)


class DeepDream(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function( input_signature=( tf.TensorSpec(shape=[None,None,3], dtype=tf.float32), tf.TensorSpec(shape=[], dtype=tf.int32), tf.TensorSpec(shape=[], dtype=tf.float32),))
    def __call__(self, img, steps, step_size):
        loss = tf.constant(0.0)
        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                tape.watch(img)
                loss = calc_loss(img, self.model)
            gradients = tape.gradient(loss, img)
            gradients /= tf.math.reduce_std(gradients) + 1e-8
            img = img + gradients*step_size
            img = tf.clip_by_value(img, -1, 1)
        return loss, img


def deprocess(img):
    img = 255*(img + 1.0)/2.0
    return tf.cast(img, tf.uint8)


def run_deep_dream_simple(deepdream, img, steps=100, step_size=0.01):
    img = keras.applications.inception_v3.preprocess_input(img)
    img = tf.convert_to_tensor(img)
    step_size = tf.convert_to_tensor(step_size)
    steps_remaining = steps
    step = 0
    while steps_remaining:
        if steps_remaining>100:
            run_steps = tf.constant(100)
        else:
            run_steps = tf.constant(steps_remaining)
        steps_remaining -= run_steps
        step += run_steps
        loss, img = deepdream(img, run_steps, tf.constant(step_size))

    return deprocess(img)



def run_deep_dream( model, original_img, steps=50, step_size=0.01, scale=1.25 ):
    img = tf.constant(np.array(original_img))
    base_shape = tf.shape(img)[:-1]
    float_base_shape = tf.cast(base_shape, tf.float32)
    for n in range(-2, 3):
        new_shape = tf.cast(float_base_shape*(scale**n), tf.int32)
        img = tf.image.resize(img, new_shape).numpy()
        img = run_deep_dream_simple(model, img=img, steps=steps, step_size=step_size)
    img = tf.image.resize(img, base_shape)
    img = tf.image.convert_image_dtype(img/255.0, dtype=tf.uint8)
    return img


def create_model():
    base_model = keras.applications.InceptionV3(include_top=False, weights='imagenet')
    names = ['mixed3', 'mixed5']
    layers = [base_model.get_layer(name).output for name in names]
    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)
    return DeepDream(dream_model)


def generate_one( deepdream_generator, img, output_path ):
    image_out = run_deep_dream( deepdream_generator, img )
    image_out = image_out.numpy()
    Image.fromarray( image_out, 'RGB' ).save( output_path )


deepdream_generator = None
def generate( image_widget, img ):
    global deepdream_generator
    if deepdream_generator is None:
        deepdream_generator = create_model()

    random_file_prefix = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 13))
    tmp_file = os.path.join( tempfile.gettempdir(), f'{random_file_prefix}_stylegan2_baby.png' )
    generate_one( deepdream_generator, img, tmp_file )
    return tmp_file




def implementation( image_widget ):
    img = PIL.Image.open(image_widget.get_snapshot_file())
    image_path = generate( image_widget, img )
    image_widget.update_content_file( image_path )


def interface():
    def detailed_implementation( image_widget ):
        def fun():
            return implementation( image_widget )
        return fun

    return 'DeepDream', detailed_implementation



