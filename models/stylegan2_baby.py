import tensorflow as tf
import tensorflow.keras as keras
from os.path import dirname, abspath
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


def lerp(a, b, t):
    out = a + (b - a) * t
    return out


def lerp_clip(a, b, t):
    out = a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)
    return out


def adjust_dynamic_range(images, range_in, range_out, out_dtype):
    scale = (range_out[1] - range_out[0]) / (range_in[1] - range_in[0])
    bias = range_out[0] - range_in[0] * scale
    images = images * scale + bias
    images = tf.clip_by_value(images, range_out[0], range_out[1])
    images = tf.cast(images, dtype=out_dtype)
    return images


def random_flip_left_right_nchw(images):
    s = tf.shape(images)
    mask = tf.random.uniform([s[0], 1, 1, 1], 0.0, 1.0)
    mask = tf.tile(mask, [1, s[1], s[2], s[3]])
    images = tf.where(mask < 0.5, images, tf.reverse(images, axis=[3]))
    return images


def preprocess_fit_train_image(images, res):
    images = adjust_dynamic_range(images, range_in=(0.0, 255.0), range_out=(-1.0, 1.0), out_dtype=tf.dtypes.float32)
    images = random_flip_left_right_nchw(images)
    images.set_shape([None, 3, res, res])
    return images


def postprocess_images(images):
    images = adjust_dynamic_range(images, range_in=(-1.0, 1.0), range_out=(0.0, 255.0), out_dtype=tf.dtypes.float32)
    images = tf.transpose(images, [0, 2, 3, 1])
    images = tf.cast(images, dtype=tf.dtypes.uint8)
    return images


def merge_batch_images(images, res, rows, cols):
    batch_size = images.shape[0]
    assert rows * cols == batch_size
    canvas = np.zeros(shape=[res * rows, res * cols, 3], dtype=np.uint8)
    for row in range(rows):
        y_start = row * res
        for col in range(cols):
            x_start = col * res
            index = col + row * cols
            canvas[y_start:y_start + res, x_start:x_start + res, :] = images[index, :, :, :]
    return canvas




def setup_resample_kernel(k):
    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
        k = np.outer(k, k)
    k /= np.sum(k)
    return k


def upfirdn_ref(x, k, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    in_height, in_width = tf.shape(x)[1], tf.shape(x)[2]
    minor_dim = tf.shape(x)[3]
    kernel_h, kernel_w = k.shape

    # Upsample (insert zeros).
    x = tf.reshape(x, [-1, in_height, 1, in_width, 1, minor_dim])
    x = tf.pad(x, [[0, 0], [0, 0], [0, up_y - 1], [0, 0], [0, up_x - 1], [0, 0]])
    x = tf.reshape(x, [-1, in_height * up_y, in_width * up_x, minor_dim])

    # Pad (crop if negative).
    x = tf.pad(x, [
        [0, 0],
        [tf.math.maximum(pad_y0, 0), tf.math.maximum(pad_y1, 0)],
        [tf.math.maximum(pad_x0, 0), tf.math.maximum(pad_x1, 0)],
        [0, 0]
    ])
    x = x[:, tf.math.maximum(-pad_y0, 0): tf.shape(x)[1] - tf.math.maximum(-pad_y1, 0),
          tf.math.maximum(-pad_x0, 0): tf.shape(x)[2] - tf.math.maximum(-pad_x1, 0), :]

    # Convolve with filter.
    x = tf.transpose(x, [0, 3, 1, 2])
    x = tf.reshape(x, [-1, 1, in_height * up_y + pad_y0 + pad_y1, in_width * up_x + pad_x0 + pad_x1])
    w = tf.constant(k[::-1, ::-1, np.newaxis, np.newaxis], dtype=x.dtype)
    x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID', data_format='NCHW')
    x = tf.reshape(x, [-1,
                       minor_dim,
                       in_height * up_y + pad_y0 + pad_y1 - kernel_h + 1,
                       in_width * up_x + pad_x0 + pad_x1 - kernel_w + 1])
    x = tf.transpose(x, [0, 2, 3, 1])

    # Downsample (throw away pixels).
    return x[:, ::down_y, ::down_x, :]


def simple_upfirdn_2d(x, k, up=1, down=1, pad0=0, pad1=0):
    output_channel = tf.shape(x)[1]
    x = tf.reshape(x, [-1, tf.shape(x)[2], tf.shape(x)[3], 1])
    x = upfirdn_ref(x, k,
                    up_x=up, up_y=up, down_x=down, down_y=down, pad_x0=pad0, pad_x1=pad1, pad_y0=pad0, pad_y1=pad1)
    x = tf.reshape(x, [-1, output_channel, tf.shape(x)[1], tf.shape(x)[2]])
    return x


def upsample_conv_2d(x, k, weight, factor, gain):
    x_height, x_width = tf.shape(x)[2], tf.shape(x)[3]
    w_height, w_width = tf.shape(weight)[0], tf.shape(weight)[1]
    w_ic, w_oc = tf.shape(weight)[2], tf.shape(weight)[3]

    # Setup filter kernel.
    k = k * (gain * (factor ** 2))
    p = (k.shape[0] - factor) - (w_width - 1)
    pad0 = (p + 1) // 2 + factor - 1
    pad1 = p // 2 + 1

    # Determine data dimensions.
    strides = [1, 1, factor, factor]
    output_shape = [1, w_oc, (x_height - 1) * factor + w_height, (x_width - 1) * factor + w_width]
    num_groups = tf.shape(x)[1] // w_ic

    # Transpose weights.
    weight = tf.reshape(weight, [w_height, w_width, w_ic, num_groups, -1])
    weight = tf.transpose(weight[::-1, ::-1], [0, 1, 4, 3, 2])
    weight = tf.reshape(weight, [w_height, w_width, -1, num_groups * w_ic])

    # Execute.
    x = tf.nn.conv2d_transpose(x, weight, output_shape, strides, padding='VALID', data_format='NCHW')
    x = simple_upfirdn_2d(x, k, pad0=pad0, pad1=pad1)
    return x


def conv_downsample_2d(x, k, weight, factor, gain):
    w_height, w_width = tf.shape(weight)[0], tf.shape(weight)[1]

    # Setup filter kernel.
    k = k * gain
    p = (k.shape[0] - factor) + (w_width - 1)
    pad0 = (p + 1) // 2
    pad1 = p // 2

    strides = [1, 1, factor, factor]
    x = simple_upfirdn_2d(x, k, pad0=pad0, pad1=pad1)
    x = tf.nn.conv2d(x, weight, strides, padding='VALID', data_format='NCHW')
    return x


def upsample_2d(x, k, factor, gain):
    # Setup filter kernel.
    k = k * (gain * (factor ** 2))
    p = k.shape[0] - factor
    pad0 = (p + 1) // 2 + factor - 1
    pad1 = p // 2

    x = simple_upfirdn_2d(x, k, up=factor, pad0=pad0, pad1=pad1)
    return x


def downsample_2d(x, k, factor, gain):
    # Setup filter kernel.
    k = k * gain
    p = k.shape[0] - factor
    pad0 = (p + 1) // 2
    pad1 = p // 2

    x = simple_upfirdn_2d(x, k, down=factor, pad0=pad0, pad1=pad1)
    return x


def compute_runtime_coef(weight_shape, gain, lrmul):
    fan_in = np.prod(weight_shape[:-1])  # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    #print( f'the fan_in has method {dir(fan_in)}' )
    #assert False, 'pause'
    #he_std = gain / np.sqrt(fan_in.value)
    he_std = gain / np.sqrt(fan_in)
    #he_std = gain / np.sqrt(fan_in.asarray(np.float32)+1.0e-10)
    init_std = 1.0 / lrmul
    runtime_coef = he_std * lrmul
    return init_std, runtime_coef


class Dense(tf.keras.layers.Layer):
    def __init__(self, fmaps, gain, lrmul, **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.fmaps = fmaps
        self.gain = gain
        self.lrmul = lrmul

    def build(self, input_shape):
        assert len(input_shape) == 2 or len(input_shape) == 4
        fan_in = np.prod(input_shape[1:])
        weight_shape = [fan_in, self.fmaps]
        init_std, runtime_coef = compute_runtime_coef(weight_shape, self.gain, self.lrmul)

        self.runtime_coef = runtime_coef

        w_init = tf.random.normal(shape=weight_shape, mean=0.0, stddev=init_std)
        self.w = tf.Variable(w_init, name='w', trainable=True)

    def call(self, inputs, training=None, mask=None):
        weight = self.runtime_coef * self.w
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.matmul(x, weight)
        return x

    def get_config(self):
        config = super(Dense, self).get_config()
        config.update({
            'fmaps': self.fmaps,
            'gain': self.gain,
            'lrmul': self.lrmul,
        })
        return config


class BiasAct(tf.keras.layers.Layer):
    def __init__(self, lrmul, act, **kwargs):
        super(BiasAct, self).__init__(**kwargs)
        assert act in ['linear', 'lrelu']
        self.lrmul = lrmul

        if act == 'linear':
            self.act = tf.keras.layers.Lambda(lambda x: tf.identity(x))
            self.gain = 1.0
        else:
            self.act = tf.keras.layers.LeakyReLU(alpha=0.2)
            self.gain = np.sqrt(2)

    def build(self, input_shape):
        self.len2 = True if len(input_shape) == 2 else False
        b_init = tf.zeros(shape=(input_shape[1],), dtype=tf.dtypes.float32)
        self.b = tf.Variable(b_init, name='b', trainable=True)

    def call(self, inputs, training=None, mask=None):
        b = self.lrmul * self.b

        if self.len2:
            x = inputs + b
        else:
            x = inputs + tf.reshape(b, shape=[1, -1, 1, 1])
        x = self.act(x)
        x = self.gain * x
        return x

    def get_config(self):
        config = super(BiasAct, self).get_config()
        config.update({
            'lrmul': self.lrmul,
            'gain': self.gain,
            'len2': self.len2,
        })
        return config


class LeakyReLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LeakyReLU, self).__init__(**kwargs)
        self.alpha = 0.2
        self.gain = np.sqrt(2)

        self.act = tf.keras.layers.LeakyReLU(alpha=self.alpha)

    def call(self, inputs, training=None, mask=None):
        x = self.act(inputs)
        x *= self.gain
        return x

    def get_config(self):
        config = super(LeakyReLU, self).get_config()
        config.update({
            'alpha': self.alpha,
            'gain': self.gain,
        })
        return config


class LabelEmbedding(tf.keras.layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super(LabelEmbedding, self).__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        weight_shape = [input_shape[1], self.embed_dim]
        # tf 1.15 mean(0.0), std(1.0) default value of tf.initializers.random_normal()
        w_init = tf.random.normal(shape=weight_shape, mean=0.0, stddev=1.0)
        self.w = tf.Variable(w_init, name='w', trainable=True)

    def call(self, inputs, training=None, mask=None):
        x = tf.matmul(inputs, self.w)
        return x

    def get_config(self):
        config = super(LabelEmbedding, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
        })
        return config


class Noise(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Noise, self).__init__(**kwargs)

    def build(self, input_shape):
        self.noise_strength = tf.Variable(initial_value=0.0, dtype=tf.dtypes.float32, trainable=True, name='w')

    def call(self, x, training=None, mask=None):
        x_shape = tf.shape(x)
        noise = tf.random.normal(shape=(x_shape[0], 1, x_shape[2], x_shape[3]), dtype=tf.dtypes.float32)

        x += noise * self.noise_strength
        return x


class MinibatchStd(tf.keras.layers.Layer):
    def __init__(self, group_size, num_new_features, **kwargs):
        super(MinibatchStd, self).__init__(**kwargs)
        self.group_size = group_size
        self.num_new_features = num_new_features

    def call(self, x, training=None, mask=None):
        group_size = tf.minimum(self.group_size, tf.shape(x)[0])
        s = tf.shape(x)

        y = tf.reshape(x, [group_size, -1, self.num_new_features, s[1] // self.num_new_features, s[2], s[3]])
        y = tf.cast(y, tf.float32)
        y -= tf.reduce_mean(y, axis=0, keepdims=True)
        y = tf.reduce_mean(tf.square(y), axis=0)
        y = tf.sqrt(y + 1e-8)
        y = tf.reduce_mean(y, axis=[2, 3, 4], keepdims=True)
        y = tf.reduce_mean(y, axis=[2])
        y = tf.cast(y, x.dtype)
        y = tf.tile(y, [group_size, 1, s[2], s[3]])
        return tf.concat([x, y], axis=1)


class FusedModConv(tf.keras.layers.Layer):
    def __init__(self, fmaps, kernel, gain, lrmul, style_fmaps, demodulate, up, down, resample_kernel, **kwargs):
        super(FusedModConv, self).__init__(**kwargs)
        assert not (up and down)
        self.fmaps = fmaps
        self.kernel = kernel
        self.gain = gain
        self.lrmul = lrmul
        self.style_fmaps = style_fmaps
        self.demodulate = demodulate
        self.up = up
        self.down = down
        self.factor = 2
        if resample_kernel is None:
            resample_kernel = [1] * self.factor
        self.k = setup_resample_kernel(k=resample_kernel)

        self.mod_dense = Dense(self.style_fmaps, gain=1.0, lrmul=1.0, name='mod_dense')
        self.mod_bias = BiasAct(lrmul=1.0, act='linear', name='mod_bias')

    def build(self, input_shape):
        x_shape, w_shape = input_shape[0], input_shape[1]
        weight_shape = [self.kernel, self.kernel, x_shape[1], self.fmaps]
        init_std, runtime_coef = compute_runtime_coef(weight_shape, self.gain, self.lrmul)

        self.runtime_coef = runtime_coef

        # [kkIO]
        w_init = tf.random.normal(shape=weight_shape, mean=0.0, stddev=init_std)
        self.w = tf.Variable(w_init, name='w', trainable=True)

    def scale_conv_weights(self, w):
        # convolution kernel weights for fused conv
        weight = self.runtime_coef * self.w     # [kkIO]
        weight = weight[np.newaxis]             # [BkkIO]

        # modulation
        style = self.mod_dense(w)                                   # [BI]
        style = self.mod_bias(style) + 1.0                          # [BI]
        weight *= style[:, np.newaxis, np.newaxis, :, np.newaxis]   # [BkkIO]

        # demodulation
        if self.demodulate:
            d = tf.math.rsqrt(tf.reduce_sum(tf.square(weight), axis=[1, 2, 3]) + 1e-8)  # [BO]
            weight *= d[:, np.newaxis, np.newaxis, np.newaxis, :]                       # [BkkIO]

        # weight: reshape, prepare for fused operation
        new_weight_shape = [tf.shape(weight)[1], tf.shape(weight)[2], tf.shape(weight)[3], -1]      # [kkI(BO)]
        weight = tf.transpose(weight, [1, 2, 3, 0, 4])                                              # [kkIBO]
        weight = tf.reshape(weight, shape=new_weight_shape)                                         # [kkI(BO)]
        return weight

    def call(self, inputs, training=None, mask=None):
        x, w = inputs
        height, width = tf.shape(x)[2], tf.shape(x)[3]

        # prepare convolution kernel weights
        weight = self.scale_conv_weights(w)

        # prepare inputs: reshape minibatch to convolution groups
        x = tf.reshape(x, [1, -1, height, width])

        if self.up:
            x = upsample_conv_2d(x, self.k, weight, self.factor, self.gain)
        elif self.down:
            x = conv_downsample_2d(x, self.k, weight, self.factor, self.gain)
        else:
            x = tf.nn.conv2d(x, weight, data_format='NCHW', strides=[1, 1, 1, 1], padding='SAME')

        # x: reshape back
        x = tf.reshape(x, [-1, self.fmaps, tf.shape(x)[2], tf.shape(x)[3]])
        return x

    def get_config(self):
        config = super(FusedModConv, self).get_config()
        config.update({
            'fmaps': self.fmaps,
            'kernel': self.kernel,
            'gain': self.gain,
            'lrmul': self.lrmul,
            'style_fmaps': self.style_fmaps,
            'demodulate': self.demodulate,
            'up': self.up,
            'down': self.down,
            'factor': self.factor,
            'k': self.k,
        })
        return config


class ResizeConv2D(tf.keras.layers.Layer):
    def __init__(self, fmaps, kernel, gain, lrmul, up, down, resample_kernel, **kwargs):
        super(ResizeConv2D, self).__init__(**kwargs)
        self.fmaps = fmaps
        self.kernel = kernel
        self.gain = gain
        self.lrmul = lrmul
        self.up = up
        self.down = down
        self.factor = 2
        if resample_kernel is None:
            resample_kernel = [1] * self.factor
        self.k = setup_resample_kernel(k=resample_kernel)

    def build(self, input_shape):
        assert len(input_shape) == 4
        weight_shape = [self.kernel, self.kernel, input_shape[1], self.fmaps]
        init_std, runtime_coef = compute_runtime_coef(weight_shape, self.gain, self.lrmul)

        self.runtime_coef = runtime_coef

        # [kernel, kernel, fmaps_in, fmaps_out]
        w_init = tf.random_normal_initializer(mean=0.0, stddev=init_std)
        self.w = tf.Variable(initial_value=w_init(shape=weight_shape, dtype='float32'), trainable=True, name='w')

    def call(self, inputs, training=None, mask=None):
        x = inputs
        weight = self.runtime_coef * self.w

        if self.up:
            x = upsample_conv_2d(x, self.k, weight, self.factor, self.gain)
        elif self.down:
            x = conv_downsample_2d(x, self.k, weight, self.factor, self.gain)
        else:
            x = tf.nn.conv2d(x, weight, data_format='NCHW', strides=[1, 1, 1, 1], padding='SAME')
        return x

    def get_config(self):
        config = super(ResizeConv2D, self).get_config()
        config.update({
            'fmaps': self.fmaps,
            'kernel': self.kernel,
            'gain': self.gain,
            'lrmul': self.lrmul,
            'up': self.up,
            'down': self.down,
            'factor': self.factor,
            'k': self.k,
        })
        return config


class ToRGB(tf.keras.layers.Layer):
    def __init__(self, in_ch, **kwargs):
        super(ToRGB, self).__init__(**kwargs)
        self.in_ch = in_ch
        self.conv = FusedModConv(fmaps=3, kernel=1, gain=1.0, lrmul=1.0, style_fmaps=self.in_ch,
                                 demodulate=False, up=False, down=False, resample_kernel=None, name='conv')
        self.apply_bias = BiasAct(lrmul=1.0, act='linear', name='bias')

    def call(self, inputs, training=None, mask=None):
        x, w = inputs
        assert x.shape[1] == self.in_ch

        x = self.conv([x, w])
        x = self.apply_bias(x)
        return x

    def get_config(self):
        config = super(ToRGB, self).get_config()
        config.update({
            'in_ch': self.in_ch,
        })
        return config


class Mapping(tf.keras.layers.Layer):
    def __init__(self, w_dim, labels_dim, n_mapping, **kwargs):
        super(Mapping, self).__init__(**kwargs)
        self.w_dim = w_dim
        self.labels_dim = labels_dim
        self.n_mapping = n_mapping
        self.gain = 1.0
        self.lrmul = 0.01

        if self.labels_dim > 0:
            self.labels_embedding = LabelEmbedding(embed_dim=self.w_dim, name='labels_embedding')

        self.normalize = tf.keras.layers.Lambda(lambda x: x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + 1e-8))

        self.dense_layers = list()
        self.bias_act_layers = list()
        for ii in range(self.n_mapping):
            self.dense_layers.append(Dense(w_dim, gain=self.gain, lrmul=self.lrmul, name='dense_{:d}'.format(ii)))
            self.bias_act_layers.append(BiasAct(lrmul=self.lrmul, act='lrelu', name='bias_{:d}'.format(ii)))

    def call(self, inputs, training=None, mask=None):
        latents, labels = inputs
        x = latents

        # embed label if any
        if self.labels_dim > 0:
            y = self.labels_embedding(labels)
            x = tf.concat([x, y], axis=1)

        # normalize inputs
        x = self.normalize(x)

        # apply mapping blocks
        for dense, apply_bias_act in zip(self.dense_layers, self.bias_act_layers):
            x = dense(x)
            x = apply_bias_act(x)

        return x

    def get_config(self):
        config = super(Mapping, self).get_config()
        config.update({
            'w_dim': self.w_dim,
            'labels_dim': self.labels_dim,
            'n_mapping': self.n_mapping,
            'gain': self.gain,
            'lrmul': self.lrmul,
        })
        return config


class SynthesisConstBlock(tf.keras.layers.Layer):
    def __init__(self, fmaps, res, **kwargs):
        super(SynthesisConstBlock, self).__init__(**kwargs)
        assert res == 4
        self.res = res
        self.fmaps = fmaps
        self.gain = 1.0
        self.lrmul = 1.0

        # conv block
        self.conv = FusedModConv(fmaps=self.fmaps, kernel=3, gain=self.gain, lrmul=self.lrmul, style_fmaps=self.fmaps,
                                 demodulate=True, up=False, down=False, resample_kernel=[1, 3, 3, 1], name='conv')
        self.apply_noise = Noise(name='noise')
        self.apply_bias_act = BiasAct(lrmul=self.lrmul, act='lrelu', name='bias')

    def build(self, input_shape):
        # starting const variable
        # tf 1.15 mean(0.0), std(1.0) default value of tf.initializers.random_normal()
        const_init = tf.random.normal(shape=(1, self.fmaps, self.res, self.res), mean=0.0, stddev=1.0)
        self.const = tf.Variable(const_init, name='const', trainable=True)

    def call(self, inputs, training=None, mask=None):
        w0 = inputs
        batch_size = tf.shape(w0)[0]

        # const block
        x = tf.tile(self.const, [batch_size, 1, 1, 1])

        # conv block
        x = self.conv([x, w0])
        x = self.apply_noise(x)
        x = self.apply_bias_act(x)
        return x

    def get_config(self):
        config = super(SynthesisConstBlock, self).get_config()
        config.update({
            'res': self.res,
            'fmaps': self.fmaps,
            'gain': self.gain,
            'lrmul': self.lrmul,
        })
        return config


class SynthesisBlock(tf.keras.layers.Layer):
    def __init__(self, in_ch, fmaps, res, **kwargs):
        super(SynthesisBlock, self).__init__(**kwargs)
        self.in_ch = in_ch
        self.fmaps = fmaps
        self.res = res
        self.gain = 1.0
        self.lrmul = 1.0

        # conv0 up
        self.conv_0 = FusedModConv(fmaps=self.fmaps, kernel=3, gain=self.gain, lrmul=self.lrmul, style_fmaps=self.in_ch,
                                   demodulate=True, up=True, down=False, resample_kernel=[1, 3, 3, 1], name='conv_0')
        self.apply_noise_0 = Noise(name='noise_0')
        self.apply_bias_act_0 = BiasAct(lrmul=self.lrmul, act='lrelu', name='bias_0')

        # conv block
        self.conv_1 = FusedModConv(fmaps=self.fmaps, kernel=3, gain=self.gain, lrmul=self.lrmul, style_fmaps=self.fmaps,
                                   demodulate=True, up=False, down=False, resample_kernel=[1, 3, 3, 1], name='conv_1')
        self.apply_noise_1 = Noise(name='noise_1')
        self.apply_bias_act_1 = BiasAct(lrmul=self.lrmul, act='lrelu', name='bias_1')

    def call(self, inputs, training=None, mask=None):
        x, w0, w1 = inputs

        # conv0 up
        x = self.conv_0([x, w0])
        x = self.apply_noise_0(x)
        x = self.apply_bias_act_0(x)

        # conv block
        x = self.conv_1([x, w1])
        x = self.apply_noise_1(x)
        x = self.apply_bias_act_1(x)
        return x

    def get_config(self):
        config = super(SynthesisBlock, self).get_config()
        config.update({
            'in_ch': self.in_ch,
            'res': self.res,
            'fmaps': self.fmaps,
            'gain': self.gain,
            'lrmul': self.lrmul,
        })
        return config


class Synthesis(tf.keras.layers.Layer):
    def __init__(self, resolutions, featuremaps, name, **kwargs):
        super(Synthesis, self).__init__(name=name, **kwargs)
        self.resolutions = resolutions
        self.featuremaps = featuremaps
        self.k = setup_resample_kernel(k=[1, 3, 3, 1])

        # initial layer
        res, n_f = resolutions[0], featuremaps[0]
        self.initial_block = SynthesisConstBlock(fmaps=n_f, res=res, name='{:d}x{:d}/const'.format(res, res))
        self.initial_torgb = ToRGB(in_ch=n_f, name='{:d}x{:d}/ToRGB'.format(res, res))

        # stack generator block with lerp block
        prev_n_f = n_f
        self.blocks = list()
        self.torgbs = list()
        for res, n_f in zip(self.resolutions[1:], self.featuremaps[1:]):
            self.blocks.append(SynthesisBlock(in_ch=prev_n_f, fmaps=n_f, res=res,
                                              name='{:d}x{:d}/block'.format(res, res)))
            self.torgbs.append(ToRGB(in_ch=n_f, name='{:d}x{:d}/ToRGB'.format(res, res)))
            prev_n_f = n_f

    def call(self, inputs, training=None, mask=None):
        w_broadcasted = inputs

        # initial layer
        w0, w1 = w_broadcasted[:, 0], w_broadcasted[:, 1]
        x = self.initial_block(w0)
        y = self.initial_torgb([x, w1])

        layer_index = 1
        for block, torgb in zip(self.blocks, self.torgbs):
            w0 = w_broadcasted[:, layer_index]
            w1 = w_broadcasted[:, layer_index + 1]
            w2 = w_broadcasted[:, layer_index + 2]

            x = block([x, w0, w1])
            y = upsample_2d(y, self.k, factor=2, gain=1.0)
            y = y + torgb([x, w2])

            layer_index += 2

        images_out = y
        return images_out

    def get_config(self):
        config = super(Synthesis, self).get_config()
        config.update({
            'resolutions': self.resolutions,
            'featuremaps': self.featuremaps,
            'k': self.k,
        })
        return config


class Generator(tf.keras.Model):
    def __init__(self, g_params, **kwargs):
        super(Generator, self).__init__(**kwargs)

        self.z_dim = g_params['z_dim']
        self.w_dim = g_params['w_dim']
        self.labels_dim = g_params['labels_dim']
        self.n_mapping = g_params['n_mapping']
        self.resolutions = g_params['resolutions']
        self.featuremaps = g_params['featuremaps']
        self.w_ema_decay = 0.995
        self.style_mixing_prob = 0.9

        self.n_broadcast = len(self.resolutions) * 2
        self.mixing_layer_indices = tf.range(self.n_broadcast, dtype=tf.int32)[tf.newaxis, :, tf.newaxis]

        self.g_mapping = Mapping(self.w_dim, self.labels_dim, self.n_mapping, name='g_mapping')
        self.broadcast = tf.keras.layers.Lambda(lambda x: tf.tile(x[:, tf.newaxis], [1, self.n_broadcast, 1]))
        self.synthesis = Synthesis(self.resolutions, self.featuremaps, name='g_synthesis')

    def build(self, input_shape):
        # w_avg
        self.w_avg = tf.Variable(tf.zeros(shape=[self.w_dim], dtype=tf.dtypes.float32), name='w_avg', trainable=False,
                                 synchronization=tf.VariableSynchronization.ON_READ,
                                 aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

    @tf.function
    def set_as_moving_average_of(self, src_net):
        beta, beta_nontrainable = 0.99, 0.0

        for cw, sw in zip(self.weights, src_net.weights):
            assert sw.shape == cw.shape
            # print('{} <=> {}'.format(cw.name, sw.name))

            if 'w_avg' in cw.name:
                cw.assign(lerp(sw, cw, beta_nontrainable))
            else:
                cw.assign(lerp(sw, cw, beta))
        return

    def update_moving_average_of_w(self, w_broadcasted):
        # compute average of current w
        batch_avg = tf.reduce_mean(w_broadcasted[:, 0], axis=0)

        # compute moving average of w and update(assign) w_avg
        self.w_avg.assign(lerp(batch_avg, self.w_avg, self.w_ema_decay))
        return

    def style_mixing_regularization(self, latents1, labels, w_broadcasted1):
        # get another w and broadcast it
        latents2 = tf.random.normal(shape=tf.shape(latents1), dtype=tf.dtypes.float32)
        dlatents2 = self.g_mapping([latents2, labels])
        w_broadcasted2 = self.broadcast(dlatents2)

        # find mixing limit index
        # mixing_cutoff_index = tf.cond(
        #     pred=tf.less(tf.random.uniform([], 0.0, 1.0), self.style_mixing_prob),
        #     true_fn=lambda: tf.random.uniform([], 1, self.n_broadcast, dtype=tf.dtypes.int32),
        #     false_fn=lambda: tf.constant(self.n_broadcast, dtype=tf.dtypes.int32))
        if tf.random.uniform([], 0.0, 1.0) < self.style_mixing_prob:
            mixing_cutoff_index = tf.random.uniform([], 1, self.n_broadcast, dtype=tf.dtypes.int32)
        else:
            mixing_cutoff_index = tf.constant(self.n_broadcast, dtype=tf.dtypes.int32)

        # mix it
        mixed_w_broadcasted = tf.where(
            condition=tf.broadcast_to(self.mixing_layer_indices < mixing_cutoff_index, tf.shape(w_broadcasted1)),
            x=w_broadcasted1,
            y=w_broadcasted2)
        return mixed_w_broadcasted

    def truncation_trick(self, w_broadcasted, truncation_psi, truncation_cutoff=None):
        ones = tf.ones_like(self.mixing_layer_indices, dtype=tf.float32)
        tpsi = ones * truncation_psi
        if truncation_cutoff is None:
            truncation_coefs = tpsi
        else:
            indices = tf.range(self.n_broadcast)
            truncation_coefs = tf.where(condition=tf.less(indices, truncation_cutoff), x=tpsi, y=ones)

        truncated_w_broadcasted = lerp(self.w_avg, w_broadcasted, truncation_coefs)
        return truncated_w_broadcasted

    def call(self, inputs, ret_w_broadcasted=False, truncation_psi=1.0, truncation_cutoff=None, training=None, mask=None):
        latents, labels = inputs

        dlatents = self.g_mapping([latents, labels])
        w_broadcasted = self.broadcast(dlatents)

        if training:
            self.update_moving_average_of_w(w_broadcasted)
            w_broadcasted = self.style_mixing_regularization(latents, labels, w_broadcasted)

        if not training:
            w_broadcasted = self.truncation_trick(w_broadcasted, truncation_psi, truncation_cutoff)

        image_out = self.synthesis(w_broadcasted)

        if ret_w_broadcasted:
            return image_out, w_broadcasted
        else:
            return image_out

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], 3, self.resolutions[-1], self.resolutions[-1]

def load_generator(ckpt_dir):

    g_params = {
        'z_dim': 512,
        'w_dim': 512,
        'labels_dim': 0,
        'n_mapping': 8,
        'resolutions': [4, 8, 16, 32, 64, 128, 256, 512, 1024],
        'featuremaps': [512, 512, 512, 512, 512, 256, 128, 64, 32],
    }

    test_latent = tf.ones((1, g_params['z_dim']), dtype=tf.float32)
    test_labels = tf.ones((1, g_params['labels_dim']), dtype=tf.float32)

    # build generator model
    generator = Generator(g_params)
    #_ = generator([test_latent, test_labels])

    ckpt = tf.train.Checkpoint(g_clone=generator)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1)
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    if manager.latest_checkpoint:
        print(f'Generator restored from {manager.latest_checkpoint}')
    return generator


def adjust_dynamic_range(images, range_in, range_out, out_dtype):
    scale = (range_out[1] - range_out[0]) / (range_in[1] - range_in[0])
    bias = range_out[0] - range_in[0] * scale
    images = images * scale + bias
    images = tf.clip_by_value(images, range_out[0], range_out[1])
    images = tf.cast(images, dtype=out_dtype)
    return images

def postprocess_images(images):
    images = adjust_dynamic_range(images, range_in=(-1.0, 1.0), range_out=(0.0, 255.0), out_dtype=tf.dtypes.float32)
    images = tf.transpose(images, [0, 2, 3, 1])
    images = tf.cast(images, dtype=tf.dtypes.uint8)
    return images



def generate_one( generator, output_path ):
    latents = np.random.randn(1, generator.z_dim).astype(np.float32)
    labels = np.random.randn(1, generator.labels_dim).astype(np.float32)
    image_out = generator( [latents, labels], training=False, truncation_psi=0.5 )
    image_out = postprocess_images(image_out)
    image_out = image_out.numpy()
    Image.fromarray( image_out[0], 'RGB' ).save( output_path )




generator = None
def generate( image_widget ):
    global generator
    if generator is None:
        image_widget.download_remote_model( "stylegan2_baby", "https://github.com/fengwang/ImagePlayer/releases/download/stylegan2_baby/checkpoint" )
        image_widget.download_remote_model( "stylegan2_baby", "https://github.com/fengwang/ImagePlayer/releases/download/stylegan2_baby/ckpt-0.data-00000-of-00001" )
        local_model_path = image_widget.download_remote_model( "stylegan2_baby", "https://github.com/fengwang/ImagePlayer/releases/download/stylegan2_baby/ckpt-0.index" )
        model_path = os.path.dirname( local_model_path )
        generator = load_generator(ckpt_dir=model_path)

    random_file_prefix = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 13))
    tmp_file = os.path.join( tempfile.gettempdir(), f'{random_file_prefix}_stylegan2_baby.png' )
    generate_one( generator, tmp_file )
    return tmp_file




def implementation( image_widget ):
    image_path = generate( image_widget )
    image_widget.update_content_file( image_path )




def interface():
    def detailed_implementation( image_widget ):
        def fun():
            return implementation( image_widget )
        return fun

    return 'RandomBaby', detailed_implementation


if __name__ == '__main__':
    img_path = generate()
    print( f'Generating image to {img_path}' )

