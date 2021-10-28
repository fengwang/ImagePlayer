
# From https://github.com/xinntao/BasicSR and https://github.com/xinntao/Real-ESRGAN

from torch import nn as nn
from torch.nn import functional as F
import argparse
import cv2
import glob
import math
import numpy as np
import os
import torch
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
from os.path import dirname, abspath
import string
import random


class Registry():
    def __init__(self, name):
        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj):
        assert (name not in self._obj_map), (f"An object named '{name}' was already registered "
                                             f"in '{self._name}' registry!")
        self._obj_map[name] = obj

    def register(self, obj=None):
        if obj is None:
            def deco(func_or_class):
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        name = obj.__name__
        self._do_register(name, obj)

    def get(self, name):
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")
        return ret

    def __contains__(self, name):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())

    def keys(self):
        return self._obj_map.keys()


DATASET_REGISTRY = Registry('dataset')
ARCH_REGISTRY = Registry('arch')
MODEL_REGISTRY = Registry('model')
LOSS_REGISTRY = Registry('loss')
METRIC_REGISTRY = Registry('metric')

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


def pixel_unshuffle(x, scale):
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)




class ResidualDenseBlock(nn.Module):

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Emperically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Emperically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x


@ARCH_REGISTRY.register()
class RRDBNet(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


upsampler = None

def resr( args_input, output_image_path, local_model_path ):
    current_folder = dirname(abspath(__file__))

    #args_model_path = os.path.join( current_folder, 'resr/pretrained_models/RealESRGAN_x4plus.pth' )
    args_model_path = local_model_path
    args_suffix = 'out'
    args_tile = 0
    args_tile_pad = 10
    args_pre_pad = 0
    args_alpha_upsampler = 'realesrgan'
    args_ext = 'png'
    args_scale = 4

    global upsampler
    if upsampler is None:
        upsampler = RealESRGANer(scale=args_scale, model_path=args_model_path, tile=args_tile, tile_pad=args_tile_pad, pre_pad=args_pre_pad)

    if os.path.isfile(args_input):
        paths = [args_input]
    else:
        paths = sorted(glob.glob(os.path.join(args_input, '*')))

    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print('Testing', idx, imgname)

        # ------------------------------ read image ------------------------------ #
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        if np.max(img) > 255:  # 16-bit image
            max_range = 65535
            print('\tInput is a 16-bit image')
        else:
            max_range = 255
        img = img / max_range
        if len(img.shape) == 2:  # gray image
            img_mode = 'L'
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA image with alpha channel
            img_mode = 'RGBA'
            alpha = img[:, :, 3]
            img = img[:, :, 0:3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if args_alpha_upsampler == 'realesrgan':
                alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
        else:
            img_mode = 'RGB'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ------------------- process image (without the alpha channel) ------------------- #
        upsampler.pre_process(img)
        if args_tile:
            upsampler.tile_process()
        else:
            upsampler.process()
        output_img = upsampler.post_process()
        output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
        if img_mode == 'L':
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

        # ------------------- process the alpha channel if necessary ------------------- #
        if img_mode == 'RGBA':
            if args_alpha_upsampler == 'realesrgan':
                upsampler.pre_process(alpha)
                if args_tile:
                    upsampler.tile_process()
                else:
                    upsampler.process()
                output_alpha = upsampler.post_process()
                output_alpha = output_alpha.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
                output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
            else:
                h, w = alpha.shape[0:2]
                output_alpha = cv2.resize(alpha, (w * args_scale, h * args_scale), interpolation=cv2.INTER_LINEAR)

            # merge the alpha channel
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
            output_img[:, :, 3] = output_alpha

        # ------------------------------ save image ------------------------------ #
        if args_ext == 'auto':
            extension = extension[1:]
        else:
            extension = args_ext
        if img_mode == 'RGBA':  # RGBA images should be saved in png format
            extension = 'png'

        save_path = output_image_path

        if max_range == 65535:  # 16-bit image
            output = (output_img * 65535.0).round().astype(np.uint16)
        else:
            output = (output_img * 255.0).round().astype(np.uint8)

        cv2.imwrite(save_path, output.astype(np.uint8))


class RealESRGANer():

    def __init__(self, scale, model_path, tile=0, tile_pad=10, pre_pad=10):
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None

        # initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
        loadnet = torch.load(model_path)
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        model.load_state_dict(loadnet[keyname], strict=True)
        model.eval()
        self.model = model.to(self.device)

    def pre_process(self, img):
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        self.img = img.unsqueeze(0).to(self.device)

        # pre_pad
        if self.pre_pad != 0:
            self.img = F.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), 'reflect')
        # mod pad
        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4
        if self.mod_scale is not None:
            self.mod_pad_h, self.mod_pad_w = 0, 0
            _, _, h, w = self.img.size()
            if (h % self.mod_scale != 0):
                self.mod_pad_h = (self.mod_scale - h % self.mod_scale)
            if (w % self.mod_scale != 0):
                self.mod_pad_w = (self.mod_scale - w % self.mod_scale)
            self.img = F.pad(self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')

    def process(self):
        try:
            # inference
            with torch.no_grad():
                self.output = self.model(self.img)
                print( f'predicted {self.output.shape}' )
        except Exception as error:
            print('Error', error)

    def tile_process(self):
        """Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                try:
                    with torch.no_grad():
                        output_tile = self.model(input_tile)
                except Exception as error:
                    print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                       output_start_x_tile:output_end_x_tile]

    def post_process(self):
        # remove extra pad
        if self.mod_scale is not None:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]
        # remove prepad
        if self.pre_pad != 0:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.pre_pad * self.scale, 0:w - self.pre_pad * self.scale]
        return self.output

import tempfile
import os
import sys
import imageio

#
# @brief Partitioning input to an image array with 256x256 pixels
#
def partition( img ):
    row, col, ch = img.shape

    nrow = (row // 256) * 256 + 512
    rl_padding = (nrow - row) // 2
    rr_padding = nrow - row - rl_padding

    ncol = (col // 256) * 256 + 512
    cl_padding = (ncol - col) // 2
    cr_padding = ncol - col - cl_padding

    nimg = np.pad( img, [(rl_padding, rr_padding), (cl_padding, cr_padding), (0, 0)], mode='reflect' )

    images_256x256 = []
    rows, cols = (nrow//128) - 1, (ncol//128) - 1
    imgs_256x256 = np.zeros( (rows, cols, 256, 256, 3) )
    for r in range( rows ):
        for c in range( cols ):
            imgs_256x256[r][c] = nimg[r*128:r*128+256, c*128:c*128+256]

    return imgs_256x256


#
# @brief Upscaling the image array with 256x256 pixels to 1024x1024 pixels.
#
def do_upscaling( local_model_path, tmp_file, images_256x256 ):
    print( f'do upscaling with {tmp_file=} and {images_256x256.shape=}' )
    row, col, *_ = images_256x256.shape
    imgs_1kx1k = np.zeros( (row, col, 1024, 1024, 3) )
    for r in range( row ):
        for c in range( col ):
            imageio.imwrite( tmp_file, np.asarray( images_256x256[r][c], dtype='uint8' ) )
            resr( tmp_file, tmp_file, local_model_path )
            imgs_1kx1k[r][c] = imageio.imread( tmp_file )
    print( f'Got {imgs_1kx1k.shape=}' )
    return imgs_1kx1k

#
# @brief Merge
#
def merge( images_1kx1k, row, col ):
    print( f'Merging input 1kx1k images {images_1kx1k.shape} with {row=} and {col=}' )
    rows, cols, *_ = images_1kx1k.shape
    tmp_img = np.zeros( (rows*512, cols*512, 3) )
    print( f'got tmp image {tmp_img.shape=}' )
    for r in range( rows ):
        for c in range( cols ):
            tmp_img[r*512:r*512+512, c*512:c*512+512] = images_1kx1k[r, c, 256:768, 256:768, :]
    pr = (rows*512 - row*4) // 2
    pc = (cols*512 - col*4) // 2
    print( f'got {pr=} and {pc=}' )
    ans = tmp_img[pr:row*4+pr, pc:col*4+pc]
    print( f'got ans {ans.shape}' )
    return ans


def implementation( image_widget ):
    local_model_path = image_widget.download_remote_model( 'RESR4X', 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth' )

    img_path = image_widget.get_snapshot_file()
    random_file_prefix = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 13))
    tmp_file = os.path.join( tempfile.gettempdir(), f'{random_file_prefix}_resr4x_cache.png' )

    # fix rgba issue
    img = imageio.imread( img_path )
    if len(img.shape) == 3  and img.shape[2] == 4:
        img = img[:,:,:3]

    row, col, _ = img.shape
    if ( row < 256 and col < 256 ):
        imageio.imwrite( img_path, np.asarray(img, dtype='uint8') )
        resr( img_path, tmp_file, local_model_path )
    else:
        images_256x256 = partition( img ) # large image from [r, c, 3] to [n, 256, 256, 3]
        images_1kx1k = do_upscaling( local_model_path, tmp_file, images_256x256 ) # to [n, 1024, 1024, 3]
        result_image = merge( images_1kx1k, row, col ) # to [r*4, c*4, 3]
        imageio.imwrite( tmp_file, np.asarray(result_image, dtype='uint8') )

    image_widget.update_content_file( tmp_file )


def interface():
    def detailed_implementation( image_widget ):
        def fun():
            return implementation( image_widget )
        return fun

    return 'RESR4X', detailed_implementation



if __name__ == '__main__':
    import tqdm
    for idx in tqdm.tqdm(range( 3931 )):
        input = f'/home/feng/Downloads/c++_layer/images/original_{str(idx+1).zfill(8)}.png'
        output = f'/home/feng/Downloads/c++_layer/upsampled/upsampled_{str(idx+1).zfill(8)}.png'
        resr( input, output )



