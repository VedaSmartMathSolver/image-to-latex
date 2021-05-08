from dataset.dataset import test_transform
import cv2
import pandas as pd
from PIL import ImageGrab
from PIL import Image
import os
import sys
import argparse
import logging
import yaml
import re
import base64

import numpy as np
import torch
from torchvision import transforms
from munch import Munch
from transformers import PreTrainedTokenizerFast
from timm.models.resnetv2 import ResNetV2
from timm.models.layers import StdConv2dSame

#from dataset.latex2png import tex2pil
from models import get_model
from utils import *

last_pic = None


def minmax_size(img, max_dimensions):
    ratios = [a/b for a, b in zip(img.size, max_dimensions)]
    if any([r > 1 for r in ratios]):
        size = np.array(img.size)//max(ratios)
        img = img.resize(size.astype(int), Image.BILINEAR)
    return img


def initialize(arguments):
    with open(arguments.config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    args = Munch(params)
    args.update(**vars(arguments))
    args.wandb = False
    args.device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'

    model = get_model(args)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))

    if 'image_resizer.pth' in os.listdir(os.path.dirname(args.checkpoint)) and not arguments.no_resize:
        image_resizer = ResNetV2(layers=[2, 3, 3], num_classes=22, global_pool='avg', in_chans=1, drop_rate=.05,
                                 preact=True, stem_type='same', conv_layer=StdConv2dSame).to(args.device)
        image_resizer.load_state_dict(torch.load(os.path.join(os.path.dirname(args.checkpoint), 'image_resizer.pth'), map_location=args.device))
        image_resizer.eval()
    else:
        image_resizer = None
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
    return args, model, image_resizer, tokenizer


def call_model(args, model, image_resizer, tokenizer):
    global last_pic
    encoder, decoder = model.encoder, model.decoder
    '''
    if args.file:
        img = Image.open(args.file)
        print(args.file,tokenizer)
    else:
        img = ImageGrab.grabclipboard()
    if img is None:
        if last_pic is None:
            print('Copy an image into the clipboard.')
            return
        else:
            img = last_pic
    else:
        last_pic = img.copy()
    '''
    #imgdata = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAc4AAAHiCAYAAABhiCz9AAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAABwdSURBVHhe7d0JuPXrXP9xkkpRkVIaDOE0R0qcVMbIFBqvJplDoVBEqLgMJRUNlDFlaKCRolJJGYqiaFKGShoQTYr17/2zf6d1dvt5nvV9nn04e/9fr+ta19lrrf3stdZvuD/3/b3v3zoX2AAAOxOcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDg5Nj4/d///c0nfuInbs4+++zN4x73uM2Vr3zlzTWvec3NbW5zm81//Md/7P0WwJkRnBwbn/M5n7N5znOes7nCFa6w+fzP//zN29/+9s2rX/3qzQUucIHN8573vL3fAjgzgpNj43u+53s2v/ALv7C56EUvunn+85+/PPbnf/7nS3D+yq/8ynIf4EwJTo6kd73rXXs/ndtP/dRPLUH5d3/3d8v9ArP7jUQBDoPg/B/Pfe5zNze5yU2W26Me9ajN2972tr1njp6//du/3bzwhS/cu3f8POlJT9p88Rd/8eZud7vb5ilPecrmne98594z73aDG9xg84Vf+IWb//zP/1zuf93Xfd3mAz7gAzZ/+Id/uNyH/f7sz/5s80Vf9EWbL//yL9/c97733bzhDW/Ye4bD8E//9E+br/7qr958zdd8zebWt7715pWvfOXeM0fX//fB+Td/8zebj/3Yj9388i//8ubXf/3XN+///u+/uec977n37NHzzGc+c3P9619/797x8opXvGJz4QtfePNLv/RLm3/8x3/cXOISl9j81m/91t6zmyVEr3SlK22+6qu+arn/93//95sP/MAPXBYHnZfWkD4M73jHOzb/+q//uneP81rHzGd+5mcuZf5XvepVS1tw7Wtfe+9ZzlSVoQKzW536L/iCL9h88Ad/8N6zR9exDc4asxrXt7zlLSddUVnvsjmxAietyvzSL/3S5ef3ln/5l39Zemm99/0jqlP5+Z//+WXkfCod0L3GP//zP59vR9i9x95f2+Gtb33r5jd/8zeXFbM9/l//9V+bD/3QD93c//733/vtzeZ3fud3Nu/7vu+7ufSlL7253/3ut4wi2peFUf7qr/5q86M/+qMH3k62eOhHfuRHNt/1Xd+1eeQjH7l585vfvPfouzXyPcyO1ute97rNZ33WZ22e+tSn7j1yYgXsepy0Pc4rfeb2Q8fliUrk5yeT7dL5ddZZZ23ufe97L/cbGV3ucpdbfuZgu7at6XjpHFw7851rTZ0cdcc2OH/1V391uRzhIz7iI5be5Il04vzDP/zD8t9f/MVfXBreH//xH9979r3jm7/5mzef8imfsrn4xS++rAqdePazn7252c1utnfvxAqiz/3cz10aic/7vM/be/T8pZPyOte5zuaSl7zk5kY3utHmBS94weajPuqjlpP23//935ee64Me9KC9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ADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAAYEJwAMCE4AGBCcADAgOAFgQHACwIDgBIABwQkAA4ITAHa22fw/HHVHFNOWDoAAAAAASUVORK5CYII=')
    imgdata = base64.b64decode(args.file)
    #print(args)
    filename = 'some_image.png'  # I assume you have a way of picking unique filenames
    img = open(filename, 'wb')
    img.write(imgdata)
    img.close()
    img = Image.open(filename)

    img = minmax_size(pad(img), args.max_dimensions)
    if image_resizer is not None and not args.no_resize:
        with torch.no_grad():
            input_image = pad(img).convert('RGB').copy()
            r, w = 1, img.size[0]
            for i in range(10):
                img = minmax_size(input_image.resize((w, int(input_image.size[1]*r)), Image.BILINEAR if r > 1 else Image.LANCZOS), args.max_dimensions)
                t = test_transform(image=np.array(pad(img).convert('RGB')))['image'][:1].unsqueeze(0)
                w = image_resizer(t.to(args.device)).argmax(-1).item()*32
                if (w/img.size[0] == 1):
                    break
                r *= w/img.size[0]
    else:
        img = np.array(pad(img).convert('RGB'))
        t = test_transform(image=img)['image'][:1].unsqueeze(0)

    im = t.to(args.device)

    with torch.no_grad():
        model.eval()
        device = args.device
        encoded = encoder(im.to(device))
        dec = decoder.generate(torch.LongTensor([args.bos_token])[:, None].to(device), args.max_seq_len,
                               eos_token=args.eos_token, context=encoded.detach(), temperature=args.temperature)
        pred = post_process(token2str(dec, tokenizer)[0])
    #print('Prediction: ',pred, '\n')
    return pred
    '''
    df = pd.DataFrame([pred])
    df.to_clipboard(index=False, header=False)
    if args.show or args.katex:
        try:
            if args.katex:
                raise ValueError
            tex2pil([f'$${pred}$$'])[0].show()
        except Exception as e:
            # render using katex
            import webbrowser
            from urllib.parse import quote
            url = 'https://katex.org/?data=' + \
                quote('{"displayMode":true,"leqno":false,"fleqn":false,"throwOnError":true,"errorColor":"#cc0000",\
"strict":"warn","output":"htmlAndMathml","trust":false,"code":"%s"}' % pred.replace('\\', '\\\\'))
            webbrowser.open(url)
    '''

#if __name__ == "__main__":
def pix2tex(instructions):  
    parser = argparse.ArgumentParser(description='Use model', add_help=False)
    parser.add_argument('-t', '--temperature', type=float, default=.333, help='Softmax sampling frequency')
    parser.add_argument('-c', '--config', type=str, default='settings/config.yaml')
    parser.add_argument('-m', '--checkpoint', type=str, default='checkpoints/weights.pth')
    parser.add_argument('-s', '--show', action='store_true', help='Show the rendered predicted latex code')
    parser.add_argument('-f', '--file', type=str, default=None, help='Predict LaTeX code from image file instead of clipboard')
    parser.add_argument('-k', '--katex', action='store_true', help='Render the latex code in the browser')
    parser.add_argument('--no-cuda', action='store_true', help='Compute on CPU')
    parser.add_argument('--no-resize', action='store_true', help='Resize the image beforehand')
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.FATAL)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    latexocr_path = os.path.dirname(sys.argv[0])
    if latexocr_path != '':
        sys.path.insert(0, latexocr_path)
        os.chdir(latexocr_path)
    
    args, *objs = initialize(args)

    #while True:
        #instructions = input('Predict LaTeX code for image ("?"/"h" for help). ')
    ins = instructions.strip().lower()
    if ins == 'x':
        #break
        return instructions
    elif ins in ['?', 'h', 'help']:
        print('''pix2tex help:

Usage:
On Windows and macOS you can copy the image into memory and just press ENTER to get a prediction.
Alternatively you can paste the image file path here and submit.

You might get a different prediction every time you submit the same image. If the result you got was close you \
can just predict the same image by pressing ENTER again. If that still does not work you can change the temperature \
or you have take another picture with another resolution.

Press "x" to close the program.
You can interrupt the model if it takes too long by pressing Ctrl+C.

Visualization:
You can either render the code into a png using XeLaTeX (see README) to get an image file back. \
This is slow and requires a working installation of XeLaTeX. To activate type 'show' or set the flag --show
Alternatively you can render the expression in the browser using katex.org. Type 'katex' or set --katex

Settings:
to toggle one of these settings: 'show', 'katex', 'no_resize' just type it into the console
Change the temperature (default=0.333) type: "t=0.XX" to set a new temperature.
            ''')
        #continue
    elif ins in ['show', 'katex', 'no_resize']:
        setattr(args, ins, not getattr(args, ins, False))
        print('set %s to %s' % (ins, getattr(args, ins)))
        #continue
    
    #elif os.path.isfile(ins):
    #    args.file = ins
    #    #print(args.file)
    #else:
    #    t = re.match(r't=([\.\d]+)', ins)
    #    if t is not None:
    #        t = t.groups()[0]
    #        args.temperature = float(t)+1e-8
    #        print('new temperature: T=%.3f' % args.temperature)
    #        continue
    
    else:
        args.file = instructions
    
    #try:
    return call_model(args, *objs)
    #except KeyboardInterrupt:
    #    pass
    #args.file = None

'''
if __name__ == "__main__":
    instructions = input('Predict LaTeX code for image ("?"/"h" for help). ')
    result = pix2tex(instructions)
    print( result )
'''