import tensorflow as tf
import os, sys
import numpy as np
import keras 
from keras import backend as K
from model.models import UNet, BayesianPredictor
import random as rn
import medpy.io
import tqdm
import argparse
import glob

# PARAMETERS
INPUT_SHAPE = [512,512,1]
N_CLASS = 6
MODEL_DIR = 'model'
WEIGHT_PATH =  os.path.join(MODEL_DIR, 'model.hdf5')
CLIM = (-400,700)
SUB = 0.5
DIV = 1./255.
MC_ITERATIONS = 10

#GPU allocation
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
K.clear_session()

def set_session(seed, gpu):
    assert gpu >= 0, "Enter number of GPUs"

    #os.environ['PYTHONHASHSEED'] = str(seed)
    #np.random.seed(seed)
    #rn.seed(seed)

    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
        gpu_options=tf.GPUOptions(
            visible_device_list = str(gpu),
            allow_growth=True
        )
    )
    #tf.set_random_seed(seed)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)


def UNET_predictor(INPUT_SHAPE, WEIGHT_PATH=WEIGHT_PATH, N_CLASS=6, MC_ITERATIONS=MC_ITERATIONS):
    model = UNet(INPUT_SHAPE, N_CLASS)
    print('UNet built...')
    model.summary()
    model.load_weights(WEIGHT_PATH)
    print('Weights loaded successfully')
    predictor = BayesianPredictor(model,INPUT_SHAPE=INPUT_SHAPE, MC_ITERATIONS=MC_ITERATIONS)
    print('Predictor ready...')
    return predictor

def preprocess(image, CLIM, SUB, DIC):
    image = (image-CLIM[0]) / (CLIM[1]-CLIM[0])
    image = np.clip(image, 0., 1.)
    image -= SUB
    image /= DIV
    return image

def segment(image, model):
    label, uncert = [], []
    for x in tqdm.tqdm(image, desc='Predicting'):
        x = np.expand_dims(x, axis=-1)
        l, u = model.predict(np.expand_dims(x, axis=0), batch_size=1, verbose=0)
        label.append(l)
        uncert.append(u)

    label = np.array(label)
    uncert = np.array(uncert)
    label = label[:,0].astype(np.int16)
    uncert = uncert[:,0].astype(np.float32)
    
    return label, uncert

def main(in_dir, out_dir, uncert_ok):
    # START SESSION
    set_session(0,0)

    # Get UNet model
    predictor = UNET_predictor(INPUT_SHAPE,
                            WEIGHT_PATH=WEIGHT_PATH,
                            N_CLASS=N_CLASS,
                            MC_ITERATIONS=MC_ITERATIONS)

    # Test on some data
    image_files = glob.glob(os.path.join(in_dir, '*.mhd'))
    for image_file in image_files:
        print('Processing %s \n' % image_file)
        # Test data
        # image, header = mhd.read(image_file)
        image, header = medpy.io.load(image_file)
        image = np.transpose(image, (2,1,0))
        print(image.shape)
        print(header.__dict__)
        image = preprocess(image, CLIM, SUB, DIV) #PREPROCESS
        # Segment image
        label, uncert = segment(image, predictor)
        # Save segmentation
        out_file = os.path.join(out_dir, os.path.basename(image_file)).replace('.mhd', '_label.mhd')
        label = np.transpose(label, (2,1,0))
        medpy.io.save(label, out_file,  header, use_compression=True)

        if uncert_ok:
            uncert = np.transpose(uncert, (2,1,0))
            uncert_file = out_file.replace('label', 'uncert')
            medpy.io.save(uncert, uncert_file, header, use_compression=True)

    # CLEAR SESSION
    K.clear_session()

if __name__ == '__main__':
    '''
    USAGE: Inference Code for Phantom Segmentation using Bayesian U-Net.
    INPUTS:
        '-i' or '--in_dir': Directory including MHD images (default: data)
        '-o' or '--out_dir': Directory to save segmentation results (default: results)
        '-u', '--uncert_ok': Flag to save the uncertainty or not (default: False)
    EXAMPLES:
        python phantom_segmentation.py -i "./data" -o "./labels" -u
        python phantom_segmentation.py --in_dir "./data" --out_dir "./labels" --uncert_ok
        python phantom_segmentation.py --in_dir "./data" --out_dir "./labels"
    '''
    parser = argparse.ArgumentParser(description='Phantom Segmentation using Bayesian U-Net')
    parser.add_argument('-i', '--in_dir',default='data', type=str)
    parser.add_argument('-o', '--out_dir', default='results', type=str)
    parser.add_argument('-u', '--uncert_ok', action='store_true')
    args = parser.parse_args()

    uncert_ok = args.uncert_ok
    in_dir = args.in_dir
    if not os.path.isdir(in_dir):
        raise RuntimeError('Input directory "%s" does not exist!' % in_dir)
    out_dir = args.out_dir
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        print('Directory %s did not exist. Now created.' % out_dir)
        main(in_dir, out_dir, uncert_ok)
    else:
        txt = input('Output directory already exists. Continue[y/n]?')
        if 'y' in txt.lower():
            main(in_dir, out_dir, uncert_ok)
        else:
            print('Quitting...')
            sys.exit()
