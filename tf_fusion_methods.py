import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

HEIGHT = 2448//2
WIDTH  = 3264//2

def convert_model(model):
    def representative_dataset():
        for _ in range(1):
          data = [np.random.rand(*inp.shape).astype(np.float32) for inp in model.inputs]
          yield data
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8  # or tf.uint8
    converter.inference_output_type = tf.uint8  # or tf.uint8
    tflite_model = converter.convert()

    with open(f'model.tflite', 'wb') as f:
        f.write(tflite_model)

def pyrUp(img):
    out = tf.image.resize(img, [img.shape[-3]*2, img.shape[-2]*2])
    return out

def pyrDown(img):
    out = tf.image.resize(img, [img.shape[-3]//2, img.shape[-2]//2])
    return out

def multires_pyramid(image, weight_map, levels):
    
    levels  = levels - 1
    imgGpyr = [image]
    wGpyr   = [weight_map]
    
    for i in range(levels):
        imgGpyr.append(pyrDown(imgGpyr[i]))
        wGpyr.append(pyrDown(wGpyr[i]))

    imgLpyr = [imgGpyr[levels]]
    
    for i in range(levels, 0, -1):
        shape = imgGpyr[i-1].shape
        imgLpyr.append(imgGpyr[i-1] - tf.image.resize(pyrUp(imgGpyr[i]), shape[-3:-1]))
    
    return imgLpyr[::-1], wGpyr
    
def merge_frames(inpsA, inpsB, inpsC, MERGE):

    if MERGE=='mean':
        func_merge = lambda x:tf.reduce_mean(x, axis=-1)
    else:
        func_merge = lambda x:tfp.stats.percentile(x, 50.0, interpolation='midpoint', axis=-1)

    a_neg, a_pos = inpsA[...,0], inpsA[...,1:]
    a_pos = func_merge(a_pos)
    inpsA = tf.stack([a_neg, a_pos], axis=-1)
   
    b_neg, b_pos = inpsB[...,0], inpsB[...,1:]
    b_pos = func_merge(b_pos)
    inpsB = tf.stack([b_neg, b_pos], axis=-1)

    c_neg, c_pos = inpsC[...,0], inpsC[...,1:]
    c_pos = func_merge(c_pos)
    inpsC = tf.stack([c_neg, c_pos], axis=-1)
    
    return inpsA, inpsB, inpsC

def FAST_YUV(NR_FRAMES, MERGE, wc, we):
    Y_inputs  = tf.keras.layers.Input((HEIGHT,WIDTH,NR_FRAMES+1), batch_size=1, name='Y_input')
    U_inputs  = tf.keras.layers.Input((HEIGHT,WIDTH,NR_FRAMES+1), batch_size=1, name='U_input')
    V_inputs  = tf.keras.layers.Input((HEIGHT,WIDTH,NR_FRAMES+1), batch_size=1, name='V_input')

    inpsY, inpsU, inpsV = Y_inputs, U_inputs, V_inputs

    if MERGE in ['mean', 'median']:
        inpsY, inpsU, inpsV = merge_frames(inpsY, inpsU, inpsV, MERGE)
        
    LEVELS = int(np.log(min(inpsY.shape[1:-1]))/np.log(2))

    # compute weights
    inpsYW = inpsY
    inpsUW = inpsU
    inpsVW = inpsV
    
    W = tf.ones_like(inpsYW)
    
    if wc==1:
        kernel = [
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ]
        if MERGE is None:
            kernel = np.array([kernel]*(NR_FRAMES+1))[...,np.newaxis]
            kernel = kernel.reshape(3,3,NR_FRAMES+1,1)
        else:
            kernel = np.array([kernel]*2)[...,np.newaxis]
            kernel = kernel.reshape(3,3,2,1)

        Cw = tf.keras.layers.DepthwiseConv2D((3,3), strides=(1,1), padding='same', use_bias=False, depthwise_initializer=lambda shape,dtype : kernel) (inpsYW)
        Cw = tf.abs(Cw)+1
        
        W = W * Cw
    
    if we==1:
        #Ew = tf.exp(-(inpsYW-0.5)**2/0.08)
        Ew = tf.abs(inpsUW+tf.keras.backend.epsilon())*tf.abs(inpsVW+tf.keras.backend.epsilon())+1
        W = W * Ew

    norm = tf.reduce_sum(W, axis=-1, keepdims=True)+tf.keras.backend.epsilon()
    weight_maps = W/norm
    
    U_fuse = tf.reduce_max(inpsU, axis=-1, keepdims=True)
    V_fuse = tf.reduce_max(inpsV, axis=-1, keepdims=True)
    
    imgLpyr, wGpyr = multires_pyramid(inpsY, weight_maps, LEVELS)

    finalPyramid = [tf.reduce_sum(im*we, axis=-1, keepdims=True) for im,we in zip(imgLpyr, wGpyr)]

    Y_fuse = finalPyramid[0]
    for i in range(LEVELS-1):
        layerx = pyrUp(finalPyramid[i+1])
        Y_fuse += tf.image.resize(layerx, (inpsY.shape[-3], inpsY.shape[-2]))
        
    model = tf.keras.Model([Y_inputs, U_inputs, V_inputs], [Y_fuse, U_fuse, V_fuse])

    convert_model(model)
    
def Mertens(NR_FRAMES, MERGE, wc, we, ws):
    R_inputs  = tf.keras.layers.Input((HEIGHT,WIDTH,NR_FRAMES+1), batch_size=1, name='R_input')
    G_inputs  = tf.keras.layers.Input((HEIGHT,WIDTH,NR_FRAMES+1), batch_size=1, name='G_input')
    B_inputs  = tf.keras.layers.Input((HEIGHT,WIDTH,NR_FRAMES+1), batch_size=1, name='B_input')

    inpsR, inpsG, inpsB = R_inputs, G_inputs, B_inputs

    if MERGE in ['mean', 'median']:
        inpsR, inpsG, inpsB = merge_frames(inpsR, inpsG, inpsB, MERGE)
        
    LEVELS = int(np.log(min(inpsR.shape[1:-1]))/np.log(2))

    # compute weights
    inpsRW = inpsR
    inpsGW = inpsG
    inpsBW = inpsB
    
    W = tf.ones_like(inpsRW)
    
    if wc==1:
        kernel = [
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ]
        if MERGE is None:
            kernel = np.array([kernel]*(NR_FRAMES+1))[...,np.newaxis]
            kernel = kernel.reshape(3,3,NR_FRAMES+1,1)
        else:
            kernel = np.array([kernel]*2)[...,np.newaxis]
            kernel = kernel.reshape(3,3,2,1)
        
        gray = tf.reduce_mean([inpsRW, inpsGW, inpsBW], axis=0)
        Cw = tf.keras.layers.DepthwiseConv2D((3,3), strides=(1,1), padding='same', use_bias=False, depthwise_initializer=lambda shape,dtype : kernel) (gray)
        Cw = tf.abs(Cw)+1
        W = W * Cw
    
    if we==1:
        func_exposure = lambda x: tf.exp(-(x-0.5)**2/0.08)
        Ew = func_exposure(inpsRW)*func_exposure(inpsGW)*func_exposure(inpsBW) + 1
        W = W * Ew
        
    if ws==1:
        Sw = tf.math.reduce_std([inpsRW, inpsGW, inpsBW], axis=0) + 1
        W = W * Sw
    
    norm = tf.reduce_sum(W, axis=-1, keepdims=True)+tf.keras.backend.epsilon()
    weight_maps = W/norm
    
    inpsRGB = tf.concat([inpsR, inpsG, inpsB], axis=0)
    imgLpyr, wGpyr = multires_pyramid(inpsRGB, weight_maps, LEVELS)

    finalPyramid = [tf.reduce_sum(im*we, axis=-1, keepdims=True) for im,we in zip(imgLpyr, wGpyr)]

    RGB_fuse = finalPyramid[0]
    for i in range(LEVELS-1):
        layerx = pyrUp(finalPyramid[i+1])
        RGB_fuse += tf.image.resize(layerx, (inpsRGB.shape[-3], inpsRGB.shape[-2]))
    RGB_fuse = tf.squeeze(RGB_fuse, axis=-1)
    RGB_fuse = tf.transpose(RGB_fuse, [2,1,0])
        
    model = tf.keras.Model([R_inputs, G_inputs, B_inputs], RGB_fuse)

    convert_model(model)
    
def SSF_BGR(NR_FRAMES, MERGE, wc, we, ws):
    R_inputs  = tf.keras.layers.Input((HEIGHT,WIDTH,NR_FRAMES+1), batch_size=1, name='R_input')
    G_inputs  = tf.keras.layers.Input((HEIGHT,WIDTH,NR_FRAMES+1), batch_size=1, name='G_input')
    B_inputs  = tf.keras.layers.Input((HEIGHT,WIDTH,NR_FRAMES+1), batch_size=1, name='B_input')

    inpsR, inpsG, inpsB = R_inputs, G_inputs, B_inputs

    if not (MERGE is None):
        inpsR, inpsG, inpsB = merge_frames(inpsR, inpsG, inpsB, MERGE)
        
    LEVELS = int(np.log(min(inpsR.shape[1:-1]))/np.log(2))

    # compute weights
    inpsRW = inpsR
    inpsGW = inpsG
    inpsBW = inpsB
    
    W = tf.zeros_like(inpsRW)
    
    if wc==1:
        kernel = [
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ]
        if MERGE is None:
            kernel = np.array([kernel]*(NR_FRAMES+1))[...,np.newaxis]
            kernel = kernel.reshape(3,3,NR_FRAMES+1,1)
        else:
            kernel = np.array([kernel]*2)[...,np.newaxis]
            kernel = kernel.reshape(3,3,2,1)
        
        gray = tf.reduce_mean([inpsRW, inpsGW, inpsBW], axis=0)
        Cw = tf.keras.layers.DepthwiseConv2D((3,3), strides=(1,1), padding='same', use_bias=False, depthwise_initializer=lambda shape,dtype : kernel) (gray)
        Cw = tf.abs(Cw)+1
        W = W + Cw
    
    if we==1:
        func_exposure = lambda x: tf.exp(-(x-0.5)**2/0.08)
        Ew = func_exposure(inpsRW)*func_exposure(inpsGW)*func_exposure(inpsBW) + 1
        W = W + Ew
        
    if ws==1:
        Sw = tf.math.reduce_std([inpsRW, inpsGW, inpsBW], axis=0) + 1
        W = W + Sw
    
    norm = tf.reduce_sum(W, axis=-1, keepdims=True)+tf.keras.backend.epsilon()
    weight_maps = W/norm
    
    # TODO: ADJUST THE KERNEL VALUES!
    Gn = tf.keras.layers.DepthwiseConv2D((5,5), strides=(1,1), padding='same', use_bias=False) (weight_maps)
    
    inpsRGB = tf.concat([inpsR, inpsG, inpsB], axis=0)
    imgLpyr, wGpyr = multires_pyramid(inpsRGB, weight_maps, 2)
    
    L1 = imgLpyr[1]*wGpyr[1]
    L1 = tf.image.resize(L1, (HEIGHT, WIDTH))
    L1 = tf.abs(L1)
    
    Gn = tf.tile(Gn ,[3,1,1,1])

    RGB_fuse = inpsRGB *(Gn + 0.2*L1)
    RGB_fuse = tf.reduce_sum(RGB_fuse, axis=-1)
    RGB_fuse = tf.transpose(RGB_fuse, [2,1,0])
        
    model = tf.keras.Model([R_inputs, G_inputs, B_inputs], RGB_fuse)

    convert_model(model)
    
def SSF_YUV(NR_FRAMES, MERGE, wc, we):
    Y_inputs  = tf.keras.layers.Input((HEIGHT,WIDTH,NR_FRAMES+1), batch_size=1, name='Y_input')
    U_inputs  = tf.keras.layers.Input((HEIGHT,WIDTH,NR_FRAMES+1), batch_size=1, name='U_input')
    V_inputs  = tf.keras.layers.Input((HEIGHT,WIDTH,NR_FRAMES+1), batch_size=1, name='V_input')

    inpsY, inpsU, inpsV = Y_inputs, U_inputs, V_inputs

    if MERGE in ['mean', 'median']:
        inpsY, inpsU, inpsV = merge_frames(inpsY, inpsU, inpsV, MERGE)
        
    LEVELS = int(np.log(min(inpsY.shape[1:-1]))/np.log(2))

    # compute weights
    inpsYW = inpsY
    inpsUW = inpsU
    inpsVW = inpsV
    
    W = tf.ones_like(inpsYW)
    
    if wc==1:
        kernel = [
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ]
        if MERGE is None:
            kernel = np.array([kernel]*(NR_FRAMES+1))[...,np.newaxis]
            kernel = kernel.reshape(3,3,NR_FRAMES+1,1)
        else:
            kernel = np.array([kernel]*2)[...,np.newaxis]
            kernel = kernel.reshape(3,3,2,1)

        Cw = tf.keras.layers.DepthwiseConv2D((3,3), strides=(1,1), padding='same', use_bias=False, depthwise_initializer=lambda shape,dtype : kernel) (inpsYW)
        Cw = tf.abs(Cw)+1
        
        W = W * Cw
    
    if we==1:
        #Ew = tf.exp(-(inpsYW-0.5)**2/0.08)
        Ew = tf.abs(inpsUW+tf.keras.backend.epsilon())*tf.abs(inpsVW+tf.keras.backend.epsilon())+1
        W = W * Ew

    norm = tf.reduce_sum(W, axis=-1, keepdims=True)+tf.keras.backend.epsilon()
    weight_maps = W/norm
    
    U_fuse = tf.reduce_max(inpsU, axis=-1, keepdims=True)
    V_fuse = tf.reduce_max(inpsV, axis=-1, keepdims=True)
    
    # TODO: ADJUST THE KERNEL VALUES!
    Gn = tf.keras.layers.DepthwiseConv2D((5,5), strides=(1,1), padding='same', use_bias=False) (weight_maps)
    
    imgLpyr, wGpyr = multires_pyramid(inpsY, weight_maps, 2)
    
    L1 = imgLpyr[1]*wGpyr[1]
    L1 = tf.image.resize(L1, (HEIGHT, WIDTH))
    L1 = tf.abs(L1)

    Y_fuse = inpsY *(Gn + 0.2*L1)
    Y_fuse = tf.reduce_sum(Y_fuse, axis=-1)
    
    model = tf.keras.Model([Y_inputs, U_inputs, V_inputs], [Y_fuse, U_fuse, V_fuse])

    convert_model(model)