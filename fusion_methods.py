import cv2
import numpy as np

def YUV_convertion_test(frames):
    ys, us, vs = [],[], []
    for img in frames:
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y,u,v = cv2.split(yuv)
        
        ys.append(y)
        us.append(u)
        vs.append(v)
        
    return cv2.cvtColor(np.stack([ys[0], us[0], vs[0]], axis=-1), cv2.COLOR_YUV2BGR)

def YUV_fusion(frames):

    assert len(frames)==2
    
    neg = np.float32(frames[0]/255.)
    pos = np.float32(frames[1]/255.)
    
    yuv_neg = cv2.cvtColor(neg, cv2.COLOR_BGR2YUV)
    y_neg, u_neg, v_neg = cv2.split(yuv_neg)
    yuv_pos = cv2.cvtColor(pos, cv2.COLOR_BGR2YUV)
    y_pos, u_pos, v_pos = cv2.split(yuv_pos)
        
    uv_neg = cv2.merge([u_neg, v_neg])-0.5
    uv_pos = cv2.merge([u_pos, v_pos])-0.5
    
    uv_fusion_neg = np.clip(np.min([uv_neg, uv_pos], axis=0), -1, 0)
    uv_fusion_pos = np.clip(np.max([uv_neg, uv_pos], axis=0), 0, 1)
    
    uv_fusion = np.clip(uv_fusion_neg + uv_fusion_pos + 0.5, 0, 1)
    
    y_blur_neg = cv2.resize(y_neg, (y_neg.shape[1]//4, y_neg.shape[0]//4), interpolation=cv2.INTER_LINEAR)
    y_blur_neg = cv2.blur(y_blur_neg, (3,3))
    y_blur_neg = cv2.resize(y_blur_neg, (y_neg.shape[1], y_neg.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    lap_image = y_neg - y_blur_neg
    y_fusion = y_pos + lap_image
    y_fusion = np.clip(y_fusion + 0.02*y_neg, 0, 1)
    
    res = cv2.cvtColor(cv2.merge([y_fusion, uv_fusion]), cv2.COLOR_YUV2BGR)
    
    res = np.uint8(255*np.clip(res, 0, 1))

    return res

def multires_pyramid(image, weight_map, levels):
    
    levels  = levels - 1
    imgGpyr = [image]
    wGpyr   = [weight_map]
    
    for i in range(levels):
        imgW = np.shape(imgGpyr[i])[1]
        imgH = np.shape(imgGpyr[i])[0]
        imgGpyr.append(cv2.pyrDown(imgGpyr[i].astype('float64')))
        
    for i in range(levels):
        imgW = np.shape(wGpyr[i])[1]
        imgH = np.shape(wGpyr[i])[0]
        wGpyr.append(cv2.pyrDown(wGpyr[i].astype('float64')))

    imgLpyr = [imgGpyr[levels]]
    
    for i in range(levels, 0, -1):
        imgW = np.shape(imgGpyr[i-1])[1]
        imgH = np.shape(imgGpyr[i-1])[0]
        imgLpyr.append(imgGpyr[i-1] - cv2.resize(cv2.pyrUp(imgGpyr[i]),(imgW,imgH)))

    return imgLpyr[::-1], wGpyr

def Mertens(images, ws, we, wc):
    
    levels = int(np.log(min(images[0].shape[:2]))/np.log(2))

    images = np.float32(images)/255.

    # compute weights
    W = np.ones_like(images[...,0]).astype('float32')
    if ws==1:
        W *= np.std(images, axis=-1)+1
    if we==1:
        W *= np.prod(np.exp(-((images - 0.5)**2)/(2*0.04)), axis=-1, dtype=np.float32)+1
    if wc==1:
        W *= np.array([np.abs(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_32F)) for img in images])+1
    
    norm = np.sum(W, axis=0)+1e-6
    weight_maps = np.stack([w/norm for w in W], axis=-1)

    image_stack = images
    D = np.shape(image_stack)[0]
    
    imgPyramids = []    
    wPyramids = []
    for i in range(D):
        imgLpyr, wGpyr = multires_pyramid(image_stack[i], weight_maps[:,:,i], levels)
        imgPyramids.append(imgLpyr)
        wPyramids.append(wGpyr)
    
    blendedPyramids = []
    for i in range(D):
        blended_multires = []
        for j in range(levels):
            blended_multires.append(imgPyramids[i][j] * wPyramids[i][j])
        blendedPyramids.append(blended_multires)
    
    finalPyramid = [] 
    for i in range(levels):
        intermediate = []
        tmp = np.zeros_like(blendedPyramids[0][i])        
        for j in range(D):
            tmp += np.array(blendedPyramids[j][i])
        intermediate.append(tmp)
        finalPyramid.append(intermediate)
    
    finalImage = []
    blended_final = np.array(finalPyramid[0][0])
    for i in range(levels-1):
        imgH = np.shape(image_stack[0])[0] 
        imgW = np.shape(image_stack[0])[1] 
        layerx = cv2.pyrUp(finalPyramid[i+1][0])
        blended_final += cv2.resize(layerx,(imgW,imgH))
    
    img = np.clip(blended_final*255, 0, 255).astype('uint8')
    
    return img

def Fast_YUV(images, wc, we):

    levels = int(np.log(min(images[0].shape[:2]))/np.log(2))

    ys, us, vs = [],[], []
    for img in images:
        yuv = cv2.cvtColor(np.float32(img)/255., cv2.COLOR_BGR2YUV)
        y,u,v = cv2.split(yuv)
        
        ys.append(y)
        us.append(u)
        vs.append(v)

    # compute weights
    W = np.ones_like(np.array(images)[...,0]).astype('float32')
    if we==1:
        W *= np.array([np.abs(u+1e-6)*np.abs(v+1e-6) for u,v in zip(us,vs)])+1
    if wc==1:
        W *= np.array([np.abs(cv2.Laplacian(y, cv2.CV_32F)) for y in ys])+1
    
    norm = np.sum(W, axis=0)+1e-6
    weight_maps = np.stack([w/norm for w in W], axis=-1)

    Uf = np.max(us, axis=0)
    Vf = np.max(vs, axis=0)

    image_stack = ys
    D = np.shape(image_stack)[0]
    
    imgPyramids = []    
    wPyramids = []
    for i in range(D):
        imgLpyr, wGpyr = multires_pyramid(image_stack[i], weight_maps[:,:,i], levels)
        imgPyramids.append(imgLpyr)
        wPyramids.append(wGpyr)
    
    blendedPyramids = []
    for i in range(D):
        blended_multires = []
        for j in range(levels):
            blended_multires.append(imgPyramids[i][j] * wPyramids[i][j])
        blendedPyramids.append(blended_multires)
    
    finalPyramid = [] 
    for i in range(levels):
        intermediate = []
        tmp = np.zeros_like(blendedPyramids[0][i])        
        for j in range(D):
            tmp += np.array(blendedPyramids[j][i])
        intermediate.append(tmp)
        finalPyramid.append(intermediate)
    
    finalImage = []
    blended_final = np.array(finalPyramid[0][0])
    for i in range(levels-1):
        imgH = np.shape(image_stack[0])[0] 
        imgW = np.shape(image_stack[0])[1] 
        layerx = cv2.pyrUp(finalPyramid[i+1][0])
        blended_final += cv2.resize(layerx,(imgW,imgH))
    
    Yf = blended_final

    img = cv2.cvtColor(np.stack([Yf, Uf, Vf], axis=-1).astype('float32'), cv2.COLOR_YUV2BGR)
    img = np.clip(img*255, 0, 255).astype('uint8')
    
    return img

def SSF_YUV(images, wc, we):
    
    levels = int(np.log(min(images[0].shape[:2]))/np.log(2))

    ys, us, vs = [],[],[]
    for img in images:
        yuv = cv2.cvtColor(np.float32(img)/255., cv2.COLOR_BGR2YUV)
        y,u,v = cv2.split(yuv)

        ys.append(y)
        us.append(u)
        vs.append(v)

    # compute weights
    W = np.zeros_like(np.array(images)[...,0]).astype('float32')
    if we==1:
        W += np.array([np.abs(u+1e-6)*np.abs(v+1e-6) for u,v in zip(us,vs)])+1
    if wc==1:
        W += np.array([np.abs(cv2.Laplacian(y, cv2.CV_32F)) for y in ys])+1
    
    norm = np.sum(W, axis=0)+1e-6
    weight_maps = np.stack([w/norm for w in W], axis=-1)

    Uf = np.max(us, axis=0)
    Vf = np.max(vs, axis=0)

    kernel = cv2.getGaussianKernel(5, sigma=levels)
    Gn = cv2.sepFilter2D(weight_maps, -1, kernel, kernel)

    image_stack = ys
    D = np.shape(image_stack)[0]
    
    imgPyramids = []    
    wPyramids = []
    for i in range(D):
        imgLpyr, wGpyr = multires_pyramid(image_stack[i], weight_maps[:,:,i], 2)
        imgPyramids.append(imgLpyr)
        wPyramids.append(wGpyr)
    
    Yf = np.zeros_like(image_stack[0])
    for i in range(D):
        Yf += image_stack[i]*( Gn[...,i] + 0.2 * np.abs(cv2.resize(imgPyramids[i][0], (Yf.shape[1],Yf.shape[0]))))

    img = cv2.cvtColor(np.stack([Yf, Uf, Vf], axis=-1).astype('float32'), cv2.COLOR_YUV2BGR)
    img = np.clip(img*255, 0, 255).astype('uint8')
    
    return img

def SSF_BGR(images, ws, we, wc):
    
    levels = int(np.log(min(images[0].shape[:2]))/np.log(2))

    images = np.float32(images)/255.

    # compute weights
    W = np.zeros_like(images[...,0]).astype('float32')
    if ws==1:
        W += np.std(images, axis=-1)+1
    if we==1:
        W += np.prod(np.exp(-((images - 0.5)**2)/(2*0.2)), axis=-1, dtype=np.float32)+1
    if wc==1:
        W += np.array([np.abs(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_32F)) for img in images])+1
    
    norm = np.sum(W, axis=0)+1e-6
    weight_maps = np.stack([w/norm for w in W], axis=-1)

    kernel = cv2.getGaussianKernel(5, sigma=levels)
    Gn = cv2.sepFilter2D(weight_maps, -1, kernel, kernel)

    image_stack = images
    D = np.shape(image_stack)[0]
    
    imgPyramids = []    
    for i in range(D):
        imgLpyr, wGpyr = multires_pyramid(image_stack[i], weight_maps[:,:,i], 2)
        imgPyramids.append(imgLpyr)
    
    SSF = np.zeros_like(image_stack[0])
    for i in range(D):
        SSF += image_stack[i]*( cv2.merge([Gn[...,i]]*3) + 0.2 * np.abs(cv2.resize(imgPyramids[i][0], (SSF.shape[1],SSF.shape[0]))))
    
    img = np.clip(SSF*255, 0, 255).astype('uint8')
    
    return img
