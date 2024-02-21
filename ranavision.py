import cv2 
import torch
import numpy as np
import torchvision
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch import *
from torchvision.transforms import v2

# ===================================================
# ========= FUNCIONES DE PREDICCION =================
# ===================================================
def getPantalla(im,model,preprocess,device='cpu'):
    def getBox(mask):
        proyx = mask.sum(axis=0)
        proyy = mask.sum(axis=1)

        minx,miny,maxx,maxy = 0,0,0,0
        for minx in range(len(proyx)):
            if proyx[minx]>0:   break

        for maxx in reversed(range(len(proyx))):
            if proyx[maxx]>0:   break
        
        for miny in range(len(proyy)):
            if proyy[miny]>0:   break

        for maxy in reversed(range(len(proyy))):
            if proyy[maxy]>0:   break
        
        return (minx,miny,maxx,maxy)
    
    transform_eval = A.Compose([A.Resize(693,520),
                            A.ToFloat(max_value=255),
                            ToTensorV2()])
    im2show = transform_eval(image=im)
    transformed = preprocess(im2show['image'])
    transformed = torch.unsqueeze(transformed,0).to(device)
    model.eval()
    mask = model(transformed)['out'].cpu().detach().numpy()[0,1]
    mask = 1*(mask>0.5)
    im2show = im2show['image'].cpu().detach().numpy()
    im2show = ( 255*np.transpose( im2show, [1,2,0] ) ).astype(np.uint8)

    box = getBox(mask)
    return im2show[box[1]:box[3],box[0]:box[2]]

def predictDigitos(pantalla,model_det,device,confianza=0.6):
    # realizar prediccion
    transform = A.Compose([A.ToFloat(max_value=255),ToTensorV2()])
    pantalla_ = transform(image=pantalla)['image']
    pantalla_ = torch.unsqueeze(pantalla_,0).to(device)
    out = model_det(pantalla_)[0]
    boxes = out['boxes']
    labels = out['labels']
    scores = out['scores']

    # ordenar prediccion
    pred = [ (l.item(),round(s.item(),3),b.cpu().detach().numpy()) for l,s,b in zip(labels,scores,boxes) if s.item() > confianza]
    pred = sorted(pred,key=lambda item:item[2][0])
    pred = [ (l,s,b) if l!=10 else (0,s,b) for l,s,b in pred ]

    return pred


# ===================================================
# =============== FUNCIONES UTILES ==================
# ===================================================

def formatDigitos(pred):
    try:
        return float(''.join([str(l) for l,s,b in pred]))
    except:
        return False

def compareDigitos(pred,number2compare,verbose=False):
    if type(number2compare)==str:
        number2compare = float(number2compare)

    conf = 0.99
    while conf>0.5:
        pred_ajustada = [ (l,s,b) for l,s,b in pred if s>conf ]
        pred_ajustada = formatDigitos(pred_ajustada)
        if verbose: 
            print(pred_ajustada)
        if pred_ajustada == number2compare:
            return conf
        conf -= 0.01
    return 0

# =========================================
# = STUFF A IMPORTAR EN ARCHIVO PRINCIPAL =
# =========================================

# # DEVICE
# device = 'cuda'         # cuda o cpu ; cuda es gpu

# # MODELO SEGMENTACION
# weights = torchvision.models.segmentation.FCN_ResNet50_Weights.DEFAULT
# preprocess = weights.transforms()
# model_seg = torchvision.models.segmentation.fcn_resnet50(num_classes=2).to(device)
# model_seg.load_state_dict(torch.load('ranavision_segmentation.pt', map_location=device))
# model_seg = model_seg.eval()

# # MODELO DETECCION
# model_det = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=11).to(device)
# model_det.load_state_dict(torch.load('ranavision_detection.pt', map_location=device))
# model_det = model_det.eval()
# pass