# logging of training

# TO DO
# set means, stds
# set label_colours
import numpy as np
import matplotlib.pyplot as plt

# set colors for annotations
Void = [0,0,0] # gray
Thinning_1 = [225,27,27] # red
Thinning_2 = [250,250,15] # yellow
Forest = [60,200,60] # green
No_Forest = [30,90,215] # blue

label_colours_5 = np.array([Void, Thinning_1, Thinning_2, Forest, No_Forest])
label_colours_4 = np.array([Void, Thinning_1, Forest, No_Forest])

#------------------------------------------------------------
# ortho
means_tams = np.array([56.12055784563426, 62.130400134006976, 53.03228547781888, 119.50916281232037], dtype='float32')
stds_tams = np.array([30.37628560708646, 30.152693706272483, 23.13718651792004, 49.301477498205074], dtype='float32')
# dsm
means_dsm = np.array([13.45]).astype(np.float32)
stds_dsm = np.array([10.386633252098674]).astype(np.float32)
# dtm
means_dtm = np.array([1446.0]).astype(np.float32)
stds_dtm = np.array([271.05322202384195]).astype(np.float32)
# slope
means_slope = np.array([22.39]).astype(np.float32)
stds_slope = np.array([11.69830556896441]).astype(np.float32)
#------------------------------------------------------------

def view_annotated(tensor, plot=True, classes=4):
    ''' set rgb colors for annotation types'''
    if classes == 5:
        label_colours = label_colours_5
    else:
        label_colours = label_colours_4
        
    temp = tensor.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, label_colours.shape[0]):
        r[temp==l]=label_colours[l,0]
        g[temp==l]=label_colours[l,1]
        b[temp==l]=label_colours[l,2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:,:,0] = (r/255.0)#[:,:,0]
    rgb[:,:,1] = (g/255.0)#[:,:,1]
    rgb[:,:,2] = (b/255.0)#[:,:,2]
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

def decode_ortho(tensor, img_type='rgb'):
    # transform dimensions for plotting [C,H,W] -> [H,W,C]
    img = tensor.permute(1, 2, 0).numpy()
    # denormalize image
    img = (stds_tams * img + means_tams).astype(np.uint8)
    #
    if img_type == 'rgb':
        # slice R,G,B
        return img[:,:,:3]
    elif img_type == 'cir':
        # change channels to NIR,R,G,B
        img = np.roll(img, 1, axis=2)
        # slice NIR,R,G
        return img[:,:,:3]

def decode_img(tensor, img_type='dsm'):
    inp = tensor.numpy()
    #mean = np.array(DSET_MEAN)
    #std = np.array(DSET_STD)
    if img_type == 'dsm':
        means = means_dsm
        stds = stds_dsm
    elif img_type == 'dtm':
        means = means_dtm
        stds = stds_dtm
    elif img_type == 'slope':
        means = means_slope
        stds = stds_slope
        
    inp = stds * inp + means
    return inp

def view_sample(loader, n):
    # get data
    inputs, targets = next(iter(loader))
    batch_size = inputs.size(0)
    # print data
    for i in range(min(n, batch_size)):
        '''
        fig, ax = plt.subplots(1,6, figsize=(21,9))
        ax[0].imshow(decode_ortho(inputs[i,:4], 'rgb'))
        ax[0].set_title("RGB")
        ax[0].axis('off')
        ax[1].imshow(decode_ortho(inputs[i,:4], 'cir'))
        ax[1].set_title("CIR")
        ax[1].axis('off')
        ax[2].imshow(decode_img(inputs[i,4], 'dsm'))
        ax[2].set_title("tree heights")
        ax[2].axis('off')
        
        ax[3].imshow(decode_img(inputs[i,5], 'dtm'))
        ax[3].set_title("DTM")
        ax[3].axis('off')
        ax[4].imshow(decode_img(inputs[i,6], 'slope'))
        ax[4].set_title("slope")
        ax[4].axis('off')
        
        ax[5].imshow(view_annotated(targets[i],plot=False))
        ax[5].set_title("Ground Truth")
        ax[5].axis('off')
        '''
        fs=15
        fig, ax = plt.subplots(2,3, figsize=(15,10))
        ax[0,0].imshow(decode_ortho(inputs[i,:4], 'rgb'))
        ax[0,0].set_title("RGB", fontsize=fs)
        ax[0,0].axis('off')
        ax[0,1].imshow(decode_ortho(inputs[i,:4], 'cir'))
        ax[0,1].set_title("CIR", fontsize=fs)
        ax[0,1].axis('off')
        ax[0,2].imshow(decode_img(inputs[i,4], 'dsm'))
        ax[0,2].set_title("CHM", fontsize=fs)
        ax[0,2].axis('off')
        
        ax[1,0].imshow(decode_img(inputs[i,5], 'dtm'))
        ax[1,0].set_title("DTM", fontsize=fs)
        ax[1,0].axis('off')
        ax[1,1].imshow(decode_img(inputs[i,6], 'slope'))
        ax[1,1].set_title("Slope", fontsize=fs)
        ax[1,1].axis('off')
        
        ax[1,2].imshow(view_annotated(targets[i],plot=False))
        ax[1,2].set_title("Ground Truth", fontsize=fs)
        ax[1,2].axis('off')

def show_predictions(inputs, targets, pred, batch_size, n):
    for i in range(min(n, batch_size)):
        fig, ax = plt.subplots(1,7, figsize=(21,9))
        ax[0].imshow(decode_ortho(inputs[i,:4], 'rgb'))
        ax[0].set_title("RGB")
        ax[0].axis('off')
        ax[1].imshow(decode_ortho(inputs[i,:4], 'cir'))
        ax[1].set_title("CIR")
        ax[1].axis('off')
        ax[2].imshow(decode_img(inputs[i,4], 'dsm'))
        ax[2].set_title("tree heights")
        ax[2].axis('off')
        
        ax[3].imshow(decode_img(inputs[i,5], 'dtm'))
        ax[3].set_title("DTM")
        ax[3].axis('off')
        ax[4].imshow(decode_img(inputs[i,6], 'slope'))
        ax[4].set_title("slope")
        ax[4].axis('off')
        
        ax[5].imshow(view_annotated(targets[i],plot=False))
        ax[5].set_title("Ground Truth")
        ax[5].axis('off')
        ax[6].imshow(view_annotated(pred[i],plot=False))
        ax[6].set_title("Prediction")
        ax[6].axis('off')
        
    plt.savefig('samples.png')