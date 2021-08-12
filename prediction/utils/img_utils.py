# logging of training

# TO DO
# set means, stds
# set label_colours
import numpy as np
import matplotlib.pyplot as plt

# set colors for annotations
Void = [0,0,0] # gray
Thinning_1 = [225,27,27] # red
#Thinning_2 = [250,250,15] # yellow
Forest = [60,200,60] # green
No_Forest = [30,90,215] # blue

#label_colours = np.array([Void, Thinning_1, Thinning_2, Forest, No_Forest])
label_colours = np.array([Void, Thinning_1, Forest, No_Forest])

means_tams = np.array([56.12055784563426, 62.130400134006976, 53.03228547781888, 119.50916281232037], dtype='float32')
stds_tams = np.array([30.37628560708646, 30.152693706272483, 23.13718651792004, 49.301477498205074], dtype='float32')

means_dsm = np.array([13.45]).astype(np.float32)
stds_dsm = np.array([10.386633252098674]).astype(np.float32)

def view_annotated(tensor, plot=True):
    ''' set rgb colors for annotation types'''
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
    if img_type == 'cir':
        # change channels to NIR,R,G,B
        img = np.roll(img, 1, axis=2)
        # slice NIR,R,G
        return img[:,:,:3]

def decode_img(tensor):
    inp = tensor.numpy()
    #mean = np.array(DSET_MEAN)
    #std = np.array(DSET_STD)
    inp = stds_dsm * inp + means_dsm
    return inp

def view_sample(loader, n):
    # get data
    inputs, targets = next(iter(loader))
    batch_size = inputs.size(0)
    # print data
    for i in range(min(n, batch_size)):
        fig, ax = plt.subplots(1,4, figsize=(15,9))
        ax[0].imshow(decode_ortho(inputs[i,:4], 'rgb'))
        ax[1].imshow(decode_ortho(inputs[i,:4], 'cir'))
        ax[2].imshow(decode_img(inputs[i,4]))
        ax[3].imshow(view_annotated(targets[i],plot=False))

def show_predictions(inputs, targets, pred, batch_size, n):
    for i in range(min(n, batch_size)):
        fig, ax = plt.subplots(1,5, figsize=(15,9))
        ax[0].imshow(decode_ortho(inputs[i,:4], 'rgb'))
        ax[1].imshow(decode_ortho(inputs[i,:4], 'cir'))
        ax[2].imshow(decode_img(inputs[i,4]))
        ax[3].imshow(view_annotated(targets[i],plot=False))
        ax[4].imshow(view_annotated(pred[i],plot=False))
