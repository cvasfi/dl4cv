import torch
import torchvision.models as models
import os

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
#plt.ioff()
# from matplotlib import pyplot as plt
# import matplotlib
# matplotlib.use('Agg')
from models import VGG, ResNetCifar10, BKVGG12, AlexNet
#source of the code : https://discuss.pytorch.org/t/understanding-deep-network-visualize-weights/2060/8

def plot_kernels(tensor, num_cols=6):
    if not tensor.ndim == 4:
        raise Exception("assumes a 4D tensor")
    if not tensor.shape[-1] == 3:
        raise Exception("last dim needs to be 3 to plot")
    num_kernels = tensor.shape[0]
    num_rows = 1 + num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols, num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows, num_cols, i + 1)
        ax1.imshow(tensor[i])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    fig.savefig('my_plot_1.png')


#vgg = models.vgg16(pretrained=True)
the_model = BKVGG12()
the_model.load_state_dict(torch.load('/wholebrain/scratch/mahsabh/DLCV/dl4cv/log/bkvgg12_epoch200.model'))
#the_model = torch.load('/wholebrain/scratch/mahsabh/DLCV/dl4cv/log/bkvgg12_epoch200.model')
print(type(the_model))
#print(the_model.keys)
#mm = vgg.double()
mm = the_model.double()
filters = mm.modules
body_model = [i for i in mm.children()][0]
layer1 = body_model[0]
#print(type(layer1))
tensor = layer1.weight.data.numpy()
#print(tensor)
#fix from here : https://stackoverflow.com/questions/47318871/valueerror-floating-point-image-rgb-values-must-be-in-the-0-1-range-while-usi
new = (1/(2*2.25)) * tensor + 0.5
plot_kernels(new)