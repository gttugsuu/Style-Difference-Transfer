import torch
import numpy

import PIL
from PIL import Image

import matplotlib.pyplot as plt

def gm_layer(gm_of_layer, iter_n, output_path):
    layer_names=['style 1', 'style 2','style diff', 'content', 'optimized', 'content optimized diff']
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(21,14))
    plt.suptitle('step: {}'.format(iter_n), fontsize=40)
    for i in range(len(gm_of_layer)):
        gm_of_layer[i] = gm_of_layer[i].squeeze()
        if torch.cuda.is_available():
            gm_of_layer[i] = gm_of_layer[i].cpu().detach().numpy()
        else:
            gm_of_layer[i] = gm_of_layer[i].detach().numpy()
        gm_of_layer[i] = gm_of_layer[i].astype(int)
        # print gram matrices
        plt.subplot(2,3,i+1)
        plt.title(layer_names[i])
        plt.imshow(gm_of_layer[i], cmap='bwr')
    # add colorbar
    plt.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    plt.colorbar(cax=cbar_ax)
    # save & close figure
    plt.savefig(output_path+'/{}.png'.format(iter_n))
    plt.close()

# Visualize all GMs on each layer
def visualize_gm_separate_layers(s1_gram_list, s2_gram_list, s_diff_gram_list, c_gram_list, o_gram_list, co_diff_gram_list, iter_n, output_path):

    s1 = s1_gram_list.copy()
    s2 = s2_gram_list.copy()
    c  = c_gram_list.copy()
    o  = o_gram_list.copy()
    s_diff = s_diff_gram_list.copy()
    co_diff = co_diff_gram_list.copy()
    
    for i in range(len(s1)):
        gm_layer([s1[i], s2[i], s_diff[i], c[i], o[i], co_diff[i]],iter_n, output_path+'/gm_layer{}'.format(i+1))
