import argparse
import time
import os
import os.path

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import torchvision
from torchvision import transforms

import PIL
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict

from utility.utility import *
from utility.vgg_network import *
from utility.gram_matrix_visualization import *
#############################################################################
# PARSER
parser = argparse.ArgumentParser(description='A Neural Algorithm of Artistic Style')
# Parser for style weights
parser.add_argument('--sw1', '-sw1', type=float,  default='1',help='sw1')
parser.add_argument('--sw2', '-sw2', type=float,  default='1',help='sw2')
parser.add_argument('--sw3', '-sw3', type=float,  default='1',help='sw3')
parser.add_argument('--sw4', '-sw4', type=float,  default='1',help='sw4')
parser.add_argument('--sw5', '-sw5', type=float,  default='1',help='sw5')
# parser for content weights
parser.add_argument('--cw1', '-cw1', type=float,  default='0',help='cw1')
parser.add_argument('--cw2', '-cw2', type=float,  default='0',help='cw2')
parser.add_argument('--cw3', '-cw3', type=float,  default='0',help='cw3')
parser.add_argument('--cw4', '-cw4', type=float,  default='0',help='cw4')
parser.add_argument('--cw5', '-cw5', type=float,  default='0',help='cw5')
# parser for input images paths and names
parser.add_argument('--style_path1',  '-style_path1',  type=str, help='Path to style image 1')
parser.add_argument('--style_path2',  '-style_path2',  type=str, help='Path to style image 2')
parser.add_argument('--content_path', '-content_path', type=str, help='Path to content image')
parser.add_argument('--style_name1',  '-style_name1',  type=str, help='Name of style image 1')
parser.add_argument('--style_name2',  '-style_name2',  type=str, help='Name of style image 2')
parser.add_argument('--content_name', '-content_name', type=str, help='Name of content image')
# parser for output path
parser.add_argument('--output_path', '-output_path', type=str, default='../../output_from_pytorch/diff_style_transfer/', help='Path to save output files')

args = parser.parse_args()
#############################################################################
# Get image paths
# Style with serifs
style_path1 = args.style_path1
style_name1 = args.style_name1
style_dir1 = style_path1 + style_name1

# Style that lacks serifs
style_path2 = args.style_path2
style_name2 = args.style_name2
style_dir2 = style_path2 + style_name2

# Content 
content_path = args.content_path
content_name = args.content_name
content_dir = content_path + content_name

# Get output path
output_path = args.output_path
try:
    os.mkdir(output_path)
except:
    pass
output_path = output_path + content_name[:-4] + '_' + style_name1[:-4] + '_' + style_name2[:-4] 

if os.path.exists(output_path):
    print('Already done this experiment!')
    exit()

# Get network
vgg = VGG()
vgg.load_state_dict(torch.load('../../Models/vgg_conv.pth'))
for param in vgg.parameters():
    param.requires_grad = False
if torch.cuda.is_available():
    vgg.cuda()

# Load images
style_image1, style_image2, content_image, opt_img = load_images(style_dir1, style_dir2, content_dir)

# Define layers, loss functions, weights and compute optimization targets

# Style layers
style_layers = ['r11','r21','r31','r41','r51'] 
# style_weights = [0,0,0, 1e3/(512**2), 10]
style_weights = [args.sw1*1e3/(64**2), args.sw2*1e3/(128**2), args.sw3*1e3/(256**2), args.sw4*1e3/(512**2), args.sw5*1e3/(512**2)]
# Content layers
content_layers = ['r12','r22','r32','r42','r52']
content_weights = [args.cw1*10, args.cw2*10, args.cw3*10, args.cw4*10, args.cw5*10]

loss_layers = style_layers + content_layers
loss_functions = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
if torch.cuda.is_available():
    loss_functions = [loss_fn.cuda() for loss_fn in loss_functions]
weights = style_weights + content_weights

# Compute optimization targets
### Style targets 
# Gram matrices of 1st style
style_targets1 = [GramMatrix()(A).detach() for A in vgg(style_image1, style_layers)]
# Gram matrices of 2nd style
style_targets2 = [GramMatrix()(A).detach() for A in vgg(style_image2, style_layers)]
# Feature responces of 1st style
style_content1 = [A.detach() for A in vgg(style_image1, content_layers)]
# Feature responces of 2nd style
style_content2 = [A.detach() for A in vgg(style_image2, content_layers)]
style_targets = []
for i in range(len(style_targets1)):
    style_targets.append((style_targets2[i] - style_targets1[i]))
style_content = []
for i in range(len(style_content1)):
    style_content.append((style_content2[i] - style_content1[i]))
### Content targets
# Feature responces of content
content_targets = [A.detach() for A in vgg(content_image, content_layers)]
# Gram matrices of content
content_styles  = [GramMatrix()(A).detach() for A in vgg(content_image, style_layers)]

# Run style transfer
make_folders(output_path)


visualize_gm_all_layers(style_targets1, style_image1, output_path+'/gramm_style1.png')
visualize_gm_all_layers(style_targets2, style_image2, output_path+'/gramm_style2.png')
visualize_gm_all_layers(style_targets, style_image2-style_image1, output_path+'/gramm_style_diff.png')
visualize_gm_all_layers(content_styles, content_image, output_path+'/gramm_content.png')


max_iter = 3000
show_iter = 100
optimizer = optim.LBFGS([opt_img]);
n_iter=[0]
loss_list = []
c_loss = []
s_loss = []

while n_iter[0] <= max_iter:

    def closure():
        optimizer.zero_grad()
        out = vgg(opt_img, loss_layers)
        content_layer_losses = []
        style_layer_losses  = []
        
        opt_style = []
        opt_content = []
        # Divide between gram matrix and feature responce
        for i, A in enumerate(out):
            if i < len(style_targets):
                opt_style.append(GramMatrix()(A))
            else:
                opt_content.append(A)

        # Difference between the gram matrices of content and optimized
        results_style = []
        for i in range(len(content_styles)):
            results_style.append((content_styles[i] - opt_style[i]))
        # Difference between feature responces of content and optimized
        results_content = []
        for i in range(len(content_targets)):
            results_content.append((content_targets[i] - opt_content[i]))
        
        # Style loss between style_target and results
        for i in range(len(content_styles)):
            style_layer_losses.append(style_weights[i]*(nn.MSELoss()(results_style[i], style_targets[i])))
        # Content loss between style_content and results
        for i in range(len(content_targets)):
            content_layer_losses.append(content_weights[i]*(nn.MSELoss()(results_content[i], style_content[i])))
        
        layer_losses = content_layer_losses + style_layer_losses

        content_loss = sum(content_layer_losses)
        style_loss   = sum(style_layer_losses)

        loss = sum(layer_losses)
        loss.backward()
        c_loss.append(content_loss)
        s_loss.append(style_loss)
        loss_list.append(loss)

        #print loss
        if n_iter[0]%show_iter == 0:
            print('Iteration: {} \nContent loss: {} \nStyle loss  : {} \nTotal loss  : {}'
            .format(n_iter[0], content_loss.item(), style_loss.item(), loss.item()))

            # Save loss graph
            plt.plot(loss_list, label='Total loss')
            plt.plot(c_loss, label='Content loss')
            plt.plot(s_loss, label='Style loss')
            plt.legend()
            plt.savefig(output_path + '/loss_graph.jpg')
            plt.close()
            # Save optimized image
            out_img = postp(opt_img.data[0].cpu().squeeze())
            out_img = PIL.ImageOps.invert(out_img)
            out_img.save(output_path + '/outputs/{}.bmp'.format(n_iter[0]))
            # Save gram matrix
            # visualize_gm_separate_layers(style_targets1,style_targets2,style_targets,
            #                                 content_styles,opt_style,results_style,n_iter[0], output_path)
            visualize_gm_all_layers(results_style, opt_img.data.cpu().data, output_path+'/gramm_results.png')
        n_iter[0] += 1
        return loss
      
    optimizer.step(closure)

# Save sum images
save_images(content_image, opt_img, style_image1, style_image2, output_path, n_iter)
# # Make gif of transformation
# save_as_gif(output_path+'/outputs/',output_path+'/',max_iter, show_iter, filetype='bmp', savename='outputs.gif')
# for i in range(5):
#     save_as_gif(output_path+'/gm_layer{}/'.format(i+1), output_path+'/', max_iter, show_iter, filetype='png', savename='layer{}.gif'.format(i+1))