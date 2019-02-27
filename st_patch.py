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

from utility.utility import postp, GramMatrix, GramMSELoss, load_images, save_images, make_folders
from utility.loss_fns import get_style_patch_weights, patch_difference, mrf_loss_fn, weight_maker, smoothnes_loss
from utility.vgg_network import VGG
#############################################################################
# PARSER
parser = argparse.ArgumentParser(description='A Neural Algorithm of Artistic Style')
# parameters
parser.add_argument('--alpha', '-alpha', type=float, default = 1, help='parameter for content loss')
parser.add_argument('--beta', '-beta', type=float, default = 1, help='parameter for style loss')
# Parser for style weights
parser.add_argument('--sw1', '-sw1', type=float,  default=1, help='sw1')
parser.add_argument('--sw2', '-sw2', type=float,  default=1, help='sw2')
parser.add_argument('--sw3', '-sw3', type=float,  default=1, help='sw3')
parser.add_argument('--sw4', '-sw4', type=float,  default=1, help='sw4')
parser.add_argument('--sw5', '-sw5', type=float,  default=1, help='sw5')
# parser for content weights
parser.add_argument('--cw1', '-cw1', type=float,  default=0, help='cw1')
parser.add_argument('--cw2', '-cw2', type=float,  default=0, help='cw2')
parser.add_argument('--cw3', '-cw3', type=float,  default=0, help='cw3')
parser.add_argument('--cw4', '-cw4', type=float,  default=1, help='cw4')
parser.add_argument('--cw5', '-cw5', type=float,  default=0, help='cw5')
# parser for input images paths and names
parser.add_argument('--image_size', '-image_size', type=int, default=256)
# parser for input images paths and names
parser.add_argument('--content_path', '-content_path', type=str, default='../input/font_contents/sanserifs/AdventPro-Medium.png')
parser.add_argument('--serif_style_path', '-serif_style_path', type=str, default='../input/font_contents/curvy_fonts/CherrySwash-Bold.png')
parser.add_argument('--nonserif_style_path', '-nonserif_style_path', type=str, default='../input/font_contents/striped_curvy_fonts/CherrySwash-Bold.png')
# parser for output path
parser.add_argument('--output_path', '-output_path', type=str, default='../output_style_difference/', help='Path to save output files')
# parser for cuda
parser.add_argument('--cuda', '-cuda', type=str, default='cuda:0', help='cuda:0 or cuda:x')

args = parser.parse_args()
#############################################################################
# Get image paths and names
# Style 1
style_dir1  = os.path.dirname(args.serif_style_path)
style_name1 = os.path.basename(args.serif_style_path)
# Style 2
style_dir2  = os.path.dirname(args.nonserif_style_path)
style_name2 = os.path.basename(args.nonserif_style_path)
# Content
content_dir  = os.path.dirname(args.content_path)
content_name = os.path.basename(args.content_path)

# Cuda device
if torch.cuda.is_available:
    device = args.cuda
else:
    device = 'cpu'
print("Using device: ", device)

# style weights
sw1=args.sw1
sw2=args.sw2
sw3=args.sw3
sw4=args.sw4
sw5=args.sw5
# Content weights
cw1=args.cw1
cw2=args.cw2
cw3=args.cw3
cw4=args.cw4
cw5=args.cw5
# Parameters
alpha = args.alpha
beta = args.beta
patch_size = 5
image_size = args.image_size
content_invert = 1
style_invert = 1
result_invert = content_invert

# Get output path
output_path = args.output_path
try:
    os.mkdir(output_path)
except:
    pass
output_path = output_path + content_name[:-4] + '_' + style_name1[:-4] + '_' + style_name2[:-4] + '/'

# Get network
vgg = VGG()
vgg.load_state_dict(torch.load('../Models/vgg_conv.pth'))
for param in vgg.parameters():
    param.requires_grad = False
vgg.to(device)

# Load images
content_image = load_images(os.path.join(content_dir, content_name), image_size, device, content_invert)
style_image1  = load_images(os.path.join(style_dir1,style_name1), image_size, device, style_invert)
style_image2  = load_images(os.path.join(style_dir2,style_name2), image_size, device, style_invert)

# Random input
opt_img = Variable(torch.randn(content_image.size()).type_as(content_image.data).to(device), requires_grad=True).to(device)
# Content input
# opt_img = Variable(content_image.data.clone(), requires_grad=True)

# Define layers, loss functions, weights and compute optimization targets
# Style layers
# style_layers = ['r11','r21','r31','r41','r51'] 
style_layers = []
style_weights = [sw*1e3/n**2 for sw,n in zip([sw1,sw2,sw3,sw4,sw5],[64,128,256,512,512])]
# Content layers
# content_layers = ['r12','r22','r32','r42','r52']
# content_weights = [cw1*1e5,cw2*1e5,cw3*1e5,cw4*1e5,cw5*1e5]
content_layers = ['r42']
content_weights = [cw4*1e4]
# Patch layers
patch_layers = ['r31','r41']


loss_layers = style_layers + content_layers + patch_layers
loss_functions = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
loss_functions = [loss_fn.to(device) for loss_fn in loss_functions]
weights = style_weights + content_weights

# Compute optimization targets

## Gram matrix targets
#### Feature maps from style layers of the style images
style1_fms_style = [A.detach() for A in vgg(style_image1, style_layers)]
style2_fms_style = [A.detach() for A in vgg(style_image2, style_layers)]
#### Difference between feature maps
style_fms_style  = [style1_fms_style[i] - style2_fms_style[i] for i in range(len(style_layers))]
#### Gram matrix of difference feature maps
gramm_style = [GramMatrix()(A) for A in style_fms_style]
#### Feature maps from style layers of the content image
content_fms_style = [A.detach() for A in vgg(content_image, style_layers)]

style_target = [GramMatrix()(style1_fms_style[i])-GramMatrix()(style2_fms_style[i]) for i in range(len(style_layers))]


## Content targets

#### Feature maps from content layers of the style images
style1_fms_content = [A.detach() for A in vgg(style_image1, content_layers)]
style2_fms_content = [A.detach() for A in vgg(style_image2, content_layers)]
#### Difference between feature maps
style_fms_content = [style1_fms_content[i] - style2_fms_content[i] for i in range(len(content_layers))]
#### Feature maps from content layers of the content image
content_fm_content = [A.detach() for A in vgg(content_image, content_layers)]


## Patch targets

#### Feature maps from patch layers of the style images
style1_fms_patch = [A.detach() for A in vgg(style_image1, patch_layers)]
style2_fms_patch = [A.detach() for A in vgg(style_image2, patch_layers)]
#### Patches extracted from the feature maps
style1_patches_lists, style1_weight_list = get_style_patch_weights(style1_fms_patch, device, k=patch_size)
style2_patches_lists, style2_weight_list = get_style_patch_weights(style2_fms_patch, device, k=patch_size)
#### Difference between corresponding patches
diff_patches_list1, diff_patches_list2 = patch_difference(style1_patches_lists,style2_patches_lists)
#### Different patches between style1 & style2
style1_different_patches256 = []
style2_different_patches256 = []
for p1, p2 in zip(style1_patches_lists[0], style2_patches_lists[0]):
    if not torch.eq(p1,p2).all():
        style1_different_patches256.append(p1)
        style2_different_patches256.append(p2)
style1_different_patches512 = []
style2_different_patches512 = []
for p1, p2 in zip(style1_patches_lists[1], style2_patches_lists[1]):
    if not torch.eq(p1,p2).all():
        style1_different_patches512.append(p1)
        style2_different_patches512.append(p2)
style1_different_patches = [style1_different_patches256, style1_different_patches512]

#### Feature maps from patch layers of the content image
content_fms_patch = [A.detach() for A in vgg(content_image, patch_layers)]
#### Patches extracted from the feature maps
content_patches_lists, content_weight_list = get_style_patch_weights(style1_fms_patch, device, k=patch_size)

#### Integrate different patches to content patches
combined_patches_lists = [content_patches_lists[0]+style1_different_patches256,content_patches_lists[1]+style1_different_patches512]
#### Create weights to use for convolution, in per channel size
weight_list = [weight_maker(style_plist,patch_size,device) for style_plist in combined_patches_lists]

# Run style transfer
make_folders(output_path)

max_iter = 600
show_iter = 50
optimizer = optim.LBFGS([opt_img])
n_iter=[0]
loss_list = []
c_loss = []
s_loss = []
p_loss = []

while n_iter[0] <= max_iter:

    def closure():
        optimizer.zero_grad()
        out = vgg(opt_img, loss_layers)
        content_layer_losses = []
        style_layer_losses  = []
        
        # Divide between style feature maps, content feature maps and patch feature maps
        opt_fms_style = out[:len(style_layers)]
        opt_fms_content = out[len(style_layers):len(style_layers)+len(content_layers)]
        opt_fms_patch = out[-len(patch_layers):]

        # Difference between feature maps on style layers
        diff_fms_style = [opt_fms_style[i] - content_fms_style[i] for i in range(len(style_layers))]
        gramm_diff = [GramMatrix()(A) for A in diff_fms_style]
        # Difference between gram matrix of feature map differences
        style_layer_losses = [style_weights[i]*(nn.MSELoss()(gramm_diff[i], gramm_style[i])) for i in range(len(style_layers))]

        ## Difference between feature maps on content layers
        fms_diff = [opt_fms_content[i] - content_fm_content[i] for i in range(len(content_layers))]
        content_layer_losses = [content_weights[i]*nn.MSELoss()(fms_diff[i],style_fms_content[i]) for i in range(len(content_layers))]


        ## Differnce between the patches
        #### Patches extracted from the opt feature maps
        opt_patches_lists, opt_weight_list = get_style_patch_weights(opt_fms_patch, device, k=patch_size)
        #### Difference between corresponding patches with content patches
        diff_patches_list1_1, diff_patches_list2_2 = patch_difference(opt_patches_lists,content_patches_lists)

        #### Patch loss
        # patch_layer_losses = []
        # for i in range(len(diff_patches_list1)):
        #     patch_layer_losses.append(nn.MSELoss()(diff_patches_list1_1[i],diff_patches_list1[i]))
        # for i in range(len(diff_patches_list2)):
        #     patch_layer_losses.append(nn.MSELoss()(diff_patches_list2_2[i],diff_patches_list2[i]))
        patch_layer_losses = [mrf_loss_fn(opt_fms_patch, combined_patches_lists, weight_list, k=patch_size)]

        # Regularzier
        regularizer = smoothnes_loss(opt_img)

        # losses
        content_loss = sum(content_layer_losses)
        style_loss   = sum(style_layer_losses)
        patch_loss   = sum(patch_layer_losses)

        loss = style_loss + content_loss + patch_loss + 0.001*regularizer
        # layer_losses = content_layer_losses + style_layer_losses + patch_layer_losses

        # # total loss
        # loss = sum(layer_losses)

        # for log
        c_loss.append(content_loss)
        s_loss.append(style_loss)
        p_loss.append(patch_loss)
        loss_list.append(loss)

        # backward calculation
        loss.backward()

        #print loss
        if n_iter[0]%show_iter == 0:
            print('Iteration: {}'.format(n_iter[0]))
            if len(content_layers)>0: print('Content loss: {}'.format(content_loss.item()))
            if len(style_layers)>0:   print('Style loss  : {}'.format(style_loss.item()))
            if len(patch_layers)>0:   print('Patch loss  : {}'.format(patch_loss))
            print('Total loss  : {}'.format(loss.item()))

            # Save loss graph
            # plt.plot(loss_list, label='Total loss')
            if len(content_layers)>0:  plt.plot(c_loss, label='Content loss')
            if len(style_layers)  >0:  plt.plot(s_loss, label='Style loss')
            if len(patch_layers)  >0:  plt.plot(p_loss, label='Patch loss')
            plt.legend()
            plt.savefig(output_path + 'loss_graph.jpg')
            plt.close()
            # Save optimized image
            out_img = postp(opt_img.data[0].cpu().squeeze(), image_size, result_invert)
            out_img.save(output_path + 'outputs/{}.bmp'.format(n_iter[0]))
        n_iter[0] += 1
        return loss
      
    optimizer.step(closure)

# Save sum images
save_images(content_image.data[0].cpu().squeeze(), opt_img.data[0].cpu().squeeze(), style_image1.data[0].cpu().squeeze(), style_image2.data[0].cpu().squeeze(), image_size, output_path, n_iter, content_invert, style_invert, result_invert)
