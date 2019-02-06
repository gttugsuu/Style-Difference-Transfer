import argparse
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

import torchvision
from torchvision import transforms

import PIL
import matplotlib.pyplot as plt
from tqdm import tqdm

from utility.utility import postp, GramMatrix, GramMSELoss, load_image, save_images, make_folders
from utility.vgg_network import VGG
from utility.loss_fns import get_style_patch_weights, smoothnes_loss, content_loss_fn, mrf_loss_fn

#############################################################################
# PARSER
parser = argparse.ArgumentParser(description='A Neural Algorithm of Artistic Style')
# parser for image size
parser.add_argument('--image_size', '-image_size', type=int, default=256)
# parser for input images paths and names
parser.add_argument('--content_path', '-content_path', type=str, default='../input/font_contents/AlegreyaSans-Light/A.png')
parser.add_argument('--style_path1', '-style_path1', type=str, default='../input/font_contents/serif/A/PT_Serif-Caption-Web-Regular.png')
parser.add_argument('--style_path2', '-style_path2', type=str, default='../input/font_contents/serif_rmv/A/PT_Serif-Caption-Web-Regular.png')
# parser for output path
parser.add_argument('--output_path', '-output_path', type=str, default='../output_style_difference/', help='Path to save output files')
# parser for cuda
parser.add_argument('--cuda', '-cuda', type=str, default='cuda:0', help='cuda:0 or cuda:x')

args = parser.parse_args()
#############################################################################

# Get image paths
# Content
content_dir  = os.path.dirname(args.content_path)
content_name = os.path.basename(args.content_path)
# Style 1
style_dir1  = os.path.dirname(args.style_path1)
style_name1 = os.path.basename(args.style_path1)
# Style 2
style_dir2  = os.path.dirname(args.style_path2)
style_name2 = os.path.basename(args.style_path2)

# Parameters
image_size = args.image_size
patch_size = 5
alpha = 1e-1
beta  = 1e-1
gamma = 1e-6
content_invert = 1
style_invert = 1
result_invert = content_invert

# Cuda device
if torch.cuda.is_available:
    device = args.cuda
else:
    device = 'cpu'
print("Using device: ", device)

# Output path
output_path = args.output_path
try:
    os.mkdir(output_path)
except:
    pass
output_path = output_path + content_name[:-4] + '_' + style_name1[:-4] + '_' + style_name2[:-4] + '/'
try:
    os.mkdir(output_path)
except:
    pass

# Get network
vgg = VGG()
vgg.load_state_dict(torch.load('../Models/vgg_conv.pth'))
for param in vgg.parameters():
    param.requires_grad = False
vgg.to(device)

# Load images
content_img = load_image(os.path.join(content_dir, content_name), image_size, device, content_invert)
style_img1 = load_image(os.path.join(style_dir1,style_name1), image_size, device, style_invert)
style_img2 = load_image(os.path.join(style_dir2,style_name2), image_size, device, style_invert)

# Random input
opt_img = Variable(torch.randn(content_img.size()).type_as(content_img.data).to(device), requires_grad=True).to(device)
# Content input
# opt_img = Variable(content_img.data.clone(), requires_grad=True)

# Define style layers
style_layers = ['r11','r21','r31','r41','r51']
style_weights = [1e3/n**3 for n in [64,128,256,512,512]]
# Define mrf layers
mrf_layers = ['r31', 'r41'] 
# Defince content layers
content_layers = ['r42']
content_weights = [1e0]
# loss layers: layers to be used by opt_img ( style_layers & mrf_layers & content_layers)
loss_layers = mrf_layers + style_layers + content_layers

# Feature maps from style image 1
mrf_fms1 = [A.detach() for A in vgg(style_img1, mrf_layers)]
# Extract style patches & create conv3d from those patches 1 
style_patches_lists1, weight_list1 = get_style_patch_weights(mrf_fms1, device, k=patch_size)
# Compute style target 1
style_targets1 = [GramMatrix()(A).detach() for A in vgg(style_img1, style_layers)]

# Feature maps from style image 2
mrf_fms2 = [A.detach() for A in vgg(style_img2, mrf_layers)]
# Extract style patches & create conv3d from those patches 2
style_patches_lists2, weight_list2 = get_style_patch_weights(mrf_fms2, device, k=patch_size)
# Compute style target 2
style_targets2 = [GramMatrix()(A).detach() for A in vgg(style_img2, style_layers)]

style_patches_lists = style_patches_lists1.copy()
weight_list = weight_list1.copy()
style_targets = style_targets1.copy()

# Difference between style patches list
for i,style_patches_list in enumerate(style_patches_lists):
    for j,style_patch in enumerate(style_patches_list):
        style_patch = torch.abs_(style_patches_lists1[i][j] - style_patches_lists1[i][j])

# Difference between 3D weights
for i,weights in enumerate(weight_list):
    weights = torch.abs_(weight_list1[i]-weight_list2[i])

# Difference between Gram matrices
for i, style_target in enumerate(style_targets):
    style_target = torch.abs_(style_targets1[i]-style_targets2[i])


# Compute content target
content_targets = [A.detach() for A in vgg(content_img, content_layers)]

# targets
targets = style_targets + content_targets
# layers weights
weights = style_weights + content_weights
# Optimizing layers
loss_fns = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
loss_fns = [loss_fn.to(device) for loss_fn in loss_fns]

# Define optimizer
optimizer = optim.LBFGS([opt_img])

n_iter = [0]
naming_it = [0]
loss_list = []
content_loss_list = []
style_loss_list = []
mrf_loss_list = []
max_iter = 2000
show_iter = 50

start = time.time()

pbar = tqdm(total=max_iter)
while n_iter[0] <= max_iter:

    def closure():
        optimizer.zero_grad()
        opt_fms = vgg(opt_img, loss_layers)
        
        # Content & style loss
        style_loss = 0
        content_loss = 0
        for a,A in enumerate(opt_fms[len(mrf_layers):]):
            one_layer_loss = weights[a] * loss_fns[a](A, targets[a])
            if a < len(style_layers):
                style_loss += one_layer_loss
            else:
                content_loss += one_layer_loss

        # MRF loss or energy function
        mrf_loss = mrf_loss_fn(opt_fms[:len(mrf_layers)], style_patches_lists, weight_list,patch_size)
                      
        # Regularzier
        regularizer = smoothnes_loss(opt_img)

        # Total loss
        # total_loss = alpha * content_loss + beta * style_loss + gamma * mrf_loss
        total_loss = alpha * content_loss + beta * style_loss + gamma * mrf_loss + 0.001*regularizer
 
        # log 
        content_loss_list.append(content_loss.item())
        style_loss_list.append(style_loss.item())
        loss_list.append(total_loss)

        # Calculate backward
        total_loss.backward()

        #print loss
        if (n_iter[0])%show_iter == 0:

            tqdm.write('Iteration: {}'.format(naming_it[0]))
            tqdm.write('Content loss : {}'.format(alpha*content_loss.item()))
            tqdm.write('Style loss   : {}'.format(beta *style_loss.item()))
            tqdm.write('MRF loss     : {}'.format(gamma*mrf_loss.item()))
            tqdm.write('Regulari loss: {}'.format(0.001*regularizer.item()))
            tqdm.write('Total loss   : {}'.format(total_loss.item()))
            # pbar.write

            # Save loss graph
            plt.plot(loss_list, label='total loss')
            plt.plot(content_loss_list, label='content loss')
            plt.plot(style_loss_list, label='style loss')
            plt.plot(mrf_loss_list, label='mrf loss')
            plt.legend()
            plt.savefig(output_path + 'loss_graph.jpg')
            plt.close()
            # Save optimized image
            out_img = postp(opt_img.data[0].cpu().squeeze(), image_size, result_invert)
            out_img.save(output_path + '{}.bmp'.format(naming_it[0]))

        n_iter[0] += 1
        naming_it[0] += 1
        pbar.update(1)
        return total_loss
    optimizer.step(closure)
pbar.close()
# Save sum images
save_images(content_img.data[0].cpu().squeeze(), style_img1.data[0].cpu().squeeze(), style_img2.data[0].cpu().squeeze(), opt_img.data[0].cpu().squeeze(), image_size, output_path, naming_it[0], content_invert, style_invert, result_invert)

end = time.time()
print("Style transfer took {} seconds overall".format(end-start))