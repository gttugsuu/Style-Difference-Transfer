import argparse
import os
import os.path

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim

import matplotlib.pyplot as plt

from utility.utility import postp, GramMatrix, GramMSELoss, load_images, save_images, make_folders
from utility.vgg_network import VGG
#############################################################################
# PARSER
parser = argparse.ArgumentParser(description='Style Difference Transfer')
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
parser.add_argument('--cw1', '-cw1', type=float,  default=1, help='cw1')
parser.add_argument('--cw2', '-cw2', type=float,  default=1, help='cw2')
parser.add_argument('--cw3', '-cw3', type=float,  default=1, help='cw3')
parser.add_argument('--cw4', '-cw4', type=float,  default=1, help='cw4')
parser.add_argument('--cw5', '-cw5', type=float,  default=1, help='cw5')
# parser for input images paths and names
parser.add_argument('--image_size', '-image_size', type=int, default=256)
# parser for input images paths and names
parser.add_argument('--serif_style_path', '-serif_style_path', type=str, default='../input/font_contents/serif_sans/U/serif/LindenHill-Italic.png')
parser.add_argument('--nonserif_style_path', '-nonserif_style_path', type=str, default='../input/font_contents/serif_sans/U/sans/LindenHill-Regular.png')
parser.add_argument('--content_path', '-content_path', type=str, default='../input/font_contents/sanserifs/U/CabinCondensed-Regular.png')
# parser for output path
parser.add_argument('--output_path', '-output_path', type=str, default='../output_style_difference/direct/', help='Path to save output files')
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
# opt_img = Variable(torch.randn(content_image.size()).type_as(content_image.data).to(device), requires_grad=True).to(device)
# Content input
opt_img = Variable(content_image.data.clone(), requires_grad=True)

# Define layers, loss functions, weights and compute optimization targets
# Style layers
style_layers = ['r12','r22','r34','r44','r54'] 
style_weights = [sw*1e3/n**2 for sw,n in zip([sw1,sw2,sw3,sw4,sw5],[64,128,256,512,1024])]
# style_weights = [1,1,1,1,1]
# Content layers
# content_layers = ['r12','r22','r32','r42','r52']
# content_layers = ['r31','r32','r33','r34','r41']
content_layers = ['r42']
content_weights = [1e3]
# content_weights = [cw1*1e4,cw2*1e4,cw3*1e4,cw4*1e4,cw5*1e4]


loss_layers = style_layers + content_layers
loss_functions = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
loss_functions = [loss_fn.to(device) for loss_fn in loss_functions]
weights = style_weights + content_weights


# Compute optimization targets
### Gram matrix targets

# Feature maps from style layers of the style images
style1_fms_style = [A.detach() for A in vgg(style_image1, style_layers)]
style2_fms_style = [A.detach() for A in vgg(style_image2, style_layers)]
# Gram matrices of style feature maps
style1_gramm = [GramMatrix()(A) for A in style1_fms_style]
style2_gramm = [GramMatrix()(A) for A in style2_fms_style]
# Difference between gram matrices of style1 and style2
gramm_style = [(style1_gramm[i] - style2_gramm[i]) for i in range(len(style_layers))]


# Difference between feature maps
#style_fms_style  = [style1_fms_style[i] - style2_fms_style[i] for i in range(len(style_layers))]
# Gram matrix of difference feature maps
#gramm_style = [GramMatrix()(A) for A in style_fms_style]


# Feature maps from style layers of the content image
content_fms_style = [A.detach() for A in vgg(content_image, style_layers)]
content_gramm = [GramMatrix()(A) for A in content_fms_style]


### Content targets
# Feature maps from content layers of the style images
style1_fms_content = [A.detach() for A in vgg(style_image1, content_layers)]
style2_fms_content = [A.detach() for A in vgg(style_image2, content_layers)]
# Difference between feature maps
style_fms_content = [(style1_fms_content[i] - style2_fms_content[i]) for i in range(len(content_layers))]
# Feature maps from content layers of the content image
content_fms_content = [A.detach() for A in vgg(content_image, content_layers)]


# Run style transfer
make_folders(output_path)

max_iter = 1000
show_iter = 50
optimizer = optim.LBFGS([opt_img])
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
        
        opt_fms_style = []
        opt_fms_content = []
        # Divide between style feature maps and content feature maps
        for i, A in enumerate(out):
            if i < len(style_layers):
                opt_fms_style.append(A)
            else:
                opt_fms_content.append(A)

        ## Difference between feature maps on style layers
#        diff_fms_style = [opt_fms_style[i] - content_fms_style[i] for i in range(len(style_layers))]
#        gramm_diff = [GramMatrix()(A) for A in diff_fms_style]
        ## Difference between gram matrix of feature map differences
#        style_layer_losses = [style_weights[i]*(nn.MSELoss()(gramm_diff[i], gramm_style[i])) for i in range(len(style_layers))]
        
        opt_gramm = [GramMatrix()(A) for A in opt_fms_style]
        gramm_diff = [(opt_gramm[i] - content_gramm[i]) for i in range(len(style_layers))]
        style_layer_losses = [style_weights[i]*nn.MSELoss()(gramm_diff[i], gramm_style[i]) for i in range(len(style_layers))]

        ## Difference between feature maps on content layers
        fms_diff = [(opt_fms_content[i] - content_fms_content[i]) for i in range(len(content_layers))]
        content_layer_losses = [content_weights[i]*nn.MSELoss()(fms_diff[i],style_fms_content[i]) for i in range(len(content_layers))]
        # content_layer_losses = [content_weights[i]*nn.MSELoss()(opt_fms_content[i],content_fms_content[i]) for i in range(len(content_layers))]
        

        # losses
        content_loss = sum(content_layer_losses)
        style_loss   = sum(style_layer_losses)
        
        # ld1 = len(str(content_loss.item()))
        # ld2 = len(str(style_loss.item()))
        # if ld1 > ld2:
        #     div = ld1 - ld2
        #     style_loss = style_loss*(10**(div))
        # else:
        #     div = ld2 - ld1
        #     content_loss = content_loss*(10**(div))
        
        
        layer_losses = content_layer_losses + style_layer_losses

        # total loss
        loss = sum(layer_losses)

        # for log
        c_loss.append(content_loss)
        s_loss.append(style_loss)
        loss_list.append(loss)

        # backward calculation
        loss.backward()

        #print loss
        if n_iter[0]%show_iter == 0:
            print('Iteration: {}'.format(n_iter[0]))
            if len(content_layers)>0: print('Content loss: {}'.format(content_loss.item()))
            if len(style_layers)>0:   print('Style loss  : {}'.format(style_loss.item()))
            print('Total loss  : {}'.format(loss.item()))

            # Save loss graph
            plt.plot(loss_list, label='Total loss')
            if len(content_layers)>0:  plt.plot(c_loss, label='Content loss')
            if len(style_layers)  >0:  plt.plot(s_loss, label='Style loss')
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
