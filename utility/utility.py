import os

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

# pre and post processing for images
img_size = 100
prep = transforms.Compose([transforms.Resize((img_size,img_size)),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x.mul_(255)),
                          ])
postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
                           transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], #add imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
                           ])
postpb = transforms.Compose([transforms.ToPILImage()])
def postp(tensor): # to clip results in the range [0,1]
    t = postpa(tensor)
    t[t>1] = 1    
    t[t<0] = 0
    img = postpb(t)
    return img

# gram matrix and loss
class GramMatrix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1,2)) 
        G.div_(h*w)
        return G

class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return(out)

# Function to load images
def load_images(style_dir1, style_dir2, content_dir):
    # Load & invert images, ordered as [style_image1, style_image2, content_image]
    img_dirs = [style_dir1, style_dir2, content_dir]
    imgs = []
    for img_dir in img_dirs:
        image = Image.open(img_dir)
        image = PIL.ImageOps.invert(image)
        image = image.convert('RGB')
        imgs.append(image)
    imgs_torch = [prep(img) for img in imgs]
    if torch.cuda.is_available():
        imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
    else:
        imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]
    style_image1, style_image2, content_image = imgs_torch
    # Initialize optimizing image as the content image
    opt_img = Variable(content_image.data.clone(), requires_grad = True)
    # opt_img = Variable(torch.rand_like(content_image), requires_grad = True)
    return style_image1, style_image2, content_image, opt_img

# Function to save images
def save_images(content_image, opt_img, style_image1, style_image2, output_path, n_iter):

    style_image1 = postp(style_image1.data[0].cpu().squeeze())
    style_image1 = PIL.ImageOps.invert(style_image1)
    style_image1.save(output_path + '/style1.bmp')

    style_image2 = postp(style_image2.data[0].cpu().squeeze())
    style_image2 = PIL.ImageOps.invert(style_image2)
    style_image2.save(output_path + '/style2.bmp')

    content_image = postp(content_image.data[0].cpu().squeeze())
    content_image = PIL.ImageOps.invert(content_image)
    content_image.save(output_path + '/content.bmp')

    # Save optimized images
    out_img = postp(opt_img.data[0].cpu().squeeze())
    out_img = PIL.ImageOps.invert(out_img)
    out_img.save(output_path + '/{}.bmp'.format(n_iter[0]))
        
    # Save summary image as [content image, optimized image, style image1, style_image2]
    images = [content_image, out_img, style_image1, style_image2]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    new_im.save(output_path + '/all.bmp')

def make_folders(output_path):
    try:
        os.mkdir(output_path)
    except:
        pass
    try:
        os.mkdir(output_path+'/outputs')
    except:
        pass
    # try:
    #     os.mkdir(output_path+'/gm_layer1')
    # except:
    #     pass
    # try:
    #     os.mkdir(output_path+'/gm_layer2')
    # except:
    #     pass
    # try:
    #     os.mkdir(output_path+'/gm_layer3')
    # except:
    #     pass
    # try:
    #     os.mkdir(output_path+'/gm_layer4')
    # except:
    #     pass
    # try:
    #     os.mkdir(output_path+'/gm_layer5')
    # except:
    #     pass


# Visualize GM of all layers with input image
def visualize_gm_all_layers(gram_list_original, original_image, output_path):

    orig_image = original_image.clone()
    if torch.cuda.is_available():
        orig_image = postp(orig_image.data[0].cpu().squeeze())
    else:
        orig_image = postp(orig_image.squeeze())
    orig_image = PIL.ImageOps.invert(orig_image)

    plt.figure(figsize=(21,14))
    plt.subplot(2,3,1)
    plt.title('original image')
    plt.imshow(orig_image)
    plt.colorbar()

    gram_list = gram_list_original.copy()
    for i in range(len(gram_list)):
        gram_list[i] = gram_list[i].squeeze()
        if torch.cuda.is_available():
            gram_list[i] = gram_list[i].cpu().detach().numpy()
        else:
            gram_list[i] = gram_list[i].numpy()
        gram_list[i] = gram_list[i].astype(int)
        # Plot gram matrices
        plt.subplot(2,3,i+2)
        plt.title('layer_{}1'.format(i+1))
        plt.imshow(gram_list[i], cmap='Blues')
        plt.colorbar()
    # plt.show()
    plt.savefig(output_path)
    plt.close()

def save_as_gif(input_path, output_path, max_iter, show_iter,filetype='bmp', savename='all.gif'):
    images = []
    for i in range(0, max_iter+1, show_iter):
        image = Image.open(input_path + '/{}.'.format(i) + filetype)
        images.append(image)
    images[0].save(output_path + savename,
            save_all=True, append_images=images[1:], optimize=False, duration=120, loop=0)