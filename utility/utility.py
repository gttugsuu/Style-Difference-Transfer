import os

import torch
from torch.autograd import Variable
import torch.nn as nn

from torchvision import transforms

import PIL
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy import ndimage
import cv2

import matplotlib.pyplot as plt



# post processing for images
postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
                           transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], #add imagenet mean
                                                std=[1,1,1]),
                            # transforms.Normalize(mean=[0,0,0], #subtract imagenet mean
                            #                         std=[1/0.5,1/0.5,1/0.5]),
                            # transforms.Normalize(mean=[-0.5,-0.5,-0.5], #subtract imagenet mean
                            #                         std=[1,1,1]),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
                           ])
postpb = transforms.Compose([transforms.ToPILImage()])

# gram matrix and loss
class GramMatrix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        Fe = input.view(b, c, h*w)
        G = torch.bmm(Fe, Fe.transpose(1,2)) 
        G.div_(h*w)
        return G

class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return(out)

def postp(tensor, image_size, invert): # to clip results in the range [0,1]
    t = postpa(tensor)
    t[t>1] = 1    
    t[t<0] = 0
    img = postpb(t)
    if invert:
        img = PIL.ImageOps.invert(img)
    img = transforms.functional.resize(img,[image_size, image_size])
    return img

def custom_postp(tensor, image_size, output_path):
    # Post processing
    t = transforms.Lambda(lambda x: x.mul_(1./255))(tensor)
    ## Save histogram
    plt.plot(torch.histc(t[0],bins=255,min=t[0].min(),max=t.max()).numpy(), color='blue')
    plt.plot(torch.histc(t[1],bins=255,min=t[1].min(),max=t.max()).numpy(), color='green')
    plt.plot(torch.histc(t[2],bins=255,min=t[2].min(),max=t.max()).numpy(), color='red')
    plt.savefig(output_path + "_before.jpg")
    ## Save before image
    bef_img = transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])(t)
    bef_img = transforms.ToPILImage()(bef_img)
    bef_img = transforms.Resize([image_size,image_size])(bef_img)
    bef_img.save(output_path + "_before.bmp")
    ## Unnormalize
    t = transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], std=[1,1,1])(t)
    ## Map to [0,1]
    # a = 0
    # b = 1
    # t = (t - t.min())*(b-a)/(t.max()-t.min()) + a
    ## Return to RGB
    t = transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])(t)
    ## Cut invalid values
    t[t>1] = 1
    t[t<0] = 0
    ## Save histogram
    plt.plot(torch.histc(t[2],bins=255,min=0,max=1).numpy(), color='blue')
    plt.plot(torch.histc(t[1],bins=255,min=0,max=1).numpy(), color='green')
    plt.plot(torch.histc(t[0],bins=255,min=0,max=1).numpy(), color='red')
    plt.savefig(output_path + "_after.jpg")
    ## Transform to PIL image
    img = transforms.ToPILImage()(t)
    ## Resize
    img = transforms.Resize([image_size,image_size])(img)
    ## Return
    return img

# Function to load images
def load_images(img_dir, img_size, device, invert):
    prep = transforms.Compose([transforms.Resize((img_size,img_size)),
                            # transforms.RandomRotation(angle),
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                            transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                                    std=[1,1,1]),
                        #    transforms.Normalize(mean=[0.5, 0.5, 0.5], #add imagenet mean
                        #                         std=[0.5,0.5,0.5]),
                            transforms.Lambda(lambda x: x.mul_(255)),
                            ])
    # Load & invert image
    image = Image.open(img_dir)
    image = image.convert('RGB')
    if invert:
        image = PIL.ImageOps.invert(image)
    # Make torch variable
    img_torch = prep(image)
    img_torch = Variable(img_torch.unsqueeze(0).to(device))
    
    return img_torch

# Function to save images
def save_images(content_image, opt_img, style_image1, style_image2, image_size, output_path, n_iter, content_invert, style_invert, result_invert):

    fnt = ImageFont.truetype('/usr/share/fonts/ubuntu/UbuntuMono-R.ttf', 13)

    # Save style image 1
    style_image1 = postp(style_image1, image_size, style_invert)
    d = ImageDraw.Draw(style_image1)
    d.text((0,0), "Style1", font=fnt, fill=(0,0,0))
    style_image1.save(output_path + 'style1.jpg')

    # Save style image 2
    style_image2 = postp(style_image2, image_size, style_invert)
    d = ImageDraw.Draw(style_image2)
    d.text((0,0), "Style2", font=fnt, fill=(0,0,0))
    style_image2.save(output_path + 'style2.jpg')

    # Save content image
    content_image = postp(content_image, image_size, content_invert)
    d = ImageDraw.Draw(content_image)
    d.text((0,0), "Content", font=fnt, fill=(0,0,0))
    content_image.save(output_path + 'content.jpg')

    # Save optimized images
    out_img = postp(opt_img, image_size, result_invert)
    d = ImageDraw.Draw(out_img)
    d.text((0,0), "Generated", font=fnt, fill=(0,0,0))
    out_img.save(output_path + '/{}.jpg'.format(n_iter))
        
    images = [style_image1, style_image2, content_image, out_img]
    # widths, heights = zip(*(i.size for i in images))
    widths = [i.size[0] for i in images]
    heights = [i.size[1] for i in images]
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

"""
Input tensor
Outputs distance transform
"""
def dist_cv2(input_tensor, device, image_size, content_invert):
    out_img = postp(input_tensor.data[0].cpu().squeeze(), image_size, content_invert)
    # out_img = PIL.ImageOps.invert(out_img)
    # out_img = PIL.ImageOps.grayscale(out_img)
    out_img = out_img.convert('L')

    img = np.asarray(out_img)
    
    img = ndimage.grey_erosion(img, size=(3,3))

    img_dist = cv2.distanceTransform(img, cv2.DIST_L2, 3)
    # plt.imshow(img_dist, cmap="Blues")
    # plt.colorbar()
    # plt.savefig("dist_img.png")
    # cv2.imwrite("dist_img.bmp", img_dist)
    cont_dist = torch.from_numpy(img_dist).float().to(device)
    f = cont_dist.unsqueeze(0)
    a = torch.cat((f,f,f),0)
    a = a.unsqueeze(0)    
    return a