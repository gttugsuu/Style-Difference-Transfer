import os
import os.path
import glob

# alphabet = 'ABCDEFGHIJKLMNOQPRSTUVWXYZ' 

style1_folder_path = '../input/font_contents/serif_sans/B/serif/'
style2_folder_path = '../input/font_contents/serif_sans/B/sans/'
content_folder_path= '../input/font_contents/sanserifs/B/'

style1_path_list = glob.glob(style1_folder_path+'*')
style2_path_list = glob.glob(style2_folder_path+'*')

# print(style1_path_list)
# print(style2_path_list)
# exit()

style1_path_list = ['../input/font_contents/serif_sans/B/serif/NotoSerif-Regular.png', '../input/font_contents/serif_sans/B/serif/PT_Serif-Caption-Web-Regular.png', '../input/font_contents/serif_sans/B/serif/NotoSerif-Bold.png', '../input/font_contents/serif_sans/B/serif/PT_Serif-Web-Italic.png']
style2_path_list = ['../input/font_contents/serif_sans/B/sans/NotoSans-Regular.png', '../input/font_contents/serif_sans/B/sans/PT_Sans-Web-Italic.png', '../input/font_contents/serif_sans/B/sans/PT_Sans-Caption-Web-Regular.png', '../input/font_contents/serif_sans/B/sans/NotoSans-Bold.png']

content_path_list = glob.glob(content_folder_path+'*')
content_path_list = ['../input/font_contents/AlegreyaSans-Light/B.png']

# print(content_path_list)
# exit()

output_path = "../output_style_difference/serif_B/"

cuda = 'cuda:0'

for content_path in content_path_list[:2]:
    for i in range(len(style1_path_list)):
        style1_path = style1_path_list[i]
        style2_path = style2_path_list[i]
        command = 'python st.py -serif_style_path {} -nonserif_style_path {} -content_path {} -output_path {} -cuda {}'.format(style1_path, style2_path, content_path, output_path, cuda)
        print(command)
        os.system(command)


# for i in range(6):
#         content_name = content_list[i]
#         style_name = style_list[i]
#         command = "python st_cnnmrf.py -content_path {} -content_name {} -style_path {} -style_name {} -output_path {} -cuda 'cuda:{}' ".format(content_folder_path, content_name, style_folder_path, style_name, output_path, cuda)
#         print(command)
#         os.system(command)
      

# #for content in content_path_list[0:5]:
# #        content_name = os.path.basename(content)
# for content_name in content_list:
#         for style_name in style_list:
# #        for style in style_path_list[0:4]:
#         #        style_name = os.path.basename(style)
#                command = "python st_cnnmrf.py -content_path {} -content_name {} -style_path {} -style_name {} -output_path {} -cuda 'cuda:{}'".format(content_folder_path, content_name, style_folder_path, style_name, output_path, cuda)
#                print(command)
#                os.system(command)
