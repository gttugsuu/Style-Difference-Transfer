import os
import os.path
import glob

import matplotlib.pyplot as plt



# available_letters = 'AHW'
# # Style with serifs
# serif_folder_path = '../../input/style_diff/serif/'
# # Style that lacks serifs
# non_serif_folder_path = '../../input/style_diff/serif_rmv/'
# # Content 
# # content_folder_path = '../../input/style_diff/contents/'
# content_folder_path = '../../input/AlegreyaSans-Light/'
# content_file_list = glob.glob(content_folder_path+'*')

serif_path = '../../input/photo-inputs/'
serif_name = 'ball2.jpeg'
non_serif_path = '../../input/photo-inputs/'
non_serif_name = 'cup_cutted.jpeg'
content_path = '../../input/photo-inputs/'
content_name = 'cup_cuttedw.jpeg'
output_path = '../../output_from_pytorch/diff_style_transfer/photo-inputs/'

command = 'python main.py -style_path1 {} -style_name1 {} -style_path2 {} -style_name2 {} -content_path {} -content_name {} -output_path {}'.format(serif_path, serif_name, non_serif_path, non_serif_name, content_path, content_name, output_path)
print(command)
os.system(command)

# counter=1
# content_number = len(content_file_list)
# # iterate over all content letters
# for content in content_file_list:
#     content_path = content[:32]
#     content_name = content[32:]
#     # print('{} of {} contents done.'.format(counter, content_number))
#     counter += 1
#     # iterate over all serif letters
#     for letter in available_letters:
#         serif_letters = glob.glob(serif_folder_path+letter+'/*')
#         for serif in serif_letters:
#             serif_path = serif[:31]
#             serif_name = serif[31:]
#             # iterate over all non serif letters
#             for n_letter in available_letters:
#                 non_serif_letters = glob.glob(non_serif_folder_path+n_letter+'/*')
#                 for non_serif in non_serif_letters:
#                     non_serif_path = non_serif[:35]
#                     non_serif_name = non_serif[35:]
#                     # define output path
#                     output_path = '../../output_from_pytorch/diff_style_transfer/'+letter+n_letter+'/'
#                     os.system('python main.py -style_path1 {} -style_name1 {} -style_path2 {} -style_name2 {} -content_path {} -content_name {} -output_path {}'
#                     .format(serif_path, serif_name, non_serif_path, non_serif_name, content_path, content_name, output_path))

