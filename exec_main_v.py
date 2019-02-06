import os
import os.path
import glob

available_letters = 'AHW'
# Style with serifs
serif_folder_path = '../../input/style_diff/serif/random_select/'
# Style that lacks serifs
non_serif_folder_path = '../../input/style_diff/serif_rmv/random_select/'
# Content 
# content_folder_path = '../../input/style_diff/contents/'
content_folder_path = '../../input/style_diff/AlegreyaSans-Light/'


content_file_list = glob.glob(content_folder_path+'*')
serif_file_list = glob.glob(serif_folder_path + '*')
non_serif_file_list = glob.glob(non_serif_folder_path + '*')

serif_path = serif_folder_path
non_serif_path = non_serif_folder_path
content_path = content_folder_path

# print('content')
# for content in content_file_list:
#     print(content[42:-4])
# print('serif')
# for serif in serif_file_list:
#     print(serif[43:-4])
# print('non_serif')
# for non_serif in non_serif_file_list:
#     print(non_serif[47:-4])

alphabet = 'ABCDEFGHIJKLMNOQPRSTUVWXYZ'

print(len(alphabet))

for letter in alphabet:
    for style in serif_file_list:
        serif_name = style[43:]
        output_path = '../../output_from_pytorch/diff_style_transfer/AlegreyaSans-Light/'+letter+'/'
        os.system('python main.py -style_path1 {} -style_name1 {} -style_path2 {} -style_name2 {} -content_path {} -content_name {} -output_path {}'
                    .format(serif_folder_path, serif_name, non_serif_folder_path, serif_name, content_folder_path, letter+'.png', output_path))
    