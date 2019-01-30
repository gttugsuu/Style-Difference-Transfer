# content_name = 'AlegreyaSans-Italic.png'
# serif_name = 'Italianno-Regular.png'
# non_serif_name = 'Italianno-Regular.png' 

# serif_path = '../../input/style_diff/serif/A/'
# non_serif_path = '../../input/style_diff/serif_rmv/A/'
# content_path = '../../input/style_diff/contents/'
# output_path = '../../output_from_pytorch/diff_style_transfer/AA/'

echo 'hello'

python main.py  -style_path1 ../../input/style_diff/serif/A/ \
                -style_name1 Italianno-Regular.png \
                -style_path2 ../../input/style_diff/serif_rmv/A/ \
                -style_name2 Italianno-Regular.png \
                -content_path ../../input/style_diff/contents/ \
                -content_name AlegreyaSans-Italic.png \
                -output_path ../../output_from_pytorch/diff_style_transfer/AA/