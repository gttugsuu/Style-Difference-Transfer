### !!!This is experimental repo. The code is so messy!!!

### Download the pre-trained weights for VGG network from [here](https://drive.google.com/open?id=1iF4oKdb-5-45AAmGIwaJyMNcjI9xJZ2i), and place it on the main folder. (~500MB)

### To run
```
python st.py -serif_style_path <path_to_style_image_1> -nonserif_style_path <path_to_style_image_2> -content_path <path_to_content_image>
```
### Other default parser arguments:
```
alpha = 0.001     # More emphasize on content loss. Override with -alpha
beta  = 0.8       # More emphasize on style loss. Override with -beta
gamma = 0.001     # More powerful constrain. Override with -gamma
EPOCH = 5000      # Set the number of epochs to run. Override with -epoch
IMAGE_WIDTH = 400 # Determine image size. Override with -width
sw1~sw5 = 1       # Weights for style layers. Override with -sw1 ~ -sw5
                  # pass 0, if you don't use style layers
cw1~cw5 = 1       # Weights for content layers. Override with -sw1 ~ -sw5
                  # pass 0, if you don't use style layers
```
### necessary imports are in utility folder
