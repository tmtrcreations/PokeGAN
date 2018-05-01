#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crop the large sprite sheet into individual sprites

Created on Sun Apr 15 2018

@author: TMTR Creations
"""

# Import the needed libraries
from PIL import Image

# Define the sprite image sizes
SMALL_SPRITE_WIDTH = 40
SMALL_SPRITE_HEIGHT = 30
COL_SIZE = 32
ROW_SIZE = 54

# Load in the large sprite sheet
img = Image.open("../Images/Sprite_Sheet.png")

# Loop through the large sprite sheet and crop
count = 1
for row_ind in range(0, ROW_SIZE):
    for col_ind in range(0, COL_SIZE):
        
        # Create the new image
        img_new = img.crop((col_ind*SMALL_SPRITE_WIDTH, row_ind*SMALL_SPRITE_HEIGHT,\
                            col_ind*SMALL_SPRITE_WIDTH+SMALL_SPRITE_WIDTH,\
                            row_ind*SMALL_SPRITE_HEIGHT+SMALL_SPRITE_HEIGHT))
        #img_new_jpg = img_new.convert("RGB")
        img_new.save("../Images/Training_Images/Single_Sprites/img"+str(count)+".png")
        img_new = img_new.transpose(Image.FLIP_LEFT_RIGHT)
        #img_new_jpg = img_new.convert("RGB")
        img_new.save("../Images/Training_Images/Single_Sprites_Flipped/img_flipped"+str(count)+".png")
        count = count + 1
        if count > 1722:
            break