# PokeGAN
Generative adversarial network for generating Pokemon sprites

## Setup
Run the script Crop_Sprite_Sheet.py first to generate the training samples. This will convert this:

<p align="center">
  <img width="50%" height="50%" src="/Images/Sprite_Sheet.png">
</p>

Into a bunch of these:

<p align="center">
  <img src="/Images/Training_Images/Single_Sprites/img1.png">
</p>

Then run the script DCGAN.py to begin training the network. The output samples for each epoch are saved in /Images/Generated_Sprites.
