# tf2_Vocal_Separation_UNet
Tensorflow 2.0 implementation for Vocal Separation UNet
Based on [SINGING VOICE SEPARATION WITH DEEP U-NET CONVOLUTIONAL NETWORKS](https://ejhumphrey.com/assets/pdf/jansson2017singing.pdf) by A. Jansson, et al
Inspired by [jnzhng/keras-unet-vocal-separation](https://github.com/jnzhng/keras-unet-vocal-separation)

## Results:

## Environment:

  python 3.6
  
  tensorflow 2.1
  
  cuda 10 for gpu

## How to run:
* Download [ccmixter vocal separation database](https://members.loria.fr/ALiutkus/kam/)
* Run preprocess_CCMixter.py
* Run train.py

## To Do:
* Save visualized results to summary
* Prediction code
* Different datasets
* Different models
