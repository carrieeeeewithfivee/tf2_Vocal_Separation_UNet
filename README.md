# tf2_Vocal_Separation_UNet
Tensorflow 2.0 implementation for Vocal Separation UNet

![Spectrogram comparison](https://github.com/carrieeeeewithfivee/tf2_Vocal_Separation_UNet/blob/master/predictions/samples.png)

Based on [SINGING VOICE SEPARATION WITH DEEP U-NET CONVOLUTIONAL NETWORKS](https://ejhumphrey.com/assets/pdf/jansson2017singing.pdf) by A. Jansson, et al. 
Inspired by [jnzhng/keras-unet-vocal-separation](https://github.com/jnzhng/keras-unet-vocal-separation). 
Full report [here](https://drive.google.com/file/d/1UguJXjXYjuKBb5Y6hKLV4rVNWd8ONHMx/view?usp=sharing)

## Results:
  Due to small dataset size (150 songs for training data VS 20,000 in original paper), our model overfits when evaluating. No data augmentation was used, which would have benefited this situation greatly. (The higher SDR is, the better)
  
![Train Val comparison](https://github.com/carrieeeeewithfivee/tf2_Vocal_Separation_UNet/blob/master/predictions/compare2.png)

Two datasets were used while training. The [ccmixter vocal separation database](https://members.loria.fr/ALiutkus/kam/) and the [MIR-1K](https://sites.google.com/site/unvoicedsoundseparation/mir-1k) dataset.

We compare the training loss of only training with ccmixter and with both datasets. The results show a significant improvement when more data is used while training.
Samples of wav files can be found in the predictions folder.
![Different dataset comparison](https://github.com/carrieeeeewithfivee/tf2_Vocal_Separation_UNet/blob/master/predictions/Fig_train.png)

## Environment:
  python 3.6
  
  tensorflow 2.1
  
  run on cuda 10.1 for gpu
  
## How to train:
* Download [ccmixter vocal separation database](https://members.loria.fr/ALiutkus/kam/) to data
* Download [MIR-1K](https://sites.google.com/site/unvoicedsoundseparation/mir-1k) to data
* Run preprocess_CCMixter.py
* Run preprocess_MIR1K.py
* Configure flags in train.py and run.

## How to run inference:
* Configure flags in inference.py and run.
* Wav files and spectrogram visualizations be found in the prediction folder. 
* If you want to run a pretrained model, download provided checkpoints [here](https://drive.google.com/drive/folders/1eV55XK8BwiVr5DWDE7fVFLRwszisFDaR?usp=sharing).

## Future work:
* Evaluate on [MedleyDB](https://zenodo.org/record/1649325#.XvgDwmozY1I).
* Add data augmentation.
* Enlarge training data.
