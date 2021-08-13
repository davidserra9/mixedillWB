# mixedillWB


![teaser](https://user-images.githubusercontent.com/37669469/129296945-ae85e148-ff4c-4e94-8887-0313a477e3e4.jpg)


Our method is built on top of the modified camera ISP proposed [here](https://github.com/mahmoudnafifi/ColorTempTuning). This repo provides the source code of our deep network proposed in our [paper](). Trained models are provided in [./models](https://github.com/mahmoudnafifi/mixedillWB/tree/main/models). 


## Training

To start training, you should first download the [Rendered WB dataset](https://github.com/mahmoudnafifi/WB_sRGB/), which includes ~65K sRGB images rendered with different color temperatures. Each image in this dataset has the corresponding ground-truth sRGB image that was rendered with an accurate white-balance correction. From this dataset, we selected 9,200 training images that were rendered with the "camera standard" photofinishing and with the following white-balance settings: tungsten (or incandescent), fluorescent, daylight, cloudy, and shade. To get this set, you need to only use images ends with the following parts: `_T_CS.png`, `_F_CS.png`, `_D_CS.png`, `_C_CS.png`, `_S_CS.png` and their associated ground-truth image (that ends with `_G_AS.png`). 

Copy all training input images in `./data/images` and copy all ground truth images in `./data/ground truth images`. Note that if you are going to train on a subset of these white-balance settings (e.g., tungsten, daylight, and shade), there is no need to have the additional white-balance settings in your training image directory. 

Then, run the following command:

`python train.py --wb-settings <WB SETTING 1> <WB SETTING 2> ... <WB SETTING N> --model-name <MODEL NAME> --patch-size <TRAINING PATCH SIZE> --batch-size <MINI BATCH SIZE> --gpu <GPU NUMBER>`

where, `WB SETTING i` should be one of the following settings: `T`, `F`, `D`, `C`, `S`, which refer to tungsten, fluorescent, daylight, cloudy, and shade, respectively. Note that daylight (`D`) should be one of the white-balance settings. For instance, to train a model using tungsten and shade white-balance settings + daylight white balance, which is the fixed setting for the high-resolution image (as described in the [paper]()), you can use this command:

`python train.py --wb-settings T D S --model-name <MODEL NAME>`

## Testing
To test trained models, use the following command:

`python test.py --wb-settings <WB SETTING 1> <WB SETTING 2> ... <WB SETTING N> --model-name <MODEL NAME> --testing-dir <TEST IMAGE DIRECTORY> --outdir <RESULT DIRECTORY> --gpu <GPU NUMBER>`

As mentioned in the paper, we apply ensembling and edge-aware smoothing (EAS) to the generated weights. To use ensembling, use `--multi-scale True`. To use EAS, use `--post-process True`. Shown below is a qualitative comparison of our results with and without the ensembling and EAS.


![weights_ablation](https://user-images.githubusercontent.com/37669469/129297902-a6b60667-d99b-4937-9c73-a58fc71378d9.jpg)


Experimentally, we found that when ensembling is used it is recommended to use an image size of 384, while when it is not used, 128x128 or 256x256 give the best results. To control the size of input images at inference time, use `--target-size`. For instance, to set the target size to 256, use `--target-size 256`. 

## Network

Our network has a [GridNet](https://arxiv.org/pdf/1707.07958.pdf)-like architecture. Our network consists of six columns and four rows. As shown in the figure below, our network includes three main units, which are: the residual unit (shown in blue), the downsampling unit (shown in green), and the upsampling unit (shown in yellow). If you are looking for the Pythorch implementation of GridNet, you can check [src/gridnet.py](https://github.com/mahmoudnafifi/mixedillWB/blob/main/src/gridnet.py).

![net](https://user-images.githubusercontent.com/37669469/129297286-b82441e3-fe02-4900-9b07-3bd0928731d2.jpg)

## Results

Given this set of rendered images, our method learns to produce weighting maps to generate a blend between these rendered images to generate the final corrected image. Shown below are examples of the produced weighting maps.

![weights](https://user-images.githubusercontent.com/37669469/129297900-c5ab58ef-bafa-409d-bdf9-bee66efa5489.jpg)


Qualitative comparisons of our results with the camera auto white-balance correction. In addition, we show the results of applying post-capture white-balance correction by using the [KNN white balance](https://github.com/mahmoudnafifi/WB_sRGB/) and [deep white balance](https://github.com/mahmoudnafifi/Deep_White_Balance).

![qualitative_5k_dataset](https://user-images.githubusercontent.com/37669469/129297898-b33ae6f9-db8f-4750-b8f9-2de00ee809ad.jpg)


Our method has a limitation in that it requires these tiny images to work (this should be given by the modified camera ISP pipeline used in our [paper]()). To process images that have already been rendered by the camera (e.g., JPEG images), we can employ one of the sRGB white-balance editing methods to synthetically generate our tiny images with the target predefined WB set in post-capture time. 

In the shown figure below, we illustrate this idea by employing the [deep white-balance editing](https://github.com/mahmoudnafifi/Deep_White_Balance) to generate the tiny images of a given sRGB camera-rendered image taken from Flickr. As shown, our method produces a better result when comparing to the camera-rendered image (i.e., traditional camera AWB) and the deep WB result for post-capture WB correction. If the input image does not have the associated tiny images (as described above), the provided source code runs automatically [deep white-balance editing](https://github.com/mahmoudnafifi/Deep_White_Balance) for you to get the tiny images. 

![qualitative_flickr](https://user-images.githubusercontent.com/37669469/129298104-9ec5186b-092f-4906-a6a4-ca8072b5b1a3.jpg)


## Dataset

![dataset](https://user-images.githubusercontent.com/37669469/129298211-2cbbdc06-915e-4d6e-9a0e-34f910e89512.jpg)

We generated a synthetic testing set to quantitatively evaluate white-balance methods on mixed-illuminant scenes. Our test set consists of 150 images with mixed illuminations. The ground-truth of each image is provided by rendering the same scene with a fixed color temperature used for all light sources in the scene and the camera auto white balance. Ground-truth images end with `_G_AS.png`, while input images ends with `_X_CS.png`, where `X` refers to the white-balance setting used to render each image. 


You can download our test set from one of the following links:
* [8-bit JPG images](https://ln4.sync.com/dl/327ce3f30/jd7rvtf6-7tgz43nf-e9ahtm3j-tv8uzxwe)
* [16-bit PNG images](https://ln4.sync.com/dl/02f0af5f0/4hhpe83r-8ymvskfz-naqpdrqt-nxvq8h4x)

