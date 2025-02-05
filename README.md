# text-autoencoders
This repo contains the code and data of the following paper:  
[Educating Text Autoencoders: Latent Representation Guidance via Denoising](https://arxiv.org/abs/1905.12777)  
*Tianxiao Shen, Jonas Mueller, Regina Barzilay, and Tommi Jaakkola*

We support plain autoencoder (AE), variational autoencoder (VAE), adversarial autoencoder (AAE), Latent-noising AAE (LAAE), and Denoising AAE (DAAE).

Once the model is trained, it can be used to generate sentences, map sentences to a continuous space, perform sentence analogy and interpolation.

<p align="center"><img width=800 src="img/example_vector_arithmetic.png"></p>
<p align="center"><img width=800 src="img/example_interpolation.png"></p>

## Video
[![overview](https://img.youtube.com/vi/ioFQ8l-Fl7c/0.jpg)](https://www.youtube.com/watch?v=ioFQ8l-Fl7c)

## Dependencies
The code has been tested in Python 3.7, PyTorch 1.1

## Download data
Download the processed Yelp and Yahoo datasets by running:
```
bash download_data.sh
```

Unzip the processed Anki English⇔Spanish dataset (by [ManyThings.org](https://www.manythings.org/bilingual/)) by running:
```
cd data
unzip anki.zip
```

Or, download and then preprocess the Anki dataset by running:
```
bash download_eng_spa_data.sh
python preproc.py --input tmp-spa-eng-data/spa.txt --output data/anki
python preproc.py --input tmp-spa-eng-data/spa.txt --output data/anki_parallel --parallel
```

## Training
The basic training command is:
```
python train.py --train data/anki/train.txt --valid data/anki/valid.txt --model_type aae --lambda_adv 10 --noise 0.3,0,0,0 --save-dir checkpoints/anki/daae
```
To train various models, use the following options:
- AE: `--model_type dae --save-dir checkpoints/yelp/ae`
- VAE: `--model_type vae --lambda_kl 0.1 --save-dir checkpoints/yelp/vae_kl0.1`
- AAE: `--model_type aae --lambda_adv 10 --save-dir checkpoints/yelp/aae`
- LAAE: `--model_type aae --lambda_adv 10 --lambda_p 0.01 --save-dir checkpoints/yelp/aae_p0.01`
- DAAE: `--model_type aae --lambda_adv 10 --noise 0.3,0,0,0 --save-dir checkpoints/yelp/daae`, where `--noise P,P,P,K` specifies word drop probability, word blank probability, word substitute probability, max word shuffle distance, respectively

Run `python train.py -h` to see all training options.

## Testing
After training, the model can be used for different tasks.

To reconstruct input data, run:
```
python test.py --reconstruct --data data/anki/test.txt --output test --checkpoint checkpoints/anki/daae/
```

To generate sentences from the model, run:
```
python test.py --sample --n 10000 --output sample.txt --checkpoint checkpoints/anki/daae/
```

To perform sentence manipulation via vector arithmetic, run:
```
python test.py --arithmetic --data data/anki/language/eng_100_train.txt,data/anki/language/spa_100_train.txt,data/anki/language/eng_100_test.txt --output eng2spa_100_test.txt --checkpoint checkpoints/anki/daae/
```
where the difference between the average latent representation of the first two data files will be applied to the third file (separated by commas), and `k` denotes the scaling factor.

To perform sentence interpolation between two data files (separated by a comma), run:
```
python test.py --interpolate --data data/yelp/interpolate/example.long,data/yelp/interpolate/example.short --output example.int --checkpoint checkpoints/yelp/daae/
```

The output file will be stored in the checkpoint directory.
