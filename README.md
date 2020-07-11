# Glow TTS

* This code is a replication of [official Glow TTS code](https://github.com/jaywalnut310/glow-tts). If you want to use Parallel WaveGAN model, I recommend that you refer to the official code.
* The following is the paper I referred:

[Kim, J., Kim, S., Kong, J., & Yoon, S. (2020). Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search. arXiv preprint arXiv:2005.11129.](https://arxiv.org/abs/2005.11129)

# Requirements

* torch >= 1.5.1
* tensorboardX >= 2.0
* librosa >= 0.7.2
* matplotlib >= 3.1.3

* Optional for loss flow
    * tensorboard >= 2.2.2

# Structure

## Training
<img src='./Figures/Training.png' width=100% height=100% />

## Inference
<img src='./Figures/Inference.png' width=81% height=81% />

# Used dataset

* <S>Currently uploaded code is compatible with the following datasets.</S>
* Multi speakers are not supported yet.
* The O mark to the left of the dataset name is the dataset actually used in the uploaded result.

|        | | Dataset   | Dataset address                                 |
|--------|-|-----------|-------------------------------------------------|
| O      | | LJSpeech  | https://keithito.com/LJ-Speech-Dataset/         |
| X      | | BC2013    | http://www.cstr.ed.ac.uk/projects/blizzard/     |
| X      | | CMU Arctic| http://www.festvox.org/cmu_arctic/index.html    |
| X      | | VCTK      | https://datashare.is.ed.ac.uk/handle/10283/2651 |
| X      | | LibriTTS  | https://openslr.org/60/                         |

# Hyper parameters
Before proceeding, please set the pattern, inference, and checkpoint paths in 'Hyper_Parameter.yaml' according to your environment.

* Sound
    * Setting basic sound parameters.
    * Some paramters like pitch are not used in current code. These are for future works.

* Use_Cython_Alignment
    * Setting which implementation of Monotonic alignment search to use
    * If `true`, the cython implementation of official code will be used.
    * If `false`, the python implementation will be used.
    * I recommend to use cython implementation because of speed.

* Encoder
    * Setting the encoder parameters

* Decoder
    * Setting the glow decoder parameters.

* WaveNet
    * Setting the parameters of Vocoder.
    * This implementation uses a pre-trained Parallel WaveGAN model.
        * https://github.com/CODEJIN/PWGAN_Torch
    * If checkpoint path is `null`, model does not exports wav files.
    * If checkpoint path is not `null`, all parameters must be matched to pre-trained Parallel WaveGAN model.

* Token path
    * Setting the token-to-index dict.
    * Pattern generator makes this file.

* Train
    * Setting the parameters of training.

* Inference_Path
    * Setting the inference path

* Checkpoint_Path
    * Setting the checkpoint path

* Log_Path
    * Setting the tensorboard log path

* Use_Mixed_Precision
    * Setting mixed precision.
    * To use, `Nvidia apex` must be installed in the environment.

* Device
    * Setting which GPU device is used in multi-GPU enviornment.
    * Or, if using only CPU, please set '-1'.
























# Generate pattern

## Command
```
python Pattern_Generate.py [parameters]
```

## Parameters

At least, one or more of datasets must be used.

* -lj <path>
    * Set the path of LJSpeech. LJSpeech's patterns are generated.
* -bc2013 <path>
    * Set the path of Blizzard Challenge 2013. Blizzard Challenge 2013's patterns are generated.    
* -cmua <path>
    * Set the path of CMU arctic. CMU arctic's patterns are generated.
* -vctk <path>
    * Set the path of VCTK. VCTK's patterns are generated.
* -libri <path>
    * Set the path of LibriTTS. LibriTTS's patterns are generated.

* -text
    * Set whether the text information save or not.
    * This is for other model. To use in Glow TTS, this option must be set.
* -eval
    * Set the evaluation pattern ratio.
    * Default is `0.001`.
* -mw
    * The number of threads used to create the pattern
    * Default is `10`.

# Run

## Command
```
python Train.py -s <int>
```

* `-s <int>`
    * The resume step parameter.
    * Default is 0.
    * When this parameter is 0, model try to find the latest checkpoint in checkpoint path.

# Result

* Now training....

# Trained checkpoint

* Now training....

# Future works

1. Applying multi-speaker