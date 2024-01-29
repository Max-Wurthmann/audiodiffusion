# Audio Diffusion

This repository aims to train and analyze a generative text-to-audio diffusion model on dataset of bird sounds.
The model consists of two components, a DDIM-based diffusion autoencoder (DiffAE) to generate a latent space and a DDIM-based text-conditional latent diffusion model (DiffGen).
For both components the pytorch implementation of [`audio-diffusion-pytorch`](https://github.com/archinetai/audio-diffusion-pytorch/) is used.
For a detailed explanation of the architecture as well as notes on training and performance please also refer to the paper that is the basis for this work: [`Moûsai`](https://arxiv.org/abs/2301.11757).


## Content

- [Audio Diffusion](#audio-diffusion)
  - [Content](#content)
  - [Setup](#setup)
  - [Models](#models)
    - [Diffusion AutoEncoder (Component 1)](#diffusion-autoencoder-component-1)
    - [Text-Conditional Diffusion Generator (Component 2)](#text-conditional-diffusion-generator-component-2)
    - [Combined Diffusion Sampler](#combined-diffusion-sampler)
  - [Training](#training)
    - [Training the Diffusion AutoEncoder](#training-the-diffusion-autoencoder)
    - [Training the Diffusion Generator](#training-the-diffusion-generator)

## Setup

Python version `3.10` was used for implementation, most if not all components should also be compatible with `3.9`.
Install the requirements as follows:

```bash
pip install -r requirements.txt
```

For the analysis of generated audio samples the additional installation/setup of the following components is necessary:
- [BirdNET](https://github.com/kahst/BirdNET-Analyzer) for bird sound detection. Note: The Python package birdnetlib did not yield the same quality as a repository clone (maybe due to a version difference).
- [Fréchet Audio Distance](https://github.com/gudgud96/frechet-audio-distance) for a general evaluation of generated audio quality.


## Models

### Diffusion Autoencoder

The diffusion autoencoder (DiffAE) encodes audio into a compressed latent for more efficient application of the DiffGen.
Encoding is done by computing a magnitude spectrogram and applying a CNN encoder.
The DiffAE can decode the compressed latent back into an audio waveform using DDIM (denoising diffusion implicit model).
With our configuration, we achieve a compression factor of 64.


```py
from audio_diffusion_pytorch import DiffusionAE, UNetV0, VDiffusion, VSampler, LTPlugin
from audio_encoders_pytorch import ME1d, TanhBottleneck

# Initialize: Variant 1
UNet = LTPlugin(
    UNetV0,
    num_filters=128,
    window_length=64,
    stride=64,
)

diffae = DiffusionAE(
    net_t=UNet,
    dim=1,
    in_channels=2,
    channels=[256, 512, 512, 512, 1024, 1024, 1024],
    factors=[1, 2, 2, 2, 2, 2, 2],
    items=[1, 2, 2, 2, 2, 2, 2],
    
    # The encoder stage using a magnitude spectrogram
    encoder=ME1d(
        in_channels=2,
        channels=512,
        multipliers=[1, 1, 1],
        factors=[2, 2],
        num_blocks=[4, 8],
        stft_num_fft=1023,
        stft_hop_length=256,
        out_channels=32,
        bottleneck=TanhBottleneck()
    ),
    inject_depth=4
)

# Initialize: Variant 2
# A shorthand for the explicit instantiation
from main.models import get_default_diffae
diffae = get_default_diffae(state_dict_file=None) # optionally pass path to a diffae state dict file to load pretraind weights

# Initialize: Variant 3
# Download a pretrained version from huggingface
# Note: This version is pretrained on music data not on our bird sound data
from main.models import get_pretrained_diffae_from_huggingface
diffae = get_pretrained_diffae_from_huggingface()

# Usage ------------------------------------------------------------------

# Calculate loss on an audio sample
audio = torch.randn(1, 2, 2**18) # [batch, in_channels, length]
loss = diffae(audio)

# Encode/decode audio
audio = torch.randn(1, 2, 2**18) # [batch, in_channels, length]
latent = diffae.encode(audio) # encode into latent. In this case of shape [1, 32, 256].
sample = diffae.decode(latent, num_steps=100) # Decode via diffusion with conditioning on latent
```

### Text-Conditional Diffusion Generator

The text-conditional diffusion generator (DiffGen) generates a latent based on a text description.
The latent can then be decoded into an audio waveform using the DiffAE.
Text-conditioning is provided by text embeddings from the (frozen) language model `t5-base`.
The DiffGen is a DDIM (denoising diffusion implicit model).

```py
# Initialize: Variant 1
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler

diffgen = DiffusionModel(
    net_t=UNetV0, 
    in_channels=32, # Number of input/output channels: We use the 32 latent channels from the AutoEncoder
    channels=[128, 256, 512, 512, 1024, 1024], # U-Net: channels at each layer
    factors=[1, 2, 2, 2, 2, 2], # U-Net: downsampling and upsampling factors at each layer
    items=[2, 2, 2, 4, 8, 8], # U-Net: number of repeating items at each layer
    attentions=[0, 0, 1, 1, 1, 1], # U-Net: attention enabled/disabled at each layer
    cross_attentions=[1, 1, 1, 1, 1, 1], # U-Net: cross-attention enabled/disabled at ea
    attention_heads=12, # U-Net: number of attention heads per attention item
    attention_features=64, # U-Net: number of attention features per attention item
    diffusion_t=VDiffusion, # The diffusion model used
    sampler_t=VSampler, # The diffusion sampler used: DDIM sampler
    embedding_max_length=128, # U-Net: text embedding length
    embedding_features=768, # U-Net: text embedding features (default for T5-base)
    use_text_conditioning=False, # put true if you want to use the models default embedder and pass text instead of embeddings
    use_embedding_cfg=True, # U-Net: enables classifier free guidance
)

# Initialize: Variant 2
# A shorthand for the explicit instantiation
from main.models import get_default_diffgen
diffgen = get_default_diffgen(state_dict_file=None) # optionally pass path to a diffgen state dict file to load pretraind weights

# Usage ------------------------------------------------------------------

# calculate loss on a latent audio sample
# get the embedding from the text you want to condition on using e.g. t5-base
embedding = torch.randn(1, 128, 768) # dummy smaple, [batch, embedding_max_length, embedding_features]

latent = torch.randn(1, 32, 256) # [batch, in_channels, length]
loss = diffgen(
    latent,
    embedding=embedding, # pass the embeddings
    embedding_mask_proba=0.1, # classifier-free guidance (CFG) probability of masking text embeddings
)

# Turn noise into new audio sample with diffusion
noise_latent = torch.randn(1, 32, 256)
sample = diffgen.sample(
    noise_latent, # pass gaussian noise to start with
    embedding=embedding, # pass the text embeddings
    embedding_scale=5.0, # guidance weight of classifier-free guidance, suggested 1 < x <= 10 (1 disables cfg in sampling)
    num_steps=100, # higher for better quality but slower latent generation
)
```

### Combined Diffusion Sampler

The combined diffusion sampler (CDS) combines the DiffAE and DiffGen to generate audio waveforms directly from text.
The text-conditional sampling process of this model first uses the text-conditional sampling process of the DiffGEn to generate a latent and then uses the DiffAE to decode the latent into the output audio.

```py
from main.models import CombinedSampler

# Initialize: Variant 1
model = CombinedSampler(
    diffae=diffae, # perviously initialized diffae
    diffgen=diffgen, # perviously initialized diffgen
    text_encoder_name="t5-base", # language model to use
    text_encoder_max_length=128, # max length of embeddings
    device="cuda:0", # device to perform sampling on (moves both diffae and diffgen to device)
)

# Initialize: Variant 2
# A shorthand for the explicit instantiation
from main.models import get_default_combined_sampler
model = get_default_combined_sampler(
    diffae_state_dict_file=None, # optionally pass path to a diffae state dict file to load pretraind weights
    diffgen_state_dict_file=None, # same for diffgen
)

# Usage ------------------------------------------------------------------

audio = model.sample(
    "Description of the Audio you want to generate", # test to condition on
    sampling_steps=100, # diffusion steps of the Diffusion Generator
    decoding_steps=100, # diffusion steps of the Diffusion AutoEncoder (decode)
    embedding_scale=5.0, # guidance weight of classifier-free guidance, suggested range 1 < x <= 10 (1 disables cfg in sampling)
    latent_length=2048, # latent length to use, longer latent results in longer audio output
) 
```

## Training

The DiffAE and DiffGen are trained separately. The combined diffusion sampler (CDS) is assembled from the components without further training. We use the `pytorch-lightning` framework and [`Hydra`](https://hydra.cc/) with `.yaml` config files to configure and instantiate the models, datamodules and trainer. The necessary classes are found in `train/main/`. The training script (`train/train.py`) uses the base config file (`train/config.yaml`). This base config can import experiment config files via the defaults list at the top of `config.yaml`. If the configurations are all set, the training process is started by running:

```bash
python train.py
```

Configurations can be changed or added in the respective config file or as a command line argument via the hydra syntax.
To give an example, a checkpoint dir can be added to the configuration via
```bash
python train.py '+ckpt="path/to/your/checkpoint.ckpt"'
```


### Training the Diffusion AutoEncoder

In order to train the DiffAE we wrote a pytorch_lightning wrapper called `DMAE` (diffusion magntude autoencoder).
We use a datamodule containing cropped stereo audio samples.
DMAE uses an [`ema-model`](https://github.com/lucidrains/ema-pytorch) for smoothing and stability during training. Thus extract the DiffAE at ```dmae.ema.ema_model``` after training and use that smoothed model for inference instead of the DiffAE at ```dmae.model```.
To train the DMAE we use the experiment config at `exp/dmae.yaml`.

### Training the Diffusion Generator

The pytorch_lightning wrapper for the text-conditional diffusion generator we want to train is called `DiffGen`.
DiffGen also uses an [`ema-model`](https://github.com/lucidrains/ema-pytorch) to obtain a smoothed version of the model in training.
The datamodule is configured to provide cropped audio samples and the corresponding labels.
DiffGen uses a `t5-base` instance (model + tokenizer) to calculate the embeddings and a pretrained DiffAE to encode the audio samples into latents. The internal model (a diffusion generator) is trained on these latents and text-conditiond on the embeddings.  
The experiment configuration can be found at `exp/diffgen.yaml`.
