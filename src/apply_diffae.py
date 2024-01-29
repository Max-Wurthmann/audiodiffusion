import torch 
from torch.utils.data import ConcatDataset, random_split, Dataset
import torchaudio
from torchaudio.transforms import Resample

print(torch.cuda.is_available())

from tqdm import tqdm
import os
import math
from IPython.display import Audio
import datasets

from train.main.models import get_diffae_from_state_dict_file
from train.main.datamodules import read_audio_file

out_dir = "/data/meta/generated"

# Load diffae from file
diffae_path = "/data/logs/ckpts/diffae_state_dict"
device = "cuda:0"
diffae = get_diffae_from_state_dict_file(diffae_path).to(device)

files = datasets.load_from_disk("/data/meta/test_data")["file"]


def up_to_closest_pow_2(x: int): 
    """'round' x up to next power of 2"""
    return 2**(math.ceil(math.log(x)/math.log(2)))
    
@torch.no_grad()
def generate(file, out_dir=out_dir, decoding_steps=100, channels=2, sr=32000):
    in_dir, name = os.path.split(file)
    out_file = os.path.join(out_dir, name)
    exists = os.path.isfile(out_file)

    if exists:
        return

    if "vogelstimmen_cds" in in_dir:
        resample = Resample(44100, sr)
    else:
        resample = None
        
        
    wave = read_audio_file(file, target_channels=channels, target_sr=sr, resample=resample)
    
    # pad up to a power of two
    curr_length = wave.shape[-1]
    total_length = up_to_closest_pow_2(curr_length)
    padding = torch.zeros(channels, total_length-curr_length)
    wave = torch.cat([wave, padding], dim=-1)
    wave = wave.unsqueeze(0).to(device)

    # apply autoencoder
    latent = diffae.encode(wave)
    decoded = diffae.decode(latent, num_steps=decoding_steps)

    torchaudio.save(uri=out_file, src=decoded.cpu().squeeze(0), sample_rate=sr)

if __name__=="__main__":
    # run analysis
    for file in tqdm(files):
        generate(file)
