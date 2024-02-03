import datasets
import pandas as pd
from tqdm import tqdm
import os
import json
import torch
import torchaudio

from train.main.models import get_default_combined_sampler

n = 1000 # number of files generated
sampling_steps = 100
decoding_steps = 100
latent_length = 2048
embedding_scale = 5
sr = 32000
batch_size = 8
model_version = "v3"

filtered = False

out_dir = "/data/tests/generated"
scale_str = str(embedding_scale).replace(".", ",")
experiment_name = f"steps{sampling_steps}_scale{scale_str}_{model_version}"
exp_dir = os.path.join(out_dir, experiment_name)
config_file = exp_dir + ".json"
os.makedirs(exp_dir, exist_ok=True)

diffgen = f"/data/logs/ckpts/diffgen_{model_version}.pth"
model = get_default_combined_sampler(diffgen_state_dict_file=diffgen)


test_data = datasets.load_from_disk("/data/meta/test_data")

if filtered:
    bird_selection = ["Parus major", "Fringilla coelebs", "Phylloscopus collybita", "Turdus merula", "Erithacus rubecula"]
    test_data = test_data.filter(lambda x: x.get("sci_name") in bird_selection)

labels = test_data["label"]
files = test_data["file"]

config = dict(
    exp=experiment_name,
    n=n,
    sampling_steps=sampling_steps,
    decoding_steps=decoding_steps,
    latent_length=latent_length,
    embedding_scale=embedding_scale,
    sr=sr,
    batch_size=batch_size,
    diffgen=diffgen,
    filtered=filtered,
)
    
print(config)
with open(config_file, "w+") as f:
    json.dump(config, f)


for i in tqdm(range(0, n, batch_size)):
    batch = labels[i:i+batch_size]
    out = model.sample(batch,
                       sampling_steps=sampling_steps,
                       decoding_steps=decoding_steps,
                       embedding_scale=embedding_scale,
                       latent_length=latent_length,
                      )
    
    for j in range(out.shape[0]):
        sample = out[j, ...]
        
        _, fname = os.path.split(files[i+j]) 
        out_file = os.path.join(exp_dir, fname)
        
        torchaudio.save(uri=out_file, src=sample.cpu(), sample_rate=sr)