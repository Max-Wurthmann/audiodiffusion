import os
from frechet_audio_distance import FrechetAudioDistance

# pann checkpoint: Cnn14_mAP=0.431
sr = 32000
frechet = FrechetAudioDistance(
    ckpt_dir="/data/fad_model_ckpt",
    model_name="pann",
    sample_rate=sr,
    verbose=True,
    audio_load_worker=2,
)

test_dir = "/data/tests"

fad_score = frechet.score(
    background_dir=None,
    background_embds_path=os.path.join(test_dir, "baseline_embeds.npy"),
    eval_dir=None,
    eval_embds_path=os.path.join(test_dir, "fad_embeds.npy"),
)

print(fad_score)

results_file = os.path.join(test_dir, "results.txt")
with open(results_file, "a+") as f:
    f.write(f"fad_combined: {fad_score}\n")
