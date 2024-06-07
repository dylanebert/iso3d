import pandas as pd
from huggingface_hub import hf_hub_download

repo_id = "diffusers-parti-prompts/karlo-v1"
file_path = hf_hub_download(
    repo_id, "data/train-00000-of-00001-7ac08d9c1ecc174b.parquet", repo_type="dataset"
)

df = pd.read_parquet(file_path)
prompts = df["Prompt"].tolist()

with open("prompts.txt", "w") as f:
    for prompt in prompts:
        full_prompt = f"{prompt}, isolated object render, with a white background"
        f.write(full_prompt + "\n")
