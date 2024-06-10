import argparse
import re
import shutil

from gradio_client import Client, handle_file
from huggingface_hub import HfApi
from tqdm import tqdm

api = HfApi()
repo_files = api.list_repo_files(repo_id="dylanebert/3d-arena", repo_type="dataset")

input_pattern = re.compile(r"^inputs/images/.*.(png|jpg|jpeg)$")
inputs = filter(input_pattern.match, repo_files)
input_files = [
    handle_file(
        f"https://huggingface.co/datasets/dylanebert/3d-arena/resolve/main/{file}"
    )
    for file in inputs
]


def run(repo_id):
    client = Client(repo_id)
    for input_file in tqdm(list(input_files)):
        filename = input_file["path"].split("/")[-1].replace(".jpg", "")
        result = client.predict(input_image=input_file)
        output_path = f"outputs/{filename}.ply"
        shutil.move(result, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True)
    args = parser.parse_args()
    run(args.repo_id)
