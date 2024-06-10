import argparse
import re

from gradio_client import Client, handle_file
from huggingface_hub import HfApi

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
    result = client.predict(input_image=input_files[0])
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True)
    args = parser.parse_args()
    run(args.repo_id)
