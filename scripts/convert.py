import re
from io import BytesIO

import requests
from huggingface_hub import HfApi
from PIL import Image
from tqdm import tqdm

api = HfApi()
repo_files = api.list_repo_files(repo_id="dylanebert/3d-arena", repo_type="dataset")

input_pattern = re.compile(r"^inputs/images/.*.(png|jpg|jpeg)$")
inputs = list(filter(input_pattern.match, repo_files))

for file in tqdm(inputs):
    filename = file.replace("inputs/images/", "").replace(".png", "")
    url = f"https://huggingface.co/datasets/dylanebert/3d-arena/resolve/main/{file}"
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    new_image = Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, (0, 0), image)
    new_image = new_image.convert("RGB").resize((512, 512))
    new_image.save(f"outputs/{filename}.jpg", "JPEG")
