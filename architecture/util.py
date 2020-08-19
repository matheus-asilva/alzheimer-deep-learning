from tqdm import tqdm
import requests
from typing import Tuple

def download_file_from_google_drive(id, path_to_file):
    URL = 'https://docs.google.com/uc?export=download'

    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)

    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    CHUNK_SIZE = 32*1024
    # TODO: this doesn't seem to work; there's no Content-Length value in header?
    total_size = int(response.headers.get('content-length', 0))

    with tqdm(desc=path_to_file, total=total_size, unit='B', unit_scale=True) as pbar:
        with open(path_to_file, 'wb') as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    pbar.update(CHUNK_SIZE)
                    f.write(chunk)

def load_image(filepath: str, input_shape: Tuple[int, int] = (224,224)):
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, input_shape)

    return image