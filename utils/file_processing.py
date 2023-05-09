import urllib.request
import os


def download_file(url, file_path):
    """Download file from url to file_path"""
    if not os.path.exists(file_path):
        # Download the dataset if it does not exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        urllib.request.urlretrieve(url, file_path)
        print(f'Downloaded {url} to {file_path}')


def download_and_extract_dataset(url, dir_path):
    """Download file from url to file_path and extract it to dir_path"""
    file_path = os.path.join(dir_path, os.path.basename(url))

    print(f'Downloading {url} to {file_path}'
          f'\nExtracting to {dir_path}'
          f'\nThis may take a few minutes...'
          )

    if not os.path.exists(dir_path):
        os.makedirs(os.path.dirname(dir_path), exist_ok=True)
        download_file(url, file_path)
        os.system(f'tar -xvjf {file_path} -C {dir_path} --strip-components 1')
        os.remove(file_path)
    else:
        print(f'{dir_path} already exists')
