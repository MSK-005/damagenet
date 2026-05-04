import yaml
import os
from pathlib import Path

def get_xbd_image_ids(path):
    """
    Given the absolute path to the dataset, retrieve all the IDs of the images. 
    Each ID has a pre- and post-disaster image. So, 2 IDs mean 4 images.
    """
    if not path.exists():
        raise Exception(f"Could not find path: {dir}")
    ids = set()
    for name in path.iterdir():
        # Get the string from file name upto the number
        name = name.name.split("_")
        name = "_".join(name[:2])
        ids.add(name)
    return sorted(list(ids))

def load_config(filename):
    config_path = get_file_path(filename=filename, folders='configs')
    root = get_root_path()

    if not config_path.exists():
       raise Exception(f"Could not open '{filename}'. All config files are located in the 'configs' directory. Please ensure correct spelling.")

    with open(config_path, 'r') as file:
        data = yaml.safe_load(file)
        # Checking if config file is a dataset file or not
        if 'system' in data:
            host = get_host()
            if host == 'kaggle':
                data['system']['root'] = Path(f"/kaggle/input/datasets/{data['system']['kaggle_username']}/{data['system']['name']}/{data['system']['subfolder_name']}")
            elif host == 'local':
                data['system']['root'] = root / data['system']['local_dir'] / data['system']['name']
            
            for mode in ['train', 'test']:
                data[mode]['abs_path'] = data['system']['root'] / data[mode]['dir']
        return data

def get_host():
    """
    Checks if code is running on Kaggle or a local machine
    """
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE', ''):
        return 'kaggle'
    return 'local'

def get_root_path():
    return Path(__file__).resolve().parents[1]

def get_file_path(filename, folders="", check_exists=False):
    root = get_root_path()
    file_path = root / folders / filename

    if check_exists and not file_path.exists():
        raise FileNotFoundError(f'Could not find file path: {file_path}')

    return file_path