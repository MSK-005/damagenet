import yaml
import os
from pathlib import Path

def load_config(filename):
   root = Path(__file__).resolve().parents[1]
   config_path = root / "configs" / filename
   if not config_path.exists():
       raise Exception(f"Could not open '{filename}'. All config files are located in the 'configs' directory. Please ensure correct spelling.")

   with open(config_path, 'r') as file:
        data = yaml.safe_load(file)
        # Checking if config file is a dataset file or not
        if 'system' in data:
            host = get_host()
            if host == 'kaggle':
                data['system']['root'] = Path(f"/kaggle/input/{data['system']['name']}")
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

