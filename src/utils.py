import yaml
import os
import torch
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
    
def resolve_checkpoint_path(model_config, stage: int) -> Path:
    """
    Resolve the path to a stage checkpoint using candidate locations in priority order:
      1. Configured model directory (local or same Kaggle session output)
      2. /kaggle/working (Kaggle session, different script)
      3. Kaggle input dataset (uploaded from a previous session)

    Raises FileNotFoundError if none are found.
    """
    filename = model_config[f'stage{stage}']['checkpoint']
    kaggle_input_path = model_config['kaggle'][f'stage{stage}_input_path']
    if kaggle_input_path:
        kaggle_input_path = Path(kaggle_input_path) / filename

    candidates = [
        Path(model_config['models'][f'stage{stage}_dir']) / filename,
        Path('/kaggle/working') / filename,
        Path(model_config['kaggle'][f'stage{stage}_input_path'])
    ]
    if kaggle_input_path:
        candidates.append(Path(kaggle_input_path))

    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        checked = '\n  '.join(str(p) for p in candidates)
        raise FileNotFoundError(
            f"Could not find Stage {stage} checkpoint '{filename}'.\n"
            f"Searched:\n  {checked}\n"
            f"Update kaggle.stage{stage}_input_path in configs/model.yaml "
            f"with your Kaggle dataset path."
        )

    print(f"Found Stage {stage} checkpoint: {path}")
    return path

def get_host():
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE', ''):
        return 'kaggle'
    if os.environ.get('SPACE_ID', ''):
        return 'huggingface'
    return 'local'

def get_root_path():
    return Path(__file__).resolve().parents[1]

def get_dir_path(folders=''):
    root = get_root_path()
    return root / folders

def get_file_path(filename, folders='', check_exists=False):
    root = get_root_path()
    file_path = root / folders / filename

    if check_exists and not file_path.exists():
        raise FileNotFoundError(f'Could not find file path: {file_path}')

    return file_path

def load_model_checkpoint(model, optimizer, scheduler, scaler, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f'Loaded model. Continuing training from epoch {epoch + 1}')
    return model, optimizer, scheduler, scaler, epoch + 1, loss

def save_model_checkpoint(model, optimizer, scheduler, scaler, epoch, loss, save_path):
    try:
        model_state_dict = model.module.state_dict()
    except AttributeError:
        model_state_dict = model.state_dict()

    torch.save({
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        "epoch": epoch,
        "loss": loss
    }, save_path)
    print(f'  Saved best model: {save_path}  (Val Loss: {loss:.4f})')
    print(f'Model checkpoint saved at epoch {epoch}')
