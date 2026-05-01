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