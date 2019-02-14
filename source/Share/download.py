import os
import platform


def mxnet_root():
    """
    :return: default data directory depending on the platform and environment variables
    """
    system = platform.system()
    if system == 'Windows':
        return os.path.join(os.environ.get('APPDATA'), 'mxnet')
    else:
        return os.path.join(os.path.expanduser("~"), '.mxnet')


def default_dir(category='model'):
    # os.getenv('MXNET_HOME', mxnet_root())
    return os.path.join(mxnet_root(), 'deep_face', category)


urls = {
    "CelebA": {
        "symbol": 'https://github.com/GalAster/DeepFace/releases/download/v1.0.0/PGGAN_CelebA-symbol.json',
        "params": 'https://github.com/GalAster/DeepFace/releases/download/v1.0.0/PGGAN_CelebA-0000.params',
        "symbol_md5": '',
        "params_md5": ''
    }
}

# download_task(model,dir=default_dir())

medel = "CelebA"
model = urls[medel]
dir = default_dir()

#

url = model["symbol"]
check = model["symbol_md5"]

# download_submit(url,check,dir=dir)

name = url.split('/')[-1]
path = os.path.join(dir, name)

print('Downloading: ' + name)
