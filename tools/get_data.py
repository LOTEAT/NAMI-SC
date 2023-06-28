'''
Author: LOTEAT
Date: 2023-06-23 16:46:28
'''

import urllib.request
import tarfile
import os.path as osp
from tqdm import tqdm


data_root = 'data'
data_info = {
    'europarl': {
        'url': 'http://www.statmt.org/europarl/v7/europarl.tgz',
        'filename': 'europarl.tgz',
        'format': 'tgz'
    }, 
    'DataShare_clean_testset': {
        'url': 'https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip?sequence=1&isAllowed=y',
        'filename': 'clean_testset_wav.zip',
        'format': 'zip'
    }
    
}


def extract(filepath, target_dir, format):
    if format == 'tgz':
        with tarfile.open(filepath, 'r:gz') as tar:
            members = tar.getmembers()
            progress = tqdm(members, desc='Extracting', ncols=80, leave=False)
            for member in progress:
                tar.extract(member, path=target_dir)
                progress.set_postfix(file=member.name)
                progress.update(1)



def download(url, filename):
    response = urllib.request.urlopen(url)
    file_size = int(response.headers['Content-Length'])
    filepath = osp.join('data', filename)
    with tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024, desc=filename, ncols=80) as pbar:
        urllib.request.urlretrieve(url, filepath, reporthook=lambda b, bsize, tsize: pbar.update(bsize))
    target_dir = osp.join(data_root, filename)
    extract(filepath, target_dir, data_info[filename]['format'])
    print("Finished!")


def get_data(data_name):
    download(data_info[data_name]['url'], data_info[data_name]['filename'])

get_data('DataShare_clean_testset')