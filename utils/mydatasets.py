import os
import os.path
import numpy as np
import pickle

import torchvision.datasets as datasets

class ListDataset(datasets.DatasetFolder):
    """
    Data Loader for images and labels both of which are written in the list file like
      root_dir/class_x/xxx.ext 0
      root_dir/class_y/yyy.ext 1

    Args:
        root (string) : Path to root directory of data
        list_file (string or list) : Path to the list file or List data.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
    
    For example,
        Data = ListDataset(LISTFILE, transform=Normalize)
        TrnData = ListDataset(LISTFILE, transform=Normalize, k_fold_cv=(5,1,'train'))
        TstData = ListDataset(LISTFILE, transform=Normalize, k_fold_cv=(5,1,'test'))
    """
    def __init__(self, root, data_list, transform=None, target_transform=None,
                 loader=datasets.folder.default_loader, k_fold_cv=None):
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        if type(data_list) is str:
            samples = self._load_list(root, data_list)
        else:
            samples = data_list

        if k_fold_cv is None:
            self.samples = samples
        else:
            k, i, mode = k_fold_cv # tupple
            trninds, tstinds = cvpartition(len(samples), k, i)
            if mode == 'train':
                self.samples = [samples[j] for j in  trninds]
            else:
                self.samples = [samples[j] for j in  tstinds]

        self.targets = [s[1] for s in self.samples]
        self.imgs = self.samples

    def _load_list(self, root, list_file):
        _, ext = os.path.splitext(list_file)
        if ext == '.txt': # ASCII text file
            samples = []
            with open(list_file) as f:
                l = f.readlines()
                for ll in l:
                    fname, id = ll.split(' ')
                    if fname[0] == '/':
                        fname = fname[1:]
                    samples.append((os.path.join(root,fname), int(id.strip())))
        else: # pickle binary file
            with open(list_file, 'rb') as f:
                samples = pickle.load(f)

        return samples
    
def cvpartition(N, k, i, seed=0):
    rs = np.random.get_state()
    np.random.seed(seed)
    inds = list(range(0,N))
    np.random.shuffle(inds)
    cvbds = np.linspace(0, N, k+1, dtype=int)
    trninds = []
    tstinds = []
    for l in range(k):
        if l == i-1:
            tstinds += inds[cvbds[l]:cvbds[l+1]]
        else:
            trninds += inds[cvbds[l]:cvbds[l+1]]
    np.random.set_state(rs)

    return trninds, tstinds