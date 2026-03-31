import os
import sys
import math
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from data.rainvec import *
from data.feature import *

TSIZE = 256
F_BACKGROUND = ['curv', 'imp', 'slope_flacc_w', 'sinkdepth', 'cos', 'sin']

def get_patch_indices(indices):
    """"""
    val_sel = np.unique(indices)
    val_sel = val_sel[val_sel.argsort()]
    # patchindices are sampled randomly and will have jumps. in the x file we want continuous index, so we need to map each entry in indices to a rownumber in the patchfile
    rowno_datafile = np.array([np.where(val_sel == x)[0] for x in indices])
    return(rowno_datafile, val_sel)


class FINNDataset(Dataset):

    def __init__(self, fx, fy, fm, l_rain_data, rowno_patchfiles, indices_r, mask=False, datatype=None):
        """
        l_rain_data: list of rain data
        rowno_patchfiles: row number of patch files. The dataset only considers a partial number of rows
        indices_r: indices of rain data
        """

        self.fx = fx
        self.fy = fy
        self.fm = fm
        self.l_rain_data = l_rain_data

        self.rowno_patchfiles = rowno_patchfiles
        self.indices_r = indices_r
        self.mask = mask

        xf2=np.memmap(fx, mode="readonly", dtype=np.float32, shape=(np.unique(rowno_patchfiles).shape[0], TSIZE, TSIZE, len(F_BACKGROUND)))
        self.xf=np.copy(xf2)
        del xf2
        yf2=np.memmap(fy, mode="readonly", dtype=np.float32, shape=(rowno_patchfiles.shape[0], TSIZE, TSIZE))
        self.yf=np.copy(yf2)
        del yf2
        if mask:
            mf2=np.memmap(fm, mode="readonly", dtype=np.float32, shape=(rowno_patchfiles.shape[0], TSIZE, TSIZE))
            self.mf=np.copy(mf2)
            del mf2

        # generate valid set: 1000 for length
        if datatype == 'train':
            self.rowno_patchfiles = self.rowno_patchfiles[1000:]
            self.indices_r = self.indices_r[1000:]
        elif datatype == 'valid':
            self.rowno_patchfiles = self.rowno_patchfiles[0:1000]
            self.indices_r = self.indices_r[0:1000]

        self.len = len(self.indices_r)
        print(f"self.len:{(self.len)}")

    def __len__(self):
        """"""
        return self.len

    def __getitem__(self, index):
        """"""
        patch_sel = self.rowno_patchfiles[index, 0]
        x = self.xf[patch_sel, :, :, :].copy()
        y = self.yf[index, :, :].copy()
        y = np.expand_dims(y, axis=2)

        r_sel = self.indices_r[index]
        r = np.zeros(shape=(NR))
        rvec = self.l_rain_data[r_sel]
        r[:] = np.array(rvec)

        x = x.astype("float32")
        r = r.astype("float32")
        y = y.astype("float32")
        if self.mask:
            m = self.mf[index, :, :].copy()
            m = np.expand_dims(m,axis=2)
            m = m.astype("float32")

        if self.mask:
            return {"x": x.transpose(2, 0, 1),
                    "r": r,
                    "y": y.transpose(2, 0, 1),
                    "m": m.transpose(2, 0, 1)}
        else:
            return {"x": x.transpose(2, 0, 1),
                    "r": r,
                    "y": y.transpose(2, 0, 1)}


def load_dataset(data_dir:str):
    """"""
    ##### Part1: 提取每一降雨事件的降雨数据，并提取特征，见libs.manage_data.make_rain_variables2
    # rain data - results for CDS storms are not used in training and validation
    npzfile = np.load(os.path.join(data_dir, 'flood', 'events.npz'))
    evnames = npzfile['evnames']
    del npzfile
    rainlist=[] # value in [0,1]
    for ii in range(evnames.shape[0]):
        if 'CDS' in evnames[ii]:
            rain=rainvec(int(evnames[ii][7:10]),'CDS_M')
        elif 'NAT' in evnames[ii]:
            rain=rainvec_nat(int(evnames[ii][4:6]))
        else:
            sys.exit('wrong event type')
        rainlist.append([x/25.0 for x in rain]) # biggest rainfall is 25mm/10min
    rainvariables=[make_rain_variables2(x) for x in rainlist]
    rvar_scale=[max([xx[x] for xx in rainvariables])-min([xx[x] for xx in rainvariables]) for x in range(NR)]
    rainvariables=[[x[xx]/rvar_scale[xx] for xx in range(NR)] for x in rainvariables]

    ##### Part2: selectors_patchlist.npy记录了一些索引信息，进行筛选
    # get_patch_indices 返回一个array(array)，记录每种值出现在哪些下标
    # create patch-extents and selectors for training with random and overlapping patches, and random combinations of patches and rain
    # this step takes one hour to execute, but needs to be performed only once
    selectors_patchlist_path=os.path.join(data_dir, "background", 'selectors_patchlist.npy')
    [select_train_rain, select_train_patch,
     select_test_rain, select_test_patch,
     patchlist2] = np.load(selectors_patchlist_path,allow_pickle=True)
    select_train_patchrows, _ = get_patch_indices(select_train_patch)
    select_test_patchrows, _ = get_patch_indices(select_test_patch)

    ##### Part3: 制作 dataset
    patchfx = os.path.join(data_dir, "background", "px2.arr")
    patchfy = os.path.join(data_dir, "background", "py2.arr")
    patchfm = os.path.join(data_dir, "background", "pm2.arr")
    patchfx_val = os.path.join(data_dir, "background", "pxval2.arr")
    patchfy_val = os.path.join(data_dir, "background", "pyval2.arr")
    patchfm_val = os.path.join(data_dir, "background", "pmval2.arr")

    train_dataset = FINNDataset(patchfx_val, patchfy_val, patchfm_val, rainvariables, select_test_patchrows, select_test_rain)
    valid_dataset = FINNDataset(patchfx_val, patchfy_val, patchfm_val, rainvariables, select_test_patchrows, select_test_rain)
    #train_dataset = FINNDataset(patchfx, patchfy, patchfm, rainvariables, select_train_patchrows, select_train_rain, datatype='train')
    #valid_dataset = FINNDataset(patchfx, patchfy, patchfm, rainvariables, select_train_patchrows, select_train_rain, datatype='valid')
    test_dataset  = FINNDataset(patchfx_val, patchfy_val, patchfm_val, rainvariables, select_test_patchrows, select_test_rain)
    return train_dataset, valid_dataset, test_dataset
