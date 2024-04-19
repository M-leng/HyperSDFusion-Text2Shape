"""
    adopted from: https://github.com/shubhtuls/PixelTransformer/blob/03b65b8612fe583b3e35fc82b446b5503dd7b6bd/data/shapenet.py
"""
import os.path
import json
import csv
import collections

import h5py
import numpy as np
from PIL import Image
from termcolor import colored, cprint
from tqdm import tqdm
import pickle

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode
import spacy

from datasets.base_dataset import BaseDataset


def create_dependency_edgelist_by_spacy_for_pyg(doc, max_len):
    # https://spacy.io/docs/usage/processing-text
    seq_len = len([token for token in doc])
    edge_list = []
    edge_list.append([0, 0])
    for token in doc:
        if token.i < seq_len:
            edge_list.append([token.i + 1, token.i + 1])
            # https://spacy.io/docs/api/token
            for child in token.children:
                if child.i < seq_len:
                    edge_list.append([token.i + 1, child.i + 1])
    for i in range(seq_len + 1, max_len):
        edge_list.append([i, i])
    return edge_list


def create_dependency_matrix_by_spacy(doc, max_len):
    # https://spacy.io/docs/usage/processing-text
    seq_len = len([token for token in doc])
    matrix = np.zeros((max_len, max_len)).astype('float32')
    if seq_len>max_len:
        seq_len = max_len
    for token in doc:
        if token.i+1 < seq_len:
            matrix[token.i+1][token.i+1] = 1
            # https://spacy.io/docs/api/token
            for child in token.children:
                if child.i+1 < seq_len:
                    matrix[token.i+1][child.i+1] = 1
                    matrix[child.i+1][token.i+1] = 1

    return matrix

# from https://github.com/laughtervv/DISN/blob/master/preprocessing/info.json
class Text2ShapeDataset(BaseDataset):

    def initialize(self, opt, phase='train', cat='all', res=64, max_seq_len=77):
        self.opt = opt
        self.max_dataset_size = opt.max_dataset_size
        self.res = res
        self.max_seq_len = max_seq_len
        dataroot = opt.dataroot

        self.text_csv = f'{dataroot}/ShapeNet/text2shape/captions.tablechair_{phase}.csv'
        self.spacy_nlp = spacy.load("en_core_web_sm")

        with open(self.text_csv) as f:
            reader = csv.reader(f, delimiter=',')
            self.header = next(reader, None)

            self.data = [row for row in reader]

        with open(f'{dataroot}/ShapeNet/info.json') as f:
            self.info = json.load(f)

        self.cat_to_id = self.info['cats']
        self.id_to_cat = {v: k for k, v in self.cat_to_id.items()}
        
        assert cat.lower() in ['all', 'chair', 'table']
        if cat == 'all':
            valid_cats = ['chair', 'table']
        else:
            valid_cats = [cat]
        
        self.model_list = []
        self.cats_list = []
        self.text_list = []
        self.edge_list = []
        edege_cache = f'{dataroot}/ShapeNet/text2shape/dense_edge_cach.pth'
        for d in tqdm(self.data, total=len(self.data), desc=f'readinging text data from {self.text_csv}'):
            id, model_id, text, cat_i, synset, subSynsetId = d
            
            if cat_i.lower() not in valid_cats:
                continue
            
            sdf_path = f'{dataroot}/ShapeNet/SDF_v1/resolution_{res}/{synset}/{model_id}/ori_sample_grid.h5'
            if not os.path.exists(sdf_path):
                continue
                # {'Chair': 26523, 'Table': 33765} vs {'Chair': 26471, 'Table': 33517}
                # not sure why there are some missing files
            if not os.path.exists(edege_cache):
                dependency = self.spacy_nlp(text)
                edge_index = create_dependency_matrix_by_spacy(dependency, self.max_seq_len)
                self.edge_list.append(edge_index)
            self.model_list.append(sdf_path)
            self.text_list.append(text)
            self.cats_list.append(synset)
        if not os.path.exists(edege_cache):
            with open(edege_cache, "wb") as fp:
                pickle.dump(self.edge_list, fp)
        else:
            with open(edege_cache, "rb") as fp:
                self.edge_list = pickle.load(fp)
        self.model_list = self.model_list[:self.max_dataset_size]
        self.text_list = self.text_list[:self.max_dataset_size]
        self.cats_list = self.cats_list[:self.max_dataset_size]
        self.edge_list = self.edge_list[:self.max_dataset_size]
        cprint('[*] %d samples loaded.' % (len(self.model_list)), 'yellow')

        self.N = len(self.model_list)
        print("N", self.N)
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):

        synset = self.cats_list[index]
        sdf_h5_file = self.model_list[index]
        text = self.text_list[index]

        #text dependcy
        edge = self.edge_list[index]
        #
        h5_f = h5py.File(sdf_h5_file, 'r')
        sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
        sdf = torch.Tensor(sdf).view(1, self.res, self.res, self.res)

        thres = self.opt.trunc_thres
        if thres != 0.0:
            sdf = torch.clamp(sdf, min=-thres, max=thres)

        ret = {
            'sdf': sdf,
            'text': text,
            'cat_id': synset,
            'cat_str': self.id_to_cat[synset],
            'path': sdf_h5_file,
            'edge': edge,
        }

        #if len(text.split(' ')) > 16:
        return ret
        #else:
            # 如果文本长度大于等于8，则重新调用__getitem__，直到找到满足条件的文本
            #return self.__getitem__(index + 1)

    def __len__(self):
        return self.N

    def name(self):
        return 'Text2ShapeDataset'

