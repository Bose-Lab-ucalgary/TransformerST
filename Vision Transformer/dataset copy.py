import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from utils import read_tiff
from contour_util import *
import numpy as np
# import torchvision
# import torchvision.transforms as transforms
import scanpy as sc
# from utils import get_data
import os
# import glob
from PIL import Image
import pandas as pd 
import scprep as scp
from PIL import ImageFile
import cv2
import json
from sklearn.preprocessing import LabelEncoder
# import mnnpy
# import seaborn as sns
# from skimage.measure import label
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

class LUNG(torch.utils.data.Dataset):
    """
    The LUNG class in this file is designed as a custom dataset for handling spatial transcriptomics data, specifically tailored for lung tissue analysis. 
    """
    def __init__(self, train=True, gene_list=None, ds=None, sr=False, fold=0):
        super(LUNG, self).__init__()
        self.r = 256 // 4
        self.label_encoder = LabelEncoder()  # Initialize label encoder

        self.train = train
        self.sr = sr

        names = ['A1','A2']

        print('Loading imgs...')
        self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in names}
   
        self.exp_dict = {}
        self.loc_dict = {}
        self.center_dict = {}
        self.gene_dict = {}
        for name in names:
            expression_data, spatial_data, gene_list = self.get_cnt(name)
            self.exp_dict[name] = expression_data
            self.loc_dict[name] = spatial_data[:, 0:2]
            self.center_dict[name] = spatial_data[:, 2:4]  # Storing last two dimensions in self.center
            self.gene_dict[name] = gene_list

        self.id2name = dict(enumerate(names))


    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i, exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp > 0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:, j])

    def __getitem__(self, index):
        """
        Retrieves and processes the data sample at the specified index.
        Args:
            index (int): Index of the data sample to retrieve.
        Returns:
            tuple:
                - patches (torch.Tensor): Tensor of shape (n_patches, patch_dim) containing normalized and flattened image patches.
                - positions (torch.LongTensor): Tensor containing the positions of each patch.
                - exps (torch.Tensor): Tensor containing the expression data associated with the sample.
        Notes:
            - The method extracts image patches centered at specified coordinates, normalizes them, and flattens them into vectors.
            - The returned tuple structure is the same for both training and evaluation modes.
        """
        
        i = index
        im = self.img_dict[self.id2name[i]]
        #im = im.permute(1, 0, 2)
        exps = self.exp_dict[self.id2name[i]]
        centers = self.center_dict[self.id2name[i]]
        patch_dim = 3 * self.r * self.r * 4

   
        n_patches = len(centers)
        # print(len(centers_org))
        patches = torch.zeros((n_patches, patch_dim))
        exps = torch.Tensor(exps)
        im_np = np.array(im)  # Convert the image object to a NumPy array
        im_torch = torch.from_numpy(im_np).float()  # Convert the NumPy array to a PyTorch tensor
        loc = self.loc_dict[self.id2name[i]]
        positions = torch.LongTensor(loc)
        min_val = torch.min(im_torch)
        max_val = torch.max(im_torch)
        for i in range(n_patches):
            center = centers[i]
            x, y = center
            patch = im_torch[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
            normalized_patch = (patch - min_val) / (max_val - min_val)
            # Flatten and store the normalized patch
            patches[i, :] = normalized_patch.flatten()
            if self.train:
                return patches,positions, exps
            else:
                return patches, positions, exps

    def __len__(self):
        return len(self.exp_dict)

    def get_img(self, name):
        img_fold = os.path.join('/media/cyzhao/New_Volume/TransformerST/data/Lung/', name,
                                'outs/spatial/full_image.tif')
        img_color = cv2.imread(img_fold, cv2.IMREAD_COLOR)
        img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

        return img_color

    def get_cnt(self, name):
        input_dir = os.path.join('/media/cyzhao/New_Volume/TransformerST/data/Lung/', name, 'outs/')
        adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5')
        adata.var_names_make_unique()
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata = adata[:, adata.var['highly_variable']]
        file_Adj = os.path.join(input_dir, "spatial/tissue_positions_list.csv")

        positions = pd.read_csv(file_Adj, header=None)
        positions.columns = [
                'barcode',
                'in_tissue',
                'array_row',
                'array_col',
                'pxl_row_in_fullres',
                'pxl_col_in_fullres',
        ]
        # Set the index to barcode for merging
        positions.set_index('barcode', inplace=True)

        # Identify overlapping columns
        overlap_columns = adata.obs.columns.intersection(positions.columns)

        # Decide how to resolve conflicts for overlapping columns
        # Here, we'll keep the column from `positions` and discard the one from `adata.obs`
        for col in overlap_columns:
            adata.obs[col] = positions[col]

        # If there are additional columns in `positions` that are not in `adata.obs`,
        # you might want to merge them into `adata.obs`
        non_overlap_columns = positions.columns.difference(adata.obs.columns)
        adata.obs = adata.obs.join(positions[non_overlap_columns], how="left")
        adata.obsm['spatial'] = adata.obs[['array_row', 'array_col','pxl_row_in_fullres', 'pxl_col_in_fullres']].to_numpy()
        expression_data = adata.X
        expression_data = expression_data.todense()
        expression_data = torch.tensor(expression_data)
        spatial_data = adata.obsm['spatial']
        gene_list = adata.var_names.tolist()
        return expression_data, spatial_data,gene_list
        # return adata

    def get_pos(self, name):

        path = self.pos_dir + '/' + name + '_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path, sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id

        return df



if __name__ == '__main__':
    dataset =LUNG(train=True,mt=False)


class ViT_HER2ST(torch.utils.data.Dataset):
    """Some Information about HER2ST"""
    def __init__(self,train=True,gene_list=None,ds=None,sr=False,fold=0, incl_patients = None):
        super().__init__()
        
        self.cnt_dir = '../../data/her2st/data/ST-cnts'
        self.img_dir = '../../data/her2st/data/ST-imgs'
        self.pos_dir = '../../data/her2st/data/ST-spotfiles'
        self.lbl_dir = '../../data/her2st/data/ST-pat/lbl'
        
        self.cnt_dir = os.path.abspath(self.cnt_dir)
        self.img_dir = os.path.abspath(self.img_dir)
        self.pos_dir = os.path.abspath(self.pos_dir)
        self.lbl_dir = os.path.abspath(self.lbl_dir)
        
        self.r = 224//4

        # gene_list = list(np.load('data/her_hvg.npy',allow_pickle=True)) 
        relative_gene_list = 'data/her_hvg_cut_1000.npy'
        abs_path = os.path.abspath(relative_gene_list)
        
        gene_list = list(np.load(abs_path,allow_pickle=True))
        self.gene_list = gene_list
        names = os.listdir(self.cnt_dir) # ['A1.tsv', 'A2.tsv',...]
        names.sort()
        names = [i[:2] for i in names] # ['A1', 'A2'..]
        self.train = train
        self.sr = sr
        
        # samples = ['A1','B1','C1','D1','E1','F1','G2','H1']
        if incl_patients is not None:
            names = [name for name in names if name in incl_patients]
        
        else:
            samples = names[1:33]

            te_names = [samples[fold]]
            tr_names = list(set(samples)-set(te_names))

            if train:
                # names = names[1:33]
                # names = names[1:33] if self.cls==False else ['A1','B1','C1','D1','E1','F1','G2']
                names = tr_names
            else:
                # names = [names[33]]
                # names = ['A1']
                # names = [ds] if ds else ['H1']
                names = te_names

        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in names}
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in names}


        self.gene_set = list(gene_list)
        self.exp_dict = {i:scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i,m in self.meta_dict.items()}
        self.center_dict = {i:np.floor(m[['pixel_x','pixel_y']].values).astype(int) for i,m in self.meta_dict.items()}
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}


        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(names))

        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5,0.5,0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])

    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i,exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp>0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:,j])


    def __getitem__(self, index):
        """
        Retrieves the data sample at the specified index.
        Depending on the mode (self.sr and self.train), this method processes image patches, their positions, and associated data.
        Args:
            index (int): Index of the data sample to retrieve.
        Returns:
            If self.sr is True:
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                    - patches: Tensor of shape (n_patches, patch_dim), containing flattened image patches.
                    - positions: Tensor of shape (n_patches, 2), containing the grid positions of each patch.
                    - centers: Tensor of shape (n_patches, 2), containing the center coordinates of each patch.
            If self.sr is False and self.train is True:
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                    - patches: Tensor of shape (n_patches, patch_dim), containing flattened image patches.
                    - positions: Tensor of shape (n_patches, 2), containing the positions of each patch.
                    - exps: Tensor of shape (n_patches, ...), containing expression data for each patch.
            If self.sr is False and self.train is False:
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                    - patches: Tensor of shape (n_patches, patch_dim), containing flattened image patches.
                    - positions: Tensor of shape (n_patches, 2), containing the positions of each patch.
                    - exps: Tensor of shape (n_patches, ...), containing expression data for each patch.
                    - centers: Tensor of shape (n_patches, 2), containing the center coordinates of each patch.
        Notes:
            - The method extracts patches from the image tensor based on the provided centers and radius (self.r).
            - When self.sr is True, patches are generated on a regular grid over the image.
            - When self.sr is False, patches are generated based on provided centers.
            - The returned tensors are suitable for use in PyTorch models.
        """
        
        i = index
        im = self.img_dict[self.id2name[i]]
        im = im.permute(1,0,2)
        # im = torch.Tensor(np.array(self.im))
        exps = self.exp_dict[self.id2name[i]]
        centers = self.center_dict[self.id2name[i]]
        loc = self.loc_dict[self.id2name[i]]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4

        if self.sr:
            centers = torch.LongTensor(centers)
            max_x = centers[:,0].max().item()
            max_y = centers[:,1].max().item()
            min_x = centers[:,0].min().item()
            min_y = centers[:,1].min().item()
            r_x = (max_x - min_x)//30
            r_y = (max_y - min_y)//30

            centers = torch.LongTensor([min_x,min_y]).view(1,-1)
            positions = torch.LongTensor([0,0]).view(1,-1)
            x = min_x
            y = min_y

            while y < max_y:  
                x = min_x            
                while x < max_x:
                    centers = torch.cat((centers,torch.LongTensor([x,y]).view(1,-1)),dim=0)
                    positions = torch.cat((positions,torch.LongTensor([x//r_x,y//r_y]).view(1,-1)),dim=0)
                    x += 56                
                y += 56
            
            centers = centers[1:,:]
            positions = positions[1:,:]

            n_patches = len(centers)
            patches = torch.zeros((n_patches,patch_dim))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                patches[i] = patch.flatten()


            return patches, positions, centers

        else:  # Usual case for TransformerST 
            n_patches = len(centers)
            
            patches = torch.zeros((n_patches,patch_dim))
            exps = torch.Tensor(exps)


            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                patches[i] = patch.flatten()

            if self.train:
                return patches, positions, exps
            else: 
                return patches, positions, exps, torch.Tensor(centers)
        
    def __len__(self):
        return len(self.exp_dict)

    def get_img(self,name):
        pre = self.img_dir+'/'+name[0]+'/'+name
        fig_name = os.listdir(pre)[0]
        path = pre+'/'+fig_name
        im = Image.open(path)
        return im

    def get_cnt(self,name):
        path = self.cnt_dir+'/'+name+'.tsv'
        df = pd.read_csv(path,sep='\t',index_col=0)

        return df

    def get_pos(self,name):
        path = self.pos_dir+'/'+name+'_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_lbl(self,name):
        # path = self.pos_dir+'/'+name+'_selection.tsv'
        path = self.lbl_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)

        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))

        return meta

    def get_overlap(self,meta_dict,gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set&set(i.columns)
        return list(gene_set)

