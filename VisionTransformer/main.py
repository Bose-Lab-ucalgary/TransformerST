# import os
# import torch
# import pandas as pd
# from sklearn import metrics
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from VisionTransformer.vis_model import lung_finetune_flex
from VisionTransformer.utils import *
# from dataset import LUNG
from VisionTransformer.dataset import ViT_HER2ST
# from PIL import Image



def main(train_group):
    # fold = 1
    tag = '-vit_1_1_cv'
    # dataset = HER2ST(train=True, fold=fold)
    # dataset = LUNG(train=True, fold=fold)
    dataset = ViT_HER2ST(train=True, incl_patients=train_group)
    train_loader = DataLoader(dataset, batch_size=1, num_workers=3, shuffle=True)
    # model=STModel(n_genes=785,hidden_dim=1024,learning_rate=1e-5)
    model = lung_finetune_flex(n_layers=5, n_genes=785, learning_rate=1e-4) # changed from 1000 to 785 genes for HER2ST
    model.phase = "reconstruction"
    # trainer = pl.Trainer(gpus=1, max_epochs=200)
    trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=200)
    trainer.fit(model, train_loader)
    for epoch, metrics in enumerate(trainer.callback_metrics):
        print(f"Epoch {epoch}: {metrics}")
    
    trainer.save_checkpoint("model/lung_last_train_" + tag + '_1.ckpt')

if __name__ == "__main__":
    main()
