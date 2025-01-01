import os
import logging

import numpy as np
import pandas as pd
import tifffile as tiff
import torch

from PIL import Image
from uni import get_encoder

logging.getLogger().setLevel(logging.INFO)

class Uni:

    def __init__(self, img_path, csv_path, out_path, batch_size=50):
        self.img_path = img_path
        self.csv_path = csv_path
        self.out_path = out_path
        self.batch_size = batch_size
        self.offset = 224 // 2
        self.load_image()
        self.load_labels()
        self.filter_labels()
        self.load_model()
        self.max_idx = len(self.ids)
        self.start_idx = self.init_csv()
    
    def load_image(self):
        """Load image into memory."""
        logging.info('Loading image.')
        self.img = tiff.imread(self.img_path).transpose(1, 2, 0)

    def load_labels(self):
        """Load cell information."""
        logging.info('Loading labels.')
        csv = pd.read_csv(self.csv_path)
        self.ids = csv['CellID'].to_numpy()
        self.loc = csv[['X_centroid', 'Y_centroid']].to_numpy(dtype=np.uint16)
    
    def filter_labels(self):
        """Filter cells near image boundaries."""
        logging.info('Filtering labels.')
        img_height, img_width, _ = self.img.shape
        inbound = (
            (self.loc[:, 0] > self.offset) &
            (self.loc[:, 0] < img_width - self.offset) &
            (self.loc[:, 1] > self.offset) &
            (self.loc[:, 1] < img_height - self.offset)
        )
        self.loc = self.loc[inbound]
        self.ids = self.ids[inbound]

    def load_model(self):
        """Load UNI."""
        logging.info('Loading model.')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.transform = get_encoder(enc_name='uni', device=self.device)
    
    def init_csv(self):
        """Initialize output file."""
        if os.path.isfile(self.out_path):
            logging.info('Resuming processing from last saved cell.')
            return len(pd.read_csv(self.out_path))
        else:
            logging.info('Creating new file.')
            cols = ['cell_idx'] + [f'uni{i}' for i in range(1, 1025)]
            pd.DataFrame(columns=cols).to_csv(self.out_path, index=False)
            return 0
    
    def run(self):
        """Main function."""
        for idx in range(self.start_idx, self.max_idx, self.batch_size):
            logging.info(f'Processing cell {idx} of {self.max_idx}.')
            self.process_batch(idx)
        logging.info('Finished processing.')

    def process_batch(self, idx):
        """Process batch."""
        indices = np.arange(idx, min(idx + self.batch_size, self.max_idx))
        ids = self.ids[indices]
        x = self.get_patches(indices)
        emb = self.embed_patches(x)
        df = pd.DataFrame(emb.cpu().numpy())
        df.insert(0, 'cell_id', ids)
        df.to_csv(self.out_path, mode='a', header=False, index=False)

    def get_patches(self, indices):
        """Generate patches."""
        x = [self.crop_cell(idx) for idx in indices]
        x = [Image.fromarray(img) for img in x]
        x = [self.transform(img) for img in x]
        return torch.stack(x)

    def crop_cell(self, idx):
        """Generate one patch."""
        xcenter, ycenter = self.loc[idx]
        xstart, xend = xcenter - self.offset, xcenter + self.offset + 1
        ystart, yend = ycenter - self.offset, ycenter + self.offset + 1
        return self.img[ystart:yend, xstart:xend]

    def embed_patches(self, x):
        """Transform patches into 1024-dimensional embeddings."""
        x = x.to(self.device)
        with torch.inference_mode():
            return self.model(x)

if __name__ == '__main__':
    uni = Uni(
        img_path='/n/scratch/users/d/daf179/melanoma/LSP26239_postHE_reg.ome.tif',
        csv_path='/n/scratch/users/d/daf179/melanoma/ML/mitosis_balanced.csv',
        out_path='embeddings.csv'
    )
    uni.run()
