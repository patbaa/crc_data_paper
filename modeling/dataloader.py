import numpy as np
from PIL import Image
from tensorflow import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for pathology images.'
    def __init__(self, fnames, labels, preprocess_input, batch_size=32, 
                 dim=(512, 512), n_channels=3, shuffle=True):
        'Initialization'
        self.fnames, self.labels, self.dim, self.n_channels = fnames, labels, dim, n_channels
        self.batch_size, self.shuffle = batch_size, shuffle
        self.preprocess_input = preprocess_input
        np.random.seed(42)
        self.on_epoch_end()

    def __len__(self):
        'Number of batches per epoch'
        return int(np.floor(len(self.fnames) / self.batch_size))

    def __getitem__(self, index):
        'Function for generating one batch of data.'
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(indices)
        return X, y
    
    def getitem(self, index):
        'Function for generating one batch of data.'
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(indices)
        return X, y

    def on_epoch_end(self):
        'Functions to run after each epoch.'
        self.indices = np.arange(len(self.fnames))
        if self.shuffle == True: 
            np.random.shuffle(self.indices)
            
    
    def spatial_augmentation(self, X):
        'Returns 1/8 change for each of the 8 symmetry orientations.'
        rnd = np.random.randint(8)
        if rnd == 0:
            return np.rot90(X, k=0)
        if rnd == 1:
            return np.rot90(X, k=1)
        if rnd == 2:
            return np.rot90(X, k=2)
        if rnd == 3:
            return np.rot90(X, k=3)
        if rnd == 4:
            return np.rot90(np.flip(X, axis=0), k=0)
        if rnd == 5:
            return np.rot90(np.flip(X, axis=0), k=1)
        if rnd == 6:
            return np.rot90(np.flip(X, axis=0), k=2)
        if rnd == 7:
            return np.rot90(np.flip(X, axis=0), k=3)
        

    def __data_generation(self, indices):
        'Lower-level generation of a single batch of data.' 
        X = np.empty((self.batch_size, ) + self.dim + (self.n_channels,))

        fnames_batch = [self.fnames[k] for k in indices]
        for idx, file in enumerate(fnames_batch):
            img = np.array(Image.open(file))
            img = self.spatial_augmentation(img)
                
            X[idx,] = self.preprocess_input(img)
        y = np.array([self.labels[k] for k in indices])

        return X, y