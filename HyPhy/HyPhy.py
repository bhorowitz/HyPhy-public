# prerequisites
import tensorflow
print("Tensorflow Version: ", tensorflow.__version__)

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Input, Dense, Lambda, concatenate,Reshape,Flatten
from keras.layers import Reshape,UpSampling3D,RepeatVector,Conv3D,MaxPool3D

from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.utils import to_categorical
from scipy.stats import norm
from keras import optimizers

import warnings
from keras.layers.merge import concatenate as concat
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import keras

from scipy import ndimage


keras.backend.set_image_data_format('channels_first')

warnings.filterwarnings('ignore')


import random
import itertools

import numpy as np
from tensorflow.keras.utils import Sequence

#Random helper functions, many depreciated...
def density_transform(x):
    return np.log10(3*x)

def velocity_transform(x):
    return x/75.0

def temp_transform(x):
    return (3.82-np.log10(x.clip(max=1000000))*3.8) +10


def inv_density_transform(y):
    return np.power(10,4*y)/3.0

def inv_temp_transform(y):
    prefac = ((4*y-13.8))/3.8
    return np.power(10,prefac)

def mean_trim(arr,min_t,max_t,min_d,max_d):
    temp_fields = arr[:,-1,:,:]
    mean_t = np.mean(temp_fields)
    t_array = np.where(temp_fields > max_t,mean_t,temp_fields)
    t_array = np.where(temp_fields < min_t,mean_t,t_array)

    den_fields = arr[:,0,:,:]
    mean_d = np.mean(den_fields)
    d_array = np.where(den_fields > max_d,mean_d,den_fields)
    d_array = np.where(den_fields < min_d,mean_d,d_array)
    
    arr[:,0,:,:,:] = d_array
    arr[:,-1,:,:,:] = t_array
    return arr

class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, list_IDs_x,list_IDs_y, normy = True,
                 to_fit=True, batch_size=32, 
                 n_channels=1, shuffle=True, inde = [0,1,2],
                 max_num = 10, fix_direction=False):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param labels: list of image labels (file names)
        :param image_path: path to images location
        
        :param to_fit: True to return X and y, False to return X only (I think always should be true?)
        :param batch_size: batch size at each iteration
        :param n_channels: number of image channels (should be 1)
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.list_IDs_x = list_IDs_x #filenames of density fields
        self.list_IDs_y = list_IDs_y #filenames of hydro fields
        self.max_num = max_num #maximum number of files to use per epoch
        self.normy = normy #normalize hydrofield quantities (y/n)?
        self.to_fit = to_fit #always True? False not implemented well...
        self.batch_size = batch_size 
        self.n_channels = n_channels #should always be one
        self.shuffle = shuffle #shuffle order of boxes
        self.st = "ijk" #for using various reflection symmetries 
        self.max = self.__len__()
        self.n = 0 #intializing get_item for random rearrangemnt
        self.fix_direction = fix_direction #use mirror/reflection symmetries
        self.perm = list(itertools.permutations(inde))
        
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        self.on_epoch_end()
        return int(np.floor(len(self.list_IDs_x) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        #choose random permulation
        
        #Choose orientation...
        y = self.y[self.batch_size*index:self.batch_size*(index+1)]
        X  = self.X[self.batch_size*index:self.batch_size*(index+1)]
        if self.fix_direction:
            ore = (0,1,2)
        else:
            ore = random.choice(self.perm)
            #print(ore)
     #   ore = (0, 1, 2)
        #print(ore)
        # Generate data
        X = self._perm_x(X,ore)
        X = np.nan_to_num(X)[:,[0,1,4],:,:]
        if self.to_fit:
            #print(y.shape)
            y = self._perm_y(y,ore)
            #y = y
            if self.normy:
                y = np.nan_to_num(self._norm(y[:,[0,1,4],:,:]))
            else:
                y = np.nan_to_num(y[:,[0,1,4],:,:])
            return [y,X],y #y*(np.random.randn(32,1,1,1,1)*0.1


    def _perm_x(self,hold,i):
        #for mirror/reflection symmetries
        Q = self.st[i[0]]+self.st[i[1]]+self.st[i[2]]
       #print('mnijk->mn'+Q)
        t1_x_temp = np.einsum('mnijk->mn'+Q, hold)
        t1_x_temp[:,[0,1, 2,3,4],:,:,:] = t1_x_temp[:,[0,i[0]+1,i[1]+1,i[2]+1,4],:,:,:]
        return t1_x_temp
    
    def _perm_y(self,hold,i):
        #for mirror/reflection symmetries

        Q = self.st[i[0]]+self.st[i[1]]+self.st[i[2]]
       # print('mnijk->mn'+Q)

        t1_y_temp = np.einsum('mnijk->mn'+Q, hold)
        t1_y_temp[:,[0,1, 2,3,4],:,:,:] = t1_y_temp[:,[0,i[0]+1,i[1]+1,i[2]+1,4],:,:,:]
        return t1_y_temp

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs_x))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
                # Generate indexes of the batch
        print(self.indexes)
        indexes = self.indexes[0:self.max_num]

        # Find list of IDs
        list_IDs_tempx = [self.list_IDs_x[k] for k in indexes]
        list_IDs_tempy = [self.list_IDs_y[k] for k in indexes]
        print(list_IDs_tempx)
        self.X = self._generate_X(list_IDs_tempx)
        self.y = self._generate_y(list_IDs_tempy)

            
    def _generate_X(self, list_IDs_temp_x):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        test_x = []
        for i in list_IDs_temp_x:
            test_x.append([np.load(i)])
       
        test_x = np.reshape(np.array(test_x),(-1,5,64,64,64))[:,:,:,:,:]
        return test_x

    def _generate_y(self, list_IDs_temp_y):
        """Generates data containing batch_size masks
        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        """
        test_y = []
        for i in list_IDs_temp_y:
            test_y.append([np.load(i)[:,:,:,:]])
        test_y = np.reshape(np.array(test_y),(-1,5,64,64,64))
                            
        test_yn = test_y[:,:,:,:] #selecting only one baryon velocity
        return test_yn
#reducing variance so all variables have similar dynamic range: otherwise loss will be ->inf :'()'

    def _norm(self,test_yn):
        #if normed = True
            
            #test_yn[:,0] = ndimage.gaussian_filter(test_yn[:,0],0.0)
            #test_yn[:,1] = ndimage.gaussian_filter(test_yn[:,1],0.0)
            #test_yn[:,2] = ndimage.gaussian_filter(test_yn[:,2],0.0)
            test_yn = mean_trim(test_yn,1000,1500000,0.01,50000)
            
            test_yn[:,0] = np.log10(test_yn[:,0].clip(min=0.01,max=50000))#density_transform(test_yn[:,0].clip(min=0.00001))/4
            test_yn[:,1] = test_yn[:,1]/100#velocity_transform(test_yn[:,1])/4
            test_yn[:,2] = np.log10(test_yn[:,2].clip(min=1000.0,max=400000))#temp_transform(test_yn[:,2].clip(min=0.00001)).clip(max=3.0)/4
            
            
            return test_yn
        
    def __next__(self):
        if self.n >= self.max:
           self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result
    
import tensorflow as tf
from keras.layers import Conv3D,Flatten,Conv3DTranspose,concatenate
from keras.layers import Input, Dense, Lambda, concatenate,Reshape,Flatten
from keras.layers import Reshape,UpSampling3D,RepeatVector,Conv3D,MaxPool3D, AvgPool3D
from keras.layers import Dropout, BatchNormalization
import keras


class HyPhy:
    
    """
    Class that holds the model for both train (cvae, and associated loss) and generation (gen),
    
    Currently takes as input just number of hidden dimensions and n_hidden, could 
    easily make more things free parameters to set...
    
    Set up slightly strangly to allow (hopefully) seamless switching between training and 
    generation, as well as changing generation size...
    
    """
    def __init__(self, n_hidden=128,z_dim=27, edge_clip=10, rec_loss_factor = 1):
        self.n_hidden = n_hidden #size of dense layers used to set mu, logvar for latent space
        self.z_dim = z_dim #number of latent space dimensions
        self.edge_clip = edge_clip #pixels to clip off of reconstructed tau for comparison
        self.rec_loss_factor = rec_loss_factor #relative weight of kl loss vs. rec loss
        self._init = 'lecun_normal'

        self.__init_encoder__() #initialize encoder layers
        self.__init_decoder__() #initialize decoder layers
        
        self.__init_cvae__() #creates training model
        #self.__init_gen__() #just call HyPhy.gen(), not made by default
    def __init_encoder__(self):
        ## dm
     
        self.inter_up0 = Conv3D(4,3,padding="same",activation="selu",kernel_initializer = self._init)
        self.inter_up0p = Conv3D(5,3,padding="same",activation="selu",kernel_initializer = self._init)
        self.encode_dm1 = Conv3D(6,2,padding="same",activation="selu",kernel_initializer = self._init)
        self.encode_dm1p = Conv3D(6,3,padding="same",activation="selu",kernel_initializer = self._init)

        self.encode_dm2 = Conv3D(8,3,padding="same",activation="selu",kernel_initializer = self._init)
        self.encode_dm3 = Conv3D(20,3,padding="same",activation="selu",kernel_initializer = self._init)
        self.encode_dm3p = Conv3D(30,3,padding="same",activation="selu",kernel_initializer = self._init)
        self.encode_dm4 = Conv3D(30,2,padding="same",activation="selu",kernel_initializer = self._init) ## linear?
        self.encode_dm4p = Conv3D(30,2,padding="same",activation="selu",kernel_initializer = self._init) ## linear?

        ## tau
    
        self.encode_tau1 = Conv3D(3,2,padding="same") 
        self.encode_tau1P = Conv3D(4,3,padding="same",activation="selu",kernel_initializer = self._init)
        self.encode_tau2 = Conv3D(6,4,padding="same",activation="selu",kernel_initializer = self._init)
        self.encode_tau2P = Conv3D(8,3,padding="same",activation="selu",kernel_initializer = self._init)
        self.encode_tau3 = Conv3D(20,5,padding="same",activation="selu",kernel_initializer = self._init)
        self.encode_tau3P = Conv3D(30,3,padding="same",activation="selu",kernel_initializer = self._init)
        self.encode_tau4 = Conv3D(30,3,padding="same",activation="selu",kernel_initializer = self._init)
        self.encode_tau4p = Conv3D(30,3,padding="same",activation="selu",kernel_initializer = self._init)
    def __init_decoder__(self):
        
        self.z_decoder2 = Conv3DTranspose(16,6,padding="same",activation="selu",kernel_initializer = self._init)#Dense(n_hidden*3, activation='relu')
        self.z_decoder2p = Conv3D(16,3,padding="same",activation="selu",kernel_initializer = self._init)
        self.z_decoder3 = Conv3DTranspose(28,3,padding="same",activation="selu",kernel_initializer = self._init)
        self.z_decoder3_mix = Conv3DTranspose(28,1,padding="same",activation="selu",kernel_initializer = self._init)
        self.z_decoder3P = Conv3D(28,2,padding="same",activation="selu",kernel_initializer = self._init)
        self.z_decoder4 = Conv3DTranspose(24,1,padding="same",activation="selu",kernel_initializer = self._init)#Dense(n_hidden*2, activation='linear')
        self.z_decoder4P = Conv3D(24,2,padding="same",activation="selu",kernel_initializer = self._init)#Dense(n_hidden*2, activation='linear')

        self.y_decoder = Conv3D(24,1,padding="same",activation="selu",kernel_initializer = self._init)#Dense(x_tr.shape[1], activation='linear')
       # self.y_decoder_BN = BatchNormalization()
        self.y_decoderP1 = Conv3D(8,5,padding="same",activation="selu",kernel_initializer = self._init)#Dense(x_tr.shape[1], activation='linear')
        self.y_decoderP2 = Conv3D(8,3,padding="same",activation="selu",kernel_initializer = self._init)#Dense(x_tr.shape[1], activation='linear')
        self.y_decoderP2_mix = Conv3D(8,2,padding="same",activation="selu",kernel_initializer = self._init)#Dense(x_tr.shape[1], activation='linear')
        self.y_decoderP3 = Conv3D(8,3,padding="same",activation="tanh")#Dense(x_tr.shape[1], activation='linear')
        self.y_decoderP4 = Conv3D(3,3,padding="same",activation="selu",kernel_initializer = self._init)#Dense(x_tr.shape[1], activation='linear')
    
    def encoder_dm(self,dm_box):
        
        #getting dm field to same size as tau field...
        dm0 = dm_box#UpSampling3D(name="dm_up1")(dm_box)
        dm0 = self.inter_up0(dm0)
        dm0 = self.inter_up0p(dm0)
        #dm0 = UpSampling3D(name="dm_up2")(dm0)
        
        #step one
        dm1 = self.encode_dm1(dm0)
        dm1 = self.encode_dm1p(dm1)
        dm1_p = MaxPool3D()(dm1)

        #step two
        dm2 = self.encode_dm2(dm1_p)
        dm2_p = MaxPool3D()(dm2)
      
        #step three
        dm3 = self.encode_dm3(dm2_p)
        dm3 = self.encode_dm3p(dm3)

        dm3_p = MaxPool3D()(dm3)

        #step four
        dm4 = self.encode_dm4(dm3_p)
        dm4 = self.encode_dm4p(dm4)

        dm4 = MaxPool3D()(dm4)
        return dm0,dm1,dm2,dm3,dm4
    
    def encoder_tau(self,x_in,dm0,dm1,dm2,dm3,dm4):
        
        tau_box = Reshape((3,64,64,64))(x_in)

        #step 1
        tau1 = self.encode_tau1(tau_box)
        tau1 = self.encode_tau1P(tau1)

        #merge + step 2
        tau1_dm1 = concatenate([dm1,tau1],axis=1)
        tau2 = self.encode_tau2(tau1_dm1)
        tau2 = self.encode_tau2P(tau2)
        tau2_p = MaxPool3D()(tau2)

        #merge + step 3

        tau2_dm2 = concatenate([dm2,tau2_p],axis=1)
        tau3 = self.encode_tau3(tau2_dm2)
        tau3 = self.encode_tau3P(tau3)
        tau3 = BatchNormalization()(tau3)
        tau3_p = MaxPool3D()(tau3)

        #merge + step 4
        tau3_dm3 = concatenate([dm3,tau3_p],axis=1)
        tau4 = self.encode_tau4(tau3_dm3)
        tau4 = MaxPool3D()(tau4) #maybe do something else here? more layers?
       # tau4 = self.encode_tau4p(tau4)
        tau4 = MaxPool3D()(tau4)
        return tau4
        
    def variational_block(self,tau4,dm4):
        
        x_encoded = concatenate([tau4,dm4])
        x_encoded = Flatten()(x_encoded)
        x_encoded = Dense(self.n_hidden, activation='selu',kernel_initializer = self._init)(x_encoded)
        x_encoded = Dropout(.10)(x_encoded)
        x_encoded = Dense(self.n_hidden//2, activation='selu',kernel_initializer = self._init)(x_encoded)
        
        self.mu = Dense(self.z_dim, activation='linear')(x_encoded)
        self.log_var = Dense(self.z_dim, activation='linear')(x_encoded)
        
        def sampling(args):
            mu, log_var = args
            eps = K.random_normal(shape=(K.shape(dm4)[0], self.z_dim), mean=0., stddev=1.0)
            return mu + K.exp(log_var/2.) * eps

        z = Lambda(sampling, output_shape=(self.z_dim,))([self.mu, self.log_var])
        return z
    
    def decoder(self,z_new, dm0,dm1,dm2,dm3,dm4):
            
        #dm4_new = Reshape((10,4,4,4))(dm4)

        z_cond = concatenate([z_new, dm4],axis=1)
    
        z_decoded = self.z_decoder2(z_cond)
        z_decoded = UpSampling3D()(z_decoded)
        z_decoded = self.z_decoder2p(z_decoded)
        z_decoded = UpSampling3D()(z_decoded)

        z_decoded = concatenate([z_decoded,dm3],axis=1)
        z_decoded = self.z_decoder3(z_decoded)
        z_decoder = self.z_decoder3_mix(z_decoded)
        z_decoded = self.z_decoder3P(z_decoded)
        z_decoded = UpSampling3D()(z_decoded)

        z_decoded = concatenate([z_decoded,dm2],axis=1)

        z_decoded = self.z_decoder4(z_decoded)
        z_decoded = self.z_decoder4P(z_decoded)

        z_decoded = UpSampling3D()(z_decoded)

        z_decoded = concatenate([z_decoded,dm1],axis=1)

        y0 = self.y_decoder(z_decoded)
       # y0 = self.y_decoder_BN(y0)
        z_decoded = concatenate([y0,dm0],axis=1)
        y0 = self.y_decoderP1(y0)
        y0 = self.y_decoderP2(y0)
        y0 = self.y_decoderP2_mix(y0)
        y0 = self.y_decoderP3(y0)
        y = self.y_decoderP4(y0)
        return y
    
    def __init_cvae__(self):
        
        self.condition = Input(shape=(3,64,64,64),name="DM_field")
        #dm density, dm velocity, redshiftt
        self.x_in = Input(shape=(3,64,64,64),name="tau_field")
        #baryon density, los velocity, temperature
        dm_box = Reshape((3,64,64,64))(self.condition)

        dm0,dm1,dm2,dm3,dm4 = self.encoder_dm(dm_box)
        tau4 = self.encoder_tau(self.x_in,dm0,dm1,dm2,dm3,dm4)
        
        z = self.variational_block(tau4,dm4)
        
        z_new = Lambda(lambda x: tf.tensordot(x, K.ones((4,4,4)), axes=0))(z)
        
        y = self.decoder(z_new, dm0,dm1,dm2,dm3,dm4)
        
        self.cvae = Model([self.x_in, self.condition], y)
    
    def cvae_loss(self,x,y):
        ec = int(self.edge_clip)
        x_f = K.flatten(x[:,:,ec:-1*ec,ec:-1*ec,ec:-1*ec])
        y_f = K.flatten(y[:,:,ec:-1*ec,ec:-1*ec,ec:-1*ec])

        reconstruction_loss = objectives.mean_squared_error(x_f, y_f)*16**3*4**3*self.rec_loss_factor

        kl_loss = 0.5 * K.sum(K.square(self.mu) + K.exp(self.log_var) - self.log_var - 1, axis = -1)
    #    out_o_bounds = tf.reduce_sum(tf.cast(tf.logical_and(y[:,0,ec:-1*ec,ec:-1*ec,ec:-1*ec]>.2,y[:,-1,ec:-1*ec,ec:-1*ec,ec:-1*ec]<4.0),tf.float32))

        rr = (y[:,0,ec:-1*ec,ec:-1*ec,ec:-1*ec] - 1.0*np.ones((32,44,44,44)))**2 + (y[:,-1,ec:-1*ec,ec:-1*ec,ec:-1*ec] - 3.5*np.ones((32,44,44,44)))**2
        out_o_bounds = tf.reduce_sum(tf.math.exp(-1*rr/0.15)*5)
        
        loss = K.mean(reconstruction_loss + kl_loss)
        return loss
    
    def kl_loss(self,x,y):
        # just for debugging
        ec = int(self.edge_clip)
        x_f = K.flatten(x[:,:,ec:-1*ec,ec:-1*ec,ec:-1*ec])
        y_f = K.flatten(y[:,:,ec:-1*ec,ec:-1*ec,ec:-1*ec])
        kl_loss = 0.5 * K.sum(K.square(self.mu) + K.exp(self.log_var) - self.log_var - 1, axis = -1)
        return kl_loss
    
    def rec_loss(self,x,y):    
    
        ec = int(self.edge_clip)
        x_f = K.flatten(x[:,:,ec:-1*ec,ec:-1*ec,ec:-1*ec])
        y_f = K.flatten(y[:,:,ec:-1*ec,ec:-1*ec,ec:-1*ec])

        reconstruction_loss = objectives.mean_squared_error(x_f, y_f)*16**3*4**3*self.rec_loss_factor
        return reconstruction_loss

    def rec_loss_exclude(self,x,y):    
    
        ec = int(self.edge_clip)
        rr = (y[:,0,ec:-1*ec,ec:-1*ec,ec:-1*ec] - 1.0*np.ones((32,44,44,44)))**2 + (y[:,-1,ec:-1*ec,ec:-1*ec,ec:-1*ec] - 3.5*np.ones((32,44,44,44)))**2
        out_o_bounds = tf.math.exp(-1*rr/0.15)*5
        #out_o_bounds = tf.reduce_sum(tf.cast(tf.logical_and(y[:,0,ec:-1*ec,ec:-1*ec,ec:-1*ec]>.20,y[:,-1,ec:-1*ec,ec:-1*ec,ec:-1*ec]<4.0),tf.float32))
        return tf.reduce_sum(out_o_bounds)
    

    def generator(self,size1 = 32,reduced_len=8):
        
        _condition = Input(shape=(3,size1,size1,size1),name="DM_field")
        _z = Input(shape=(self.z_dim,))
        #x_in = Input(shape=(3,64,64,64),name="tau_field")
        
        dm0,dm1,dm2,dm3,dm4 = self.encoder_dm(_condition)
        
        _z_new = Lambda(lambda x: tf.tensordot(x, K.ones((reduced_len,reduced_len,reduced_len)), axes=0))(_z)
        
        _y = self.decoder(_z_new, dm0,dm1,dm2,dm3,dm4)
        
        self.gen = Model([_condition,_z], _y)
        
