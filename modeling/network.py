from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50

def get_model(name, n_out):
    if name == 'ResNet50':
        model = ResNet50(include_top=False, weights='imagenet', 
                         input_shape=(512, 512, 3), pooling='max')
    
        inp = model.input
        out = model.layers[-1].output
        out = Dense(n_out, activation='sigmoid', name='predictions')(out)
        return Model(inp, out)