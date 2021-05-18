import os.path
import pathlib
import sys
import time
import sys
import numpy as np
import pickle
from keras.models import load_model
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv1D, AveragePooling1D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn import preprocessing
from exploit_pred import *
from matplotlib.pyplot import figure

if len(sys.argv) != 2:
    raise Exception("Please supply training set id to train on ...")
training_set_id = sys.argv[1]
if training_set_id not in ['all', '1234', '0and11']:
    training_set_id = f"{int(training_set_id):02d}"

### Scripts based on ASCAD github : https://github.com/ANSSI-FR/ASCAD

def check_file_exists(file_path):
        if os.path.exists(file_path) == False:
                print("Error: provided file path '%s' does not exist!" % file_path)
                sys.exit(-1)
        return

def shuffle_data(profiling_x,label_y):
    l = list(zip(profiling_x,label_y))
    random.shuffle(l)
    shuffled_x,shuffled_y = list(zip(*l))
    shuffled_x = np.array(shuffled_x)
    shuffled_y = np.array(shuffled_y)
    return (shuffled_x, shuffled_y)


### CNN network
def cnn_architecture(input_size=4000,learning_rate=0.00001,classes=256):
        
        # Designing input layer
        input_shape = (input_size,1)
        img_input = Input(shape=input_shape)
        
        # 1st convolutional block
        x = Conv1D(2, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1')(img_input)
        x = BatchNormalization()(x)
        x = AveragePooling1D(2, strides=2, name='block1_pool')(x)

        x = Flatten(name='flatten')(x)

        # Classification layer
        x = Dense(2, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
        
        # Logits layer
        x = Dense(classes, activation='softmax', name='predictions')(x)

        # Create model
        inputs = img_input
        model = Model(inputs, x, name='dpacontest_v4')
        optimizer = Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
        return model

#### Training model
def train_model(X_profiling, Y_profiling, X_test, Y_test, model, save_file_name, epochs=150, batch_size=100):
    check_file_exists(os.path.dirname(save_file_name))

    # Save model every epoch
    save_model = ModelCheckpoint(save_file_name)

    # Get the input layer shape
    input_layer_shape = model.get_layer(index=0).input_shape

    # Sanity check
    if input_layer_shape[1] != len(X_profiling[0]):
        print("Error: model input shape %d instead of %d is not expected ..." % (input_layer_shape[1], len(X_profiling[0])))
        sys.exit(-1)

    Reshaped_X_profiling, Reshaped_X_test  = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1)),X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    callbacks=[save_model]      
    
    history = model.fit(x=Reshaped_X_profiling, y=to_categorical(Y_profiling, num_classes=256), validation_data=(Reshaped_X_test, to_categorical(Y_test, num_classes=256)), batch_size=batch_size, verbose = 1, epochs=epochs, callbacks=callbacks)
    return history


#################################################
#################################################

#####            Initialization            ######

#################################################
#################################################

# Our folders
root = "./"
DPAv4_trained_models_folder = root+"DPAv4_trained_models/"
history_folder = root+"training_history/"
predictions_folder = root+"model_predictions/"
pathlib.Path(DPAv4_trained_models_folder).mkdir(exist_ok=True, parents=True)
pathlib.Path(history_folder).mkdir(exist_ok=True, parents=True)
pathlib.Path(predictions_folder).mkdir(exist_ok=True, parents=True)

# Choose the hyperparameter's values
nb_epochs = 50
batch_size = 50
input_size = 4000
learning_rate = 1e-3

start = time.time()

def fetch_data(_set_id):
    if _set_id == 'all':
        _X = []
        _Y = []
        for _i in range(16):
            (X_profiling, Y_profiling) = fetch_data(_set_id=f"{_i:02d}")
            _X.append(X_profiling)
            _Y.append(Y_profiling)
        return (np.concatenate(_X, axis=0), np.concatenate(_Y, axis=0))
    elif _set_id == '1234':
        _X = []
        _Y = []
        for _i in range(4):
            (X_profiling, Y_profiling) = fetch_data(_set_id=f"{_i:02d}")
            _X.append(X_profiling)
            _Y.append(Y_profiling)
        return (np.concatenate(_X, axis=0), np.concatenate(_Y, axis=0))
    elif _set_id == '0and11':
        _X = []
        _Y = []
        for _i in [0, 11]:
            (X_profiling, Y_profiling) = fetch_data(_set_id=f"{_i:02d}")
            _X.append(X_profiling)
            _Y.append(Y_profiling)
        return (np.concatenate(_X, axis=0), np.concatenate(_Y, axis=0))
    else:
        _ = int(_set_id)
        DPAv4_data_folder = root + f"DPAv4_dataset_{_set_id}/"
        # Load the profiling traces
        # (X_profiling, Y_profiling), (X_attack, Y_attack), (plt_profiling, plt_attack) = (np.load(DPAv4_data_folder + 'profiling_traces_dpav4.npy'), np.load(DPAv4_data_folder + 'profiling_labels_dpav4.npy')), (np.load(DPAv4_data_folder + 'attack_traces_dpav4.npy'), np.load(DPAv4_data_folder + 'attack_labels_dpav4.npy')), (np.load(DPAv4_data_folder + 'profiling_plaintext_dpav4.npy'), np.load(DPAv4_data_folder + 'attack_plaintext_dpav4.npy'))
        (X_profiling, Y_profiling) = (np.load(DPAv4_data_folder + 'profiling_traces_dpav4.npy'), np.load(DPAv4_data_folder + 'profiling_labels_dpav4.npy'))
        return (X_profiling, Y_profiling)

# Shuffle data
(X_profiling, Y_profiling) = fetch_data(_set_id=training_set_id)
(X_profiling, Y_profiling) = shuffle_data(X_profiling, Y_profiling)

X_profiling = X_profiling.astype('float32')
# X_attack = X_attack.astype('float32')

#Standardization + Normalization (between 0 and 1)
scaler_1 = preprocessing.StandardScaler()
X_profiling = scaler_1.fit_transform(X_profiling)
# X_attack = scaler.transform(X_attack)

scaler_2 = preprocessing.MinMaxScaler(feature_range=(0,1))
X_profiling = scaler_2.fit_transform(X_profiling)
# X_attack = scaler.transform(X_attack)
# X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))


#################################################
#################################################

####                Training               ######

#################################################
#################################################
#
# # Choose your model
model = cnn_architecture(input_size=input_size, learning_rate=learning_rate)
model_name=f"DPA-contest_v4_trained_on_set_{training_set_id}"
# model = load_model(DPAv4_trained_models_folder + model_name)

print('\n Model name = '+model_name)


print("\n############### Starting Training #################\n")

# Record the metrics
if training_set_id == 'all':
    train_examples = 4000 * 16
elif training_set_id == '1234':
    train_examples = 4000 * 4
else:
    train_examples = 4000
history = train_model(
    X_profiling[:train_examples],
    Y_profiling[:train_examples],
    X_profiling[train_examples:],
    Y_profiling[train_examples:],
    model,
    DPAv4_trained_models_folder + model_name,
    epochs=nb_epochs,
    batch_size=batch_size
)
end=time.time()

print('Execution Time = %d'%(end-start))

print("\n############### Training Done #################\n")

# Save the DL metrics (loss and accuracy)
with open(history_folder + 'history_' + model_name, 'wb') as file_pi:
    pickle.dump(history.history, file_pi)


#################################################
#################################################

####               Prediction              ######

#################################################
#################################################


plot.clf()
# plot.rcParams['figure.figsize'] = (2000, 1000)
figure(figsize=(12, 10), dpi=400)
plot.ylim(-5, 125)
plot.grid(True)
plot.xlabel('Number of traces')
plot.ylabel('Rank')
# legend = plot.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=1, ncol=1)
title = plot.title(
    f"Rank plots for 16 fixed key datasets for model {model_name}"
)

nb_traces_attacks = 30
nb_attacks = 100

for set_id in [f"{_:02d}" for _ in range(16)]:
    DPAv4_data_folder = root + f"DPAv4_dataset_{set_id}/"
    real_key = np.load(DPAv4_data_folder + "key.npy")
    mask = np.load(DPAv4_data_folder + "mask.npy")
    att_offset = np.load(DPAv4_data_folder + "attack_offset_dpav4.npy")
    (X_attack, Y_attack, plt_attack) = (
        np.load(DPAv4_data_folder + 'attack_traces_dpav4.npy'), np.load(DPAv4_data_folder + 'attack_labels_dpav4.npy'),
        np.load(DPAv4_data_folder + 'attack_plaintext_dpav4.npy')
    )
    X_attack = X_attack.astype('float32')
    X_attack = scaler_1.transform(X_attack)
    X_attack = scaler_2.transform(X_attack)
    X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))

    print("\n############### Starting Predictions #################\n")

    predictions = model.predict(X_attack)

    print("\n############### Predictions Done #################\n")

    np.save(predictions_folder + 'predictions_' + model_name + f'_attacked_on_set_{set_id}.npy', predictions)

    #################################################
    #################################################

    ####            Perform attacks            ######

    #################################################
    #################################################

    print("\n############### Starting Attack on Test Set #################\n")

    avg_rank = perform_attacks(nb_traces_attacks, predictions, nb_attacks, plt=plt_attack, key=real_key, mask=mask, offset=att_offset, byte=0, filename=model_name, savefig=False)

    print("\n t_GE = ")
    print(np.where(avg_rank<=0))

    if set_id == training_set_id:
        plot_type = '-'
    else:
        plot_type = ':'
        if training_set_id == '1234':
            if int(set_id) < 4:
                plot_type = '-'
        if training_set_id == '0and11':
            if int(set_id) in [0, 11]:
                plot_type = '-'

    line_plot, = plot.plot(avg_rank, plot_type, label=f'set_{set_id} (key {real_key[0]:03d})')

    print("\n############### Attack on Test Set Done #################\n")

# plot and save to disk
legend = plot.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=1, ncol=1)
plot.savefig(
    f"fig/rank_all_sets_with_{nb_attacks}_attacks_for_model_{model_name}.svg",
    format='svg',
    # dpi=1200,
    bbox_extra_artists=(legend,),
    bbox_inches='tight'
)
