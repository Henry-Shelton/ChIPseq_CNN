'''
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~# HYPERAMETER OPT #~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

#HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
#HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
METRIC_ACCURACY = 'accuracy'

def hparams_config():
    hp.hparams_config(
        hparams=[HP_OPTIMIZER], #HP_NUM_UNITS, HP_DROPOUT]
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
    )
 '''

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~# HYPERPARAMETERS #~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

print("\n \n ################### LOADING HYPERPARAMETERS ###################\n \n")

#BUILDMODEL / DEFINEMODEL / CALLPEAKS: 

batch_size = 1                        #define input size for train/test data


#BUILDMODEL / DEFINEMODEL:

threshold_division = 40               #output bias [10 for beta / 50 for alpha]
learning_rate = 0.0005               #optimiser #default 0.00003 #optimum 0.0005 (lower = longer to intercest train/test, .0005 intersect at 4k)
filter_size_a = 101                   #smoothing filter (101) 
filter_size_b = 301                   #smoothing filter (301)

#BUILDMODEL:

generations = 8000                   #upper bound for range of model training steps
eval_every = 20                      #assess every n generations
class_threshold = 0.6               #loss & dropout probability + getStat() .55


#.5, .55, .6, .65, .7, .75


#DEFINEMODEL:
target_size = 12000                   #1: placeholders / bam, label, generef inputs for training

conv1_ref_features = 4                #2: stem layers / conv1_ref_features

conv1_features = 16                   #2: stem layers / main stem layer 
conv2_features = 32                   #2: stem layers / main stem layer 

conv1a_features = 16                  #3: inception layers / inception 1
conv1b_features = 16                  #3: inception layers / inception 1
conv1c_features = 16                  #3: inception layers / inception 1
convMax1_features = 16                #3: inception layers / inception 1
convAvg1_features = 16                #3: inception layers / inception 1

conv2a_features = 32                  #3: inception layers / inception 2
conv2b_features = 32                  #3: inception layers / inception 2
conv2c_features = 32                  #3: inception layers / inception 2
convMax2_features = 32                #3: inception layers / inception 2
convAvg2_features = 32                #3: inception layers / inception 2

conv3a_features = 48                  #3: inception layers / inception 3
conv3b_features = 48                  #3: inception layers / inception 3
conv3c_features = 48                  #3: inception layers / inception 3

conv4a_features = 128                 #3: inception layers / inception 4
conv4b_features = 128                 #3: inception layers / inception 4
conv4c_features = 128                 #3: inception layers / inception 4

conv5a_features = 192                 #3: inception layers / inception 5
conv5b_features = 192                 #3: inception layers / inception 5
conv5c_features = 192                 #3: inception layers / inception 5

conv6a_features = 512                 #3: inception layers / inception 6 
conv6b_features = 512                 #3: inception layers / inception 6
conv6c_features = 512                 #3: inception layers / inception 6
conv6d_features = 512                 #3: inception layers / inception 6

conv7a_features = 720                 #3: inception layers / inception 7
conv7b_features = 720                 #3: inception layers / inception 7
conv7c_features = 720                 #3: inception layers / inception 7
conv7d_features = 720                 #3: inception layers / inception 7

resulting_width = 1                   #4: fully connected (final conv size)
fully_connected_size1 = 500           #4: fully connected

fully_connected_size2 = 250           #4: fully connected + #5: output layer

max_pool_size_stem = 2                #6.1: definemodel / peakPredictConvModel()
init_depth = 1                        #6.1: definemodel / peakPredictConvModel() + generateOutput()
topK_set_a = 6000                     #6.1: definemodel / aggregatedLoss() / loss_a 
topK_set_b = 3000                     #6.1: definemodel / aggregatedLoss() / loss_b

#~~~~~~~~~# UNUSED #~~~~~~~~~#

#pnRate_threshold = 50
#evaluation_size = 1
#topK_set_c = 1500
#topK_set_d = 750
#max_pool_size_ref1 = 2
#max_pool_size_ref2 = 2
#max_pool_size_ref3 = 2
#convMax3_features = 64
#convAvg3_features = 64
#convMax4_features = 32
#convAvg4_features = 32
#convMax5_features = 128
#convAvg5_features = 128
#max_pool_size1 = 2
#max_pool_size2 = 2
#max_pool_size3 = 2
#max_pool_size4 = 3
#max_pool_size5 = 2
#max_pool_size6 = 2             