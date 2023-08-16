print(" \n ################### DEFINE MODEL ###################\n ")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~# MODEL MODULES #~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
print("[INFO] importing definemodel modules\n")

import tensorflow as tf #2.13.0

from scipy.signal import gaussian
import numpy as np

#get rid of depreciated errors
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #tf 2.x

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) #np

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from buildmodel.hyperparameters import *

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

'''

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~# HYPERPARAMETERS #~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import hyperparameters #import hyperparams
print("[INFO] importing definemodel hyperparams\n")

from hyperparameters import batch_size, learning_rate, filter_size_a, filter_size_b, init_depth, target_size, topK_set_a, topK_set_b

from hyperparameters import conv1_ref_features
from hyperparameters import conv1_features, conv2_features
from hyperparameters import conv1a_features, conv1b_features, conv1c_features, convMax1_features, convAvg1_features
from hyperparameters import conv2a_features, conv2b_features, conv2c_features, convMax2_features, convAvg2_features
from hyperparameters import conv3a_features, conv3b_features, conv3c_features
from hyperparameters import conv4a_features, conv4b_features, conv4c_features
from hyperparameters import conv5a_features, conv5b_features, conv5c_features
from hyperparameters import conv6a_features, conv6b_features, conv6c_features, conv6d_features
from hyperparameters import conv7a_features, conv7b_features, conv7c_features, conv7d_features
from hyperparameters import max_pool_size_stem
from hyperparameters import resulting_width
from hyperparameters import fully_connected_size1
from hyperparameters import fully_connected_size2
from hyperparameters import threshold_division

'''

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~# DEFINE MODEL #~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


print("[INFO] defining model\n")

#global scope for trained model tensor objects to quickly summarised/saved
        
def tensorSummaries(tensor, name):
    with tf.name_scope(name):
        # Calculate the mean value
        mean = tf.reduce_mean(tensor)
        tf.summary.scalar('mean', mean)

        with tf.name_scope('stddev'):
            # Calculate the standard deviation
            stddev = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))
        tf.summary.scalar('stddev', stddev)

        # Calculate the maximum and minimum values
        tf.summary.scalar('max', tf.reduce_max(tensor))
        tf.summary.scalar('min', tf.reduce_min(tensor))

        # Create a histogram summary
        tf.summary.histogram('histogram', tensor)


#~~~~~~~~~~# PLACEHOLDERS #~~~~~~~~~~#

print("[INFO] 1 - defining placeholders\n")
#data to be assigned/fed into these placeholder variables later on (from tf v1)

#loss/dropout probs
tf.compat.v1.disable_eager_execution() #allow tf.placeholder v1 in a v2 script without errors

is_train_step = tf.compat.v1.placeholder(tf.bool)
loss_weight = tf.compat.v1.placeholder(tf.float32)
class_threshold = tf.compat.v1.placeholder(tf.float32)
p_dropout = tf.compat.v1.placeholder(tf.float32)

#bam, label, generef inputs for training - [HP: batch_size, target_size]
input_data_train = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, target_size, 1), name="trainingData")
input_data_eval = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, target_size, 1), name="testData")

label_data_train = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, 1, target_size))
label_data_eval = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, 1, target_size))

input_ref_data_train = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, target_size, 1), name="TrainRefData")
input_ref_data_eval = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, target_size, 1), name="TestRefData")

#smoothing filter - [HP: filter_size_b]
smoothing_filter = tf.constant(gaussian(filter_size_b,50)/np.sum(gaussian(filter_size_b,50)),tf.float32, shape=[filter_size_b,1,1], name='smoothing_filter')



#~~~~~~~~~~# STEM LAYERS #~~~~~~~~~~#

print("[INFO] 2 - defining stem layeers\n")
#initial layers of most CNNs - play critical role in learning the most local features (large volume so seperate from body)

#refgen depth stem - [HP: conv1_ref_features]
conv1_ref_weight = tf.get_variable("Conv_REF_1", shape=[3, 1, conv1_ref_features], initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))

#main stem layers - [HP: conv1_features, conv2_features]
conv1_weight = tf.get_variable("Conv_STEM1", shape=[5, 1, conv1_features], initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))
conv2_weight = tf.get_variable("Conv_STEM2", shape=[3, conv1_features, conv2_features], initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))
conv1_bias = tf.get_variable("Conv_STEM1_bias", shape=[conv1_features], initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))
conv2_bias = tf.get_variable("Conv_STEM2_bias", shape=[conv2_features], initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))



#~~~~~~~~~~# INCEPTION LAYERS #~~~~~~~~~~#

print("[INFO] 3 - defining inception layers filters\n")
#inception network layers for deep CNN  architecture (aka GoogLeNet) - consists of 1x1, 3x3, 5x5, max pooling and output concat

#inception 1 - [HP: conv1a_features, conv1b_features, conv1c_features, convMax1_features, convAvg1_features]
layer1_width = conv2_features + conv1_ref_features

conv1a_weight = tf.get_variable("Conv_1A", shape=[4, layer1_width, conv1a_features], initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))
conv1b_weight = tf.get_variable("Conv_1B", shape=[3, layer1_width, conv1b_features], initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))
conv1c_weight = tf.get_variable("Conv_1C", shape=[2, layer1_width, conv1c_features], initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))
print("TESTTTTTTTTT: {}".format(conv1a_weight.shape))
convMax1_weight = tf.get_variable("Conv_max_W1", shape=[2, layer1_width, convMax1_features], initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))
convAvg1_weight = tf.get_variable("Conv_avg_W1", shape=[2, layer1_width, convAvg1_features], initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))

conv1a_bias = tf.get_variable("conv1a_bias", shape=[conv1a_features], initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))
conv1b_bias = tf.get_variable("conv2b_bias", shape=[conv1b_features], initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))
conv1c_bias = tf.get_variable("conv3c_bias", shape=[conv1c_features], initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))

#inception 2 - [HP: conv2a_features, conv2b_features, conv2c_features, convMax2_features, convAvg2_features]
layer2_width = conv1a_features + conv1b_features + conv1c_features + (layer1_width*2)

conv2a_weight = tf.get_variable("Conv_2A", shape=[4, layer2_width, conv2a_features], initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))
conv2b_weight = tf.get_variable("Conv_2B", shape=[3, layer2_width, conv2b_features], initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))
conv2c_weight = tf.get_variable("Conv_2C", shape=[2, layer2_width, conv2c_features], initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))

convMax2_weight = tf.get_variable("Conv_max_W2", shape=[2, layer2_width, convMax2_features], initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))
convAvg2_weight = tf.get_variable("Conv_avg_W2", shape=[2, layer2_width, convAvg2_features], initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))

#inception 3 - [HP: conv3a_features, conv3b_features, conv3c_features]
layer3_width = conv2a_features + conv2b_features + conv2c_features + (layer2_width *2 )

conv3a_weight = tf.get_variable("Conv_3A", shape=[5, layer3_width, conv3a_features],initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))
conv3b_weight = tf.get_variable("Conv_3B", shape=[4, layer3_width, conv3b_features],initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))
conv3c_weight = tf.get_variable("Conv_3C", shape=[3, layer3_width, conv3c_features],initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))

#inception 4 - [HP: conv4a_features, conv4b_features, conv4c_features]
layer4_width = conv3a_features + conv3b_features + conv3c_features + layer3_width

conv4a_weight = tf.get_variable("Conv_4A", shape=[4, layer4_width, conv4a_features], initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))
conv4b_weight = tf.get_variable("Conv_4B", shape=[3, layer4_width, conv4b_features], initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))
conv4c_weight = tf.get_variable("Conv_4C", shape=[2, layer4_width, conv4c_features],initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))

#inception 5 - [HP: conv5a_features, conv5b_features, conv5c_features]
layer5_width = conv4a_features + conv4b_features + conv4c_features + layer4_width

conv5a_weight = tf.get_variable("Conv_5A", shape=[4, layer5_width, conv5a_features],initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))
conv5b_weight = tf.get_variable("Conv_5B", shape=[3, layer5_width, conv5b_features],initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))
conv5c_weight = tf.get_variable("Conv_5C", shape=[2, layer5_width, conv5c_features],initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))

#inception 6 - [HP: conv6a_features, conv6b_features, conv6c_features, conv6d_features]
layer6_width = conv5a_features + conv5b_features + conv5c_features + layer5_width

conv6a_weight = tf.get_variable("Conv_6A", shape=[10, layer6_width, conv6a_features],initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))
conv6b_weight = tf.get_variable("Conv_6B", shape=[8, layer6_width, conv6b_features],initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))
conv6c_weight = tf.get_variable("Conv_6C", shape=[6, layer6_width, conv6c_features],initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))
conv6d_weight = tf.get_variable("Conv_6D", shape=[5, layer6_width, conv6d_features],initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))

#inception 7 - [HP: conv7a_features, conv7b_features, conv7c_features, conv7d_features]
layer7_width = conv6a_features + conv6b_features + conv6c_features + conv6d_features

conv7a_weight = tf.get_variable("Conv_7A", shape=[10, layer7_width, conv7a_features],initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))
conv7b_weight = tf.get_variable("Conv_7B", shape=[8, layer7_width, conv7b_features],initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))
conv7c_weight = tf.get_variable("Conv_7C", shape=[6, layer7_width, conv7c_features],initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))
conv7d_weight = tf.get_variable("Conv_7D", shape=[5, layer7_width, conv7d_features],initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))



#~~~~~~~~~~# FULLY CONNECTED LAYER #~~~~~~~~~~#

print("[INFO] 4 - defining fully connecting inception layer - flatten into single vector\n")
#connect outputs of inception conv layers together to calculate probability output in 1D

#fully connected - [HP: resulting_width, fully_connected_size1, fully_connected_size2]
layer_full_width = conv7a_features + conv7b_features + conv7c_features + conv7d_features
final_conv_size = resulting_width * ( layer_full_width )

full1_weight = tf.get_variable("Full_W1", shape=[final_conv_size, fully_connected_size1],initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))
full2_weight = tf.get_variable("Full_W2", shape=[fully_connected_size1, fully_connected_size2], initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))



#~~~~~~~~~~# OUTPUT LAYER #~~~~~~~~~~#

print("[INFO] 5 - defining output layer - final probabilities\n")
#compute final probabilities and full output for the model

#output - [HP: fully_connected_size2, threshold_division]
output_weight = tf.get_variable("Full_Output", shape=[fully_connected_size2, threshold_division], initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform='True'))
output_bias = tf.Variable(tf.truncated_normal([threshold_division], stddev=0.1, dtype=tf.float32))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~# MODEL TRAINING OUTPUT/PREDICTION/LOSS/TEST FUNCTIONS #~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

print("[INFO] 6 - defining model functions\n")

#defining CNN architecture to return 2 final output tensors (training setps + final output tensors):
  #peakPredictConvModel() -> returns final_threshold_output


#define concat layers for incepton network:
  #concatLayer_A() -> returns concats
  #concatLayer_B() -> returns concats
  #concatLayer_C() -> returns concats
  

#define loss function:
  #aggregatedLoss() -> returns loss_a


#generate output:
  #generateOutout() -> returns result_tensor
  


#~~~~~~~~~~# DEF MODEL FUNCTIONS #~~~~~~~~~~#

#CNN architecture
print("[INFO] 6.1 - defining CNN architecture\n")
def peakPredictConvModel(input_data_depth, input_data_ref, test=False, smoothing=True, normalize=False, eps=0.00001):

    input_data_depth = input_data_depth + init_depth 

    if smoothing:
        depth_tensor_max = tf.nn.pool(input_data_depth, [filter_size_a], strides=[1], padding='SAME', pooling_type='MAX')
        depth_tensor_smooth = tf.nn.conv1d(depth_tensor_max, smoothing_filter, stride=1, padding='SAME')
        input_data_depth = depth_tensor_smooth

    if normalize:
        depth_mean, depth_var = tf.nn.moments(input_data_depth, [1])
        input_data_depth = (input_data_depth -  depth_mean)/tf.sqrt(depth_var + eps)
    
    print("input chipseq data shape: {}".format(input_data_depth.shape))
    print("input ref gene data shape: {}".format(input_data_ref.shape))
    #Stem of read depth data
    conv1 = tf.nn.conv1d(input_data_depth, conv1_weight, stride=1, padding='SAME')
    conv1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train_step, data_format='NHWC', decay=0.9, zero_debias_moving_mean=True)
    conv1 = tf.nn.relu(conv1)
    print("Stem 1 : {}".format(conv1.shape))

    conv2 = tf.nn.conv1d(conv1, conv2_weight, stride=1, padding='SAME')
    conv2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train_step, data_format='NHWC', decay=0.9, zero_debias_moving_mean=True)
    conv2 = tf.nn.relu(conv2)
    max_pool1 = tf.nn.pool(conv2, [max_pool_size_stem], strides=[max_pool_size_stem],
            padding='SAME', pooling_type='MAX')
    print("Stem 2 : {}".format(conv2.shape))

    #Stem of ref gene data
    conv1_ref = tf.nn.conv1d(input_data_ref, conv1_ref_weight, stride=1, padding='SAME')
    conv1_ref = tf.contrib.layers.batch_norm(conv1_ref, is_training=is_train_step, data_format="NHWC", decay=0.9, zero_debias_moving_mean=True)
    conv1_ref = tf.nn.relu(conv1_ref)
    max_pool1_ref = tf.nn.pool(conv1_ref, [max_pool_size_stem], strides=[max_pool_size_stem],
            padding='SAME', pooling_type='MAX')
    print("Stem_Ref 1 : {}".format(max_pool1_ref.shape))
    
    #stem concat
    input_concat = tf.concat([max_pool1, max_pool1_ref],axis = 2) #max_pool1
    print("Stem_concat : {}".format(input_concat.shape))

    # Inception modules 1 to 6
    concat1 = concatLayer_A(input_concat, conv1a_weight, convMax1_weight, conv1b_weight, convAvg1_weight, conv1c_weight, 2)
    concat2 = concatLayer_A(concat1, conv2a_weight, convMax2_weight, conv2b_weight, convAvg2_weight, conv2c_weight, 2)
    concat3 = concatLayer_B(concat2, conv3a_weight, conv3b_weight, conv3c_weight, 3)
    concat3 = concat3 + tf.nn.pool(tf.concat([concat1 for x in range(4)], axis=2), [6], strides=[6], pooling_type='AVG', padding='SAME')
    concat4 = concatLayer_B(concat3, conv4a_weight, conv4b_weight, conv4c_weight, 2)
    concat5 = concatLayer_B(concat4, conv5a_weight, conv5b_weight, conv5c_weight, 2)
    concat5 = concat5 + tf.nn.pool(tf.concat([concat3 for x in range(3)], axis=2), [4], strides=[4], pooling_type='AVG', padding='SAME')
    concat6 = concatLayer_C(concat5, conv6a_weight, conv6b_weight, conv6c_weight, conv6d_weight, 5)
    concat7 = concatLayer_C(concat6, conv7a_weight, conv7b_weight, conv7c_weight, conv7d_weight, 5)
    concat7 = concat7 + tf.nn.pool(tf.concat([concat5 for x in range(2)], axis=2), [25], strides=[25], pooling_type='AVG', padding='SAME')

    concat7 = tf.nn.pool(concat7, [5], strides=[5], padding='SAME', pooling_type='AVG')
    print("Final Avg pool : {}".format(concat7.shape))

    final_conv_shape = concat7.get_shape().as_list()
    final_shape = final_conv_shape[1] * final_conv_shape[2]
    flat_output = tf.reshape(concat7, [final_conv_shape[0] , final_shape])

    fully_connected1 = tf.matmul(flat_output, full1_weight)
    fully_connected1 = tf.contrib.layers.batch_norm(fully_connected1,is_training=is_train_step, decay=0.9, zero_debias_moving_mean=True)
    fully_connected1 = tf.nn.relu(fully_connected1)
    print("Fully connected A :{}".format(fully_connected1.shape))

    fully_connected2 = tf.matmul(fully_connected1, full2_weight)
    fully_connected2 = tf.contrib.layers.batch_norm(fully_connected2,is_training=is_train_step, decay=0.9, zero_debias_moving_mean=True)
    fully_connected2 = tf.nn.relu(fully_connected2)
    print("Fully connected B :{}".format(fully_connected2.shape))

    final_threshold_output = tf.matmul(fully_connected2, output_weight)
    final_threshold_output = tf.add(final_threshold_output, output_bias)
    print("Threshold Output :{}".format(final_threshold_output.shape))

    return final_threshold_output


#concat layers
print("[INFO] 6.2 - defining concat functions for conv inception layers\n")
def concatLayer_A(source_layer, conv1_w, conv_max_w, conv2_w, conv_avg_w, conv3_w, pooling_size):

    conv1 = tf.nn.conv1d(source_layer, conv1_w, stride=2, padding='SAME')
    conv1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train_step, data_format='NHWC', decay=0.9, zero_debias_moving_mean=True)
    conv1 = tf.nn.relu(conv1)

    conv2 = tf.nn.conv1d(source_layer, conv2_w, stride=2, padding='SAME')
    conv2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train_step, data_format='NHWC', decay=0.9, zero_debias_moving_mean=True)
    conv2 = tf.nn.relu(conv2)

    conv3 = tf.nn.conv1d(source_layer, conv3_w, stride=2, padding='SAME')
    conv3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train_step, data_format='NHWC', decay=0.9, zero_debias_moving_mean=True)
    conv3 = tf.nn.relu(conv3)

    max_pool = tf.nn.pool(source_layer, [pooling_size], strides=[pooling_size,],
            padding='SAME', pooling_type='MAX')

    avg_pool = tf.nn.pool(source_layer, [pooling_size], strides=[pooling_size,],
            padding='SAME', pooling_type='AVG')

    concat = tf.concat([conv1, avg_pool, conv2, max_pool, conv3], axis=2)
    print("Concat Type A :{}".format(concat.shape))
    return concat

def concatLayer_B(source_layer, conv1_w, conv2_w, conv3_w, pooling_size):

    conv1 = tf.nn.conv1d(source_layer, conv1_w, stride=pooling_size, padding='SAME')
    conv1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train_step, data_format='NHWC', decay=0.9, zero_debias_moving_mean=True)
    conv1 = tf.nn.relu(conv1)

    conv2 = tf.nn.conv1d(source_layer, conv2_w, stride=pooling_size, padding='SAME')
    conv2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train_step, data_format='NHWC', decay=0.9, zero_debias_moving_mean=True)
    conv2 = tf.nn.relu(conv2)

    conv3 = tf.nn.conv1d(source_layer, conv3_w, stride=pooling_size, padding='SAME')
    conv3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train_step, data_format='NHWC', decay=0.9, zero_debias_moving_mean=True)
    conv3 = tf.nn.relu(conv3)

    max_pool = tf.nn.pool(source_layer, [pooling_size], strides=[pooling_size],
            padding='SAME', pooling_type='MAX')

    concat = tf.concat([conv1, conv2, conv3,  max_pool], axis=2)
    print("Concat Type B :{}".format(concat.shape))
    return concat

def concatLayer_C(source_layer, conv1_w, conv2_w, conv3_w, conv4_w, pooling_size):

    conv1 = tf.nn.conv1d(source_layer, conv1_w, stride=pooling_size, padding='SAME')
    conv1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train_step, data_format='NHWC', decay=0.9, zero_debias_moving_mean=True)
    conv1 = tf.nn.relu(conv1)

    conv2 = tf.nn.conv1d(source_layer, conv2_w, stride=pooling_size, padding='SAME')
    conv2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train_step, data_format='NHWC', decay=0.9, zero_debias_moving_mean=True)
    conv2 = tf.nn.relu(conv2)

    conv3 = tf.nn.conv1d(source_layer, conv3_w, stride=pooling_size, padding='SAME')
    conv3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train_step, data_format='NHWC', decay=0.9, zero_debias_moving_mean=True)
    conv3 = tf.nn.relu(conv3)

    conv4 = tf.nn.conv1d(source_layer, conv4_w, stride=pooling_size, padding='SAME')
    conv4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train_step, data_format='NHWC', decay=0.9, zero_debias_moving_mean=True)
    conv4 = tf.nn.relu(conv4)

    concat = tf.concat([conv1, conv2, conv3, conv4], axis=2)
    print("Concat Type C :{}".format(concat.shape))
    return concat



#loss function
print("[INFO] 6.3 - defining final loss function\n")
def aggregatedLoss(label_data_train, prediction_before_sigmoid):

    loss_a = tf.reduce_mean(tf.nn.top_k(tf.nn.weighted_cross_entropy_with_logits(targets=label_data_train, logits=prediction_before_sigmoid,
        pos_weight=tf.maximum(1.,(loss_weight))), k = topK_set_a).values)
    tf.summary.scalar("Top_{}_Loss".format(topK_set_a),loss_a)

    loss_b = tf.reduce_mean(tf.nn.top_k(tf.nn.weighted_cross_entropy_with_logits(targets=label_data_train, logits=prediction_before_sigmoid,
        pos_weight=tf.maximum(1.,(loss_weight))), k = 750).values)
    tf.summary.scalar("Top_{}_Loss".format(topK_set_b),loss_b)

    return loss_a


#generate output
print("[INFO] 6.4 - defining generate output function\n")
def generateOutput(threshold_tensor, depth_tensor, div=10, input_size=12000, batch_size_in=batch_size, smoothing=True, normalize=False, t=1, eps=0.00001):

    depth_tensor = depth_tensor + init_depth 
    
    if smoothing:
        depth_tensor_max = tf.nn.pool(depth_tensor, [filter_size_a], strides=[1], padding='SAME', pooling_type='MAX')
        depth_tensor_smooth = tf.nn.conv1d(depth_tensor_max, smoothing_filter, stride=1, padding='SAME')
        depth_tensor = depth_tensor_smooth

    if normalize:
        depth_mean, depth_var = tf.nn.moments(depth_tensor, [1])
        depth_tensor = (depth_tensor -  depth_mean)/tf.sqrt(depth_var + eps)
        #depth_max = tf.reduce_max(depth_tensor)
        #depth_tensor = ((depth_tensor - depth_mean)/(depth_max + 0.00001))*100
        
    depth_tensor = tf.reshape(depth_tensor,[batch_size_in ,div, input_size//div])
    threshold_tensor = tf.reshape(threshold_tensor,[batch_size_in,div,1])
    print("depth tensor :{}".format(depth_tensor.shape))
    print("threshold tensor :{}".format(threshold_tensor.shape))

    result_tensor = tf.subtract(depth_tensor, threshold_tensor, name="results")
    print("result tensor before:{}".format(result_tensor.shape))
    result_tensor = tf.reshape(result_tensor,[batch_size_in, 1, input_size])
    print("result tensor after:{}\n".format(result_tensor.shape))

    if not (t == 1):
        result_tensor = result_tensor * t 

    return result_tensor

'''
#parameter testing
total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    trainable_variable_name = variable.name
    shape = variable.get_shape()
    #print(len(shape))
    variable_parameters = 1
    for dim in shape:
        #print("dim", dim)
        variable_parameters *= dim.value
    #print("Num params in layer:", variable_parameters)
    print(trainable_variable_name, shape, variable_parameters)
    total_parameters += variable_parameters
print("\nTotal num params in model:", total_parameters, "\n")
'''

#~~~~~~~~~~# TENSOR GRAPH FOR TRAINING STEPS #~~~~~~~~~~#

print("[INFO] 7 - defining tensor graph for training steps\n")

model_output  = peakPredictConvModel(input_data_train, input_ref_data_train)
prediction_before_sigmoid = generateOutput(model_output, input_data_train, div=threshold_division, smoothing=True)
prediction = tf.nn.sigmoid(prediction_before_sigmoid)
loss = aggregatedLoss(label_data_train, prediction_before_sigmoid)
#loss = topKAverageLoss(label_data_train, prediction_before_sigmoid)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)


with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)





'''
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~# hparam testing #~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import tensorflow as tf


with tf.control_dependencies(update_ops, hparams):
    optimizer = hyperparameters.HP_OPTIMIZER.value
    train_step = optimizer.minimize(loss)
    metrics = ['accuracy']
'''








#~~~~~~~~~~# TENSOR GRAPH FOR TEST STEPS #~~~~~~~~~~#

print("[INFO] 8 - defining tensor graph for test steps\n")

test_model_output = peakPredictConvModel(input_data_eval, input_ref_data_eval)
test_prediction = tf.nn.sigmoid(generateOutput(test_model_output, input_data_eval, div=threshold_division, smoothing=True))


#MODEL NAME = model_output