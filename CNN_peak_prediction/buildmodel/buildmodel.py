#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~# MODEL MODULES #~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# test = validation set

print("\n[INFO] importing buildmodel modules")

# disable the tf AVX CPU warning, doesn't take advantage of AVX/FMA to run faster
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import shutil
import logging
logger = logging.getLogger("ConvLog")
logging.basicConfig(filename="/nobackup/kfwc76/DISS/CNN_Peak_Calling_algo/buildmodelogger.txt", level=logging.DEBUG)
logging.debug('logging active')
import glob
import ntpath

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
logging.getLogger('matplotlib.font_manager').disabled = True
import random
import math

from scipy.ndimage.filters import maximum_filter1d
from scipy.signal import gaussian

#tensorboard in
from tensorflow.summary import FileWriter
FileWriter = tf.summary.FileWriter('./logs/training')
FileWriter_2 = tf.summary.FileWriter('./logs/validation')

from sklearn.metrics import f1_score


#saving multiple files with same name
import time
import string



'''
#HPARAM TESTING

HPARAM_file_writer = tf.summary.FileWriter('./logs/hparam_tuning')
summary = tf.compat.v1.Summary()
HPARAM_file_writer.add_summary(summary)

#Configure hyperparameter tracking
hyperparameters.hparams_config()
'''

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from buildmodel.definemodel import *
from buildmodel.hyperparameters import *


'''
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~# HYPERPARAMETERS #~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#import hyperparams
print("[INFO] importing buildmodel hyperparams\n")

from hyperparameters import learning_rate, threshold_division, eval_every, generations, batch_size, class_threshold, filter_size_a, filter_size_b



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~# RUN DEFINEMODEL SCRIPT -> IMPORT PREVIOUSLY DEFINED ARCHITECTURE + VARIABLES #~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


import definemodel #import variables defined during defining model functions
print("[INFO] importing defined model structure placeholder variables\n")

loss = definemodel.loss                                      #run() / training()
prediction = definemodel.prediction                          #run() / training()
test_prediction = definemodel.test_prediction                #run() / training()
train_step = definemodel.train_step                          #run() / training()
input_ref_data_train = definemodel.input_ref_data_train      #run() / training()
label_data_train = definemodel.label_data_train              #run() / training()
loss_weight = definemodel.loss_weight                        #run() / training()

input_data_train = definemodel.input_data_train              #training()

is_train_step = definemodel.is_train_step                    #training() / visualisePeakResult()
p_dropout = definemodel.p_dropout                            #training() / visualisePeakResult()

input_data_eval = definemodel.input_data_eval                #training() / visualisePeakResult()
input_ref_data_eval = definemodel.input_ref_data_eval        #training() / visualisePeakResult()

label_data_eval = definemodel.label_data_eval                #training() / visualisePeakResult()

model_output = definemodel.model_output
'''


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~# DEF FUNCTIONS FOR RUN() #~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

print("[INFO] defining functions for run()\n")


#0.5 - for #1, from utility.utilities import extractChrClass
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(path)


#1 - extract a chromosome number and a class number from label file names
def extractChrClass(dir):

    chr_list = set()
    #a = glob.glob(dir + "*")
    #print(a)
    for ct_file in glob.glob(dir + "/*.ct"):
        #chr_list.add(ct_file.rsplit('/', 1)[1].split('_')[0])
        chr_list.add(path_leaf(ct_file).split('_')[0])

    data_direction = {}
    for chr in chr_list:
        cls_list = []
        for ct_file in glob.glob(dir + "/" + chr + "_*.ct"):
            #cls_list.append(ct_file.rsplit('/', 1)[1].split('_')[1])
            #b = path_leaf(ct_file).split('_')[1]
            cls_list.append(path_leaf(ct_file).split('_')[1])
        data_direction[chr] = cls_list

    return data_direction


#2 - split training data, if Kfold = 0, just split into 2 parts
def splitTrainingData(data_list, label_list, ref_list, Kfold=15):

    print("\n[INFO] total benchmark labeled data: {} \n".format(len(data_list)))

    size = len(data_list)
    counter = size / Kfold

    test_data = []
    test_label = []
    test_ref = []

    for i in range(Kfold - 1):
        test_data_temp = []
        test_ref_temp = []
        test_label_temp = []
        while True:
            if counter <= 0:
                test_ref.append(test_ref_temp)
                test_data.append(test_data_temp)
                test_label.append(test_label_temp)
                counter = size // Kfold
                break

            pop_index = random.randint(0, len(data_list)-1)
            test_ref_temp.append(ref_list.pop(pop_index))
            test_data_temp.append(data_list.pop(pop_index))
            test_label_temp.append(label_list.pop(pop_index))
            counter -= 1

    test_data.append(data_list)
    test_ref.append(ref_list)
    test_label.append(label_list)

    return test_data, test_label, test_ref


#2.5 - define functions for training()
#pnRate() - return the The ratio of Negative#/ Positive#, used for weights of loff func to adjust between sens and spec
def pnRate(targets, batch_size_in=batch_size):

    positive_count = 0.

    for i in range(batch_size_in):
        for index in range(len(targets[i][0])):
            if targets[i][0][index] > 0.5:
                positive_count += 1

    # For the label only has negative target_size samples.
    if positive_count == 0.:
        return 1

    negative_count = len(targets[0][0])*batch_size_in - positive_count

    ### TODO :: adjust these SHITTY EQUATION.
    return (negative_count / positive_count)


#getStat() -  
def getStat(logits, targets, batch_size_in=batch_size, num_grid=0):

    logits = logits.reshape(batch_size_in, num_grid)
    targets = targets.reshape(batch_size_in, num_grid)

    TP = 0.
    TN = 0.
    FN = 0.
    FP = 0.

    for i in range(batch_size_in):
        for index in range(len(logits[0])):
            if (logits[i][index]) >= class_threshold and targets[i][index] >= class_threshold:
                TP += 1
            elif (logits[i][index]) >= class_threshold and targets[i][index] < class_threshold:
                FP += 1
            elif (logits[i][index]) < class_threshold and targets[i][index] >= class_threshold:
                FN += 1
            elif (logits[i][index]) < class_threshold and targets[i][index] < class_threshold:
                TN += 1
            else:
                pass

    #print(TP,"   ",FP,"   ",FN,"   ",TN) 
    
    if TP + FN == 0:
        sens = 0  # Set sensitivity to 0 when there are no positive instances.
    else:
        sens = TP / (TP + FN)

    if TP + FP == 0:
        prec = 0  # Set precision to 0 when there are no true positive predictions.
    else:
        prec = TP / (TP + FP)

    return {
        'sens': sens,
        'prec': prec,
        'spec': TN / (TN + FP),
        'acc': (TP + TN) / (TP + TN + FN + FP)
    }



#3 - training()
def training(sess, loss, prediction, test_prediction, train_step, train_data_list, train_label_list
        , train_ref_list, test_data_list, test_label_list, test_ref_list, logger, num_grid, step_num):

    train_loss = []
    loss_containor_for_mean = []
    
    train_spec = []
    train_sens = []
    train_acc = []
    train_f1 = []
    spec_containor_for_mean = []
    sens_containor_for_mean = []
    acc_containor_for_mean = []
    f1_containor_for_mean = []
    
    test_spec = []
    test_sens = []
    test_acc = []
    test_f1 = []
    test_spec_containor_for_mean = []
    test_sens_containor_for_mean = []
    test_acc_containor_for_mean = []
    test_f1_containor_for_mean = []
    
    saver = tf.train.Saver(max_to_keep=10)
    

    
    #pnRate_sum = 0
    #for i in range(len(train_data_list)):
    #    rand_y = []
    #    rand_y.append(np.repeat(train_label_list[i][['peak']].values.transpose(),5))
    #    rand_y = np.array(rand_y).reshape(label_data_train.shape)
    #    pnRate_sum += pnRate(rand_y)
    #pnRate_mean = pnRate_sum/len(train_data_list)
    #logger.info("average PN rate : {}".format(pnRate_mean))
    
    # Start of the training process
    for i in range(generations):
        rand_index = np.random.choice(len(train_data_list), size=batch_size)

        rand_x = []
        rand_ref = []
        rand_y = []
        
        for j in range(batch_size):
            rand_x.append(train_data_list[rand_index[j]]['readCount'].values)
            rand_ref.append(train_ref_list[rand_index[j]]['refGeneCount'].values)
            rand_y.append(np.repeat(train_label_list[rand_index[j]][['peak']].values.transpose(),5))

        rand_x = np.array(rand_x).reshape(input_data_train.shape)
        rand_ref = np.array(rand_ref).reshape(input_ref_data_train.shape)
        rand_y = np.array(rand_y).reshape(label_data_train.shape)
        
        
        #p_dropout rate
        train_dict = {input_data_train: rand_x, label_data_train: rand_y, input_ref_data_train: rand_ref, loss_weight:pnRate(rand_y), is_train_step:True, p_dropout:0.6}
                #loss_weight:pnRate_mean, is_train_step:True, p_dropout:0.6}
        
        
        
        '''
        #VISUALISE INPUT DATA PER GEN
        
        print("######### INPUT DATA SHAPES ######### ")
        print("depth shape: {}".format(rand_x.shape))
        print()
        print("refseq shape: {}".format(rand_ref.shape))
        print()
        print("label shape: {}".format(rand_y.shape))
        print()

        if not os.path.exists("graphs"):
            os.makedirs("graphs")
                  
        # Create a figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    
        # Plot x
        axs[0].plot(rand_x[0, :, 0])
        axs[0].set_title('ChIP-seq readCount: (1, 12000, 1)')
    
        # Plot ref
        axs[1].plot(rand_ref[0, :, 0])
        axs[1].set_title('Ref Seq: (1, 12000, 1)')
    
        # Plot y
        axs[2].plot(rand_y[0, 0, :])
        axs[2].set_title('Peak Label: (1, 1, 12000)')
    
        # Adjust spacing between subplots
        plt.tight_layout()
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        input_graphs_time = f"graphs/inputdata_graph_gen_{timestamp}.png"

        # Save the graph with an epoch-specific filename
        plt.savefig(input_graphs_time)
        
        # Close the current figure to avoid memory leakage
        plt.close(fig)
        
        ########################################
        '''
        
        
        
        sess.run([train_step], feed_dict=train_dict)

        temp_train_loss, temp_train_preds = sess.run([loss, prediction], feed_dict=train_dict)
        temp_train_stat = getStat(temp_train_preds, rand_y, num_grid=num_grid)

        loss_containor_for_mean.append(temp_train_loss)
        
        spec_containor_for_mean.append(temp_train_stat['spec'])      #train spec
        if temp_train_stat['sens'] != -1:
            sens_containor_for_mean.append(temp_train_stat['sens'])  #train sens
        acc_containor_for_mean.append(temp_train_stat['acc'])        #train acc
        #DIVISION BY 0 ERRORS
        if (temp_train_stat['prec'] + temp_train_stat['sens']) == 0:
            train_f1_result = 0
        else:
            train_f1_result = 2 * ((temp_train_stat['prec'] * temp_train_stat['sens']) / (temp_train_stat['prec'] + temp_train_stat['sens']))
        f1_containor_for_mean.append(train_f1_result)    
               
        
        
        # Recording results of test data
        eval_index = np.random.choice(len(test_data_list), size=batch_size)

        eval_x = []
        eval_ref = []
        eval_y = []


        for j in range(batch_size):
            eval_x.append(test_data_list[eval_index[j]]['readCount'].values)
            eval_ref.append(test_ref_list[eval_index[j]]['refGeneCount'].values)
            eval_y.append(np.repeat(test_label_list[eval_index[j]][['peak']].values.transpose(),5))

        eval_x = np.array(eval_x).reshape(input_data_eval.shape)
        eval_ref = np.array(eval_ref).reshape(input_ref_data_eval.shape)
        eval_y = np.array(eval_y).reshape(label_data_eval.shape)

        test_dict = {input_data_eval: eval_x, label_data_eval: eval_y, input_ref_data_eval: eval_ref
                , is_train_step:False}
                
        
        test_preds = sess.run(test_prediction, feed_dict=test_dict)
        temp_test_stat = getStat(test_preds, eval_y, num_grid=num_grid)
        
        test_spec_containor_for_mean.append(temp_test_stat['spec'])      #test spec
        if temp_test_stat['sens'] != -1:
            test_sens_containor_for_mean.append(temp_test_stat['sens'])  #test sens
        test_acc_containor_for_mean.append(temp_test_stat['acc'])        #test acc
        #DIVISION BY 0 ERRORS
        if (temp_test_stat['prec'] + temp_test_stat['sens']) == 0:
            test_f1_result = 0
        else:
            test_f1_result = 2 * ((temp_test_stat['prec'] * temp_test_stat['sens']) / (temp_test_stat['prec'] + temp_test_stat['sens']))
        test_f1_containor_for_mean.append(test_f1_result)                        #test f1
       
        
        if (i + 1) % eval_every == 0:
        
            loss_mean = sum(loss_containor_for_mean)/float(len(loss_containor_for_mean))
                 
            if len(sens_containor_for_mean) == 0:
                sens_mean = -1.
            else:
                sens_mean = sum(sens_containor_for_mean)/float(len(sens_containor_for_mean))
            spec_mean = sum(spec_containor_for_mean)/float(len(spec_containor_for_mean))
            acc_mean = sum(acc_containor_for_mean)/float(len(acc_containor_for_mean))            
            f1_mean = sum(f1_containor_for_mean)/float(len(f1_containor_for_mean))
        
            if len(test_sens_containor_for_mean) == 0:
                test_sens_mean = -1.
            else:
                test_sens_mean = sum(test_sens_containor_for_mean)/float(len(test_sens_containor_for_mean))
            test_spec_mean = sum(test_spec_containor_for_mean)/float(len(test_spec_containor_for_mean)) 
            test_acc_mean = sum(test_acc_containor_for_mean)/float(len(test_acc_containor_for_mean))
            test_f1_mean = sum(test_f1_containor_for_mean)/float(len(test_f1_containor_for_mean))
                        
            
            ######TRAIN SET STATS###### means for prints

            train_loss.append(loss_mean)
            train_spec.append(spec_mean)
            train_sens.append(sens_mean)
            train_acc.append(acc_mean)
            train_f1.append(f1_mean)
            
            ######VALIDATION SET STATS###### means for prints
            
            test_spec.append(test_spec_mean)
            test_sens.append(test_sens_mean)
            test_acc.append(test_acc_mean)
            test_f1.append(test_f1_mean)
            
            #tensorboard tensors/stats
            summary = tf.Summary()
            
            
            #TRAINING SET GRAPHS
            
            #per gen
            #summary.value.add(tag='.Validation Set: Spec (stats)', simple_value=temp_train_stat['spec'])
            #FileWriter.add_summary(summary, global_step=i+1)
            #summary.value.add(tag='.Validation Set: Sens (stats)', simple_value=temp_train_stat['sens'])
            #FileWriter.add_summary(summary, global_step=i+1)
            #summary.value.add(tag='.Training Set: Accuracy (stats)', simple_value=temp_train_stat['acc'])
            #FileWriter.add_summary(summary, global_step=i+1)

            #mean
            summary.value.add(tag='!Training Set: Mean loss (mean)', simple_value=loss_mean)
            
            summary.value.add(tag='Spec (mean)', simple_value=spec_mean)
            summary.value.add(tag='Sens (mean)', simple_value=sens_mean)
            summary.value.add(tag='Accuracy (mean)', simple_value=acc_mean)    
            summary.value.add(tag='F1 Score (mean)', simple_value=f1_mean)
            
            FileWriter.add_summary(summary, global_step=i+1)
            #flush values every generation
            FileWriter.flush()


            #VALIDATION SET GRAPHS
             
            #per gen     
            #summary.value.add(tag='Validation Set: Spec', simple_value=temp_test_stat['spec'])
            #FileWriter.add_summary(summary, global_step=i+1)
            #summary.value.add(tag='Validation Set: Sens', simple_value=temp_test_stat['sens'])
            #FileWriter.add_summary(summary, global_step=i+1)
            #summary.value.add(tag='Validation Set: Accuracy', simple_value=temp_test_stat['acc'])
            #FileWriter.add_summary(summary, global_step=i+1)
            
            #mean
            summary.value.add(tag='Spec (mean)', simple_value=test_spec_mean)
            summary.value.add(tag='Sens (mean)', simple_value=test_sens_mean)
            summary.value.add(tag='Accuracy (mean)', simple_value=test_acc_mean)
            summary.value.add(tag='F1 Score (mean)', simple_value= test_f1_mean)
            
            FileWriter_2.add_summary(summary, global_step=i+1)
            #flush values every generation
            FileWriter_2.flush()


            #clear values every generation
            loss_containor_for_mean.clear()
            
            spec_containor_for_mean.clear()
            sens_containor_for_mean.clear()
            acc_containor_for_mean.clear()
            f1_containor_for_mean.clear()
            
            test_spec_containor_for_mean.clear()
            test_sens_containor_for_mean.clear()  
            test_acc_containor_for_mean.clear()
            test_f1_containor_for_mean.clear()

            
            #save training process to logger/console
            #logger.info('Generation # {}. Loss: {:.2f}. Test: SENS:{:.2f} SPEC:{:.2f}| Train: SENS:{:.2f} SPEC:{:.2f}\n'.format(i+1, loss_mean, test_sens_mean, test_spec_mean, sens_mean, spec_mean))
            print()
            
            print('GEN: [{}]  |   mean loss: [{:.2f}] \nTRAIN [means]:      sens: {:.2f} | spec: {:.2f} | acc: {:.2f} | f1: {:.2f} \nVALIDATION [means]: sens: {:.2f} | spec: {:.2f} | acc: {:.2f} | f1: {:.2f} \n'.format(i+1, loss_mean, sens_mean, spec_mean, acc_mean, f1_mean, test_sens_mean, test_spec_mean, test_acc_mean, test_f1_mean))
            
        if i % 100 == 0 and i != 1:
            save_path = saver.save(sess, os.getcwd() + "/models/model{}.ckpt".format(step_num,step_num), global_step=i)
            
        
    print("\n \n ################### PRODUCING TRAINING GRAPHICS + STATS ###################\n ")
    visualizeTrainingProcess(eval_every, generations, test_sens, test_spec, test_acc, test_f1, train_sens, train_spec, train_loss, train_acc, train_f1, K_fold=str(step_num))

            
    #print("\n \n ################### PRODUCING PEAK PREDICTION RESULT PLOTS ###################\n ")
    #visualizePeakResult(batch_size, input_data_eval, num_grid, label_data_eval, sess, test_data_list
    #        , test_label_list, test_ref_list, test_prediction, k=len(test_data_list), K_fold=str(step_num))

    logger.info("Saving CNNPeaks")
    save_path = saver.save(sess, os.getcwd() + "/models/model{}.ckpt".format(step_num,step_num))
    logger.info("Model saved in path : %s" % save_path)
    print("[INFO] model saved in path : %s" % save_path)
    
    
    #tensorboard out
    FileWriter.add_graph(sess.graph)
    
    print("F1 SCORE RESULTS DIF FOR TENSORBOARD AND PNG OUTPUTS (DIF EQUATIONS)")
    
    #BIG BOY MODEL RESULTS
    print("RESULTS: \n")
    
    print("MAX TEST F1 SCORE", max(test_f1))
    print("MEAN TEST F1 SCORE", sum(test_f1) / len(test_f1), "\n")
    
    print("MAX ACCURACY", max(test_acc))
    print("MEAN ACCURACY SCORE", sum(test_acc) / len(test_acc), "\n")




#4 - visuslisation of the training process
#creation of mpl figures (loss function + accuracy)

def visualizeTrainingProcess(eval_every, generations, test_sens, test_spec, test_acc, test_f1, train_sens, train_spec, train_loss, train_acc, train_f1, K_fold =""):

    eval_indices = range(0, generations, eval_every)

    for i in range(len(train_sens) - 1):
        if train_sens[i+1] == -1:
            train_sens[i+1] = train_sens[i]
        if test_sens[i+1] == -1:
            test_sens[i+1] = test_sens[i]


    #for comparing different runs without overriding pre-existing graphs
    # Generate a timestamp (you can customize the format if needed)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Combine the timestamp and random string to create a unique filename
    #for F1 scores 
    f1_unique_filename = 'models/model_{}/!F1-score-per-generation_{}.png'.format(K_fold, timestamp)
    #paper_f1
    f1_paper_unique_filename = 'models/model_{}/!PAPER_F1-score-per-generation_{}.png'.format(K_fold, timestamp)
    #accuracy
    acc_unique_filename = 'models/model_{}/!accuracy-score-per-generation_{}.png'.format(K_fold, timestamp)


    plt.plot(eval_indices, train_loss, 'k-')
    plt.title('Cross entropy Loss per generation')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.savefig('models/model_{}/!loss-per-generation.png'.format(K_fold), dpi=400)
    #plt.show()
    plt.clf()
    print("\n[INFO] loss/generation.png")

    plt.plot(eval_indices, test_sens, label='Test Set sensitivity')
    plt.plot(eval_indices, train_sens, label='Train Set sensitivity')
    plt.title('Train and Test Sensitivity')
    plt.xlabel('Generation')
    plt.ylabel('Sensitivity')
    plt.legend(loc='lower right')
    plt.ylim([0,1])
    plt.savefig('models/model_{}/!sensitivity-per-generation.png'.format(K_fold))
    #plt.show()
    plt.clf()
    print("\n[INFO] sensitivity/generation.png")

    plt.plot(eval_indices, test_spec, label='Test Set specificity')
    plt.plot(eval_indices, train_spec, label='Train Set specificity')
    plt.title('Train and Test specificity')
    plt.xlabel('Generation')
    plt.ylabel('Specificity')
    plt.legend(loc='lower right')
    plt.ylim([0,1])
    plt.savefig('models/model_{}/!specificity-per-generation.png'.format(K_fold))
    #plt.show()
    plt.clf()
    print("\n[INFO] specificity/generation.png")

    plt.plot(eval_indices, train_acc, label='Test Set Accuracy Score')
    plt.plot(eval_indices, test_acc, label='Train Set Accuracy Score')
    plt.title('Accuracy Score')
    plt.xlabel('Generation')
    plt.ylabel('Accuracy Score')
    plt.legend(loc='lower right')
    #plt.ylim([0,1])
    plt.savefig(acc_unique_filename)
    #plt.show()
    plt.clf()
    print("\n[INFO]", acc_unique_filename)  

    plt.plot(eval_indices, train_f1, label='Test Set F1 Score')
    plt.plot(eval_indices, test_f1, label='Train Set F1 Score')
    plt.title('F1 Score')
    plt.xlabel('Generation')
    plt.ylabel('F1 Score')
    plt.legend(loc='lower right')
    plt.ylim([0,1])
    #plt.savefig('models/model_{}/!F1-score-per-f1_unique_filename.png'.format(K_fold), dpi=400)
    plt.savefig(f1_unique_filename, dpi=400)
    #plt.show()
    plt.clf()
    print("\n[INFO", f1_unique_filename, "\n") 
    
    plt.plot(eval_indices, [2*(test_sens[i]*test_spec[i])/(test_sens[i]+test_spec[i]) for i in range(len(test_sens))], label='Test Set F1 Score')
    plt.plot(eval_indices, [2*(train_sens[i]*train_spec[i])/(train_sens[i]+train_spec[i]) for i in range(len(train_sens))], label='Train Set F1 Score')
    plt.title('F1 Score')
    plt.xlabel('Generation')
    plt.ylabel('F1 Score')
    plt.legend(loc='lower right')
    plt.ylim([0,1])
    plt.savefig(f1_paper_unique_filename, dpi=400)
    #plt.show()
    plt.clf()
    
     


################## D E A C T I V A T E D ##############################



#4.5 - define functions for visualizePeakResult() and patternVis()
#For output of model, probabilities of a final vector will be changed s binary values by checking whether elements of vector are higher or lower than class_threshold that defined in hyperparameters
def classValueFilter(output_value):

    class_value_list = []

    for index in range(output_value.shape[2]):
        if output_value[0][0][index] >= class_threshold:
            class_value_list.append(1)
        elif output_value[0][0][index] < class_threshold:
            class_value_list.append(0)

    return class_value_list


#5 - visualision of peak results
def visualizePeakResult(batch_size, input_data_eval, num_grid, label_data_eval, sess, test_data_list, test_label_list,
                        test_ref_list, test_prediction, k = 1, K_fold="", min_peak_size=1, max_peak_num=50):

    if k > 0:
        for i in range(k):
            show_x = test_data_list[i]['readCount'].values
            show_x = show_x.reshape(input_data_eval.shape)

            show_ref = test_ref_list[i]['refGeneCount'].values
            show_ref = show_ref.reshape(input_data_eval.shape)

            show_y = test_label_list[i][['peak']].values.transpose()
            show_y = np.repeat(show_y, 5)
            show_y = show_y.reshape(label_data_train.shape)
            show_dict = {input_data_eval: show_x, input_ref_data_eval: show_ref, label_data_eval: show_y,
                         is_train_step: False, p_dropout:1}
            show_preds = sess.run(test_prediction, feed_dict=show_dict)

            show_x = show_x.reshape(num_grid)
            smoothing_filter = gaussian(filter_size_b, 50) / np.sum(gaussian(filter_size_b, 50))
            show_x = maximum_filter1d(show_x, filter_size_a)  ## MAX POOL to extract boarder lines
            show_x = np.convolve(show_x, smoothing_filter, mode='same')  ## Smoothing boarder lines

            show_ref = show_ref.reshape(num_grid)
            
            plt.subplot(2,1,2, autoscale_on=True,position=[0,2,1,0.2])
            plt.imshow(show_preds.reshape(num_grid)[np.newaxis,:], cmap="jet", vmax=1, vmin=0, aspect="auto")
            
            #classvaluefilter()
            show_preds = classValueFilter(show_preds)
            show_y = classValueFilter(show_y)

            ############# Peak post processing ##########
            peak_num = 0

            if show_preds[0] > 0:
                peak_size = 1
                peak_num += 1
            else:
                peak_size = 0

            for pred_index in range(len(show_preds)-1):
                if show_preds[pred_index+1] > 0:
                    peak_size += 1
                else:
                    if peak_size < min_peak_size:
                        for j in range(peak_size):
                            show_preds[pred_index-j] = 0
                        peak_size=0
                    else:
                        peak_num += 1
                        peak_size = 0

            if peak_num > max_peak_num:
                for j in range(len(show_preds)):
                    show_preds[j] = 0
            #############################################

            y_index = []
            y = []
            for index in range(len(show_preds)):
                if show_preds[index] > 0:
                    y_index.append(index)
                    y.append(show_y[index])

            ref_index = []
            ref = []
            for index in range(len(show_ref)):
                if show_ref[index] > 0:
                    ref_index.append(index)
                    ref.append(-1)
            
            plt.subplot(2,1,1,position=[0,2.2,1,1])
            plt.plot(show_x.reshape(num_grid).tolist(),'k', markersize=2, linewidth=1)
            plt.plot(y_index,y, 'r.', label='Model prediction')
            plt.plot(ref_index, ref, 'b|', markersize=8)

            onPositive = False
            start = 0
            end = 0
            for j in range(len(show_y)):
                if show_y[j] == 1 and not onPositive:
                    start = j
                    onPositive = True
                elif (show_y[j] == 0 or j == len(show_y)-1) and onPositive:
                    end = j
                    onPositive = False
                    plt.axvspan(start, end, color='red', alpha=0.3)


            plt.title('Peak prediction result by regions')
            plt.xlabel('Regions')
            plt.ylabel('Read Count')
            plt.legend(loc='upper right')
            plt.savefig('models/model_{}/peak{}.png'.format(K_fold,i), dpi=400)
            #plt.show()
            plt.clf()
    print("[INFO] saved peak figures: models/model_{}/xxx.png \n".format(K_fold))


#6 patternVis() ?


################################################





#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~# RUN() - BUILD CNN MODEL #~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


print("[INFO] finalising run() function\n")

def run(dir_name, logger, num_grid=12000, K_fold_in=10, cross_valid = False, label_fix_num=None):
    
    print("\n \n ################### BUILDING + TRAINING MODEL ###################\n \n")
    
    PATH = os.path.abspath(dir_name)

    dir_list = os.listdir(PATH)

    for dir in dir_list:
        True

    #1 - extractChrClass()
    input_list = {}
    for dir in dir_list:
        dir = PATH + '/' + dir
        input_list[dir] = extractChrClass(dir)
    
    print("|--------| BUILD PARAMS |--------| \n")
    print(" -> number of generations: [{}]".format(generations), "\n")
    print(" -> evaluation every: [{}]".format(eval_every), "\n")
    print(" -> learning rate: [{}]".format(learning_rate), "\n")
    print(" -> threshold division: [{}]".format(threshold_division), "\n")
    print(" -> smoothing filter size A: [{}]".format(filter_size_a), "\n")
    print(" -> smoothing filter size B: [{}]\n".format(filter_size_b))
    print("|-------------------------------| \n")

    #Define Session and Initialize Graph
    init_glob = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    sess = tf.Session()
    sess.run(init_glob)
    sess.run(init_local)

    #Data Load and Split for Cross validation 
    input_data_names = []
    ref_data_names = []
    label_data_names = []

    for dir in input_list:
        label_num = 0
        for chr in input_list[dir]:
            for cls in input_list[dir][chr]:
                label_num += 1
                input_file_name = "{}/{}_{}_grid{}.ct".format(dir, chr, cls, num_grid)
                ref_file_name = "{}/ref_{}_{}_grid{}.ref".format(dir, chr, cls, num_grid)
                label_file_name = "{}/label_{}_{}_grid{}.lb".format(dir, chr, cls, num_grid)
                input_data_names.append(pd.read_csv(input_file_name))
                ref_data_names.append(pd.read_csv(ref_file_name))
                label_data_names.append(pd.read_csv(label_file_name))
        logger.info("DIRECTORY (TARGET) [{}]# of labels : <{}>".format(label_num,dir))
        dir_target_file = os.path.basename(dir)
        print("[INFO] target directory: <{}> | num labels present = [{}]".format(dir_target_file,label_num))

    if label_fix_num != None:
        step = 0
        fin = len(input_data_names)
        while fin-step > label_fix_num:
            rand_idx = random.randint(0, fin - step - 1)
            input_data_names.pop(rand_idx)
            ref_data_names.pop(rand_idx)
            label_data_names.pop(rand_idx)
            step += 1
            
    #print(label_data_names) #ALL ARE CLASSIFIED AS NOPEAK
    
    K_fold = K_fold_in
    
    #2 - splitTrainingData()
    input_data_list, label_data_list, ref_data_list = splitTrainingData(input_data_names, label_data_names
            , ref_data_names, Kfold=K_fold)
    if not os.path.isdir(os.getcwd() + "/models"):      ### Make directory for save the model
        os.mkdir(os.getcwd() + "/models")

    #Training start with cross validation
    for i in range(K_fold):

        training_data = []
        training_ref = []
        training_label = []
        test_data = []
        test_ref = []
        test_label = []
        for j in range(K_fold):
            if i == j:
                test_data += input_data_list[j]
                test_ref += ref_data_list[j]
                test_label += label_data_list[j]
            else:
                training_data += input_data_list[j]
                training_ref += ref_data_list[j]
                training_label += label_data_list[j]

        if not os.path.isdir(os.getcwd() + "/models/model_{}".format(i)):
            os.mkdir(os.getcwd() + "/models/model_{}".format(i))

        logger.info("[{}]# of labels for training, [{}]# of labels for testing.".format(len(training_data), len(test_data)))
        print("TRAINING label num = [{}] \nTESTING label num = [{}]".format(len(training_data), len(test_data)), "\n")
        
        #print("   TP   |   FP   |   FN   |   TN")
        
        #modules imported from definemodel
        
        
        #RUN AND TRAIN THE MODEL
        
        
        
        
        
        #3 - training()
        training(sess, loss, prediction, test_prediction, train_step, training_data, training_label
                , training_ref, test_data, test_label, test_ref, logger, num_grid, i)
                



                
        
        if cross_valid == False:
            break

    

    
    

    print("\n \n ################### DONE ###################\n \n")
    
    
    print("[INFO] for model summary: \n")
    print("tensorboard dev upload --logdir ./logs \\")
    print("--name \" OPTIONAL \" \\")
    print("--description \" OPTIONAL \" ")
    print("\n")
    
    print("[INFO] before next run: \n")
    print("rm -rf /nobackup/kfwc76/DISS/CNN_Peak_Calling_algo/buildmodel/logs/validation/ \\")
    print("rm -rf /nobackup/kfwc76/DISS/CNN_Peak_Calling_algo/buildmodel/logs/training/")
    print("\n")
    
    

#~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~# RUN() #~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~#
print("[INFO] initialising buildmodel run() \n")

run('/nobackup/kfwc76/DISS/CNN_Peak_Calling_algo/buildmodel/trainingdata', 
    logger, 
    num_grid=12000, 
    K_fold_in=5, 
    cross_valid = False, 
    label_fix_num=None
    )


