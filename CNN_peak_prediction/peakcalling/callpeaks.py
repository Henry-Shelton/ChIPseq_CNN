#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~# MODEL MODULES #~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
print("\n[INFO] importing callpeaks modules \n")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logger = logging.getLogger("ConvLog")
logging.basicConfig(filename="/nobackup/kfwc76/DISS/CNN_Peak_Calling_algo/callpeakslogger.txt", level=logging.DEBUG)
logging.debug('logging active')

import subprocess as sp
import progressbar as pgb
import multiprocessing
from multiprocessing import cpu_count, Process, Manager

import pysam
import random
import string
import time
import math
import pyximport #for importing cython functions

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from buildmodel import buildmodel
from buildmodel.definemodel import *
from buildmodel.hyperparameters import *


'''
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~# RUN PREPROC/HYPERPARAM/DEFINEMODEL/BUILDMODEL SCRIPTS #~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

print("[INFO] importing model hyperparams\n")
print("[INFO] importing model placeholder variables\n")

import definemodel

#hyperparams
batch_size = definemodel.batch_size
threshold_division = definemodel.threshold_division               
learning_rate = definemodel.threshold_division               
filter_size_a = definemodel.filter_size_a                   
filter_size_b = definemodel.filter_size_b                   
generations = definemodel.generations                      
eval_every = definemodel.eval_every                       
class_threshold = definemodel.class_threshold                 

#learned model variables
test_model_output = definemodel.test_model_output
test_prediction = definemodel.test_prediction
is_train_step = definemodel.test_prediction
input_data_eval = definemodel.test_prediction
input_ref_data_eval = definemodel.test_prediction
'''


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~# DEF FUNCTIONS FOR RUN() #~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

print("IF FAIL: LD_LIBRARY_PATH=/nobackup/kfwc76/DISS/CNN_Peak_Calling_algo/peakcalling/bamdepth/htslib_1_9/ \n")
print("[INFO] importing cython functions\n")

#1 generateReadcounts() -
from readCounts import generateReadcounts


#2 generateRefcounts() - 
from readCounts import generateRefcounts


#3 predictionToBedString() - 
from bedGen import predictionToBedString


#3.1 writeBed() -

def writeBed(output_file, peaks, logger, printout=False):
    if not os.path.isfile(output_file):
        bed_file = open(output_file, 'w')
    else:
        bed_file = open(output_file, 'a')

    for peak in peaks:
        bed_file.write(peak)
        if printout == True:
            logger.info("{}".format(peak[:-1]))
            print("{}".format(peak[:-1]))
            
            
#4 callpeak - 
def call_peak(chr_no, chr_table, chr_lengths, file_name, ref_data_df, input_data, input_data_ref,
        logger, num_grid, prediction, sess, window_size, 
        pgb_on=False, 
        window_start=1, 
        window_end=None, 
        window_chunk=100, 
        bed_name=None
        ):

    window_count = window_start
    if window_end == None:
        window_end = chr_lengths[chr_no]

    stride = window_size / num_grid
    eval_counter = 0
    if bed_name == None:
        output_file_name = "{}.bed".format(file_name.rsplit('.')[0])
    else:
        output_file_name = bed_name
    logger.info("Output bed file name : {}".format(output_file_name))
    peaks = []

    logger.info("Length of [{}] is : {}".format(chr_table[chr_no], window_end))

    ref_data = ref_data_df.values
    
    while True:
        ### Make genomic segment for each window ( # of window: window_chunk )
        if (eval_counter % window_chunk) == 0:
            ### ProgressBar
            if pgb_on:
                bar = pgb.ProgressBar(max_value=window_chunk)
            logger.info("Generate Read Counts . . .")
            
            ### If remained genomic regions are enough to create the large window chunk.
            if window_count + window_size*window_chunk < window_end:
                read_count_chunk = generateReadcounts(window_count,window_count+window_size*window_chunk-1, chr_table[chr_no], file_name, num_grid, window_chunk)
                end = window_count+window_size*window_chunk -1
            
            ### If a size of window chunk is larger than remained genome.
            else:
                window_n = int((window_end - window_count)/ window_size)
                if window_n < 1:
                    logger.info("The remained region is less than window size.")
                    break
                read_count_chunk = generateReadcounts(window_count,
                        window_count+window_size*window_n, chr_table[chr_no], file_name, num_grid, window_n)
                end = window_end
            logger.info("Calling . . . :[{}:{}-{}]".format(chr_table[chr_no],window_count,end))

        ### END OF PEAK CALLING FOR ONE CHROMOSOME :: write remained predicted peaks.
        if window_count + window_size > window_end:
            logger.info("Peak calling for [{}] is done.".format(chr_table[chr_no]))
            writeBed(output_file_name, peaks, logger, printout=False)
            break

        read_count_by_grid = read_count_chunk[eval_counter*num_grid:(eval_counter+1)*num_grid].reshape(input_data_eval.shape)
        ref_data_by_grid = generateRefcounts(window_count, window_count+window_size,
                ref_data, num_grid).reshape(input_ref_data_eval.shape)
        
        result_dict = {input_data_eval: read_count_by_grid, input_ref_data_eval: ref_data_by_grid, is_train_step: False}
        preds = sess.run(prediction, feed_dict=result_dict)
        class_value_prediction = np.array(preds.reshape(num_grid))
        
        peaks += predictionToBedString(class_value_prediction, chr_table[chr_no], window_count, stride,
                num_grid, read_count_by_grid.reshape(num_grid), 10, 50)
        
        eval_counter += 1

        if pgb_on:
            bar.update(eval_counter)

        if eval_counter == window_chunk:
            writeBed(output_file_name, peaks, logger, printout=False)
            eval_counter = 0
            peaks =[]

        window_count += window_size



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~# RUN() - CALL PEAKS #~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def run(input_bam, logger, window_size=100000, num_grid=0, model_name=None, regions=None, genome=None, bed_name=None):

    input_data = tf.placeholder(tf.float32, shape=(batch_size, num_grid, 1), name="testData")
    input_data_ref = tf.placeholder(tf.float32, shape=(batch_size, num_grid, 1), name="TestRefData")

    sess = tf.Session()

    model_output = test_model_output
    prediction = test_prediction

    if model_name == None:
        model_name = "model0"
    saver = tf.train.Saver()
    saver.restore(sess, os.getcwd() + "/models/{}.ckpt".format(model_name))
    
    #from tensorflow.python.tools import inspect_checkpoint as chkp
    #checkpoint_path = os.path.join(os.getcwd(), "trained_models", trained_model_name, "{}.ckpt".format(model_name))
    #chkp.print_tensors_in_checkpoint_file(checkpoint_path, tensor_name='', all_tensors=True)
    
    print("################################# TRAINED MODEL USED #################################\n")
    
    #print("[INFO] model source:", os.getcwd() + "/trained_models/{}/{}.ckpt".format(trained_model_name, model_name))
    #print("[INFO] {}/{}.ckpt will be used during peak calling".format(trained_model_name, model_name), "\n\n")
        
        
        
        
    '''
    #make index file for test data (if not already existant)
    if not os.path.isdir(input_bam[:-4]):
        os.makedirs(input_bam[:-4])

    if not os.path.isfile(input_bam + '.bai'):
        logger.info("Creating index file of [{}]".format(input_bam))
        preProcessing.createBamIndex(input_bam)
        logger.info("[{} was created.]".format(input_bam+".bai"))
    else:
        logger.info("[" + input_bam + "] already has index file.")
    '''
    
    
    input_bam = os.path.abspath(input_bam)

    bam_alignment = pysam.AlignmentFile(input_bam , 'rb', index_filename=input_bam + '.bai')
    chr_lengths = bam_alignment.lengths
    
    chr_table = list(bam_alignment.references)
    
    
    print("################################# PEAK CALLING IN PROGRESS #################################\n")
    
    #NOT THIS ONE USUALLY
    if regions is not None:
        logger.info("Specific calling regions was defined :: {}".format(regions))
        regions = regions.split(':')
        chromosome = regions[0]
        regions = regions[1].split('-')
        call_start = regions[0]
        call_end = regions[1]

        for i in range(len(chr_table)):
            if chr_table[i] == chromosome:
                chr_no = i
                break

        if call_start == 's':
            call_start = 1
        else:
            call_start = int(call_start)

        if call_end == 'e':
            call_end = chr_lengths[chr_no]
        else:
            call_end = int(call_end)

        logger.info("Chromosome<{}> , <{}> to <{}>".format(chromosome, call_start, call_end))
        print("Chromosome<{}> , <{}> to <{}>".format(chromosome, call_start, call_end))

        ref_data_df = pd.read_csv("geneRef/{}.bed".format(chromosome), names=['start','end'] , header=None, usecols=[1,2], sep='\t')
        logger.info("Peak calling in chromosome {}:".format(chromosome))
        call_peak(chr_no, chr_table, chr_lengths, input_bam, ref_data_df, input_data, input_data_ref,
                logger, num_grid, prediction, sess, window_size, pgb_on=True, window_start=call_start, window_end=call_end, bed_name=bed_name)
    
    else:
        
        
        print("input_bam:", input_bam)
        print("input_bam_index:", input_bam + ".bai")                     
        print("input_data:", input_data)             
        print("input_data_ref:", input_data_ref)
        print("num_grid:", num_grid)        
        print("prediction:", prediction)        
        print("sess:", sess)        
        print("window_size:", window_size)        
        print()
                
                
        for chr_no in range(len(chr_table)):
            if os.path.isfile("geneRef/{}.bed".format(chr_table[chr_no])):
                ref_data_df = pd.read_csv("geneRef/{}.bed".format(chr_table[chr_no]), names=['start','end'] , header=None, usecols=[1,2], sep='\t')
            else:
                logger.info("Chromosome {} is invalid.".format(chr_table[chr_no]))
                continue
                #ref_data_df = pd.DataFrame(header=None)
                
            logger.info("Peak calling in chromosome {}:".format(chr_table[chr_no]))
            print("[INFO] peak calling in chromosome {}:".format(chr_table[chr_no]))

            #call_peak()
            call_peak(chr_no, chr_table, chr_lengths, input_bam, ref_data_df, input_data, input_data_ref,
                    logger, num_grid, prediction, sess, window_size, pgb_on=True, bed_name=bed_name)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~# RUN() #~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~#

print("[INFO] initialising callpeaks run() \n")

#PICK TEST DATA FILE.BAM
run("/nobackup/kfwc76/DISS/CNN_Peak_Calling_algo/peakcalling/test_data/AKAP8L.bam",
logger, 
window_size=100000, 
num_grid=12000, 
model_name=None, 
regions=None, 
genome=None, 
bed_name=None)
    
print("move to outputs for model test data evaluation, rename to [TF]_[CELLLINE].bed")
print("################################# PEAK CALLING COMPLETE #################################\n")