print("PREPROCESSING MODULE")
#import modules
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import re
import glob
import random
import time
import logging
logger = logging.getLogger("ConvLog")
logging.basicConfig(filename="/nobackup/kfwc76/DISS/CNN_Peak_Calling_algo/preproclogger.txt", level=logging.DEBUG)
logging.debug('logging active')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import subprocess as sp
import progressbar as pgb

import pysam
import pandas as pd
import numpy as np

from multiprocessing import cpu_count, Process, Manager
from sklearn.cluster import DBSCAN

#This preprocessing step will create alignment read count data from input directory dir_name was specified by a user. 
#The results will be saved in directories which have same name with input bam files.
#e.g) /H3K4me3_1_K562.bam `s results is saved in training_data/H3K4me3_1_K562/chrn_n_gridxxxx.csv

#REQUIRES SAMTOOLS + BAMTOOLS LOADED (BIOINFO MODULE)

###########################################
#defining functions for preprocessing run()
###########################################


#1-bamtools index file creation: bamtools index -in xxx.bam [def createBamIndex(input_bam)...]


#2-check if .bam name and label df same name
def is_same_target(bam_file_name, label_data_df):
    return bam_file_name.rsplit('/',1)[1].split('_')[0] == label_data_df.rsplit('/',1)[1].split('_')[0]


#3-load label file: 1) list of labeled files 2) return label_pd.df
def loadLabel(label_file_name):
    label_col = ['chr', 'start', 'end', 'peakStat', 'cellType']
    label_data_frame = pd.DataFrame(columns=label_col)

    label_file = open(label_file_name, 'r')
    raw_label_data = label_file.readlines()
    print("############### \n")
    print("Appending label_file:", label_file_name, "\n")

    label_data = []

    for peak in raw_label_data:
        if peak == "\r\n":
            raw_label_data.remove(peak)

    for peak in raw_label_data:
        if '#' not in peak and 'chr' in peak:
            peak = peak.rstrip('\r\n')
            label_data.append(peak)

    label_data.sort()

    for i in range(len(label_data)):
        label_data[i] = re.split(':|-| ', label_data[i], 4)


##### FIRST ISSUE #####

        for index in range(len(label_data[i])-1, -1, -1):
            #print(label_data[i][index] == '')
            if label_data[i][index] == '':
                label_data[i].pop(index)
                
        if len(label_data[i]) == 4:
            label_data[i].append('None')

        label_data[i][1] = int(label_data[i][1].replace(',',''))
        label_data[i][2] = int(label_data[i][2].replace(',',''))

    
    additional_data = pd.DataFrame(columns=label_col, data=label_data)
    label_data_frame = label_data_frame.append(additional_data, ignore_index=True)
    label_file.close()
    
    label_data_frame['chr'] = label_data_frame.chr.astype('category')
    label_data_frame['start'] = label_data_frame.start.astype(int)
    label_data_frame['end'] = label_data_frame.end.astype(int)
    label_data_frame['peakStat'] = label_data_frame.peakStat.astype('category')
    
    label_data_frame.to_csv("./label_data_frame.csv")
    
    return label_data_frame


#4-cluster pd.df labed data with DBSCAN (hyperparam) (changes original df)
def clusteringLabels(label_data_df, bp_eps):

    label_data_df['class'] = None
    chr_list = set(label_data_df['chr'].tolist())

    for chr in chr_list:
        chr_df = label_data_df.loc[label_data_df.chr == chr].copy()
        feature_df = chr_df[['start']]

        DBSCAN_model = DBSCAN(eps = bp_eps, min_samples=1)  ## It does not allow noisy element with minsample = 1
        predict = pd.DataFrame(DBSCAN_model.fit_predict(feature_df), columns=['class'])

        label_data_df.loc[label_data_df.chr == chr, 'class'] = predict['class'].tolist()


#5-maketrainfrags - bam alignment files -> slice into frags with label regions -> len of sliced bam files is sum of labeles regions

#5.1-define functions used
#create samtools style region strings "chrN:zzz,zzz,zzz-yyy,yyy,yyy"
def createRegionStr(chr, start, end=None):
    if end == None:
        return str(chr) + ":" + str(int(start)) + "-" + str(int(start))
    elif end is not None:
        return str(chr) + ":" + str(int(start)) + "-" + str(int(end))

#5.2-module load bioinformatics -> module load samtools [REQUIRED]

#5.3-makeRefGeneTAGS(): refgene tag creation, return pd.df with regions from this
def makeRefGeneTags(refGene_df, start, end, stride, num_grid, dataType='pd'):

    refGene_depth_list = []

    start_point = 0

    for step in range(num_grid):
        location = int(start + stride*step)
        index = start_point
        depth = 0
        while True:
            if len(refGene_df) <= index :
                start_point = index
                refGene_depth_list.append(depth)
                break

            if location < refGene_df.iloc[index]['start']:
                start_point = index
                refGene_depth_list.append(depth)
                break
            elif refGene_df.iloc[index]['start'] <= location and location <= refGene_df.iloc[index]['end']:
                depth = 1
            index += 1

    if dataType == 'pd':
        return pd.DataFrame(refGene_depth_list, columns=['refGeneCount'], dtype=int)
    elif dataType == 'list':
        return refGene_depth_list
    else:
        return refGene_depth_list


#5.4-clustrering labels 
def clusteringLabels(label_data_df, bp_eps):
    """
    Clustering pandas label data with DBSCAN.
    It has hyperparameter that define maximum distance between
    cluster elements in the same group.
    :param label_data_df:
    :param bp_eps: size of base points
    :return: None. it will change the original data frame "label_data".
    """

    label_data_df['class'] = None
    chr_list = set(label_data_df['chr'].tolist())

    for chr in chr_list:
        chr_df = label_data_df.loc[label_data_df.chr == chr].copy()
        feature_df = chr_df[['start']]

        DBSCAN_model = DBSCAN(eps = bp_eps, min_samples=1)  ## It does not allow noisy element with minsample = 1
        predict = pd.DataFrame(DBSCAN_model.fit_predict(feature_df), columns=['class'])

        label_data_df.loc[label_data_df.chr == chr, 'class'] = predict['class'].tolist()

#5.5-maketrainfrags()
def makeTrainFrags(bam_file, label_data_df, searching_dist, num_grid, cell_type, logger):

    num_grid_label = num_grid // 5

    chr_list = set(label_data_df['chr'].tolist())

    if not os.path.isdir(bam_file[:-4]):
        os.makedirs(bam_file[:-4])

    if not os.path.isfile(bam_file + '.bai'):
        createBamIndex(bam_file)
        logger.info("Creating index file of [" + bam_file + "]")
    else:
        logger.info("[" + bam_file + "] already has index file.")

    for chr in chr_list:
        label_data_by_chr = label_data_df[label_data_df['chr'] == chr]
        class_list = set(label_data_by_chr['class'].tolist())
        refGenePd = pd.read_table("geneRef/{}.bed".format(chr), names=['chr','start','end'] ,header=None, usecols=[0,1,2])

        for cls in class_list:
            label_data_by_class = label_data_by_chr[label_data_by_chr['class'] == cls]
            region_start = int(label_data_by_class.head(1)['start'])
            region_end = int(label_data_by_class.tail(1)['end'])
            region_size = region_end - region_start

            if region_size > searching_dist * 0.8:
                left_dist = int(region_size/10)
                right_dist = int(region_size/10)
            else:
                left_dist = random.randint(0, searching_dist)  # Additional window is non-deterministic.
                right_dist = searching_dist - left_dist

            region_start -= left_dist
            region_end += right_dist
            region_size = region_end - region_start

            stride = region_size / num_grid             # that can be elimenated.
            stride_label = region_size / num_grid_label

            logger.debug("STRIDE :" + str(stride) + "           REGION SIZE :" + str(region_size))

#.1-createRegionStr()
#.2-load SAMTOOLS
            samtools_call = ['samtools depth -aa -r {} {} > tmp_depth'.format(
                createRegionStr("{}".format(chr), int(region_start), int(region_end)),bam_file)]
            sp.call(samtools_call, shell=True)
            depth_data = pd.read_table('tmp_depth', header=None, usecols=[2], names=['readCount'])
            #print(depth_data['readCount'].values)  #last one is 0 which is causing the error
            logger.debug(depth_data['readCount'].values)


####### SECOND ISSUE ####### REMEMBER TO LOAD BIOINFO AND SAMTOOLS MODULES

            read_count_list = []
            for step in range(num_grid): #0-1999 num_grid param = 12000
                
                readCount_values = depth_data['readCount'].values
                #print(len(readCount_values))
                
                if len(readCount_values) > 0:
                    value = readCount_values[int(step * stride)]
                    # Perform operations with the 'value' variable
                else:
                    # Handle the case when the array is empty
                    print("The readCount_values array is empty.")
                    pass


                read_count_list.append(
                    int(readCount_values[int(step * stride)])
                    )
                
                #step_stride_int = int(step * stride)
                #print(step_stride_int)  #0-50000...
                
                #depth_data_readcount_int = depth_data['readCount']
                #print(depth_data_readcount_int)  #pd.df
                
                #combined_depthdata_stepstride_int = int(depth_data_readcount_int.iloc[step_stride_int])

                #read_count_list.append(combined_depthdata_stepstride_int)



            read_count_by_grid = pd.DataFrame(read_count_list, columns=['readCount'])
            os.remove('tmp_depth')

            output_count_file = bam_file[:-4] + "/" + str(chr) + "_" + str(cls) + "_grid" + str(num_grid)+".ct"
            output_label_file = bam_file[:-4] + "/label_" + str(chr) + "_" + str(cls) + "_grid" + str(num_grid)+".lb"
            output_refGene_file = bam_file[:-4] + "/ref_" + str(chr) + "_" + str(cls) + "_grid" + str(num_grid)+".ref"

            output_label_df_bef = pd.DataFrame(columns=['startGrid','endGrid'])
            output_label_df_bef['startGrid'] = (label_data_by_class['start'] - region_start) / stride_label
            output_label_df_bef['endGrid'] = (label_data_by_class['end'] - region_start) / stride_label

            output_label_df = pd.DataFrame(columns=['peak', 'noPeak'], dtype=int, index=range(num_grid_label))
            output_label_df['peak'] = 0
            output_label_df['noPeak'] = 1
            
            #ISSUEEEEE ALL NOPEAKS = 1
            
            index_count = 0
            for index, row in output_label_df_bef.iterrows():
                #print("if", cell_type, "in", str(label_data_by_class['cellType'].iloc[index_count]), "then UPDATE")
                #if cell_type in str(label_data_by_class['cellType'].iloc[index_count]):
                if(1):
                    output_label_df.loc[int(row['startGrid']):int(row['endGrid']), 'peak'] = 1
                    output_label_df.loc[int(row['startGrid']):int(row['endGrid']), 'noPeak'] = 0
                else:
                    pass                                    
                index_count += 1
                
                
            #.3makerefGeneTags()
            sub_refGene = makeRefGeneTags(
                refGenePd[(refGenePd['start'] > region_start)&(refGenePd['end'] < region_end)],
                region_start, region_end, stride, num_grid)

            sub_refGene.to_csv(output_refGene_file)
            read_count_by_grid.to_csv(output_count_file)
            output_label_df.to_csv(output_label_file)

            logger.info("["+output_count_file+"] is created.")
            logger.info("["+output_label_file+"] is created.")
            logger.info("["+output_refGene_file+"] is created.")


###########################################
#preprocessing function run()
###########################################


#preprocessing function to run all functions above in order
#create alignment read counts -> save to .csv files in bam dirs
#functions defined below

def run(dir_name, logger, bp_eps=30000, searching_dist=60000, num_grid=12000):

    PATH = os.path.abspath(dir_name)
    bam_files = glob.glob(PATH + '/aligned_bam/*.bam')
    label_files = glob.glob(PATH + '/labelfiles_txt/*.txt')
    #print(bam_files, label_files)
    print("########################################################################")

    MAX_CORE = cpu_count()
    processes = []

    print("Creating small fragment files for training in input dir.")

    for bam_file in bam_files:
        for label_file in label_files:
        
            #is_same_target()
            if is_same_target(bam_file, label_file) == True:
                logger.info("Find a matched pair between labeled data and an alignment file.")
                logger.info("Shared target :: " + bam_file.rsplit('/',1)[1].split('_')[0])
                
                #loadLabel()
                label_data = loadLabel(label_file)
                cellType_string = bam_file[:-4].rsplit('_',1)[1]
                
                #clusteringLabels()
                print("DBSACN clustering raw label data with base point eps:[ " + str(bp_eps) + " ]\n")
                clusteringLabels(label_data, bp_eps)

                #makeTrainFrags()
                print("Making fragments for training with <searching distance, grid> : [ " \
                            + str(searching_dist) + ", "+ str(num_grid)+" ]\n")
                makeTrainFrags(bam_file, label_data, searching_dist, num_grid, cellType_string, logger)
                
    print("successfully formed read count training data from .bam files and labels")
    print("saved to input bam dir -> mv to training data dir")
                
run('/nobackup/kfwc76/DISS/CNN_Peak_Calling_algo/DATA/0_preproc/' ,logger)                                                                                            