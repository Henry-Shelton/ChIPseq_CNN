print("\n ############### MODEL TEST SET EVALUATION METRICS ############### \n")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~# MODEL MODULES #~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import logging
logger = logging.getLogger("ConvLog")
logging.basicConfig(filename="/nobackup/kfwc76/DISS/CNN_Peak_Calling_algo/outputs/logger.txt", level=logging.DEBUG)
logging.debug('logging active')

import copy
import random

import os


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~# INPUTS #~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

input_bed_file = "AKAP8L_Hep2G.bed"
input_label_file = "AKAP8L_Hep2G.bam"

print("INPUT PREDICTED PEAKS FILE =", input_bed_file)
print("INPUT TRUTH LABEL FILE =", input_label_file, "\n")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#1 - loadPeak()
#from utility.loadPeak import run as loadPeak

#regular bed
def bed_file_load(input_bed, chrom = None):
	"""loading bed files and translate to Python Object"""
	bed_file = open(input_bed,'r')
	peak_data = bed_file.readlines()

	peak_table = ['chr','region_s','region_e','peak_name','score']
	peak_labels = []

	if len(peak_data) is 0:
		return []

	while True:
		if len(peak_data) > 0 and peak_data[0][0] == '#':
			del peak_data[0]
		else:
			break

	for peak in peak_data:
		peak_labels.append(dict(zip(peak_table,peak.split())))

	return peak_labels


#ENCODE narrowPeak
def narrow_peak_file_load(input_Npeak, chrom = None):
	""""""
	Npeak_file = open(input_Npeak, 'r')
	peak_data = Npeak_file.readlines()

	peak_table = ['chr','region_s','region_e','name','score','strand','signalValue','pValue','qValue','peak']
	peak_labels = []

	for peak in peak_data:
		peak_labels.append(dict(zip(peak_table,peak.split())))

	return peak_labels


#ENCODE broadPeak
def broad_peak_file_load(input_Bpeak, chrom = None):
	""""""
	Bpeak_file = open(input_Bpeak, 'r')
	peak_data = Bpeak_file.readlines()

	peak_table = ['chr','region_s','region_e','name','score','strand','signalValue','pValue','qValue']
	peak_labels = []

	for peak in peak_data:
		peak_labels.append(dict(zip(peak_table,peak.split())))

	return peak_labels


#ENCODE gappedPeak
def gapped_peak_file_load(input_Gpeak, chrom = None):
	pass


def loadPeak(input_file_name):
	format = input_file_name.rsplit('.', 1)[1]

	if format == "bed":
		return bed_file_load(input_file_name)
	elif format == "narrowPeak":
		return narrow_peak_file_load(input_file_name)
	elif format == "broadPeak":
		return broad_peak_file_load(input_file_name)
	elif format == "gappedPeak":
		return gapped_peak_file_load(input_file_name)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#2 - loadLabel()
#from utility.loadLabel import run as loadLabel

def parse_cellType(file_name):
    """
    Parsing file_name and extracting cell-type
    input file must be EXPERIMENT_AREA_CELL-TYPE.bam so bamtools create
    EXPERIMENT_AREA_CELL-TYPE.REF_chrN.PEAK
    :param file_name:
    :return: cell_type
    """
    parse = file_name.split('.')[0].rsplit('_')

    return parse[len(parse)-1]


def parse_chr(file_name):
    """
    Parsing file_name and extracting chromosome
    input file must be EXPERIMENT_AREA_CELL-TYPE.bam so bamtools create
    EXPERIMENT_AREA_CELL-TYPE.REF_chrN.PEAK
    :param file_name:
    :return: chromosome
    """

    file_name = file_name.rsplit('.',1)[0]

    file_name = file_name.rsplit('.',1)[0]
    file_name = file_name.rsplit('_',1)
    chromosome = file_name[1]

    return chromosome


def parse_peak_labels(peak_labels, chromosome_num, cell_type):
    """
    :param peak_labels:
    :param chromosome_num:
    :param cell_type:
    :param cpNum_data:
    :return:
    """

    labels = []
    label_table = ['regions', 'peakStat', 'cellType']

    #parse the text file to python list
    for peak in peak_labels:
        containor = []
        containor.append(peak.split(':')[0])
        containor.append(peak.split(':')[1].split(' ',2))
        labels.append(containor)

    #this list will be return value.
    result_labels_list = []

    #check the condition ( chromosome ) and change to python map
    for label in labels:
        if label[0] == chromosome_num:
            label_map = dict(zip(label_table, label[1]))
            result_labels_list.append(label_map)

    if len(result_labels_list) == 0:
        #print "there are matched label data. so cannot handle it"
        return -1

    for label in result_labels_list:
        if len(label) == 2 or not cell_type.lower() in label['cellType'].lower():
            label['peakStat'] = 'noPeak'

    for label in result_labels_list:
        label['regions'] = label['regions'].split('-')
        label['regions'][0] = int(label['regions'][0].replace(',',''))
        label['regions'][1] = int(label['regions'][1].replace(',',''))


    return result_labels_list


def peak_label_load(label_file_name):
    """loading Validation Set Files and translate to Python Object."""
    valid_file = open(label_file_name, 'r')
    peak_data = valid_file.readlines()
    peak_labels = []

    for peak in peak_data:
        if peak == "\r\n":
            peak_data.remove(peak)

    for peak in peak_data:
        if '#' not in peak and 'chr' in peak:
            peak = peak.rstrip('\r\n')
            peak_labels.append(peak)

    valid_file.close()
    return peak_labels


def loadLabel(file_name, input_chromosome = None, input_cellType = None):
    """
    :param validSet:
    :param file_name:
    :param input_chromosome:
    :param input_cellType:
    :return:
    """

    if input_chromosome is None:
        chromosome = parse_chr(file_name)
    else:
        chromosome = input_chromosome

    if input_cellType is None:
        cellType = parse_cellType(file_name)
    else:
        cellType = input_cellType

    validSet = peak_label_load(file_name)

    peak_labels = parse_peak_labels(validSet, chromosome, cellType)

    # cannot found label about selected area.
    if peak_labels is -1:
        return -1
    
    return peak_labels


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#3 - calculateError()
#from utility.calculateError import run as calculateError

def calculate_error(peak_data, labeled_data, strong_call = True):
    """
    calculate actual error by numbering to wrong label
    :param peak_data:
        python map is parsed and it is from result of peak calling algorithm
        like a MACS.
    :param labeled_data:
        python map is parsed and it is from labeled data file.
    :return:
        return python tuple ( number of incorrect label , number of whole label )
    """

    # sum of error label
    scores = 0.0

    # number of Error label about each error type.
    FP = 0.0
    FN = 0.0
    TP = 0.0
    TN = 0.0

    # number of label which can occur error about each error type.
    possible_FP = 0.0
    possible_FN = 0.0

    for label in labeled_data:
        if label['peakStat'].lower() == 'peaks':
            possible_FN += 1
            state = is_peak(peak_data, label['regions'], weak_predict=True)

            if state == "False Negative":
                FN += 1
            else:
                scores += state
                TP += 1

        elif (label['peakStat'].lower() == 'peakstart') or (label['peakStat'].lower() == 'peakend'):
            #if strong_call is False:
            #    possible_FP += 1
            possible_FN += 1
            state = is_peak(peak_data, label['regions'], weak_predict= not strong_call)

            if state == "False Positive":
                FP += 1
            elif state == "False Negative":
                FN += 1
            else:
                scores += state

        elif label['peakStat'].lower() == 'nopeak':
            possible_FP += 1

            state = is_noPeak(peak_data, label['regions'])
            if not (state == True):
                FP += 1
                scores += state
            else:
                scores += 1
                TN += 1

        else:
            print("label type error :::{}".format(label['peakStat']))
            exit()

    #print("possible FN {} possible FP {}".format(possible_FN,possible_FP))
    #print("FN_Error: {} FP_Error: {}\n".format(FN,FP))

    FNFP_dict = {"negativeNum": possible_FN , "positiveNum" : possible_FP, "FN" : FN, "FP" : FP, "TN" : TN , "TP" : TP}

    return len(labeled_data) - scores, len(labeled_data) , FNFP_dict


def is_peak(target, labelRegion, tolerance=0, weak_predict=False):
    """
    Checking the label is peak or not.
    :param target:
    :param labelRegion:
    :param tolerance:
    :param weak_predict:
    :return:
    """

    index = len(target) // 2
    min_index = 0
    max_index = len(target)

    # if find correct one, return True
    while True:
        correct_ness = is_same(target, labelRegion, index, tolerance)

        if correct_ness is 'less':
            max_index = index
            index = (min_index + index) // 2
        elif correct_ness is 'upper':
            min_index = index
            index = (max_index + index) // 2
        # find correct regions
        else:
            if (weak_predict == True):
                return 1 #calculate_sum_of_weights(index, target, tolerance, labelRegion, mode='bonus')

            # find one peak
            else:
                if (index + 1) is not len(target) \
                        and is_same(target, labelRegion, index + 1, tolerance) is 'in' \
                        or is_same(target, labelRegion, index - 1, tolerance) is 'in':
                    return "False Positive"
                else:
                    return 1 # + bonus_weight(labelRegion, target[index], 'peakStart')

        if max_index <= min_index + 1:
            if is_same(target, labelRegion, index, tolerance) is 'in':
                return 1 # + bonus_weight(labelRegion, target[index], 'peakStart')
            else:
                return "False Negative"


def is_noPeak(target, value, tolerance=0):
    """
    :param target:
    :param value:
    :param tolerance:
    :return:
    """
    region_min = value[0]
    region_max = value[1]

    index = int(len(target) // 2)
    min_index = 0
    max_index = int(len(target))
    steps = 1

    while True:
        find_matched = is_same(target, value, index, tolerance)

        if find_matched is 'less':
            max_index = index
            index = (min_index + index) // 2
        elif find_matched is 'upper':
            min_index = index
            index = (max_index + index) // 2
        # find correct regions , so it is fail
        else:
            return calculate_sum_of_weights(index, target, tolerance, value, mode='bias')

        if abs(float(target[index]['region_e']) - region_min) < 5 * steps or steps > 1000:
            break
        steps += 1

    # correct label ( no peak )
    if not (index + 1 >= len(target)):
        if float(target[index + 1]['region_s']) + tolerance > region_max \
                and float(target[index]['region_e']) + tolerance < region_min:
            return True
        else:
            return True

    # false negative no peak ( there is peak )
    else:
        return calculate_sum_of_weights(index, target, tolerance, value, mode='bias')


def calculate_sum_of_weights(index, target, tolerance, value, mode=None):
    peaks = []
    num_of_peaks = 1
    front_check = 1
    back_check = 1
    peaks.append(target[index])
    ## front seek
    while True:
        if index + front_check < len(target):
            if (is_same(target, value, index + front_check, tolerance) is 'in'):
                num_of_peaks += 1
                peaks.append(target[index + front_check])
            else:
                break
            front_check += 1
        else:
            break

    ## back seek
    while True:
        if index - back_check is not (-1):
            if (is_same(target, value, index - back_check, tolerance) is 'in'):
                num_of_peaks += 1
                peaks.append(target[index - back_check])
            else:
                break
            back_check += 1
        else:
            break

    if mode is 'bonus':
        return 1 + bonus_weight(value, peaks, 'peaks')
    elif mode is 'bias':
        return bonus_weight(value, peaks, 'nopeak')


def is_same(target, value, index, tolerance):
    """
    this function check label value whether bigger than index or lower than index
    :param target:
    :param value:
    :param index:
    :param tolerance:
    :return:
    """

    if value[1] + tolerance < int(target[index]['region_s']):
        return 'less'
    elif value[0] - tolerance > int(target[index]['region_e']):
        return 'upper'
    else:
        return 'in'


def bonus_weight(label, target, case):
    """
    label is raw of label data set.
    Target is dict or list of dict.

    :param label:
    :param target:
    :param case:
    :return:
    """
    length_label = (label[1] - label[0]) / 2
    center_label = label[1] + length_label

    if case == "peakStart":

        target['region_e'] = float(target['region_e'])
        target['region_s'] = float(target['region_s'])

        length_target = (target['region_e'] - target['region_s']) / 2
        center_target = target['region_s'] + length_target

        distance_c = abs(center_label - center_target)
        distance_l = abs(length_label - length_target)

        weight = (1.0 / (1 + distance_c / length_label)) * (1.0 / (1 + distance_l / length_target))

        return weight

    elif (case == "peaks") or (case == "nopeak"):

        weight_sum = 0

        for peak in target:
            length_peak = (float(peak['region_e']) - float(peak['region_s'])) / 2
            center_peak = float(peak['region_s']) + length_peak

            distance_c = abs(center_label - center_peak)
            distance_l = abs(length_label - length_peak)

            weight = (1.0 / (1 + distance_c / length_label)) * (1.0 / (1 + distance_l / length_peak))

            weight_sum += weight

        n = len(target)

        weight_mean = float(weight_sum) / float(n)
        penalty = 2 * (1.0 - ((n ** 0.5) / (1.0 + n ** 0.5)))

        if case == "peaks":
            return weight_mean * penalty
        elif case == "nopeak":
            return (-1) * (weight_mean * penalty)
        else:
            print("caseError")
            exit(0)


def calculateError(input_peaks, input_labels):
    """
    This is the module for calculation Error by comparing between
    labeled data and the input file.
    :param input_file_name:
        This parameter is file name that result of peak detectors.
    :param input_labels:
        It is python map and it is already parsed which means
        having specific cell type and chromosome.
    :return:
        Accuracy of result file.
    """


    # case of input label size is 0, error num error rate is zero.
    if input_labels is -1:
        return 0, 0, None

    if input_peaks is -1:
        return 0, 0, None

    if len(input_peaks) is 0:
        return 0, 0, None

    if len(input_labels) is 0:
        return 0, 0, None

    return calculate_error(input_peaks, input_labels)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def run(input_bed_file, input_label_file, logger):
    peaks = loadPeak(input_bed_file)
    error = 0
    total = 0
    N = 0.
    P = 0.
    FN = 0.
    FP = 0.
    TN = 0.
    TP = 0.
    
    cell_type = input_bed_file.rsplit('_',1)[1].split('.')[0]
    
    for i in range(22):
        chr_labels = loadLabel(input_label_file, input_chromosome="chr{}".format(i + 1), input_cellType = cell_type)
        #print(chr_labels)
        chr_peaks = list(filter(lambda peak: peak['chr'] == 'chr{}'.format(i + 1), peaks))
        #logger.info("chr{}".format(i + 1))
        temp_x, temp_y, FNFP = calculateError(chr_peaks, chr_labels)
        if temp_x == 0 and temp_y == 0:
            continue
        error += temp_x
        total += temp_y
        N += FNFP['negativeNum']
        P += FNFP['positiveNum']
        FN += FNFP['FN']
        FP += FNFP['FP']
        TN += FNFP['TN']
        TP += FNFP['TP']


    print("\n\npossible # of Negatives: {} , possible # of Positives: {}".format(N, P))
    print("\nP: {}, N: {}".format(TP+FP, TN+FN))
    print("\nTP: {}, TN: {}, FP: {}, FN: {}".format(TP, TN, FP, FN))
    print("\nACC: {} , FN_Rate: {} , FP_Rate: {}".format((TP+TN)/(TP+TN+FP+FN), FN /(TN+FN), FP/(FP+TP)))
    print("\nSensitivity: {} , Specificity: {}\n\n".format(TP/(TP+FN), TN/(TN+FP)))
    sens = TP/(TP+FN)
    spec = TN/(TN+FP)
    print("\nF1 Score: {}\n\n".format((2*sens*spec)/(sens+spec)))

run(input_bed_file, input_label_file, logger)