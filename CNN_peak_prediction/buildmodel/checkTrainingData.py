#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~# PREREQ TEST/TRAIN INDEX LISTS #~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import matplotlib.pyplot as plt
import ntpath

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(path)


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

def expandingPrediction(input_list, multiple=5):

    expanded_list = []
    for prediction in input_list:
        for i in range(multiple):
            expanded_list.append(prediction)

    return expanded_list
    

def checkTrainingData(dir_name, num_grid=12000):
    PATH = os.path.abspath(dir_name)

    dir_list = os.listdir(PATH)

    for dir in dir_list:
        dir = PATH + '/' + dir
        print("DIRECTORY (TARGET) : <" + dir +">")

    input_list = {}
    for dir in dir_list:
        dir = PATH + '/' + dir
        input_list[dir] = extractChrClass(dir)

#PREREQ FUNCTIONS OR THESE INDEX
    train_data_list = []
    train_label_list = []
    for dir in input_list:
        for chr in input_list[dir]:
            for cls in input_list[dir][chr]:
                input_file_name = "{}/{}_{}_grid{}.ct".format(dir, chr, cls, num_grid)
                ref_file_name = "{}/ref_{}_{}_grid{}.ref".format(dir, chr, cls, num_grid)
                label_file_name = "{}/label_{}_{}_grid{}.lb".format(dir, chr, cls, num_grid)

                reads = pd.read_csv(input_file_name)['readCount'].values.reshape(num_grid)
                label = pd.read_csv(label_file_name)['peak'].values.transpose()
                label = expandingPrediction(label)


                #check dodgy peaks
                
                #plt.plot(reads,'k')
                #plt.plot(label,'r.')
                #plt.title('{} {}-{}'.format(dir,chr,cls))
                #plt.show()
                #plt.savefig('/nobackup/kfwc76/DISS/CNN_Peak_Calling_algo/buildmodel/trainingdata/checktrainingdata.png')

                #if input("save(1) or delete(0)  ::") == '0':
                #    os.remove(input_file_name)
                #    os.remove(label_file_name)
                #    os.remove(ref_file_name)
    print(train_data_list)
    print("[INFO] [OPTIONAL] manually choose dodgy peaks here \n")
    print("[INFO] train_data_list / train_data_list created \n")
    
checkTrainingData('/nobackup/kfwc76/DISS/CNN_Peak_Calling_algo/buildmodel/trainingdata/', num_grid=12000)