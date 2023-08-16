print("formatting script for McGill label layout")

input_folder = '/nobackup/kfwc76/DISS/CNN_Peak_Calling_algo/DATA/0_preproc/lanceotron_files/'
output_folder = '/nobackup/kfwc76/DISS/CNN_Peak_Calling_algo/DATA/0_preproc/lanceotron_files/formatted_labels/'

import os

def format_position(position):
    return f"{int(position):,}".replace(",", ",")

def process_line(line):
    parts = line.strip().split("\t")
    formatted_position1 = format_position(parts[1])
    formatted_position2 = format_position(parts[2])
    return f"{parts[0]}:{formatted_position1}-{formatted_position2} {parts[3]}"

def process_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        current_chr = None
        for line in infile:
            formatted_line = process_line(line)
            if current_chr != formatted_line.split(':')[0]:
                if current_chr:
                    outfile.write('\n')
                current_chr = formatted_line.split(':')[0]
            outfile.write(formatted_line + '\n')

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith('.txt'):
        input_file = os.path.join(input_folder, filename)
        print("processing input file:", input_file)
        output_file = os.path.join(output_folder, filename)
        process_file(input_file, output_file)
        print("done")