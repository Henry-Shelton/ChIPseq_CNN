# Read the input file
input_filename = 'raw_AKAP8L_labels.txt'
output_filename = 'formatted_AKAP8L_labels.txt'

with open(input_filename, 'r') as input_file, open(output_filename, 'w') as output_file:
    for line in input_file:
        fields = line.strip().split('\t')
        chromosome = fields[0]
        start = '{:,}'.format(int(fields[1]))
        end = '{:,}'.format(int(fields[2]))
        output_line = f'{chromosome}:{start}-{end} peaks\n'
        output_file.write(output_line)

print(f"Formatted peaks written to {output_filename}")