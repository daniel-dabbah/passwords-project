from tqdm import tqdm

def extract_first_100_lines(input_file, output_file):
    # Open the input file in read mode
    with open(input_file, 'r') as infile:
        # Open the output file in write mode
        with open(output_file, 'w') as outfile:
            # Use tqdm to create a progress bar for 100 lines
            for i, line in enumerate(tqdm(infile, total=100, desc="Extracting lines")):
                if i < 100:
                    outfile.write(line)
                else:
                    break  # Stop once 100 lines have been processed

# Specify the input and output files
input_file = 'pwned-passwords-ordered-by-count.txt'
output_file = 'output_100_lines.txt'

# Call the function to extract the first 100 lines
extract_first_100_lines(input_file, output_file)

print(f'The first 100 lines have been extracted to {output_file}')
