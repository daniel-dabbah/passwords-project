import os
import random
from tqdm import tqdm

def sample_passwords(file_path, sample_size, max_line_length):
    sampled_passwords = []
    file_size = os.path.getsize(file_path)
    
    sampled_offsets = sorted(random.sample(range(file_size), sample_size))
    
    with open(file_path, 'rb') as file:
        for offset in tqdm(sampled_offsets, desc="Sampling passwords"):
            file.seek(offset)
            # Move to the start of the next line
            file.readline()
            password = file.readline().decode('utf-8', errors='ignore').strip()
            if password:
                sampled_passwords.append(password)
                if len(sampled_passwords) >= sample_size:
                    break

    return sampled_passwords

def save_sampled_passwords(sampled_passwords, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for password in sampled_passwords:
            file.write(password + '\n')

# Usage
file_path = 'antipublic.txt'
sample_size = 100000
max_line_length = 32  # Assuming maximum line length including newline characters
output_file = 'passwords.txt'

sampled_passwords = sample_passwords(file_path, sample_size, max_line_length)
save_sampled_passwords(sampled_passwords, output_file)

print(f'Sampled passwords saved to {output_file}')

