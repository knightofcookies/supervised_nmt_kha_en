import os

# Paths to the directories
parts_dir = '../datasets/parts'  # Folder containing original text files
translated_dir = '../datasets/translated'  # Folder containing translated text files

# Output file paths
en_output_file = 'en.txt'
kha_output_file = 'kha.txt'

# Function to combine corresponding lines from both directories
def combine_files_with_matching_lines(parts_directory, translated_directory, en_output, kha_output):
    with open(en_output, 'w', encoding='utf-8') as en_outfile, open(kha_output, 'w', encoding='utf-8') as kha_outfile:
        for i in range(1, 4000 + 1):
            part_file_path = os.path.join(parts_directory, f'part{i}.txt')
            translated_file_path = os.path.join(translated_directory, f'part{i}_en_to_kha.txt')

            # Check if both part and translated files exist
            if os.path.exists(part_file_path) and os.path.exists(translated_file_path):
                with open(part_file_path, 'r', encoding='utf-8') as part_file, \
                     open(translated_file_path, 'r', encoding='utf-8') as translated_file:

                    # Read lines from both files
                    part_lines = part_file.readlines()
                    translated_lines = translated_file.readlines()

                    # Only process lines that have a corresponding translation
                    for part_line, translated_line in zip(part_lines, translated_lines):
                        en_outfile.write(part_line.strip() + '\n')  # Write line from original part
                        kha_outfile.write(translated_line.strip() + '\n')  # Write corresponding translated line

print(f"Combining files into {en_output_file} and {kha_output_file}...")

# Combine files while only keeping matching lines
combine_files_with_matching_lines(parts_dir, translated_dir, en_output_file, kha_output_file)
