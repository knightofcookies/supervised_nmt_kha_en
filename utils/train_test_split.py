import random

en_file_path = 'en_nits.txt'
kha_file_path = 'kha_nits.txt'

with open(en_file_path, 'r', encoding='utf-8') as en_file, \
     open(kha_file_path, 'r', encoding='utf-8') as kha_file:
    en_lines = en_file.readlines()
    kha_lines = kha_file.readlines()

assert len(en_lines) == len(kha_lines), "Files must have the same number of lines"

data = list(zip(en_lines, kha_lines))

random.shuffle(data)

train_size = int(0.9 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

with open('en_nits_train.txt', 'w', encoding='utf-8') as train_en_file, \
     open('kha_nits_train.txt', 'w', encoding='utf-8') as train_kha_file:
    for en_line, kha_line in train_data:
        train_en_file.write(en_line)
        train_kha_file.write(kha_line)

with open('en_nits_test.txt', 'w', encoding='utf-8') as test_en_file, \
     open('kha_nits_test.txt', 'w', encoding='utf-8') as test_kha_file:
    for en_line, kha_line in test_data:
        test_en_file.write(en_line)
        test_kha_file.write(kha_line)

print("Train-test split complete!")
