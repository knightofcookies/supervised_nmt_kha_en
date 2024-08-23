import random


def extract_random_lines(input_file, output_file, num_lines):
    with open(input_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    random_lines = ["en\t\t\t\t\tkha\n"] + random.sample(lines[1:], num_lines)

    with open(output_file, "w", encoding="utf-8") as file:
        file.writelines(random_lines)


extract_random_lines("../datasets/parallel_corpus.tsv", "../datasets/pc_15k.tsv", 15000)
