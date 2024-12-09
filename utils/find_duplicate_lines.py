def find_duplicate_lines(filename):
    """
    Reads a text file and prints the line numbers of duplicate lines.

    Args:
        filename (str): The name of the file to read.
    """

    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()

    line_numbers = {}
    for i, line in enumerate(lines, 1):
        if line in line_numbers:
            line_numbers[line].append(i)
        else:
            line_numbers[line] = [i]

    for line, numbers in line_numbers.items():
        if len(numbers) > 1:
            print(f"Duplicate line: '{line.strip()}' found on lines: {numbers}")


# Example usage:
filename = "../datasets/translated/part2500_en_to_kha.txt"  # Replace with your file name
find_duplicate_lines(filename)
