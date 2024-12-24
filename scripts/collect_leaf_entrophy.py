import re

def collect_average_leaf_entropy(log_file_path):
    # Regular expression to match the Average Leaf Entropy value
    entropy_pattern = re.compile(r'Average Leaf Entrophy: (\d+\.\d+)')

    # List to store collected entropy values
    entropy_values = []

    # Open and read the log file
    with open(log_file_path, 'r') as log_file:
        for line in log_file:
            match = entropy_pattern.search(line)
            if match:
                entropy_values.append(float(match.group(1)))

    return entropy_values


# Example usage
log_file_path = '/path/to/your/logfile.txt'
entropy_values = collect_average_leaf_entropy(log_file_path)
print(entropy_values)