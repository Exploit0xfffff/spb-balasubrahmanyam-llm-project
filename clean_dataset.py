import re

def clean_dataset(input_file, output_file):
    with open(input_file, 'r') as file, open(output_file, 'w') as outfile:
        for line in file:
            print(f"Original line: {line.strip()}")
            match = re.match(r'^\s*\d+\s*', line)
            if match:
                print(f"Match found: {match.group()}")
            cleaned_line = re.sub(r'^\s*\d+\s*', '', line)
            print(f"Cleaned line: {cleaned_line.strip()}")
            if cleaned_line.strip():
                outfile.write(cleaned_line + '\n')

if __name__ == "__main__":
    input_file = 'spb_texts.txt'
    output_file = 'cleaned_spb_texts.txt'
    clean_dataset(input_file, output_file)
