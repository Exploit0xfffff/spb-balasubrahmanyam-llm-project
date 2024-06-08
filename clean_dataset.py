import re

def clean_dataset(input_file, output_file):
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    with open(input_file, 'r') as file, open(output_file, 'w') as outfile:
        for line in file:
            print(f"Original line: {line.strip()}")
            match = re.match(r'^\s*\d+\s*', line)
            if match:
                print(f"Match found: {match.group()}")
            cleaned_line = re.sub(r'^\s*\d+\s*', '', line)
            print(f"Cleaned line: {cleaned_line.strip()}")
            if cleaned_line.strip():
                outfile.write(cleaned_line + ' ')
                print(f"Written to outfile: {cleaned_line.strip()}")

def format_for_pretraining(input_file, output_file):
    print(f"Formatting file for pre-training: {input_file}")
    with open(input_file, 'r') as file, open(output_file, 'w') as outfile:
        text = file.read()
        formatted_text = re.sub(r'\s+', ' ', text).strip()
        outfile.write(formatted_text)
        print(f"Formatted text written to: {output_file}")

if __name__ == "__main__":
    input_file = 'spb_texts.txt'
    cleaned_output_file = 'cleaned_spb_texts.txt'
    pretraining_output_file = 'formatted_spb_texts.txt'
    clean_dataset(input_file, cleaned_output_file)
    format_for_pretraining(cleaned_output_file, pretraining_output_file)
