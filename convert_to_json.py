import json

def convert_to_json(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        lyrics = file.readlines()

    data = {'lyrics': [line.strip() for line in lyrics]}

    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    input_file = 'telugu_lyrics_dataset.txt'
    output_file = 'telugu_lyrics_dataset.json'
    convert_to_json(input_file, output_file)
