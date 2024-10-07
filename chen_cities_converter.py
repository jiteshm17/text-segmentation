import utils
from pathlib import Path
from argparse import ArgumentParser
import os
import wiki_utils

def main(args):
    utils.read_config_file(args.config)
    utils.config.update(vars(args))  # Update config with args as a dictionary

    file_path = args.input
    output_folder_path = args.output
    special_delim_sign_path = args.sign

    # Open and read the special delimiter sign file
    with open(special_delim_sign_path, "r", encoding='utf-8') as file:
        special_delim_sign = file.read().split("\n")[0]

    # Open and read the input file
    with open(file_path, "r", encoding='utf-8') as file:
        raw_content = file.read()

    sentences = [s for s in raw_content.strip().split("\n") if s]

    last_doc_id = 0
    last_topic = ""
    result_file_path = None

    for sentence in sentences:
        first_comma_index = sentence.index(',')
        second_comma_index = sentence[first_comma_index + 1:].index(',') + first_comma_index + 1
        current_doc_id = sentence[:first_comma_index]
        sign_index = sentence.index(special_delim_sign)
        start_sentence_index = sign_index + 1
        actual_sentence = sentence[start_sentence_index:]
        current_topic = sentence[second_comma_index + 1:sign_index]

        # Handle new document id and create new file for it
        if current_doc_id != last_doc_id:
            last_doc_id = current_doc_id
            print('New file index:', last_doc_id)
            if result_file_path:
                result_file.close()

            result_file_path = os.path.join(output_folder_path, f"{current_doc_id}.text")
            result_file = open(result_file_path, "w", encoding='utf-8')
            last_topic = ""

        # Write new topic to file if changed
        if current_topic != last_topic:
            last_topic = current_topic
            level = 1 if current_topic == "TOP-LEVEL SEGMENT" else 2
            result_file.write(wiki_utils.get_segment_seperator(level, current_topic) + ".\n")

        if '\n' in sentence:
            print('Backslash in sentence')

        # Write actual sentence to file
        result_file.write(actual_sentence + "\n")

    if result_file_path:
        result_file.close()

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--config', help='Path to config.json', default='config.json')
    parser.add_argument('--input', help='Chen text file', required=True)
    parser.add_argument('--output', help='Folder for converted files', required=True)
    parser.add_argument('--sign', help='File containing special delimiter sign', required=True)

    main(parser.parse_args())