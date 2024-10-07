import utils
from argparse import ArgumentParser
import os
import wiki_utils

def main(args):
    utils.read_config_file(args.config)
    utils.config.update(vars(args))  # Use vars(args) for dictionary-like access

    file_path = args.input
    segments_path = args.segment
    output_folder_path = args.output

    # Read the segments content file
    with open(segments_path, "r", encoding='utf-8') as file:
        segments_content = file.read()

    # Read the input file
    with open(file_path, "r", encoding='utf-8') as file:
        raw_content = file.read()

    sentences = [s for s in raw_content.strip().split("\n") if s]
    segments = [s for s in segments_content.strip().split("\n") if s]

    if len(sentences) != len(segments):
        print("len(sentences) != len(segments)")
        return

    last_doc_id = 0
    last_topic = ""
    result_file_path = None

    for i in range(len(sentences)):
        sentence = sentences[i]
        segment = segments[i].split("\r")[0]

        first_comma_index = segment.index(',')
        second_comma_index = segment[first_comma_index + 1:].index(',') + first_comma_index + 1
        current_doc_id = segment[:first_comma_index]
        current_topic = segment[second_comma_index + 1:]

        # Handle new document id and create a new file for it
        if current_doc_id != last_doc_id:
            last_doc_id = current_doc_id
            print('New file index:', last_doc_id)
            if result_file_path:
                result_file.close()

            result_file_path = os.path.join(output_folder_path, f"{current_doc_id}.text")
            result_file = open(result_file_path, "w", encoding='utf-8')
            last_topic = ""

        # Write new topic to the file if changed
        if current_topic != last_topic:
            last_topic = current_topic
            level = 1 if current_topic == "TOP-LEVEL SEGMENT" else 2
            result_file.write(wiki_utils.get_segment_seperator(level, current_topic) + ".\n")

        # Write the actual sentence to the file
        result_file.write(sentence + "\n")

    if result_file_path:
        result_file.close()

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--config', help='Path to config.json', default='config.json')
    parser.add_argument('--input', help='Chen text file', required=True)
    parser.add_argument('--segment', help='Regina segmentation file', required=True)
    parser.add_argument('--output', help='Folder for converted files', required=True)

    main(parser.parse_args())