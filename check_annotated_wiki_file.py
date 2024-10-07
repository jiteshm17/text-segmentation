from argparse import ArgumentParser
from wiki_loader import read_wiki_file
import pandas as pd
import accuracy
from annotate_wiki_file import get_files
import os
from glob import glob

graphseg_delimiter = "=========="

def generate_segmentation_template(path, output_path):
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:  # Use context manager
        sentences, _, _ = read_wiki_file(path, None, False)
        sentences = [' '.join(s) + '.' for s in sentences]
        df = pd.DataFrame({'Sentences': sentences, 'Cut here': [0] * len(sentences)})
        df = df[['Sentences', 'Cut here']]
        df.to_excel(writer, sheet_name='segment')

def target_place_to_list(targets):
    list_of_targets = [1 if i in targets else 0 for i in range(targets[-1] + 1)]
    list_of_targets[-1] = 1  # Ensure the last sentence is marked as the end
    return list_of_targets

def get_graphseg_segments(file_path):
    with open(str(file_path), "r", encoding='utf-8') as file:
        raw_content = file.read()

    sentences = [s for s in raw_content.strip().split("\n") if s and s != "\n"]
    h = []

    for sentence in sentences:
        if sentence == graphseg_delimiter:
            if h:
                h[-1] = 1
        else:
            h.append(0)

    h[-1] = 1  # Correct segmentation for the last sentence
    return h

def get_xlsx_segments(xlsx_path):
    df = pd.read_excel(xlsx_path)
    outputs = df['Cut here'].values
    outputs[-1] = 1  # Ensure the last sentence is marked as the end
    return outputs

def get_gold_segments(path):
    sentences, targets, _ = read_wiki_file(path, None, remove_preface_segment=True, return_as_sentences=True, ignore_list=True, remove_special_tokens=False, high_granularity=False)
    return target_place_to_list(targets)

def get_sub_folders_for_graphseg(folder):
    sub_folders = [os.path.join(folder, o) for o in os.listdir(folder) if os.path.isdir(os.path.join(folder, o))]
    print(sub_folders)
    return sub_folders

def analyze_folder(wiki_folder, xlsx_folder, is_graphseg, use_xlsx_sub_folders=False):
    acc = accuracy.Accuracy()

    input_files = get_files(wiki_folder)
    if use_xlsx_sub_folders:
        annotated_files_folders = [os.path.join(xlsx_folder, f) for f in os.listdir(xlsx_folder) if os.path.isdir(os.path.join(xlsx_folder, f))]
    else:
        annotated_files_folders = [xlsx_folder]

    for file in input_files:
        id = os.path.basename(file)
        file_name = f"{id}.xlsx" if not is_graphseg else id
        xlsx_file_paths = [os.path.join(folder, file_name) for folder in annotated_files_folders]
        print(xlsx_file_paths)
        print(file)

        for xlsx_file_path in xlsx_file_paths:
            if os.path.isfile(xlsx_file_path):
                if is_graphseg:
                    tested_segments = get_graphseg_segments(xlsx_file_path)
                else:
                    tested_segments = get_xlsx_segments(xlsx_file_path)
            else:
                tested_segments = None

            gold_segments = get_gold_segments(file)
            if tested_segments is not None and len(tested_segments) != len(gold_segments):
                print("(len(tested_segments) != len(gold_segments))")
                print("Stopping run")
                return 1000, 1000

            if tested_segments is not None:
                acc.update(tested_segments, gold_segments)

    # Print results
    calculated_pk, calculated_windiff = acc.calc_accuracy()
    print('Finished testing.')
    print(f'Pk: {calculated_pk:.4f}.')
    print()

    return calculated_pk, calculated_windiff

def result_to_file(pk_list, wd_list, path_list, result_file_path):
    with pd.ExcelWriter(result_file_path, engine='xlsxwriter') as writer:  # Use context manager
        df = pd.DataFrame({'pk': pk_list, 'wd': wd_list, 'folders': path_list})
        df.to_excel(writer, sheet_name='annotated_result')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', help='wiki folder, truth', type=str)
    parser.add_argument('--xlsx_path', help='folder with xlsx files', type=str)
    parser.add_argument('--graphseg', help='to calculate graphseg pk', action='store_true')

    args = parser.parse_args()
    pk_list = []
    wd_list = []
    path_list = []

    if args.graphseg:
        graphseg_folders = get_sub_folders_for_graphseg(args.xlsx_path)
        for folder in graphseg_folders:
            pk, wd = analyze_folder(args.path, folder, args.graphseg)
            pk_list.append(pk)
            wd_list.append(wd)
            path_list.append(folder)
    else:
        pk, wd = analyze_folder(args.path, args.xlsx_path, args.graphseg, use_xlsx_sub_folders=True)
        pk_list.append(pk)
        wd_list.append(wd)
        path_list.append(args.xlsx_path)

    # Write result to file
    result_to_file(pk_list, wd_list, path_list, os.path.join(args.xlsx_path, "result_pk.xlsx"))