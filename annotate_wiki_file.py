from argparse import ArgumentParser
from wiki_loader import read_wiki_file
import pandas as pd
from pathlib import Path  # Use pathlib, not pathlib2
import os

def get_files(path):
    all_objects = Path(path).rglob('*')  # Use rglob for '**/*' pattern
    files = (str(p) for p in all_objects if p.is_file())
    return files

def generate_segmentation_template(path, output_path):
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:  # Use context manager for ExcelWriter
        sentences, _, _ = read_wiki_file(path, None, remove_preface_segment=True, return_as_sentences=True, ignore_list=True, remove_special_tokens=False)
        df = pd.DataFrame({'Cut here': [0] * len(sentences), 'Sentences': sentences})
        df = df[['Cut here', 'Sentences']]
        df.to_excel(writer, sheet_name='segment')

def generate_test_article(path, output_path):
    sentences, _, _ = read_wiki_file(path, None, remove_preface_segment=True, return_as_sentences=True, ignore_list=True, remove_special_tokens=False, 
                                     high_granularity=False)
    article_text = "\n".join(sentences)
    with open(output_path, "w", encoding='utf-8') as f:  # Use context manager and specify encoding
        f.write(article_text)

def generate_folder(input_folder, output_folder, to_text):
    counter = 0
    input_files = get_files(input_folder)
    for file in input_files:
        id = os.path.basename(file)
        file_name = f"{id}.xlsx" if not to_text else id
        output_file = os.path.join(output_folder, file_name)
        if to_text:
            generate_test_article(file, output_file)
        else:
            generate_segmentation_template(file, output_file)
        counter += 1
    print(f'Generated {counter} files')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', help='input folder path', default='/home/michael/Downloads/migo/68943', type=str)
    parser.add_argument('--output_path', help='output folder path', default='blah.xlsx', type=str)
    parser.add_argument('--toText', help='output to text files?', action='store_true')
    args = parser.parse_args()

    generate_folder(args.path, args.output_path, args.toText)