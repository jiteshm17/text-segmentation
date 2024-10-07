import os
from argparse import ArgumentParser
import subprocess
import re
from pathlib2 import Path
from random import shuffle, seed, uniform
import math
from shutil import move
import utils
import wiki_utils
import text_manipulation
import wiki_thresholds
import json

logger = utils.setup_logger(__name__, 'processor_log.log', True)
doc_split_delimiter = "</doc>"
id_parts = 7

seed(1234)

wikipedia_namespaces = ['Category', 'File', 'Ru', 'Wikipedia', 'Talk', 'User', 'MediaWiki', 'Template', 'Help', 'Portal', 
                        'Book', 'Draft', 'Education Program', 'TimedText', 'Module', 'Gadget', 'Gadget definition', 
                        'Media', 'Special']

disambiguation_pattern = '(disambiguation)'

global num_sentences_for_avg
global sum_sentences_for_avg
num_sentences_for_avg = 0
sum_sentences_for_avg = 0

def count_str_occurrences(text, findStr):
    return len(text.split(findStr)) - 1

def get_file_path(id):
    id_str = str(id).zfill(id_parts)
    return os.path.join(id_str[:2], id_str[2:4], id_str[4:6])

def process_header(header):
    id_match = re.search(r'<doc id="(\d+)" url', header)
    id = id_match.groups()[0]

    title_match = re.search(r'title="(.*)">', header)
    title = title_match.groups()[0]

    not_valid = title.isdigit() or any(title.startswith(prefix + ':' or prefix + ' talk:') 
                                       for prefix in wikipedia_namespaces) or title.endswith(disambiguation_pattern)

    return id, not not_valid

def get_sections(content):
    lines = content.split('\n')
    section = ""
    sections = [wiki_utils.get_segment_separator(1, "preface.")]
    for line in lines:
        if wiki_utils.is_separator_line(line):
            if len(section) > 0:
                sections.append(section)
            section = ""
            sections.append(line)
        else:
            section += line + '\n'

    if len(section) > 0:
        sections.append(section)

    return sections

def process_section(section, id):
    global num_sentences_for_avg, sum_sentences_for_avg
    sentences = text_manipulation.split_sentences(section, id)
    section_sentences = []
    num_lists, num_sentences, num_formulas, num_codes = 0, 0, 0, 0
    last_sentence_was_list = False

    for sentence in sentences:
        is_list_sentence = wiki_utils.get_list_token() + "." == sentence.encode('utf-8')
        if '\n' in sentence:
            logger.info(f"DocId: {id}   backslash in sentence: {sentence}")
        if wiki_utils.get_list_token() in sentence and (wiki_utils.get_list_token() + ".") != sentence.encode('utf-8'):
            num_lists += 1
            last_sentence_was_list = True
            logger.info(f"DocId: {id}     Special case 1: {sentence}")
            continue
        elif is_list_sentence:
            if last_sentence_was_list:
                continue
            last_sentence_was_list = True
            num_lists += 1
        else:
            last_sentence_was_list = False
            sentence_words = text_manipulation.extract_sentence_words(sentence)
            if len(sentence_words) < wiki_thresholds.min_words_in_sentence:
                continue
            sum_sentences_for_avg += len(sentence_words)
            num_sentences_for_avg += 1

        num_formulas += count_str_occurrences(sentence, wiki_utils.get_formula_token())
        num_codes += count_str_occurrences(sentence, wiki_utils.get_codesnippet_token())
        num_sentences += 1
        section_sentences.append(sentence)

    valid_section = True
    error_message = None

    if num_sentences < wiki_thresholds.min_sentence_in_section:
        valid_section = False
        error_message = "Sentences count in section is too low"

    if num_sentences > 0:
        lists_percentage = float(num_lists) / float(num_sentences)
        if lists_percentage >= wiki_thresholds.max_list_in_section_percentage:
            valid_section = False
            error_message = f"List percentage in section is too high: {lists_percentage}"

    section_text = ''.join(section_sentences)
    if len(re.findall('[a-zA-Z]', section_text)) < wiki_thresholds.min_section_char_count:
        valid_section = False
        error_message = "Char count in section is too low"

    if num_formulas >= wiki_thresholds.max_section_formulas_count:
        valid_section = False
        error_message = f"Number of formulas in section is too high: {num_formulas}"

    if num_codes >= wiki_thresholds.max_section_code_snippet_count:
        valid_section = False
        error_message = f"Number of code snippets in section is too high: {num_codes}"

    return valid_section, section_sentences, error_message

def is_valid_article(valid_section_count, section_count):
    if valid_section_count < wiki_thresholds.min_valid_section_count:
        return False, f"Valid section count is too low: {valid_section_count}"

    valid_section_percentage = float(valid_section_count) / float(section_count)
    if valid_section_percentage < wiki_thresholds.min_valid_section_percentage:
        return False, f"Valid section percentage is too low: {valid_section_percentage}"

    return True, ""

def max_level_in_article(content):
    max_level = -1
    for line in content:
        if wiki_utils.is_separator_line(line):
            current_level = wiki_utils.get_segment_level(line)
            if current_level > max_level:
                max_level = current_level
    return max_level

def delete_empty_segment_headers(content):
    num_of_deletions = 0
    max_level = max_level_in_article(content)
    for handle_level in range(max_level, 0, -1):
        last_section_level = -1
        last_section_header = True
        for i in range(len(content) - 1, -1, -1):
            section = content[i]
            if wiki_utils.is_separator_line(section):
                section_level = wiki_utils.get_segment_level(section)
                if section_level == handle_level:
                    is_empty = last_section_header
                    if is_empty and last_section_level <= section_level:
                        del content[i]
                        num_of_deletions += 1
                last_section_level = section_level
                last_section_header = True
            else:
                last_section_header = False

    return content, num_of_deletions

def vec_to_text(sections_with_headers):
    return '\n'.join(sections_with_headers)

def process_content(content, id):
    sections_with_headers = get_sections(content)
    article_lines = []
    section_count = 0
    valid_section_count = 0

    for section in sections_with_headers:
        if wiki_utils.is_separator_line(section):
            article_lines.append(section)
        else:
            is_valid_section, section_sentences, message = process_section(section, id)
            section_count += 1
            if is_valid_section:
                valid_section_count += 1
                article_lines.extend(section_sentences)
            else:
                logger.info(f'Invalid section in article id: {id}    Reason: {message}    Content: {vec_to_text(section_sentences).strip()}')

    is_valid, reason = is_valid_article(valid_section_count, section_count)

    if is_valid:
        article_content, _ = delete_empty_segment_headers(article_lines)
        adjusted_content_text = vec_to_text(article_content)
    else:
        adjusted_content_text = ""

    return is_valid, adjusted_content_text, reason

def process_article(article):
    non_empty_lines = [l for l in article.strip().split("\n") if l != ""]
    header = non_empty_lines[0]
    id, is_valid_header = process_header(header)

    if not is_valid_header:
        logger.info(f'Invalid header in doc id: {id}     header:   {header}')
        return "", id, False

    content = "\n".join(non_empty_lines[2:])
    is_valid_content, processed_content, debug = process_content(content, id)
    if not is_valid_content:
        logger.info(f'Invalid article in doc id: {id}.  {debug}\n\n')
    else:
        logger.info(f'Valid article , id: {id}\n\n')

    return processed_content, id, is_valid_content

def process_wiki_file(path, output_folder, train_ratio, test_ratio, forbidden_train_ids):
    train_size, dev_size, test_size = 0, 0, 0
    with open(path, "r") as file:
        raw_content = file.read()

    articles = [s for s in raw_content.decode('utf-8').strip().split(doc_split_delimiter) if len(s) > 0]
    created_articles_count = 0
    processed_articles_count = 0

    for article in articles:
        processed_article, id, is_valid = process_article(article)
        processed_articles_count += 1
        if not is_valid:
            continue
        random_num = uniform(0, 1)
        if (random_num > train_ratio and random_num <= train_ratio + test_ratio) or int(id) in forbidden_train_ids:
            partition = "test"
            test_size += 1
        elif random_num > train_ratio + test_ratio:
            partition = "dev"
            dev_size += 1
        else:
            partition = "train"
            train_size += 1
        output_sub_folder = os.path.join(output_folder, partition, get_file_path(id))
        if not os.path.exists(output_sub_folder):
            os.makedirs(output_sub_folder)
        output_file_path = os.path.join(output_sub_folder, str(id))
        with open(output_file_path, "w") as output_file:
            output_file.write(processed_article.encode('utf-8'))
        created_articles_count += 1

    return created_articles_count, processed_articles_count, train_size, dev_size, test_size

def get_forbidden_train_ids():
    with open('wikicities_article_names_to_ids') as f:
        wiki_cities = json.load(f)

    with open('wikielements_article_names_to_ids') as f:
        wiki_elements = json.load(f)

    forbidden_train_ids = [int(v) for d in (wiki_cities, wiki_elements) for v in d.values()]
    return set(forbidden_train_ids)

def get_wiki_files(path):
    all_objects = Path(path).glob('**/*')
    return (str(p) for p in all_objects if p.is_file())

def process_wiki_folder(input_folder, output_folder, train_ratio, test_ratio):
    total_train_size, total_dev_size, total_test_size = 0, 0, 0
    folders = [o for o in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, o))]
    total_created_articles, total_processed_articles = 0, 0
    previous_debug = 0
    forbidden_train_ids = get_forbidden_train_ids()

    for folder in folders:
        full_folder_path = os.path.join(input_folder, folder)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        files = get_wiki_files(full_folder_path)
        for file in files:
            created_articles, processed_articles, train_size, dev_size, test_size = process_wiki_file(
                file, output_folder, float(train_ratio), float(test_ratio), forbidden_train_ids
            )
            total_train_size += train_size
            total_dev_size += dev_size
            total_test_size += test_size
            total_created_articles += created_articles
            total_processed_articles += processed_articles
            if total_created_articles - previous_debug > 2500:
                previous_debug = total_created_articles
                print(f'Created {total_created_articles} wiki articles, out of {total_processed_articles} processed articles')

    total_samples = total_train_size + total_dev_size + total_test_size
    print(f'total_samples = {total_samples}')
    print(f"#train = {total_train_size}, ratio: {total_train_size / float(total_samples):.2f}")
    print(f"#dev = {total_dev_size}, ratio: {total_dev_size / float(total_samples):.2f}")
    print(f"#test = {total_test_size}, ratio: {total_test_size / float(total_samples):.2f}")

def move_wiki_file(src, folder, partition):
    file = os.path.relpath(src, folder)
    dst_file = os.path.join(folder, partition, file)
    dstdir = os.path.dirname(dst_file)
    if not os.path.exists(dstdir):
        os.makedirs(dstdir)
    move(src, dst_file)

def remove_empty_folders(path, remove_root=True):
    if not os.path.isdir(path):
        return

    files = os.listdir(path)
    for f in files:
        fullpath = os.path.join(path, f)
        if os.path.isdir(fullpath):
            remove_empty_folders(fullpath)

    files = os.listdir(path)
    if len(files) == 0 and remove_root:
        os.rmdir(path)

def train_test_dev(dest_folder, train_size, test_size):
    train_size_ratio = float(train_size)
    test_size_ratio = float(test_size)
    dev_size_ratio = 1 - train_size_ratio - test_size_ratio

    print(dest_folder, train_size, test_size)

    all_files = []
    if not os.path.exists(dest_folder):
        print("Output folder does not exist")
        return
    folders = [o for o in os.listdir(dest_folder) if os.path.isdir(os.path.join(dest_folder, o))]
    for folder in folders:
        full_folder_path = os.path.join(dest_folder, folder)
        files = get_wiki_files(full_folder_path)
        all_files.extend(files)

    shuffle(all_files)

    train_size = int(math.floor(len(all_files) * train_size_ratio))
    dev_size = int(math.floor(len(all_files) * dev_size_ratio))

    for i in range(train_size):
        move_wiki_file(all_files[i], dest_folder, partition="train")

    if dev_size > 0:
        for i in range(train_size, train_size + dev_size):
            move_wiki_file(all_files[i], dest_folder, partition="dev")

    for i in range(train_size + dev_size, len(all_files)):
        move_wiki_file(all_files[i], dest_folder, partition="test")

    print(f"#train = {train_size}")
    print(f"#dev = {dev_size}")
    print(f"#test = {len(all_files) - train_size - dev_size}")

    remove_empty_folders(dest_folder)

def main(args):
    global num_sentences_for_avg, sum_sentences_for_avg
    if not os.path.exists(args.temp):
        os.makedirs(args.temp)

    cmd = ['python', str(Path(__file__).parent / 'wiki_extractor.py'), '-s', '-o', args.temp, '--article_count', str(args.article_count), '--lists']
    print(cmd)

    if args.processes:
        cmd += ['--processes', args.processes]

    cmd += [args.input]

    if not args.no_extractor:
        subprocess.call(cmd)
        print("Finished extractor")

    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    process_wiki_folder(args.temp, args.output, args.train, args.test)

    print(f"Number of processed sentences: {num_sentences_for_avg}")
    print(f"avg len sentence = {sum_sentences_for_avg / float(num_sentences_for_avg)}")
    print('done')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', help='Path to wikipedia dump', required=True)
    parser.add_argument('--processes', help='Num of processors to use in wiki_extractor')
    parser.add_argument('--no_extractor', help='Skip wiki-extractor', action='store_true')
    parser.add_argument('--temp', help='folder to save temporal files', required=True)
    parser.add_argument('--output', help='output folder', required=True)
    parser.add_argument('--train', help='train size ratio', required=True)
    parser.add_argument('--test', help='test size ratio', required=True)
    parser.add_argument('--article_count', help='max number of wikipedia articles to extract', default=1000000)
    main(parser.parse_args())