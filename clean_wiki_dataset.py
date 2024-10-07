from pathlib import Path
import wiki_processor
from argparse import ArgumentParser

def remove_malicious_files(dataset_path):
    # Read the malicious file IDs from the file
    with open('malicious_wiki_files', 'r') as f:
        malicious_file_ids = f.read().splitlines()

    # Define paths for test, train, and dev datasets
    test_path = Path(dataset_path).joinpath('test')
    train_path = Path(dataset_path).joinpath('train')
    dev_path = Path(dataset_path).joinpath('dev')

    deleted_file_count = 0

    # Iterate over the malicious file IDs and delete the corresponding files
    for file_id in malicious_file_ids:
        file_path_suffix = Path(wiki_processor.get_file_path(file_id)).joinpath(file_id)
        
        if test_path.joinpath(file_path_suffix).exists():
            test_path.joinpath(file_path_suffix).unlink()  # Use .unlink() to delete a file
            deleted_file_count += 1

        elif train_path.joinpath(file_path_suffix).exists():
            train_path.joinpath(file_path_suffix).unlink()
            deleted_file_count += 1

        elif dev_path.joinpath(file_path_suffix).exists():
            dev_path.joinpath(file_path_suffix).unlink()
            deleted_file_count += 1

        else:
            raise Exception(f'Malicious file is not included in the dataset: {file_id}')

    print(f'Deleted {deleted_file_count} files. Malicious file count: {len(malicious_file_ids)}')

def main(args):
    remove_malicious_files(args.path)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', help='Path to dataset', required=True)
    main(parser.parse_args())