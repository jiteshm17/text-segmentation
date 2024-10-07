import os
from pathlib import Path
from argparse import ArgumentParser
from shutil import move

def remove_empty_folders(path, remove_root=True):
    if not os.path.isdir(path):
        return

    # Remove empty subfolders
    for f in os.listdir(path):
        fullpath = os.path.join(path, f)
        if os.path.isdir(fullpath):
            remove_empty_folders(fullpath)

    # If folder is empty, delete it
    if not os.listdir(path) and remove_root:
        os.rmdir(path)

def convert_choi_to_by_seg_length(path):
    folders = [o for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]

    for folder in folders:
        full_folder_path = os.path.join(path, folder)
        seg_folders = [o for o in os.listdir(full_folder_path) if os.path.isdir(os.path.join(full_folder_path, o))]

        for seg_folder in seg_folders:
            full_seg_folder_path = os.path.join(full_folder_path, seg_folder)
            converted_path_list = full_seg_folder_path.split(os.sep)

            converted_path = os.path.sep.join(converted_path_list[:-2] + [converted_path_list[-1], converted_path_list[-2]])
            if not os.path.exists(converted_path):
                os.makedirs(converted_path)

            all_objects = Path(full_seg_folder_path).rglob('*')  # Use rglob for recursive search
            files = (str(p) for p in all_objects if p.is_file())

            for file in files:
                target = os.path.join(converted_path, os.path.basename(file))
                move(file, target)

            print(f"Removing empty folder: {full_seg_folder_path}")
            remove_empty_folders(full_seg_folder_path)

def main(args):
    convert_choi_to_by_seg_length(args.input)
    print('done')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', help='Path to choi dataset', required=True)
    main(parser.parse_args())