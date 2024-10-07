from pathlib import Path
import os

# Define paths and settings
root = '/home/adir/Projects/text-segmentation-2017/data/choi/1/3-5'
output = '/home/adir/Projects/text-segmentation-2017/data/part_choi/'
delimiter = '=========='
truth = '********************************************'

# Get all .ref files recursively from the root directory
textfiles = list(Path(root).rglob('*.ref'))

counter = 0

# Iterate over all text files
for file in textfiles:
    counter += 1
    with file.open('r', encoding='utf-8') as f:
        raw_text = f.read()

    # Replace the old delimiter with the new "truth" separator
    new_text = raw_text.replace(delimiter, truth)

    # Create a new file path for the modified content
    new_file_path = os.path.join(output, f"{counter}_{file.name}")

    # Write the new content to the new file
    with open(new_file_path, "w", encoding='utf-8') as f:
        f.write(new_text)

print('done')