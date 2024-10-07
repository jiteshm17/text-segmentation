import json

# Define the configuration data
jsondata = {
    "word2vecfile": "/Users/jitesh/Downloads/text-segmentation/data/word2vec/GoogleNews-vectors-negative300.bin",
    "choidataset": "/home/omri/code/text-segmentation-2017/data/choi",
    "wikidataset": "/home/omri/datasets/wikipedia/process_dump_r"
}

# Write the data to config.json
with open('config.json', 'w') as f:
    json.dump(jsondata, f, indent=4)  # Added indent for better readability