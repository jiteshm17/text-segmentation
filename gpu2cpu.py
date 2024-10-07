import torch
from argparse import ArgumentParser
from pathlib import Path

def main(args):
    input_path = Path(args.input)
    
    # Load the model from the input file (in binary mode)
    with input_path.open('rb') as f:
        model = torch.load(f, map_location=torch.device('cpu'))  # Ensure loading to CPU

    model = model.cpu()  # Ensure the model is on CPU

    # Determine the output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / (input_path.stem + '_cpu' + input_path.suffix)

    # Save the CPU model to the output file
    with output_path.open('wb') as f:
        torch.save(model, f)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to original model file', required=True)
    parser.add_argument('-o', '--output', help='Output path for the CPU model')
    args = parser.parse_args()

    main(args)