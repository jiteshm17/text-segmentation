import os
import subprocess
from timeit import default_timer as timer
import utils
from argparse import ArgumentParser

def main(input, output, jar_path, threshold, min_segment):
    # Create an output folder based on the threshold and min_segment
    output_folder = os.path.join(output, f'graphseg_output_{min_segment}_{threshold}')

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Setup logger
    logger = utils.setup_logger(__name__, os.path.join(output_folder, 'graphseg_timer.log'), delete_old=True)

    # Prepare the command
    cmd = ['java', '-jar', jar_path, input, output_folder, str(threshold), str(min_segment)]
    print(cmd)

    # Measure execution time
    start = timer()
    subprocess.call(cmd)  # Use subprocess to execute the command
    end = timer()

    # Log the results
    logger.info(f'Running with params: threshold={threshold}, min_segment={min_segment}')
    logger.info(f'Execution time (seconds): {end - start}')
    
    print(f'Execution time (seconds): {end - start}')
    print('Done')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', help='Input folder path',
                        default='/home/adir/Projects/data/wikipedia/wiki4_no_seperators', type=str)
    parser.add_argument('--output', help='Output folder path',
                        default='/home/adir/Projects/data/wikipedia/wiki4_output_graphseg/', type=str)
    parser.add_argument('--jar', help='Graphseg jar file path',
                        default='/home/adir/Projects/graphseg/binary/graphseg.jar', type=str)
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--min_segment', type=int, required=True)

    args = parser.parse_args()

    main(args.input, args.output, args.jar, args.threshold, args.min_segment)