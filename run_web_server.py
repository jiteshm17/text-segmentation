from argparse import ArgumentParser
from utils import config, read_config_file
from webapp import app

def main(args):
    # Read configuration from the config file
    read_config_file(args.config)
    config.update(vars(args))  # Use vars(args) to convert argparse.Namespace to a dictionary

    # Run the web server
    app.run(debug=True, port=args.port)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', help='Is cuda?', action='store_true')
    parser.add_argument('--model', help='Model file path', required=True)
    parser.add_argument('--config', help='Path to config.json', default='config.json')
    parser.add_argument('--test', help='Use fake word2vec', action='store_true')
    parser.add_argument('--port', type=int, help='Port to listen on', default=5000)

    args = parser.parse_args()

    main(args)