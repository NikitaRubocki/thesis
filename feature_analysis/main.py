import argparse
import json
import logging as log
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from feature_scorer import FeatureScorer


def get_args():

    parser = argparse.ArgumentParser('Feature Analysis Program',
                                     description='A program that outputs importance scores given a dataset. '
                                                 'IMPORTANT: Target label must be the FIRST column in the dataset.')

    parser.add_argument('data', type=str,
                        help='CSV data file to run the analysis on (with target label as the first column)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show progress messages as the program runs')
    parser.add_argument('-f', '--fast', dest='fast', action='store_true',
                        help='Run a quicker version of the ensemble (excludes the slowest analyses) if specified')
    parser.add_argument('-t', dest='type', metavar='task_type', choices=['classification', 'regression'], required=True,
                        help='The type of task the dataset represents (regression or classification)')
    parser.add_argument('-j', dest='json', metavar='json_output', type=str, default=False,
                        help='Output the rankings in a json file if a file name is passed in')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    if args.verbose:
        log.basicConfig(format='%(message)s', level=log.INFO)

    # load csv file and encode labels if categorical
    log.info('Reading in csv file...')
    df = pd.read_csv(args.data)
    if args.type == 'classification':
        df.iloc[:, 0] = LabelEncoder().fit_transform(df.iloc[:, 0])
    log.info('File reading complete.')

    # feature scoring, optionally save to json
    log.info('\nStarting feature scoring (this may take a while)...')
    scores = FeatureScorer(df, args.type, args.fast).importance_scores()
    scores = {key: dict(sorted(val.items(), key=lambda x: x[1], reverse=True))
              for key, val in scores.items()}
    log.info('Feature scoring complete.')
    if args.json is not False:
        with open(args.json, 'w') as fp:
            json.dump(scores, fp, indent=4)
            log.info(f'Importance scores have been generated and output to {args.json}')
    else:
        print('Feature Scores:')
        print(json.dumps(scores, indent=4))
