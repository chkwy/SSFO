
import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model-id', type=str, default='', help='model id')
    parser.add_argument('--gpu_num', type=int, default=8, help='GPU number')
    # dataset path
    parser.add_argument(
        '--data-folder',
        type=str,
        default='data',
        help='path to the datafolder of ChatRAG Bench')
    parser.add_argument(
        '--output-folder',
        type=str,
        default='outputs',
        help='path to the datafolder of ChatRAG Bench')
    parser.add_argument('--eval-dataset', type=str, default='')
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.6,
        help='temperature for sampling')
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.9,
        help='top-p sampling parameter')

    # others
    parser.add_argument('--out-seq-len', type=int, default=64)
    parser.add_argument('--num-ctx', type=int, default=5)
    parser.add_argument('--max-tokens', type=int, default=64)

    args = parser.parse_args()

    return args
