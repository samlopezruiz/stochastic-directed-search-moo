import getopt
import sys


def get_input_args():
    input_args = {'model_ix': 2}

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'x', ['model_ix='])
    except getopt.GetoptError:
        print('valid options are: -x <model_ix>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-x", "--model_ix"):
            input_args['model_ix'] = int(arg)

    return input_args
