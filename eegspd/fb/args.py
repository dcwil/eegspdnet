import argparse

parser = argparse.ArgumentParser(description='FB Search')
parser.add_argument('--results-dir', required=True, type=str)

# dataset
parser.add_argument('-d', '--dataset', type=str, required=True)
parser.add_argument('-mo', '--mode', choices=['valid', 'eval'])

parser.add_argument('-s', '--seeds', nargs='+', default=[114, 236, 7934])

parser.add_argument('--n-filters', '-nf', required=True, type=int)
parser.add_argument('--num-samples', '-n', required=True, type=int)
parser.add_argument('--metric', '-m', default='logeuclid', choices=['euclid', 'logeuclid', 'riemann'])
parser.add_argument('--classifier', '-clf', default='mdm', choices=['svm_linear', 'mdm'])
parser.add_argument('--time-budget-s', '-t', default=43200, type=int)
parser.add_argument('--fb-type', default='spec', choices=['spec', 'ind'])
parser.add_argument('--search-alg', default='bayes', choices=['bayes', 'hyperopt'])
parser.add_argument('--cov-estimator', default='scm', choices=['scm', 'oas'])
parser.add_argument('--spacing', default='rand', choices=['rand', 'mel', 'hz'])
parser.add_argument('--remove-interband', action='store_true')
parser.add_argument('--keep-inter-elec', action='store_true')

