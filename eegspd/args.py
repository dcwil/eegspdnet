import argparse

parser = argparse.ArgumentParser(description='EEG-SPD')

# dataset
parser.add_argument('-d', '--dataset', type=str, required=True)
parser.add_argument('-mo', '--mode', choices=['valid', 'eval'])

# essential
parser.add_argument('-lr', '--learning-rate', type=float, dest='lr', required=True)
parser.add_argument('-wd', '--weight-decay', type=float, dest='wd', required=True)
parser.add_argument('-nf', '--n-filters', type=int, required=False)  # required if model is not FBSPDNet
parser.add_argument('-m', '--model', type=str, required=True)
parser.add_argument('-fbp', '--fb-path', type=str, required=False)  # required if model is FBSPDNet
parser.add_argument('-o', '--optim', choices=['Adam', 'RiemannianAdam'], default='RiemannianAdam')

parser.add_argument('-s', '--seeds', nargs='+', default=[114, 236, 7934])
parser.add_argument('-n', '--n-epochs', type=int, default=1000)
parser.add_argument('-ck', '--checkpoints', nargs='+', default=[100, 500])

parser.add_argument('--results-dir', required=True, type=str)

# non-essential
parser.add_argument('--dropout', dest='final_layer_drop_prob', type=float, default=0)
parser.add_argument('--bimap-sizes-k', type=float, default=2)
parser.add_argument('--bimap-sizes-n', type=int, default=3)

# analyses
parser.add_argument('--run-analyses', action='store_true')
parser.add_argument('--post-hoc-analysis', action='store_true')
parser.add_argument('--exclude-analyses', nargs='+', default=list())
# TODO: add arguments for specifying subset of analyses
