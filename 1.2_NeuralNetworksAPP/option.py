import argparse

parser = argparse.ArgumentParser(description='Sonar Target Detection Based on Deep Neural Networks')

# data specifications
parser.add_argument('--data_loader_path', type=str, default='E:\2.0_UWSL_Datasets\test', help='dataset path')

parser.add_argument('--cpu', action='store_true', help='use cpu only')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers to load data')
parser.add_argument('--rest_time', type=int, default=0, help='time(s) to rest')
parser.add_argument('--seed', type=int, default=0, help='Random seeds')
parser.add_argument('--num_of_sources', type=int, default=32, help='Number of reading sources at once')

# model specifications
parser.add_argument('--model', type=str, default='mtl_cnn', help='model name')  # which model to use.

# training specifications
parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training')
parser.add_argument('--epoch', type=int, default=10, help='Maximum number of training cycles')
parser.add_argument('--test_only', type=str, default='False', help='set this option to test the model')
parser.add_argument('--plot_only', type=str, default='False', help='set this option to plot the model loss')
parser.add_argument('--exp_only', type=str, default='False', help='set this option to exp the model')
parser.add_argument('--load_model_path', type=str, default='mtl_cnn_epoch_0.20', help='loading weight file name')

# optimization specifications
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')

# log specification
parser.add_argument('--save_file', type=str, default='-', help='file name to save')
parser.add_argument('--load', type=str, default='.', help='file name to load')
parser.add_argument('--resume', type=str, default='False', help='resume from specific checkpoint')
parser.add_argument('-f', type=str, default="Read additional parameters")

args = parser.parse_args()

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
