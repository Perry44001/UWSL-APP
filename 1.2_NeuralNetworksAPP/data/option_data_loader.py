import argparse
# Define hyper parameters for debugging dataloader
parser = argparse.ArgumentParser(description='Option for Data loader')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
parser.add_argument('--file_path', type=str, default='1.1.3.1 Training Set/(1) One environment training set'
                    , help='Loading folder name')
parser.add_argument('--length_freq', type=int, default=151, help='Length of frequency vector')
parser.add_argument('--num_of_receiver', type=int, default=19, help='Length of receiver')
parser.add_argument('--SNR_range', type=list, default=[10, 15], help='SNR range for training')
parser.add_argument('--num_read_sources', type=int, default=1024, help='Number of reading sources in one go')
# parser.add_argument('--Sr', type=list, default=[10], help='Simulated data at source range Sr for test (km)')
# parser.add_argument('--Sd', type=list, default=[10], help='Simulated data at source depth Sd for test (m)')
parser.add_argument('--SNR', type=float, default=10, help='SNR for test in simulated data')
parser.add_argument('--i_file', type=int, default=0, help='Index of reading file for simulated data')
parser.add_argument('--run_mode', type=str, default='train',
                    help='Run mode (train, test on simulated data set or measured data')
parser.add_argument('--model', type=str, default='mtl_unet', help='CNNs model name (mtl_cnn, mtl_unet, xception)')
args_d = parser.parse_args()

for arg in vars(args_d):
    if vars(args_d)[arg] == 'True':
        vars(args_d)[arg] = True
    elif vars(args_d)[arg] == 'False':
        vars(args_d)[arg] = False