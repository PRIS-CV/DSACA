import argparse

parser = argparse.ArgumentParser(description='DSACA for regression based-method')

parser.add_argument('--workers', type=int, default=4,
                    help='load data workers')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='start epoch for training')
parser.add_argument('--batch_size', type=int, default=1,
                    help='input batch size for training')
parser.add_argument('--lr', type=float, default=1e-5 * 1,
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=20000,
                    help='number of epochs to train')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--best_pred', type=int, default=1e5,
                    help='best pred')
parser.add_argument('--lr_step', type=list, default=[70,90],
                    help='drop learning rate by 10')
parser.add_argument('--max_epoch', type=int, default=300,
                    help='which epoch to test only')
parser.add_argument('--lamd', type=float, default=1.0,
                    help='Cross Entropy rate')
parser.add_argument('--weight_decay', type=float, default=5 * 1e-4,
                    help='weight decay')
parser.add_argument('--print_freq', type=int, default=100,
                    help='print frequency')

# user
###################################################################################
parser.add_argument('--train_dataset', type=str, default='VisDrone',
                    help='choice train dataset')
parser.add_argument('--pre', type=str, default='./pre_trained/VisDrone_class8.pth',
                    help='pre-trained model directory')
parser.add_argument('--task_id', type=str, default='./save_file_model/VisDrone_class8',
                    help='save checkpoint directory')
parser.add_argument('--test_dataset', type=str, default='VisDrone',
                    help='choice train dataset')
parser.add_argument('--gpu_id', type=str, default='1',
                    help='gpu id')
###################################################################################

args = parser.parse_args()
