import argparse
import torch.optim as optim

optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}

class Config(object):
    def __init__(self, kwargs):
        for key, value in kwargs.items():
            if key=='optimizer':
                value=optimizer_dict[value]

            setattr(self, key, value)

def get_config(**optional_kwargs):
    parser = argparse.ArgumentParser()

    parser.add_argument('--granularity', type=int, default=48)
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--dataset_dir", type=str, default=r'/mnt/data/DEAP')
    parser.add_argument('--save_path', type=str, default=r'/mnt/CONIRepo/fusion/dapkls/')
    parser.add_argument('--batch_size',type=int, default=128)
    parser.add_argument('--label_type', type=int, default=0, help='0: valence, 1: arousal')
    parser.add_argument('--model_name', type=str, default='DGCNN')
    parser.add_argument('--gpu_no', type=int,default=0)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--in_channels', type=int, default=128)
    parser.add_argument('--num_electrodes', type=int, default=32)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--learning_rate',type=float,default=1e-4)
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--num_epochs',type=int, default=500)
    parser.add_argument('--da', action='store_true', help='apply domain adaptation')
    parser.add_argument('--num_nodes', type=int, default=170)

    args = parser.parse_args()
    args = vars(args)
    args.update(optional_kwargs)

    return Config(args)