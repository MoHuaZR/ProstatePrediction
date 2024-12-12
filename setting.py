'''
Configs for training & testing
Written by Whalechen
'''

import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root',
        default='',
        type=str,
        help='Root directory path of data')
    parser.add_argument(
        '--excel_path',
        default='',
        type=str,
        help='patient_info_excel')    
    parser.add_argument(
        '--json_path',
        default='',
        type=str,
        help='json_path')    
    parser.add_argument(
        '--mixed_clinic',
        action='store_true',
        help='mix the clnical data')  
    parser.add_argument(
        '--focalloss',
        action='store_true',
        help='focal loss') 
    parser.add_argument(
        '--regualarization',
        action='store_true',
        help='loss regualarization') 
    parser.add_argument(
        '--weight_for_negative_class',
        default = 2.0,
        type=float,
        help='weight of negative') 
    parser.add_argument(
        '--clinic_dimension',
        default=6,
        type=int,
        help='the dimension of clnical data')  
    parser.add_argument(
        '--n_seg_classes',
        default=2,
        type=int,
        help="Number of segmentation classes"
    )
    parser.add_argument(
        '--if_transform',
        default=False,
        type=bool,
        help="transform or not"
    )
    parser.add_argument(
        '--learning_rate',  # set to 0.001 when finetune
        default=0.0001,
        type=float,
        help=
        'Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument(
        '--num_workers',
        default=16,
        type=int,
        help='Number of jobs')
    parser.add_argument(
        '--batch_size', default=200, type=int, help='Batch Size')
    parser.add_argument(
        '--phase', default='train', type=str, help='Phase of train or test')
    parser.add_argument(
        '--n_epochs',
        default=1000,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--validation_split',
        default=0.3,
        type=float,
        help='percentage of validation data')
    parser.add_argument(
        '--input_D',
    default=64,
        type=int,
        help='Input size of depth')
    parser.add_argument(
        '--input_H',
        default=64,
        type=int,
        help='Input size of height')
    parser.add_argument(
        '--input_W',
        default=64,
        type=int,
        help='Input size of width')
    parser.add_argument(
        '--resume_path',
        default= '',
        type=str,
        help= 'Path for resume model.'
    )
    parser.add_argument(
        '--pretrain_path',
        default='',
        type=str,
        help=
        'Path for pretrained model.'
    )
    parser.add_argument(
        '--new_layer_names',
        default= ['mlp'],
        type=list,
        help='New layer except for backbone')
    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument(
        '--gpu_id',
        nargs='+',
        type=int,              
        help='Gpu id lists')
    parser.add_argument(
        '--model',
        default='resnet',
        type=str,
        help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument(
        '--model_depth',
        default=18,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    parser.add_argument(  #3407
        '--manual_seed', default=3407, type=int, help='Manually set random seed')
    parser.add_argument(
        '--ci_test', action='store_true', help='If true, ci testing is used.')
    args = parser.parse_args()
    args.save_folder = "/depth_50_64_v1/{}_{}".format(args.model, args.model_depth)
    
    return args
