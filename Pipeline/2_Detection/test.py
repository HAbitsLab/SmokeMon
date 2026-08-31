import argparse
import collections

import pandas as pd
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser

import torch.nn as nn

from torchvision.models.mobilenetv2 import mobilenet_v2

#from model.mobilenet_v2_3d import get_model
from model.mobilenet_v2_3d_slowfast import get_model
from model.resnet_3d import resnet18



def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        p_list=[config["data_loader"]['args']['p_list']],
        label_type= "label",
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=8,
        GT_sessions = False,
        #transform = True,
        temporal_window=31,
    )


    # build model architecture

    # --------------------#
    #      ResNet18      #
    # --------------------#
    #model = resnet18(sample_size=32,sample_duration=3, num_classes=1)

    # Thermal images have one channel. Replacing the first 3d convolution to match the number of channels
    #model.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)

    # --------------------#
    #    MobileNetV2 3D   #
    # --------------------#
    model= get_model(num_classes=1)

    # Thermal images have one channel. Replacing the first 3d convolution to match the number of channels
    #model.features[0] = nn.Conv3d(1, 32, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1), bias=False)

    # --------------------#
    # Read from config file#
    # --------------------#
    # model = config.init_obj('arch', module_arch)

    # --------------------#
    #    MobileNetV2      #
    # --------------------#
    # model = mobilenet_v2(num_classes=1)

    # Thermal images have one channel. Replacing the first 2d convolution to match the number of channels
    # model.features[0] = nn.Conv2d(11, 32, 3, 3)

    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    all_preds = {"timestamp":[],"pred":[], "pred_sig":[],"b_label":[],"labels":[]}

    with torch.no_grad():
        for i, (data, target, b_label, p_timestamp) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            m = nn.Sigmoid()
            pred = m(output)

            pred = pred.flatten()
            pred[pred > 0.5] = 1
            pred[pred <= 0.5] = 0

            #
            # save sample images, or do something with output here
            #
            #pred = torch.argmax(output, dim=1)
            # pred[pred == 2] = 1
            # target[target == 2] = 1
            all_preds["timestamp"] += p_timestamp[1].cpu().tolist()
            all_preds["labels"] += target.cpu().tolist()
            all_preds["pred"] += output.cpu().tolist()
            all_preds["pred_sig"] += pred.cpu().tolist()
            all_preds["b_label"] += b_label.cpu().tolist()


            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size




    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: met(torch.tensor(all_preds["pred"]), torch.tensor(all_preds["labels"])) for i, met in enumerate(metric_fns)
    })

    # calculating baseline metrics
    log.update({
        "b_" + met.__name__: met(torch.tensor(all_preds["b_label"]), torch.tensor(all_preds["labels"])) for i, met in enumerate(metric_fns)
    })

    logger.info(log)



    results = pd.DataFrame(all_preds)
    results = results.set_index("timestamp",drop=True)
    results.to_csv(config.resume.parent / "results.csv")




if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['-p', '--participant'], type=str, target='data_loader;args;p_list'),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
