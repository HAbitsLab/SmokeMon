import argparse
import torch
import os
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from logger import TensorboardWriter
import torchvision
import numpy as np
import pandas as pd

def metric_helper(output, target):
    with torch.no_grad():
        output_bin=((output.reshape(-1))>0.5)*1
        print(output_bin.sum())
        target_bin=((target.reshape(-1))>0.5)*1
        print(target_bin.sum())
        #error count
        err=(torch.logical_xor(output_bin, target_bin)*1).sum()
        #false negative count
        falseNeg=np.array([1 for i in range(target_bin.shape[0]) if target_bin[i]==1 and output_bin[i]==0]).sum()
        #total predictions
        total=target_bin.shape[0]
        #total cases labeled positive
        truth=target_bin.sum()
    return total, err, falseNeg, truth

def Precision(output, target):
    total, err, falseNeg, truth=metric_helper(output, target)
    falsePos=err-falseNeg
    return (truth-falseNeg)/(truth-falseNeg+falsePos)*100 if (truth-falseNeg+falsePos) != 0 else 0

def Recall(output, target):
    total, err, falseNeg, truth=metric_helper(output, target)
    falsePos=err-falseNeg
    # print(total, err, falseNeg, truth)
    return (truth-falseNeg)/(truth)*100 if truth != 0 else 0
def F1(output, target):
    p=precision(output, target)
    r=recall(output, target)
    return 2*p*r/(p+r) if (p+r) != 0 else 0


def main(config):
    logger = config.get_logger('test')
    writer = TensorboardWriter(os.path.join(config.log_dir,'test/'), logger, True)
    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['in_lab_dir'],
        config['data_loader']['args']['in_wild_dir'],
        config['data_loader']['args']['in_lab_test_id'],
        config['data_loader']['args']['in_wild_test_id'],
        batch_size=64,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )
    
    #prepare path for filtering_reversal
    in_wild_list=['P0', 'P1', 'P2', 'P3', 'P4']
    data_dir_path=os.path.join(config['data_loader']['args']['in_wild_dir'], in_wild_list[config['data_loader']['args']['in_wild_test_id']], 'clean')
    

    # build model architecture
    model = config.init_obj('arch', module_arch)
    #logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    
    #model.load_state_dict(state_dict)
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    #total_metrics = torch.zeros(len(metric_fns))
    total = 0
    err = 0
    falseNeg = 0
    truth = 0
    ac_positive = 0
    ac_negative = 0
    
    y_list=torch.tensor([])
    y_hat_list=torch.tensor([])
    index_list=torch.tensor([])
    with torch.no_grad():
        for i, (data, target, index) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            #mimic real-life behavior: preprocessing of data before inputting into model
            #frames with max temperature lower than 40C are discarded
            
            y_list=torch.cat([y_list, target.view(-1).cpu()])
            y_hat_list=torch.cat([y_hat_list, output.view(-1).cpu()])
            index_list=torch.cat([index_list, index.cpu()])
            
            #custom tracker
            target_bin=(target>0.5)*1
            output_bin=(output>0.3)*1
            
            ##writer writing to tensorboard
            #writer.set_step(i, 'test')
            #fn_list=[i for i in range(output_bin.shape[0]) if target_bin[i]==1 and output_bin[i]==0]
            #if len(fn_list)>0:
            #    fn_img_grid = torchvision.utils.make_grid((data.cpu())[fn_list, :494].reshape(-1, 1, 19, 26), nrow=4)
            #    writer.add_image('false negative images', fn_img_grid)
            #fp_list=[i for i in range(output_bin.shape[0]) if target_bin[i]==0 and output_bin[i]==1]
            #if len(fp_list)>0:
            #    fp_img_grid = torchvision.utils.make_grid((data.cpu())[fp_list, :494].reshape(-1, 1, 19, 26), nrow=4)
            #    writer.add_image('false positive images', fp_img_grid)
            
            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            #for i, metric in enumerate(metric_fns):
            #    total_metrics[i] += metric(output_bin.to('cpu'), target_bin.to('cpu')) * batch_size
            to, e, fn, tr = module_metric.metric_helper(output_bin.to('cpu'), target_bin.to('cpu'))
            total += to
            err += e
            falseNeg += fn
            truth += tr
            ac_positive += len([1 for i in target if i>0.5])/1.0
            ac_negative += len([1 for i in target if i<0.5])/1.0

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    
    #original
    falsePos=err-falseNeg
    precision = ((truth-falseNeg)/(truth-falseNeg+falsePos)*100).item() if (truth-falseNeg+falsePos) != 0 else 0
    recall = ((truth-falseNeg)/(truth)*100).item() if truth != 0 else 0
    print("With in filtered data")
    log.update({
        #met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
        "precision": precision,
        "recall": recall,
        "f1": (2*precision*recall/(precision+recall)) if (precision+recall) != 0 else 0,
        "actual_positive": ac_positive,
        "actual_negative": ac_negative
    })
    logger.info(log)
    
    def tensor2CSV(y, y_hat, index, name):
        index=np.array(index[:,0].int(), dtype='int64')*10000000+np.array(index[:,1].int(), dtype='int64')
        y_data = pd.DataFrame(index, columns=['index'])
        y_data['y'] = y
        y_data['y_hat']=y_hat
        y_data.to_csv(os.path.join(config.log_dir,'test/')+name+'.csv')
        return y_data
        
    label=tensor2CSV(y_list, y_hat_list, index_list, "latest")
    
    #for filter reversal
    labelDense=pd.DataFrame()
    all_label=label[(label['y_hat']>0.3)]
    data_dir_list= np.sort([i for i in os.listdir(data_dir_path) if i[0].isdigit()])
    #generate concatenated label
    for data_dir in data_dir_list:
            
        if not data_dir.startswith('.'):
            final_path = os.path.join(data_dir_path, data_dir)
            #print(f'Loading from path {final_path}')
            label_name = [i for i in os.listdir(final_path) if i.startswith('CP')][0]
                
            #Data inversion for all subjects not P0
            labels_ts = pd.read_csv(os.path.join(final_path, label_name), index_col=0).iloc[10:]
            labelDense=pd.concat([labelDense, labels_ts])
            
    #generate all cases
    partial_label=all_label
    #partial_label=partial_label.drop(['Unnamed: 0', 'y', 'y_hat'], axis=1)
    labelDense=labelDense.reset_index()
    label_expand=np.array([1 if i in set(partial_label['index']) else 0 for i in labelDense['0'] ])
    label_expand2=np.array([0 if label_expand[i]==0 or (label_expand[i-2:i+1].sum()<3 and label_expand[i-1:i+2].sum()<3 and label_expand[i:i+3].sum()<3) else 1 for i in range(len(label_expand))])
    labelDense['pred']=label_expand2
    precision=Precision(torch.tensor(labelDense['pred'].values), torch.tensor(labelDense['label'].values))
    recall=Recall(torch.tensor(labelDense['pred'].values), torch.tensor(labelDense['label'].values))
    print("With all data")
    log.update({
        #met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
        "precision": precision,
        "recall": recall,
        "f1": (2*precision*recall/(precision+recall)) if (precision+recall) != 0 else 0,
        "actual_positive": ac_positive,
        "actual_negative": ac_negative
    })
    logger.info(log)
    


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
