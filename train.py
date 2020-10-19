import os
import time
import numpy as np
import torch
from config import params
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from libs import slowfastnet
from libs.eval import evaluate
from libs.dataset import VideoDataset,collate_function
from libs.utils import Progbar,AverageMeter,read_annotations
from libs.transforms import create_train_transforms,create_val_transforms

def train(model, train_dataloader, current_epoch, criterion, optimizer, writer):
    batch_time = AverageMeter()
    progbar = Progbar(len(train_dataloader.dataset), stateful_metrics=['epoch', 'config', 'lr'])
    model.train()
    end = time.time()
    for step, (labels, imgs, img_paths) in enumerate(train_dataloader):
        optimizer.zero_grad()
        numm = imgs.shape[0]
        imgs = imgs.reshape((-1, imgs.shape[-4], imgs.shape[-3], imgs.shape[-2], imgs.shape[-1])).permute((0, 2, 1, 3, 4))
        imgs = Variable(imgs, requires_grad=True).cuda()
        labels = Variable(labels, requires_grad=False).cuda().reshape(-1)
        outputs = model(imgs).reshape(-1)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        progbar.add(numm, values=[('epoch', current_epoch),('train_loss', loss.item())])
        writer.add_scalar('train_loss_epoch', loss.item(), current_epoch * len(train_dataloader.dataset) // train_dataloader.batch_size + step)

def validation(model, val_dataloader, current_epoch, criterion, writer, run_type='val',model_dir='./checkpoint'):
    batch_time = AverageMeter()
    losses = AverageMeter()
    model.eval()
    progbar = Progbar(len(val_dataloader.dataset), stateful_metrics=['run-type'])
    end = time.time()
    with torch.no_grad():
        probs = np.array([])
        gt_labels = np.array([])
        names = []
        for step, (labels, imgs, img_paths) in enumerate(val_dataloader):
            names.extend(img_paths)
            imgs = imgs.cuda()
            outputs = model(imgs).reshape(-1).cpu()
            loss = criterion(outputs,labels.float())
            losses.update(loss)
            outputs = torch.sigmoid(outputs).cpu().numpy().reshape(-1)
            probs = np.concatenate((probs, outputs), axis=0)
            gt_labels = np.concatenate((gt_labels, labels.data.numpy().reshape(-1)), axis=0)
            assert gt_labels.shape == probs.shape

            progbar.add(imgs.size(0),
                        values=[('run-type', run_type), ('val_loss', loss.item())])  # ,('batch_time', batch_time.val)])

            batch_time.update(time.time() - end)
            end = time.time()
    metrixs = evaluate(gt_labels, probs > 0.5, probs)
    writer.add_scalar('val_loss_epoch', losses.avg, current_epoch)
    writer.add_scalar('val_f1',metrixs['f1'])
    print("Epoch: {} f1: {}".format(current_epoch, metrixs['f1']))
    torch.save({
        'epoch': current_epoch + 1,
        'state_dict': model.state_dict(),
    }, model_dir + '/model_last.pth.tar')



def main():
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = params['gpu']
    cudnn.benchmark = True
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    logdir = os.path.join(params['log'], cur_time)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    writer = SummaryWriter(log_dir=logdir)

    print("Loading dataset")
    train_dataloader = DataLoader(
        VideoDataset(annotations=read_annotations(params['train_set']),
                     transforms=create_train_transforms(params['size'])),
        batch_size=params['batch_size'],
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_function,
        num_workers=8,
        )
    val_dataloader = DataLoader(
        VideoDataset(annotations=read_annotations(params['val_set']),
                     transforms=create_val_transforms(params['size'])),
        batch_size=params['batch_size'] * 2,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_function,
        num_workers=8,
        drop_last=False
        )

    print("load model")
    model = slowfastnet.resnet50(class_num=1)
    
    if params['pretrained'] is not None:
        pretrained_dict = torch.load(params['pretrained'], map_location='cpu')
        try:
            model_dict = model.module.state_dict()
        except AttributeError:
            model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print("load pretrain model")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    model = model.cuda()
    #model = nn.DataParallel(model, device_ids=params['gpu'])  # multi-Gpu

    criterion = nn.BCEWithLogitsLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=params['momentum'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params['step'], gamma=0.1)

    model_save_dir = os.path.join(params['save_path'], cur_time)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    for epoch in range(params['epoch_num']):
        train(model, train_dataloader, epoch, criterion, optimizer, writer)
        if epoch % 2== 0:
            validation(model, val_dataloader, epoch, criterion, optimizer, writer)
        scheduler.step()
        if epoch % 1 == 0:
            checkpoint = os.path.join(model_save_dir,
                                      str(epoch) + ".pth.tar")
            torch.save(model.module.state_dict(), checkpoint)

    writer.close

if __name__ == '__main__':
    main()
