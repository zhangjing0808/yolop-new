import argparse
import os, sys
import math
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import pprint
import time
import torch
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import amp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np
from lib.utils import DataLoaderX, torch_distributed_zero_first
from tensorboardX import SummaryWriter

import lib.dataset as dataset
from lib.config import cfg
from lib.config import update_config
from lib.core.loss import get_loss
from lib.core.function import train
from lib.core.function import validate
from lib.core.general import fitness
from lib.models import get_net
from lib.utils import is_parallel
from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger, select_device
from lib.utils import run_anchor


def parse_args():  # 定义解析命令行参数的函数
    parser = argparse.ArgumentParser(description='Train Multitask network')  # 创建一个 ArgumentParser 对象，描述为训练多任务网络

    # general
    # parser.add_argument('--cfg',
    #                     help='experiment configure file name',
    #                     required=True,
    #                     type=str)

    # philly
    parser.add_argument('--modelDir',  # 添加模型目录参数
                        help='model directory',  # 参数说明
                        type=str,  # 参数类型为字符串
                        default='')  # 默认值为空字符串
    parser.add_argument('--logDir',  # 添加日志目录参数
                        help='log directory',  # 参数说明
                        type=str,  # 参数类型为字符串
                        default='runs/')  # 默认值为 'runs/' 目录
    parser.add_argument('--dataDir',  # 添加数据目录参数
                        help='data directory',  # 参数说明
                        type=str,  # 参数类型为字符串
                        default='')  # 默认值为空字符串
    parser.add_argument('--prevModelDir',  # 添加前一个模型目录参数
                        help='prev Model directory',  # 参数说明
                        type=str,  # 参数类型为字符串
                        default='')  # 默认值为空字符串

    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')  # 添加一个布尔参数，用于指示是否使用同步批量归一化（SyncBatchNorm），仅在分布式数据并行（DDP）模式下可用
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')  # 添加一个整数参数，用于指定当前进程的本地排名，默认为 -1
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')  # 添加一个浮点数参数，用于设置对象置信度阈值，默认为 0.001
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')  # 添加一个浮点数参数，用于设置 NMS 的 IOU 阈值，默认为 0.6

    args = parser.parse_args()  # 解析命令行参数并将结果存储在 args 对象中

    return args  # 返回解析后的参数对象


def main():
    # set all the configurations
    args = parse_args()  # 解析命令行参数
    update_config(cfg, args)  # 更新配置对象 cfg

    # Set DDP variables
    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1  # 从环境变量中获取 WORLD_SIZE，如果不存在则默认为 1
    global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1  # 从环境变量中获取 RANK，如果不存在则默认为 -1

    rank = global_rank  # 将全局排名赋值给 rank 变量
    #print(rank)
    # TODO: handle distributed training logger
    # set the logger, tb_log_dir means tensorboard logdir

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, cfg.LOG_DIR, 'train', rank=rank)

    if rank in [-1, 0]:
        logger.info(pprint.pformat(args))  # 打印命令行参数
        logger.info(cfg)  # 打印配置信息

        writer_dict = {
            'writer': SummaryWriter(log_dir=tb_log_dir), # 创建 TensorBoard 写入器
            'train_global_steps': 0,  # 训练全局步骤计数器初始化为 0
            'valid_global_steps': 0,  # 验证全局步骤计数器初始化为 0
        }
    else:  # 如果当前进程不是主进程
        writer_dict = None  # 不创建 TensorBoard 写入器

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # bulid up model
    # start_time = time.time()
    print("begin to bulid up model...")
    # DP mode
    device = select_device(logger, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU* len(cfg.GPUS)) if not cfg.DEBUG \
        else select_device(logger, 'cpu')

    if args.local_rank != -1:
        assert torch.cuda.device_count() > args.local_rank
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
    
    print("load model to device")
    model = get_net(cfg).to(device)# 根据配置获取模型并将其加载到指定设备上
    # print("load finished")
    # model = model.to(device)
    # print("finish build model")
    

    # define loss function (criterion) and optimizer
    criterion = get_loss(cfg, device=device) # 获取损失函数，传入配置对象和设备对象
    optimizer = get_optimizer(cfg, model) # 获取优化器，传入配置对象和模型对象


    # load checkpoint model
    best_perf = 0.0  # 初始化最佳性能指标为 0.0
    best_model = False  # 初始化最佳模型标志为 False
    last_epoch = -1  # 初始化最后一个epoch编号为 -1，表示尚未开始训练

    Encoder_para_idx = [str(i) for i in range(0, 17)]  # 编码器部分的参数索引列表
    Det_Head_para_idx = [str(i) for i in range(17, 25)]  # 检测头部分的参数索引列表
    Da_Seg_Head_para_idx = [str(i) for i in range(25, 34)]  # 驾驶区域分割头部分的参数索引列表
    Ll_Seg_Head_para_idx = [str(i) for i in range(34,43)]  # 车道线分割头部分的参数索引列表

    # 定义学习率衰减函数 / 使用余弦退火策略，该策略在训练过程中逐渐减小学习率
    lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * \
                   (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF  # cosine
    # 创建学习率调度器 / LambdaLR 允许使用 lambda 函数来调整学习率
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # 设置开始训练的初始 / epoch 编号 如果配置中指定了开始 epoch，则使用该值；否则，从 0 开始
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH

    if rank in [-1, 0]:  # 如果当前进程是主进程（rank -1 表示本地模式，rank 0 表示分布式模式的第一个进程）
        checkpoint_file = os.path.join(  # 构建检查点文件的完整路径
            os.path.join(cfg.LOG_DIR, cfg.DATASET.DATASET), 'checkpoint.pth'
        )
        if os.path.exists(cfg.MODEL.PRETRAINED):  # 检查预训练模型文件是否存在
            logger.info("=> loading model '{}'".format(cfg.MODEL.PRETRAINED)) # 记录加载预训练模型的信息
            checkpoint = torch.load(cfg.MODEL.PRETRAINED)  # 加载预训练模型的检查点
            begin_epoch = checkpoint['epoch']  # 获取检查点中的 epoch 编号
            # best_perf = checkpoint['perf']
            last_epoch = checkpoint['epoch']  # 获取检查点中的 epoch 编号
            model.load_state_dict(checkpoint['state_dict'])  # 加载模型的状态字典
            optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器的状态字典
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(  # 记录加载检查点的信息
                cfg.MODEL.PRETRAINED, checkpoint['epoch']))
            #cfg.NEED_AUTOANCHOR = False     #disable autoanchor
        
        if os.path.exists(cfg.MODEL.PRETRAINED_DET):  # 检查预训练的检测分支模型文件是否存在
            logger.info("=> loading model weight in det branch from '{}'".format(cfg.MODEL.PRETRAINED))  # 记录加载预训练检测分支模型的信息
            det_idx_range = [str(i) for i in range(0,25)]  # 创建一个包含检测分支参数索引的列表
            model_dict = model.state_dict()  # 获取模型当前的状态字典
            checkpoint_file = cfg.MODEL.PRETRAINED_DET  # 获取预训练检测分支模型的文件路径
            checkpoint = torch.load(checkpoint_file)  # 加载预训练检测分支模型的检查点
            begin_epoch = checkpoint['epoch']  # 获取检查点中的 epoch 编号
            last_epoch = checkpoint['epoch']  # 获取检查点中的 epoch 编号
            checkpoint_dict = {k: v for k, v in checkpoint['state_dict'].items() if k.split(".")[1] in det_idx_range}  # 从检查点中筛选出检测分支的参数
            model_dict.update(checkpoint_dict)  # 将检测分支的参数更新到模型的状态字典中
            model.load_state_dict(model_dict)  # 将更新后的状态字典加载到模型中
            logger.info("=> loaded det branch checkpoint '{}' ".format(checkpoint_file))  # 记录加载检测分支检查点的信息
        
        if cfg.AUTO_RESUME and os.path.exists(checkpoint_file): # 如果配置中设置了自动恢复，并且检查点文件存在
            logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            begin_epoch = checkpoint['epoch']
            # best_perf = checkpoint['perf']
            last_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer = get_optimizer(cfg, model)
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                checkpoint_file, checkpoint['epoch']))
            #cfg.NEED_AUTOANCHOR = False     #disable autoanchor
        # model = model.to(device)

        if cfg.TRAIN.SEG_ONLY:  # Only train two segmentation branchs
            logger.info('freeze encoder and Det head...')
            for k, v in model.named_parameters():
                v.requires_grad = True  # train all layers
                if k.split(".")[1] in Encoder_para_idx + Det_Head_para_idx:
                    print('freezing %s' % k)
                    v.requires_grad = False

        if cfg.TRAIN.DET_ONLY:  # Only train detection branch
            logger.info('freeze encoder and two Seg heads...')
            # print(model.named_parameters)
            for k, v in model.named_parameters():
                v.requires_grad = True  # train all layers
                if k.split(".")[1] in Encoder_para_idx + Da_Seg_Head_para_idx + Ll_Seg_Head_para_idx:
                    print('freezing %s' % k)
                    v.requires_grad = False

        if cfg.TRAIN.ENC_SEG_ONLY:  # Only train encoder and two segmentation branchs
            logger.info('freeze Det head...')
            for k, v in model.named_parameters():
                v.requires_grad = True  # train all layers 
                if k.split(".")[1] in Det_Head_para_idx:
                    print('freezing %s' % k)
                    v.requires_grad = False

        if cfg.TRAIN.ENC_DET_ONLY or cfg.TRAIN.DET_ONLY:    # Only train encoder and detection branchs
            logger.info('freeze two Seg heads...')
            for k, v in model.named_parameters():
                v.requires_grad = True  # train all layers
                if k.split(".")[1] in Da_Seg_Head_para_idx + Ll_Seg_Head_para_idx:
                    print('freezing %s' % k)
                    v.requires_grad = False


        if cfg.TRAIN.LANE_ONLY: 
            logger.info('freeze encoder and Det head and Da_Seg heads...')
            # print(model.named_parameters)
            for k, v in model.named_parameters():
                v.requires_grad = True  # train all layers
                if k.split(".")[1] in Encoder_para_idx + Da_Seg_Head_para_idx + Det_Head_para_idx:
                    print('freezing %s' % k)
                    v.requires_grad = False

        if cfg.TRAIN.DRIVABLE_ONLY:
            logger.info('freeze encoder and Det head and Ll_Seg heads...')
            # print(model.named_parameters)
            for k, v in model.named_parameters():
                v.requires_grad = True  # train all layers
                if k.split(".")[1] in Encoder_para_idx + Ll_Seg_Head_para_idx + Det_Head_para_idx:
                    print('freezing %s' % k)
                    v.requires_grad = False
        
    if rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=cfg.GPUS)
        # model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    # # DDP mode
    if rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank,find_unused_parameters=True)


    # assign model params
    model.gr = 1.0  # 为模型的某个属性赋值，可能是比例因子或缩放因子
    model.nc = 1  # 为模型的另一个属性赋值，可能是类别数量或通道数量
    # print('bulid model finished')

    print("begin to load data")
    # Data loading
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = eval('dataset.' + cfg.DATASET.DATASET)( # 动态创建数据集实例，类名由配置中的 DATASET 参数确定
        cfg=cfg,  # 传递配置对象
        is_train=True,  # 指定这是训练数据集
        inputsize=cfg.MODEL.IMAGE_SIZE,  # 指定输入图像的大小
        transform=transforms.Compose([ # 创建一个变换序列
            transforms.ToTensor(),  # 将图像转换为 PyTorch 张量
            normalize,  # 应用标准化变换
        ])
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if rank != -1 else None # 如果处于分布式训练环境中，创建分布式采样器

    train_loader = DataLoaderX( # 创建自定义数据加载器 DataLoaderX 实例，用于训练
        train_dataset,  # 使用上面创建的训练数据集
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),  # 每个 GPU 的批次大小乘以 GPU 数量
        shuffle=(cfg.TRAIN.SHUFFLE & rank == -1),  # 如果配置要求打乱数据且不在分布式环境中，则打乱数据
        num_workers=cfg.WORKERS,  # 设置加载数据的工作线程数
        sampler=train_sampler,  # 使用分布式采样器，如果处于分布式环境
        pin_memory=cfg.PIN_MEMORY,  # 如果配置要求，将数据加载到锁定的内存中，以加快数据传输到 GPU 的速度
        collate_fn=dataset.AutoDriveDataset.collate_fn  # 使用自定义的 collate 函数来处理数据批次
    )
    num_batch = len(train_loader) # 计算训练数据加载器中的批次总数

    if rank in [-1, 0]:
        valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
            cfg=cfg,
            is_train=False,
            inputsize=cfg.MODEL.IMAGE_SIZE,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )

        valid_loader = DataLoaderX(
            valid_dataset,
            batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
            shuffle=False,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            collate_fn=dataset.AutoDriveDataset.collate_fn
        )
        print('load data finished')
    
    if rank in [-1, 0]:
        if cfg.NEED_AUTOANCHOR:
            logger.info("begin check anchors")
            run_anchor(logger,train_dataset, model=model, thr=cfg.TRAIN.ANCHOR_THRESHOLD, imgsz=min(cfg.MODEL.IMAGE_SIZE))
        else:
            logger.info("anchors loaded successfully")
            det = model.module.model[model.module.detector_index] if is_parallel(model) \
                else model.model[model.detector_index]
            logger.info(str(det.anchors))

    # training
    num_warmup = max(round(cfg.TRAIN.WARMUP_EPOCHS * num_batch), 1000) # 定义预热周期数，至少为1000步
    scaler = amp.GradScaler(enabled=device.type != 'cpu') # 创建一个GradScaler对象，用于混合精度训练中的梯度缩放
    print('=> start training...')
    for epoch in range(begin_epoch+1, cfg.TRAIN.END_EPOCH+1): # 训练循环，从begin_epoch+1开始，到cfg.TRAIN.END_EPOCH结束
        if rank != -1: # 如果处于分布式训练环境，设置当前epoch的采样器
            train_loader.sampler.set_epoch(epoch)
        # train for one epoch
        train(cfg, train_loader, model, criterion, optimizer, scaler,
              epoch, num_batch, num_warmup, writer_dict, logger, device, rank) # 训练一个epoch
        
        lr_scheduler.step() # 更新学习率

        # evaluate on validation set
        if (epoch % cfg.TRAIN.VAL_FREQ == 0 or epoch == cfg.TRAIN.END_EPOCH) and rank in [-1, 0]: # 如果是验证周期的倍数，或者已经是最后一个epoch，并且在主进程中，执行验证
            # print('validate')
            da_segment_results,ll_segment_results,detect_results, total_loss,maps, times = validate(
                epoch,cfg, valid_loader, valid_dataset, model, criterion,
                final_output_dir, tb_log_dir, writer_dict,
                logger, device, rank
            ) # 调用验证函数，获取验证结果
            fi = fitness(np.array(detect_results).reshape(1, -1))  #目标检测评价指标

            msg = 'Epoch: [{0}]    Loss({loss:.3f})\n' \
                      'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n' \
                      'Lane line Segment: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})  mIOU({ll_seg_miou:.3f})\n' \
                      'Detect: P({p:.3f})  R({r:.3f})  mAP@0.5({map50:.3f})  mAP@0.5:0.95({map:.3f})\n'\
                      'Time: inference({t_inf:.4f}s/frame)  nms({t_nms:.4f}s/frame)'.format(
                          epoch,  loss=total_loss, da_seg_acc=da_segment_results[0],da_seg_iou=da_segment_results[1],da_seg_miou=da_segment_results[2],
                          ll_seg_acc=ll_segment_results[0],ll_seg_iou=ll_segment_results[1],ll_seg_miou=ll_segment_results[2],
                          p=detect_results[0],r=detect_results[1],map50=detect_results[2],map=detect_results[3],
                          t_inf=times[0], t_nms=times[1]) # 构建并打印训练信息的字符串
            logger.info(msg)

            # if perf_indicator >= best_perf:
            #     best_perf = perf_indicator
            #     best_model = True
            # else:
            #     best_model = False

        # save checkpoint model and best model
        if rank in [-1, 0]:
            savepath = os.path.join(final_output_dir, f'epoch-{epoch}.pth') # 构建保存检查点的文件路径
            logger.info('=> saving checkpoint to {}'.format(savepath))
            save_checkpoint(
                epoch=epoch,
                name=cfg.MODEL.NAME,
                model=model,
                # 'best_state_dict': model.module.state_dict(),
                # 'perf': perf_indicator,
                optimizer=optimizer,
                output_dir=final_output_dir, # 输出目录
                filename=f'epoch-{epoch}.pth' # 文件名，包含 epoch 编号
            )
            save_checkpoint(
                epoch=epoch,
                name=cfg.MODEL.NAME,
                model=model,
                # 'best_state_dict': model.module.state_dict(),
                # 'perf': perf_indicator,
                optimizer=optimizer,  # 优化器对象
                output_dir=os.path.join(cfg.LOG_DIR, cfg.DATASET.DATASET),  # 输出目录，通常是日志目录和数据集名称的组合
                filename='checkpoint.pth'
            )

    # save final model
    if rank in [-1, 0]:
        final_model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        ) # 构建最终模型状态文件的路径
        logger.info('=> saving final model state to {}'.format(
            final_model_state_file)
        )
        model_state = model.module.state_dict() if is_parallel(model) else model.state_dict()
        torch.save(model_state, final_model_state_file)
        writer_dict['writer'].close()
    else:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()