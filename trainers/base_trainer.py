import torch
from .common_config import get_optimizer, get_transformations, get_train_dataset, get_criterion, get_dataloader, get_eval_dataset
import os
import time 
import numpy as np
import torch.distributed as dist
import cv2

class BASE_TRAINER(object):
    def __init__(self, args, model , device = None) -> None:
        self.args = args
        self.model = model 
        if device == None:
            raise ValueError("Please set the device")
        self.device = device

    def train(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        args = self.args
        model = self.model

        scheduler, optimizer = get_optimizer(args, model)
        if args.resume:
            if os.path.exists(args.ckpt_path):
                ckpt = torch.load(args.ckpt_path, map_location = "cpu")
                state_dict = ckpt['model_dict']
                optimizer_state = ckpt["optimizer"]
                scheduler_state = ckpt["scheduler"]
            else:
                raise ValueError(f"check the ckpt path:{args.ckpt_path}")
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
            batch_iter = ckpt["batch_iter"]
        else:
            batch_iter = 0

        train_transforms, val_transforms = get_transformations(args)
        dataset = get_train_dataset(args, train_transforms)

        if args.ddp:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, drop_last=False)
        else:
            sampler = torch.utils.data.RandomSampler(dataset, drop_last = False)

        dataloader = get_dataloader(args, dataset, sampler)
        criterions = get_criterion(args)
        start = time.time()

        while batch_iter <= args.max_batch_iter:
            sampler.set_epoch(batch_iter)
            for img, depth, scale  in dataloader:
                msg = "time:{:<30}".format(time.strftime("%Y-%m-%d %p %H:%M:%S",time.localtime()))
                img = img.to(self.device)
                scale = scale.to(self.device).reshape(-1,1,1,1)
                b,c,h,w = img.shape
                depth = depth.to(self.device).reshape(-1,1,h,w)
                if batch_iter == 0:
                    self.visualize(depth)
                prediction = model(img).reshape(-1,1,h,w)
                if args.ignore_depth_scale:
                    scale = 1.0
                msg += "| batch_iter:{:<20}".format(batch_iter)
                loss = 0
                for criterion in criterions:
                    if criterion == "sigloss":
                        loss_ = criterions[criterion](prediction, depth, scale=scale)
                        loss_ *= 10

                    if criterion == "l1loss":
                        loss_ = criterions[criterion](prediction, depth, scale=scale)
                        loss_ *= 1
                    
                    if criterion == 'gradl1loss':
                        loss_ = criterions[criterion](prediction, depth)
                        loss_ *= 100
                    
                    if criterion == "ssim":
                        mask = (depth > args.min_depth_eval) & (depth < args.max_depth_eval)
                        loss_ ,_ = criterions[criterion](prediction, depth, mask = mask)
                        if batch_iter > args.max_batch_iter//2:
                            loss_ *= 1
                        else:
                            loss_ *= 10

                    if criterion == 'meadstd':
                        loss_ = criterions[criterion](prediction, depth, scale = scale)
                        if batch_iter > args.max_batch_iter//2:
                            loss_ *= 0.1

                    if criterion == "msgil_norm_loss":
                        loss_ = criterions[criterion](prediction, depth, scale = scale)
                        if batch_iter > args.max_batch_iter//2:
                            loss_ *= 0.1

                    if criterion == 'edge_ranking_loss':
                        mask = (depth > args.min_depth_eval) & (depth < args.max_depth_eval)
                        loss_ = criterions[criterion](prediction, depth, img, masks = mask)
                        loss_ *= 10

                    msg += "| {}:{:<25}".format(criterion, loss_)
                    loss += loss_
                msg += "| total_loss:{:<25}".format(loss)
                
                if args.local_rank == 0 and batch_iter % args.log_freq  == 0:
                    msg += "| learning_rate:{:<25}".format(optimizer.param_groups[0]['lr'])
                    print(msg)

                optimizer.zero_grad()
                loss.backward()

                
                if args.local_rank == 0 and batch_iter % args.save_freq == 0:
                    save_name = os.path.join(args.exp_name_pre, args.ckpt_name) + "_iter_{}.pth".format(batch_iter)
                    self.save_ckpt(save_name, model , optimizer, scheduler, batch_iter)

                if args.do_online_eval and batch_iter % args.eval_freq == 0:
                    self.online_eval(args, self.device, model = model, transform=val_transforms)
                    model.train()

                optimizer.step()
                scheduler.step()
                batch_iter += 1

    def visualize(self, data, pre = ""):
        b,c,h,w = data.shape
        for i in range(b):
            img = data[i,0:1,:,:]
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0),
                size=(480,640),
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()
            #img = img.unsqueeze(1).squeeze().cpu().numpy()
            img = (255*(img - img.min())/(img.max() - img.min())).astype("uint8")
            cv2.imwrite("{}_{}.jpg".format(pre, i), img)

    def save_ckpt(self, save_name, model, optimizer, scheduler, batch_iter):
        ckpt = {
            "model_dict": model.state_dict(),
            "optimzer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "batch_iter": batch_iter
        }
        torch.save(ckpt, save_name)

    def online_eval(self, args, device, model = None, transform = None):
        abs_depth_eval_metrics = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']
        eval_measures = torch.zeros(10).to(device)
        def compute_errors(gt, pred):
            thresh = np.maximum((gt / pred), (pred / gt))
            d1 = (thresh < 1.25).mean()
            d2 = (thresh < 1.25 ** 2).mean()
            d3 = (thresh < 1.25 ** 3).mean()

            rms = (gt - pred) ** 2
            rms = np.sqrt(rms.mean())

            log_rms = (np.log(gt) - np.log(pred)) ** 2
            log_rms = np.sqrt(log_rms.mean())

            abs_rel = np.mean(np.abs(gt - pred) / gt)
            sq_rel = np.mean(((gt - pred) ** 2) / gt)

            err = np.log(pred) - np.log(gt)
            silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

            err = np.abs(np.log10(pred) - np.log10(gt))
            log10 = np.mean(err)

            return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]
        
        dataset = get_eval_dataset(args, transform)
        if args.ddp:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, drop_last=False)
        else:
            sampler = torch.utils.data.RandomSampler(dataset, drop_last = False)

        dataloader = get_dataloader(args, dataset, sampler)
        for img, depth  in dataloader:
            img = img.to(self.device)
            if args.dataset == "IPHONE":
                b,c,h,w  = depth.shape
            else:
                b,h,w = depth.shape
            depth = depth.to(self.device).reshape(-1,1,h,w)
            target_size = (h,w)
           

            with torch.no_grad():
                model.eval()
                if not args.ignore_depth_scale:
                    pred_depth = model(img).reshape(-1,1,img.shape[2],img.shape[3]) * args.scale
                else:
                    pred_depth = model(img).reshape(-1,1,img.shape[2],img.shape[3])
                
                pred_depth = torch.nn.functional.interpolate(pred_depth,size=target_size,mode="bicubic",align_corners=False)
            
            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = depth.cpu().numpy().squeeze()
            pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
            pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
            valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

            measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])
            eval_measures[:9] += torch.tensor(measures).to(device)
            eval_measures[9] += 1

        if args.ddp:
            dist.all_reduce(eval_measures, op = dist.ReduceOp.SUM)

        eval_measures_cpu = eval_measures.cpu()
        cnt = eval_measures_cpu[9].item()
        eval_measures_cpu /= cnt
        if args.ddp and args.local_rank == 0:
            print('Computing errors for {} eval samples'.format(int(cnt)))
            print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                     'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                     'd3'))
            for i in range(8):
                print('{:7.3f}, '.format(eval_measures_cpu[i]), end='')
            print('{:7.3f}'.format(eval_measures_cpu[8]))
        

    
    
    
