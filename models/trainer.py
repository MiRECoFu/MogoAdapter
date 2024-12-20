import torch
from collections import defaultdict
import torch.optim as optim
from collections import OrderedDict
from mogo_models.transformers.transformotion import Transformotion
from utils.utils import *
from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
from utils.eval_t2m import evaluation_mask_transformer, evaluation_res_transformer
from models.tools import *
import time
from einops import rearrange, repeat
import wandb
import random

def def_value():
    return 0.0

class Trainer:
    def __init__(self, args, transformotion: Transformotion, vq_model, mogo_clip, mogo_adapter):
        self.opt = args
        self.transformotion = transformotion
        self.vq_model = vq_model
        self.mogo_clip = mogo_clip
        self.mogo_adapter = mogo_adapter
        self.device = args.device
        self.vq_model.eval()
        self.mogo_clip.eval()
        self.transformotion.eval()
        for param in self.vq_model.parameters():
            param.requires_grad = False
        for param in self.mogo_clip.parameters():
            param.requires_grad = False
        for param in self.transformotion.parameters():
            param.requires_grad = False
        self.pad_id = args.num_tokens
        wandb.init(
            # set the wandb project where this run will be logged
            project="mogo_adapter",

            # track hyperparameters and run metadata
            config={
            "learning_rate": self.opt.lr,
            "epochs": self.opt.max_epoch,
            }
        )
        if args.is_train:
            self.logger = SummaryWriter(args.log_dir)
            
    
    def save(self, file_name, ep, total_it):
        mogo_adapter_state_dict = self.mogo_adapter.state_dict()
        vq_model_weights = [e for e in mogo_adapter_state_dict.keys() if e.startswith('vq_model.')]
        for e in vq_model_weights:
            del mogo_adapter_state_dict[e]
        transformotion_weights = [e for e in mogo_adapter_state_dict.keys() if e.startswith('transformotion.')]
        for e in transformotion_weights:
            del mogo_adapter_state_dict[e]
        mogo_clip_weights = [e for e in mogo_adapter_state_dict.keys() if e.startswith('mogo_clip.')]
        for e in mogo_clip_weights:
            del mogo_adapter_state_dict[e]
        state = {
            'mogo_adapter': mogo_adapter_state_dict,
            'opt_mogo_adapter': self.opt_mogo_adapter.state_dict(),
            'scheduler':self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)
        
    def forward(self, batch_data1, batch_data2):

        _captions1, motions1, m_lens1 = batch_data1
        m_lens1 = m_lens1.detach().long().to(self.device)
        # captions = captions.to(self.device)
        batch_size = motions1.shape[0]
        motions1 = motions1.detach().float().to(self.device)
        code_idx1, _motion_emb1 = self.vq_model.encode(motions1)
        # captions1 = [random.choice(["A person", "A person is moving", "A person is standing", "A person is walking"]) for _ in range(batch_size)]
        # captions1 = [" "] * batch_size
        # print(captions1)

        # _ce_loss, _acc, _pred_id, output, logits, all_attends_out = self.transformotion(captions1, code_idx1, m_lens1, code_idx1.clone(), has_adapter=True)

        captions2, motions2, m_lens2 = batch_data2
        
        motions2 = motions2.detach().float().to(self.device)
        code_idx2, _motion_emb2 = self.vq_model.encode(motions2)
        refer_label = code_idx2.clone()
        motion_code2 = code_idx2[:, :, 0]
        # batch_size = motions.shape[0]
        # input_motion_logits = all_attends_out.to(self.device)
        input_motion_logits = self.transformotion.tok_emb(code_idx1)
        # input_motion_logits = input_motion_logits.permute(0, 1, 3, 2) # (b, len, dim, q)
        motion_code_feature2 = self.mogo_clip.encode_motion_code(motion_code2).to(self.device)
        res_features, res_all_out = self.mogo_adapter(input_motion_logits, motion_code_feature2)
        ce_loss, pred_id, acc = cal_performance(res_all_out, refer_label, m_lens2, self.pad_id)
        wandb.log({"Train/loss": ce_loss, "Train/acc": acc})
        return ce_loss, acc
    
    def update(self, batch_data1, batch_data2):
        loss, acc = self.forward(batch_data1, batch_data2)

        self.opt_mogo_adapter.zero_grad()
        loss.backward()
        self.opt_mogo_adapter.step()
        # torch.nn.utils.clip_grad_norm_(self.transformotion.parameters(), 0.25)
        self.scheduler.step()

        return loss.item(), acc
    
    
    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        missing_keys, unexpected_keys = self.mogo_adapter.load_state_dict(checkpoint['mogo_adapter'], strict=False)
        assert len(unexpected_keys) == 0
        # assert all([k.startswith('clip_model.') for k in missing_keys])

        try:
            self.opt_mogo_adapter.load_state_dict(checkpoint['opt_mogo_adapter']) # Optimizer

            self.scheduler.load_state_dict(checkpoint['scheduler']) # Scheduler
        except:
            print('Resume wo optimizer')
        return checkpoint['ep'], checkpoint['total_it']
    
    def train(self, train_loader1, train_loader2, val_loader, eval_val_loader1, eval_val_loader2, eval_wrapper, plot_eval):
        self.transformotion.to(self.device)
        self.vq_model.to(self.device)
        self.mogo_clip.to(self.device)

        self.opt_mogo_adapter = optim.AdamW(self.mogo_adapter.parameters(), lr=self.opt.lr, weight_decay=0.0)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.opt_mogo_adapter,
                                        800000, eta_min=3e-6)
        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')  # TODO
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader1)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader1), len(val_loader)))
        logs = defaultdict(def_value, OrderedDict())
        # best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_mask_transformer(
        #     self.opt.save_root, eval_val_loader1, eval_val_loader2, self.transformotion, self.vq_model, self.mogo_adapter, self.mogo_clip, self.logger, epoch,
        #     best_fid=100, best_div=100,
        #     best_top1=0, best_top2=0, best_top3=0,
        #     best_matching=100, eval_wrapper=eval_wrapper,
        #     plot_func=plot_eval, save_ckpt=False, save_anim=True
        # )
        best_fid, best_div, best_top1, best_top2, best_top3, best_matching = 100, 100, 0, 0, 0, 100
        best_acc = 0.
        while epoch < self.opt.max_epoch:
            self.transformotion.eval()
            self.vq_model.eval()
            self.mogo_clip.eval()
            self.mogo_adapter.train()
            for i, (batch_data, batch_data2) in enumerate(zip(train_loader1, train_loader2)):
                it += 1
                # if it < self.opt.warm_up_iter:
                #     self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                loss, acc = self.update(batch_data1=batch_data, batch_data2=batch_data2)
                logs['loss'] += loss
                logs['acc'] += acc
                logs['lr'] += self.opt_mogo_adapter.param_groups[0]['lr']
                wandb.log({"Train/lr": self.opt_mogo_adapter.param_groups[0]['lr']})
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    # self.logger.add_scalar('val_loss', val_loss, it)
                    # self.l
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            epoch += 1
            
            if epoch % 20 == 0 or epoch == 1:
                best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_mask_transformer(
                    self.opt.save_root, eval_val_loader1, eval_val_loader2, self.transformotion, self.vq_model, self.mogo_adapter, self.mogo_clip, self.logger, epoch,
                    best_fid=best_fid, best_div=best_div,
                    best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                    best_matching=best_matching, eval_wrapper=eval_wrapper,
                    plot_func=plot_eval, save_ckpt=False, save_anim=True
                )