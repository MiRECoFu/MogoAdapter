import torch
from collections import defaultdict
import torch.optim as optim
from collections import OrderedDict
from mogo_models.transformers.transformotion import Transformotion
from utils.utils import *
from os.path import join as pjoin
from utils.eval_t2m import evaluation_mask_transformer, evaluation_res_transformer
from models.tools import *
import time
from einops import rearrange, repeat
import wandb

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

        captions1, motions1, m_lens1 = batch_data1
        m_lens1 = m_lens1.detach().long().to(self.device)
        # captions = captions.to(self.device)
        motions1 = motions1.detach().float().to(self.device)
        code_idx1, _motion_emb1 = self.vq_model.encode(motions1)
        

        _ce_loss, _acc, pred_id, output, logits, all_attends_out = self.transformotion(captions1, code_idx1, m_lens1, code_idx1.clone())

        captions2, motions2, m_lens2 = batch_data2
        # wandb.log({"Train/loss": ce_loss, "Train/acc": acc})
        motions2 = motions2.detach().float().to(opt.device)
        code_idx2, _motion_emb2 = vq_model.encode(motions2)
        refer_label = code_idx2.clone()
        motion_code2 = code_idx2[:, :, 0]
        # batch_size = motions.shape[0]
        input_motion_logits = all_attends_out.to(opt.device)
        motion_code_feature2 = self.mogo_clip.encode_motion_code(motion_code2).to(opt.device)
        text_feature2 = self.mogo_clip.encode_text(captions2).to(opt.device)
        res_features, res_all_out = mogo_adapter(all_attends_out, text_feature2, motion_code_feature2)
        ce_loss, pred_id, acc = cal_performance(res_all_out, refer_label, m_lens2, self.pad_id)
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
        missing_keys, unexpected_keys = self.transformotion.load_state_dict(checkpoint['mogo_adapter'], strict=False)
        assert len(unexpected_keys) == 0
        # assert all([k.startswith('clip_model.') for k in missing_keys])

        try:
            self.opt_t2m_transformer.load_state_dict(checkpoint['opt_mogo_adapter']) # Optimizer

            self.scheduler.load_state_dict(checkpoint['scheduler']) # Scheduler
        except:
            print('Resume wo optimizer')
        return checkpoint['ep'], checkpoint['total_it']
    
    def train(self, train_loader1, train_loader2, val_loader, eval_val_loader, eval_wrapper, plot_eval):
        self.transformotion.to(self.device)
        self.vq_model.to(self.device)

        self.opt_mogo_adapter = optim.AdamW(self.mogo_adapter.parameters(), lr=self.opt.lr, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.opt_t2m_transformer,
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
        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_mask_transformer(
            self.opt.save_root, eval_val_loader, self.transformotion, self.vq_model, self.logger, epoch,
            best_fid=100, best_div=100,
            best_top1=0, best_top2=0, best_top3=0,
            best_matching=100, eval_wrapper=eval_wrapper,
            plot_func=plot_eval, save_ckpt=False, save_anim=True
        )
        # best_fid, best_div, best_top1, best_top2, best_top3, best_matching = 100, 100, 0, 0, 0, 100
        best_acc = 0.