import logging
import os
import yaml
import pickle
import torch
import torch.nn as nn
import pandas as pd
from itertools import chain
from argparse import ArgumentParser, Namespace
from transformers import (
    ElectraModel,
    ElectraTokenizer,
    AutoConfig,
)
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from sales_forecasting_data import SalesDataModule

MODEL_PATH = 'sales_seq_electra-small_model'
MAX_DEVICE = 4

class SeqNet(nn.Module):
    def __init__(self, model_type, input_dim, hidden_dim, output_dim, n_seq, 
                 drop_prob=0.2, add_total=False, use_bert=False, transformer_path=''):
        super(SeqNet, self).__init__()
        
        if use_bert:
            config = AutoConfig.from_pretrained(MODEL_PATH, return_dict=True)
            self.bert = ElectraModel.from_pretrained(MODEL_PATH)
            self.bert.eval()
            hidden_size = 2 * hidden_dim
            self.bert_layer = nn.Linear(config.hidden_size, hidden_dim)
        else:
            hidden_size = hidden_dim
            
        if add_total:
            self.total_layer = nn.Linear(n_seq, hidden_dim)
            hidden_size += hidden_dim
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_seq = n_seq
        self.model_type = model_type
        self.hidden = None
        if len(transformer_path) > 0:
            self.transformer = pickle.load(open(transformer_path, 'rb'))
        else:
            self.transformer = None
        
        if model_type == 'gru':
            self.seq_net = nn.GRU(input_dim, hidden_dim, n_seq, batch_first=True, dropout=drop_prob)
        elif model_type == 'lstm':
            self.seq_net = nn.LSTM(input_dim, hidden_dim, n_seq, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_size, output_dim)
        self.relu = nn.ReLU()
        
    def preprocess(self, batch_input, batch_hidden, batch_total=None, batch_bert_input=None, batch_tr_output=None, batch_output=None):
        batch_size = len(batch_input[0][0])
        device = batch_input[0][0].device
        
        input = torch.stack(list(map(lambda y: torch.stack(list(map(lambda x: torch.stack(x)[:, y], batch_input))).T, range(batch_size)))).float()
        tr_output = None if batch_tr_output is None else torch.stack(list(chain(*batch_tr_output))).T.float()
        output = None if batch_output is None else torch.stack(batch_output).T.float()
        if batch_bert_input is None:
            bert_input = None
        else:
            bert_input = []
            for i in range(batch_size):
                days_list = []
                for j in range(self.n_seq):
                    singles = []
                    for k in range(self.input_dim):
                        singles.append(torch.stack(batch_bert_input[j][k])[:, i])
                    days_list.append(torch.stack(singles))
                bert_input.append(torch.stack(days_list))
            bert_input = torch.stack(bert_input)
        total = None if batch_total is None else torch.stack(list(chain(*batch_total))).T.float()
        hidden = batch_hidden.data.float().to(device) if self.model_type == 'gru' else tuple([e.data.float().to(device) for e in batch_hidden])
        return input, hidden, total, bert_input, tr_output, output
        
    def forward(self, input, hidden, total=None, bert_input=None, tr_output=None, output=None):
        input, hidden, total, bert_input, tr_label, label = self.preprocess(input, hidden, total, bert_input, tr_output, output)
        
        out, hidden = self.seq_net(input, hidden)
        if bert_input is None:
            out = out[:, -1]
        else:
            bert_out = []
            for single in bert_input:
                cls_hiddens = []
                for seqs in single:
                    with torch.no_grad():
                        cls_hiddens.append(self.bert(seqs).last_hidden_state[:, 0, :])
                bert_out.append(torch.stack(cls_hiddens).sum(0))
            bert_out = self.bert_layer(torch.stack(bert_out)).mean(1)
            out = torch.cat([out[:, -1], bert_out], -1)
        if total is not None:
            total_out = self.total_layer(total)
            out = torch.cat([out, total_out], -1)
            
        out = self.fc(self.relu(out))
        
        if self.transformer is None:
            org_out = None
        else:
            resizes = list(out.shape) + [1]
            out_ = out.reshape(*resizes).cpu().detach().numpy()
            org_out = torch.stack([ torch.tensor(self.transformer.inverse_transform(o).squeeze()) for o in out_ ]).to(out.device)
        
        return out, hidden, tr_label, label, org_out
    
    def init_hidden(self, batch_size, weight):
        if self.model_type == 'gru':
            hidden = weight.new(self.n_seq, batch_size, self.hidden_dim).zero_()
        else:
            hidden = (weight.new(self.n_seq, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_seq, batch_size, self.hidden_dim).zero_())
        return hidden
    
    
class Base(pl.LightningModule):
    def __init__(self, hparams, **kwargs) -> None:
        super(Base, self).__init__()
        self.save_hyperparameters(hparams)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('-f', default=None)
        parser.add_argument('--lr', type=float, default=5e-5)
        parser.add_argument('--warmup_ratio', type=float, default=0.1)
        parser.add_argument('--seed', type=int, default=1234)
        parser.add_argument('--train_file', type=str, default="")
        parser.add_argument('--val_file', type=str, default="")
        parser.add_argument('--test_file', type=str, default="")
        parser.add_argument('--workers', type=int, default=4)
        parser.add_argument('--overwrite_cache', action='store_true', default=False)
        parser.add_argument('--max_length', type=int, default=32)
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--gpu_list', type=str, default='')
        
        parser.add_argument('--item', type=int, default=0)
        parser.add_argument('--transformer_path', type=str, default='')
        parser.add_argument('--model_type', type=str, default='gru or lstm')
        parser.add_argument('--input_dim', type=int, default=10)
        parser.add_argument('--output_dim', type=int, default=10)
        parser.add_argument('--hidden_dim', type=int, default=20)
        parser.add_argument('--n_seq', type=int, default=7)
        parser.add_argument('--drop_prob', type=float, default=0.2)
        parser.add_argument('--use_bert', type=bool, default=False)
        parser.add_argument('--add_total', type=bool, default=False)
        
        return parser
    
    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        # warm up lr
        num_workers = self.hparams.num_nodes if self.hparams.num_nodes is not None else 1
        data_len = len(self.train_dataloader().dataset)
        logging.info(f'number of workers {num_workers}, data length {data_len}')
        num_train_steps = int(data_len / (self.hparams.batch_size * num_workers) * self.hparams.max_epochs)
        logging.info(f'num_train_steps : {num_train_steps}')
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        logging.info(f'num_warmup_steps : {num_warmup_steps}')
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]
    
    
class MAPE(nn.Module):
    def __init__(self):
        super(MAPE, self).__init__()
        
    def forward(self, labels, outs):
        return (((labels - outs) / labels).abs()).mean()
    

class Train_SalesSeqNet(Base):
    def __init__(self, hparams, **kwargs):
        super(Train_SalesSeqNet, self).__init__(hparams, **kwargs)
        
        self.save_hyperparameters(hparams)
        
        self.model = SeqNet(hparams.model_type, hparams.input_dim, hparams.hidden_dim, hparams.output_dim,
                            hparams.n_seq, hparams.drop_prob, hparams.add_total, hparams.use_bert, hparams.transformer_path)
        self.model.train()
        self.mse = nn.MSELoss()
        self.mape = MAPE()
        
        self.epoch = -1
        self.log_filepath = self.hparams.log_filepath
        self.log_file = None
        self.prior_test = False
        self.hidden = None
        self.weight = next(self.model.parameters()).data
        
        if self.hparams.gpus is None:
            self.agg_device = torch.device('cpu')
        elif self.hparams.gpus == -1:
            self.agg_device = torch.device('cuda:0')
        else:
            min_dv = min(self.hparams.gpus)
            self.agg_device = torch.device(f'cuda:{min_dv}')
            
    def add_hidden(self, batch):
        batch_size = len(batch['input'][0][0])
        if self.hidden is None:
            self.hidden = self.model.init_hidden(batch_size, self.weight)
        batch['hidden'] = self.hidden
        if 'total' not in batch.keys():
            batch['total'] = None
        if 'bert_input' not in batch.keys():
            batch['bert_input'] = None
        return batch
    
    def forward(self, inputs):
        return self.model(input=inputs['input'], hidden=inputs['hidden'], total=inputs['total'], bert_input=inputs['bert_input'], 
                          tr_output=inputs['tr_output'], output=inputs['output'])
    
    def compute_losses(self, tr_label, out, label, org_out):
        mse, mape = self.mse(tr_label, out).to(self.agg_device), 0.0 if org_out is None else self.mape(label, org_out).to(self.agg_device)
        loss = mse + mape
        return loss, mse, mape
        
    def training_step(self, batch, batch_idx):
        inputs = self.add_hidden(batch)
        out, self.hidden, tr_label, label, org_out = self(inputs)
        loss, _, _ = self.compute_losses(tr_label, out, label, org_out)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = self.add_hidden(batch)
        out, self.hidden, tr_label, label, org_out = self(inputs)
        loss, mse, mape = self.compute_losses(tr_label, out, label, org_out)
        return {'loss': loss, 'mse': mse, 'mape': mape}
    
    def validation_epoch_end(self, outputs):
        losses, mses, mapes = [], [], []
        for i in outputs:
            losses.append(i['loss'])
            mses.append(i['mse'])
            mapes.append(i['mape'])
        loss, mse, mape = torch.stack(losses).mean(), torch.stack(mses).mean(), torch.stack(mapes).mean()
        self.log_dict({'val_loss': loss, 'val_mse': mse, 'val_mape': mape}, prog_bar=True)
        print(f'epoch={self.epoch}, val_loss={float(loss)}, val_mse={float(mse)}, val_mape={float(mape)}')
        self.log_file = open(self.log_filepath, 'a')
        if self.epoch == -1:
            if not self.prior_test:
                self.log_file.write(f'epoch,val_loss,val_mse,val_mape\n')
        else:
            self.log_file.write(f'{self.epoch},{float(loss)},{float(mse)},{float(mape)}\n')
        self.log_file.close()
        self.epoch += 1
        
    def test_step(self, batch, batch_idx):
        inputs = self.add_hidden(batch)
        out, self.hidden, tr_label, label, org_out = self(inputs)
        loss, mse, mape = self.compute_losses(tr_label, out, label, org_out)
        return {'loss': loss, 'mse': mse, 'mape': mape}
    
    def test_epoch_end(self, outputs):
        losses, mses, mapes = [], [], []
        for i in outputs:
            losses.append(i['loss'])
            mses.append(i['mse'])
            mapes.append(i['mape'])
        loss, mse, mape = torch.stack(losses).mean(), torch.stack(mses).mean(), torch.stack(mapes).mean()
        self.log_dict({'test_loss': loss, 'test_mse': mse, 'test_mape': mape}, prog_bar=True)
        self.log_file = open(self.log_filepath, 'a')
        if self.epoch == -1:
            self.log_file.write(f'epoch,val_loss,val_mse,val_mape\n')
            self.prior_test = True
        self.log_file.write(f'-1,{float(loss)},{float(mse)},{float(mape)}\n')
        self.log_file.close()
        
        
def train(dict_args={}):
    # ------------
    # args
    # ------------
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Base.add_model_specific_args(parser)
    args = parser.parse_args()
    user = os.environ['USER']
    
    if len(dict_args) > 0:
        for k, v in dict_args.items():
            setattr(args, k, v)
    
        if args.gpu_list == '':
            if args.gpus > -1:
                setattr(args, 'gpus', str(args.gpus))
        else:
            gpu_list = list(map(lambda x: int(x), args.gpu_list.split(';')))
            setattr(args, 'gpus', gpu_list)
        
        setattr(args, 'model_name', f'item{args.item}_{args.model_type}_bert{args.use_bert}_total{args.add_total}_{args.lr}_{args.seed}')
        setattr(args, 'root_dir', f'/home/{user}/logs_{args.model_name}')
        setattr(args, 'log_filepath', os.path.join(args.root_dir, 'performance_log.txt'))
        logging.info(args)
        
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    data_module = SalesDataModule(
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
        workers=args.workers,
        overwrite_cache=args.overwrite_cache,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_bert=args.use_bert,
        add_total=args.add_total,
        transformer_path=args.transformer_path
    )

    # ------------
    # model
    # ------------
    model = Train_SalesSeqNet(args)

    # ------------
    # training
    # ------------
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss',
                                                       dirpath=args.root_dir,
                                                       filename='model_chp/{epoch:02d}-{train_loss:.3f}-{val_loss:.3f}-{val_mse:.3f}-{val_mape:.3f}',
                                                       verbose=True,
                                                       save_last=False,
                                                       mode='min',
                                                       save_top_k=10)
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.root_dir, 'tb_logs'))
    lr_logger = pl.callbacks.LearningRateMonitor()
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=10)
    trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger, callbacks=[checkpoint_callback, lr_logger, early_stop_callback])
    trainer.test(model, test_dataloaders=data_module.test_dataloader())
    trainer.fit(model, data_module)
    trainer.test(ckpt_path="best", test_dataloaders=data_module.test_dataloader())
    

def evaluate_models(log_dir_path):
    last_version = os.popen(f'ls {log_dir_path}/tb_logs/default/').readlines()[-1][:-1]
    
    with open(f'{log_dir_path}/tb_logs/default/{last_version}/hparams.yaml', 'r') as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)
    args = Namespace(**hparams)
    model_names = os.popen(f'ls {log_dir_path}/model_chp/').readlines()
    
    perf_df = pd.read_csv(f'{log_dir_path}/performance_log.txt', sep=',', header=0)
    perf_pairs = list(perf_df.loc[perf_df['epoch'] > -1].values)
    best_epoch = sorted(perf_pairs, key=lambda x: x[1])[0][0]
    last_epoch = perf_pairs[-1][0]
    model_files = dict()
    for i, epoch in enumerate([best_epoch, last_epoch]):
        if i == 0:
            tag = 'best'
        else:
            tag = 'last'
        matched = list(filter(lambda x: x.startswith('epoch={0:02d}'.format(int(epoch))), model_names))
        if len(matched) == 1:
            model_files[tag] = f'{log_dir_path}/model_chp/{matched[0][:-1]}'
    print(model_files)
    
    data_module = SalesDataModule(
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
        workers=args.workers,
        overwrite_cache=args.overwrite_cache,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_bert=args.use_bert,
        add_total=args.add_total,
        transformer_path=args.transformer_path
    )
    
    trainer = pl.Trainer.from_argparse_args(args)
    model_result = dict()
    for k, v in model_files.items():
        lmmodel = Train_SalesSeqNet.load_from_checkpoint(checkpoint_path=v, hparams=args)
        model_result[k] = trainer.test(lmmodel, test_dataloaders=data_module.test_dataloader())
    return model_result