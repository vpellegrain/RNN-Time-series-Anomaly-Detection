import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import shutil
from pathlib import Path

class TimeSeriesLSTMStochastic(nn.Module):
    
    def __init__(self, args, feature_dim, embed_dim, hidden_dim, out_dim, nb_layers, dropout):
        super(TimeSeriesLSTMStochastic, self).__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.nb_layers = nb_layers
        self.dropout = dropout
        self.device = args.device
        
        self.encoder = nn.Linear(self.feature_dim, self.embed_dim)
        self.rnn = getattr(nn, 'LSTM')(self.embed_dim, self.hidden_dim, self.nb_layers, dropout = self.dropout)
        self.decoder = nn.Linear(self.hidden_dim, self.out_dim)
        self.do = nn.Dropout(self.dropout)
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        
    def init_hidden(self):
        weight = next(self.parameters()).data
        return (torch.zeros(self.nb_layers,1,self.hidden_dim).to(self.device), torch.zeros(self.nb_layers,1,self.hidden_dim).to(self.device))
        
    def forward(self, input, hidden, return_hiddens):
        emb = self.do(self.encoder(input))
        #emb = emb.view(-1, self.batch_size, model.hidden_dim)
        out, hidden = self.rnn(emb.unsqueeze(1), hidden)
        out = self.do(out)
        dec = self.decoder(out).squeeze()
        
        if return_hiddens:
            return dec,hidden,out
        
        return dec, hidden
    
    def repackage_hidden(self,h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == tuple:
            return tuple(self.repackage_hidden(v) for v in h)
        else:
            return h.detach()

    def save_checkpoint(self,state, is_best):
        print("=> saving checkpoint ..")
        args = state['args']
        checkpoint_dir = Path('../../../workdir/pellegrainv/3Tanks/save')
        checkpoint_dir.mkdir(parents=True,exist_ok=True)
        checkpoint = checkpoint_dir.joinpath(args.filename).with_suffix('.pth')

        torch.save(state, checkpoint)
        if is_best:
            model_best_dir = Path(checkpoint_dir,'model_best')
            model_best_dir.mkdir(parents=True,exist_ok=True)

            shutil.copyfile(checkpoint, model_best_dir.joinpath(args.filename).with_suffix('.pth'))

        print('=> checkpoint saved.')

    def extract_hidden(self, hidden):
        if self.rnn_type == 'LSTM':
            return hidden[0][-1].data.cpu()  # hidden state last layer (hidden[1] is cell state)
        else:
            return hidden[-1].data.cpu()  # last layer

    def initialize(self,args,feature_dim, out_dim):
        self.__init__(args, feature_dim=feature_dim,
                           embed_dim = args.embed_dim,
                           hidden_dim = args.hidden_dim,
                           nb_layers = args.nb_layers,
                           dropout = args.dropout,
                           out_dim = out_dim)
        self.to(args.device)

    def load_checkpoint(self, args, checkpoint, feature_dim, out_dim):
        start_epoch = checkpoint['epoch'] +1
        best_val_loss = checkpoint['best_loss']
        args_ = checkpoint['args']
        args_.resume = args.resume
        args_.pretrained = args.pretrained
        args_.epochs = args.epochs
        args_.save_interval = args.save_interval
        args_.prediction_window_size=args.prediction_window_size
        self.initialize(args_, feature_dim=feature_dim, out_dim = out_dim)
        self.load_state_dict(checkpoint['state_dict'])

        return args_, start_epoch, best_val_loss