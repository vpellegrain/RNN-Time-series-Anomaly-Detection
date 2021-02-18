import argparse
import time
import torch
import torch.nn as nn
import preprocess_data
from model.model import TimeSeriesLSTMStochastic
from torch import optim
from matplotlib import pyplot as plt
from pathlib import Path
from anomalyDetector import fit_norm_distribution_param
from dataloading import TimeSeriesDataset

parser = argparse.ArgumentParser(description='PyTorch RNN Prediction Model on Time-series Dataset')
parser.add_argument('--data', type=str, default='/mnt/d/vpell/Documents/Thèse/Python/data/3Tanks/capteurs_labels',
                    help='type of the dataset (ecg, gesture, power_demand, space_shuttle, respiration, nyc_taxi')
parser.add_argument('--filename', type=str, default='5 epochs',
                    help='filename of the dataset')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, SRU)')
parser.add_argument('--augment', type=bool, default=False,
                    help='augment')
parser.add_argument('--noise_ratio', type=float, default=0.05,
                    help='augment')                    
parser.add_argument('--embed_dim', type=int, default=32,
                    help='size of rnn input features')
parser.add_argument('--hidden_dim', type=int, default=32,
                    help='number of hidden units per layer')
parser.add_argument('--nb_layers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--res_connection', action='store_true',
                    help='residual connection')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight decay')
parser.add_argument('--clip', type=float, default=10,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=5,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=64, metavar='N',
                    help='eval_batch size')
parser.add_argument('--bptt', type=int, default=50,
                    help='sequence length')
parser.add_argument('--teacher_forcing_ratio', type=float, default=0.7,
                    help='teacher forcing ratio (deprecated)')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights (deprecated)')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--device', type=str, default='cuda',
                    help='cuda or cpu')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='report interval')
parser.add_argument('--save_interval', type=int, default=5, metavar='N',
                    help='save interval')
parser.add_argument('--save_fig', action='store_true',
                    help='save figure')
parser.add_argument('--resume','-r',
                    help='use checkpoint model parameters as initial parameters (default: False)',
                    action="store_true")
parser.add_argument('--pretrained','-p',
                    help='use checkpoint model parameters and do not train anymore (default: False)',
                    action="store_true")
parser.add_argument('--prediction_window_size', type=int, default=10,
                    help='prediction_window_size')
args = parser.parse_args()
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


###############################################################################
# Training code
###############################################################################
def get_batch(args,source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len] # [ seq_len * batch_size * feature_size ]
    target = source[i+1:i+1+seq_len] # [ (seq_len x batch_size x feature_size) ]
    return data, target

def generate_output(args,epoch, model, gen_dataset, disp_uncertainty=True,startPoint=500, endPoint=3500):
    if args.save_fig:
        # Turn on evaluation mode which disables dropout.
        model.eval()
        hidden = model.init_hidden(1)
        outSeq = []
        upperlim95 = []
        lowerlim95 = []
        with torch.no_grad():
            for i in range(endPoint):
                if i>=startPoint:
                    # if disp_uncertainty and epoch > 40:
                    #     outs = []
                    #     model.train()
                    #     for i in range(20):
                    #         out_, hidden_ = model.forward(out+0.01*Variable(torch.randn(out.size())).cuda(),hidden,noise=True)
                    #         outs.append(out_)
                    #     model.eval()
                    #     outs = torch.cat(outs,dim=0)
                    #     out_mean = torch.mean(outs,dim=0) # [bsz * feature_dim]
                    #     out_std = torch.std(outs,dim=0) # [bsz * feature_dim]
                    #     upperlim95.append(out_mean + 2.58*out_std/np.sqrt(20))
                    #     lowerlim95.append(out_mean - 2.58*out_std/np.sqrt(20))

                    out, hidden = model.forward(out, hidden)

                    #print(out_mean,out)

                else:
                    out, hidden = model.forward(gen_dataset[i].unsqueeze(0), hidden)
                outSeq.append(out.data.cpu()[0][0].unsqueeze(0))


        outSeq = torch.cat(outSeq,dim=0) # [seqLength * feature_dim]

        target= preprocess_data.reconstruct(gen_dataset.cpu(), TimeseriesData.mean, TimeseriesData.std)
        outSeq = preprocess_data.reconstruct(outSeq, TimeseriesData.mean, TimeseriesData.std)
        # if epoch>40:
        #     upperlim95 = torch.cat(upperlim95, dim=0)
        #     lowerlim95 = torch.cat(lowerlim95, dim=0)
        #     upperlim95 = preprocess_data.reconstruct(upperlim95.data.cpu().numpy(),TimeseriesData.mean,TimeseriesData.std)
        #     lowerlim95 = preprocess_data.reconstruct(lowerlim95.data.cpu().numpy(),TimeseriesData.mean,TimeseriesData.std)

        plt.figure(figsize=(15,5))
        for i in range(target.size(-1)):
            plt.plot(target[:,:,i].numpy(), label='Target'+str(i),
                     color='black', marker='.', linestyle='--', markersize=1, linewidth=0.5)
            plt.plot(range(startPoint), outSeq[:startPoint,i].numpy(), label='1-step predictions for target'+str(i),
                     color='green', marker='.', linestyle='--', markersize=1.5, linewidth=1)
            # if epoch>40:
            #     plt.plot(range(startPoint, endPoint), upperlim95[:,i].numpy(), label='upperlim'+str(i),
            #              color='skyblue', marker='.', linestyle='--', markersize=1.5, linewidth=1)
            #     plt.plot(range(startPoint, endPoint), lowerlim95[:,i].numpy(), label='lowerlim'+str(i),
            #              color='skyblue', marker='.', linestyle='--', markersize=1.5, linewidth=1)
            plt.plot(range(startPoint, endPoint), outSeq[startPoint:,i].numpy(), label='Recursive predictions for target'+str(i),
                     color='blue', marker='.', linestyle='--', markersize=1.5, linewidth=1)

        plt.xlim([startPoint-500, endPoint])
        plt.xlabel('Index',fontsize=15)
        plt.ylabel('Value',fontsize=15)
        plt.title('Time-series Prediction on ' + args.data + ' Dataset', fontsize=18, fontweight='bold')
        plt.legend()
        plt.tight_layout()
        plt.text(startPoint-500+10, target.min(), 'Epoch: '+str(epoch),fontsize=15)
        save_dir = Path('result',args.data,args.filename).with_suffix('').joinpath('fig_prediction')
        save_dir.mkdir(parents=True,exist_ok=True)
        plt.savefig(save_dir.joinpath('fig_epoch'+str(epoch)).with_suffix('.png'))
        #plt.show()
        plt.close()
        return outSeq

    else:
        pass



def evaluate_1step_pred(args, model, test_dataset):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    with torch.no_grad():
        hidden = model.init_hidden(args.eval_batch_size)
        for nbatch, i in enumerate(range(0, test_dataset.size(0) - 1, args.bptt)):

            inputSeq, targetSeq = get_batch(args,test_dataset, i)
            outSeq, hidden = model.forward(inputSeq, hidden)

            loss = criterion(outSeq.view(args.batch_size,-1), targetSeq.view(args.batch_size,-1))
            hidden = model.repackage_hidden(hidden)
            total_loss+= loss.item()

    return total_loss / nbatch

def train(model, dataset, epoch):
    epoch_loss1=[]
    epoch_loss2=[]
    epoch_loss3=[]
    total_1 = 0
    total_2 = 0
    total_3 = 0
    with torch.enable_grad():
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0
        for e in range(epoch):
            epoch_loss = []
            for i, (x, y, idx) in enumerate(dataset):
                epoch_loss1.append(0)
                epoch_loss2.append(0)
                epoch_loss3.append(0)
                epoch_loss.append(0)
                hidden = model.init_hidden()
                hidden_ = model.init_hidden()
                optimizer.zero_grad()
                outVal = x[0]
                outVals=[]
                hids1 = []
                for _ in range(x.size(0)):
                    outVal, hidden_, hid = model(outVal.unsqueeze(0), hidden_,return_hiddens=True)
                    outVals.append(outVal)
                    hids1.append(hid)
                hids1 = torch.cat(hids1, dim=0)
                outSeq1 = torch.cat(outVals,dim=0)
                loss1 = criterion(outSeq1, y.view(-1))
                total_1 += loss1.item()
                epoch_loss1[-1]+=loss1.item()
                
                outSeq2, hidden, hids2 = model(x, hidden, return_hiddens=True)
                loss2 = criterion(outSeq2, y)
                total_2 += loss2.item()
                epoch_loss2[-1]+=loss2.item()
                
                loss3 = criterion(hids1.detach(), hids2.detach())
                total_3 += loss3.item()
                epoch_loss3[-1]+=loss3.item()
                
                loss = loss1+loss2+loss3
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
                total_loss += loss.item()
                epoch_loss[-1] += loss.item()
                if i % 10 == 0 and i > 0:
                    print(i, total_loss / 10, total_1/10, total_2 /10, total_3/10)
                    total_loss = 0
                    total_1 = 0
                    total_2 = 0
                    total_3 = 0
                #except:
                 #   print(idx, x_len)
            epoch_loss[-1]/=len(dataset)
            epoch_loss1[-1]/=len(dataset)
            epoch_loss2[-1]/=len(dataset)
            epoch_loss3[-1]/=len(dataset)
            print("EPOCH {:3d} : loss = {:5.4f} ,loss1 = {:5.4f} , loss2 = {:5.4f} , loss3 = {:5.4f} ".format(e, epoch_loss[-1], epoch_loss1[-1], epoch_loss2[-1], epoch_loss3[-1]))
    return epoch_loss,epoch_loss1,epoch_loss2,epoch_loss3

def evaluate(args, model, test_dataset):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for i, (x, y, idx) in enumerate(test_dataset):
            hidden = model.init_hidden()
            hidden_ = model.init_hidden()
            '''Loss1: Free running loss'''
            outVal = x[0]
            outVals=[]
            hids1 = []
            for _ in range(x.size(0)):
                outVal, hidden_, hid = model(outVal.unsqueeze(0), hidden_,return_hiddens=True)
                outVals.append(outVal)
                hids1.append(hid)
            outSeq1 = torch.cat(outVals,dim=0)
            hids1 = torch.cat(hids1,dim=0)
            loss1 = criterion(outSeq1, y.view(-1))

            '''Loss2: Teacher forcing loss'''
            outSeq2, hidden, hids2 = model(x, hidden, return_hiddens=True)
            loss2 = criterion(outSeq2, y)

            '''Loss3: Simplified Professor forcing loss'''
            loss3 = criterion(hids1.detach(), hids2.detach())

            '''Total loss = Loss1+Loss2+Loss3'''
            loss = loss1+loss2+loss3

            total_loss += loss.item()

    return total_loss / (i+1)




###############################################################################
# Load data
###############################################################################

train_dataset = TimeSeriesDataset(args.data, args.augment, args.noise_ratio)
test_dataset = TimeSeriesDataset(args.data, args.augment, args.noise_ratio, train = False, clean = False, trainset=train_dataset)
train_dataset_full = TimeSeriesDataset(args.data, args.augment, args.noise_ratio, train = True, clean = False, trainset=train_dataset)
Path('save',args.data,'checkpoint')

###############################################################################
# Build the model
###############################################################################
feature_dim = train_dataset[0][0].shape[1]
model = TimeSeriesLSTMStochastic(feature_dim, args.embed_dim, args.hidden_dim, args.nb_layers, args.dropout).to(args.device)
optimizer = optim.Adam(model.parameters(), lr= args.lr,weight_decay=args.weight_decay)
criterion = nn.MSELoss()

# Loop over epochs.
if args.resume or args.pretrained:
    print("=> loading checkpoint ")
    checkpoint = torch.load(Path('save', args.data, 'checkpoint', args.filename).with_suffix('.pth'))
    args, start_epoch, best_val_loss = model.load_checkpoint(args,checkpoint,feature_dim)
    optimizer.load_state_dict((checkpoint['optimizer']))
    del checkpoint
    epoch = start_epoch
    print("=> loaded checkpoint")
else:
    epoch = 1
    start_epoch = 1
    best_val_loss = float('inf')
    print("=> Start training from scratch")
print('-' * 89)
print(args)
print('-' * 89)

if not args.pretrained:
    # At any point you can hit Ctrl + C to break out of training early.
    try:

        train(args,model,train_dataset,args.epochs)
        val_loss = evaluate(args,model,test_dataset)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | '.format(epoch, (time.time() - epoch_start_time),                                                                                        val_loss))
        print('-' * 89)

        #generate_output(args,epoch,model,gen_dataset,startPoint=1500)

        if epoch%args.save_interval==0:
            # Save the model if the validation loss is the best we've seen so far.
            is_best = val_loss < best_val_loss
            best_val_loss = min(val_loss, best_val_loss)
            model_dictionary = {'epoch': epoch,
                                'best_loss': best_val_loss,
                                'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'args':args
                                }
            model.save_checkpoint(model_dictionary, is_best)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


# Calculate mean and covariance for each channel's prediction errors, and save them with the trained model
print('=> calculating mean and covariance')
means, covs = list(),list()
for channel_idx in range(model.feature_dim):
    mean, cov = fit_norm_distribution_param(args,model,train_dataset,channel_idx)
    means.append(mean), covs.append(cov)
model_dictionary = {'epoch': max(epoch,start_epoch),
                    'best_loss': best_val_loss,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'args': args,
                    'means': means,
                    'covs': covs
                    }
model.save_checkpoint(model_dictionary, True)
print('-' * 89)
