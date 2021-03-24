from torch.autograd import Variable
import torch
import numpy as np
from tqdm import tqdm
import itertools

def fit_norm_distribution_param(args, model, train_dataset):
    errors = []
    for i in tqdm(range(len(train_dataset))):
        sample = train_dataset[i][0]
        if len(sample)<10:
            continue
        predictions = []
        organized = []
        with torch.no_grad():
            # Turn on evaluation mode which disables dropout.
            model.eval()
            pasthidden = model.init_hidden()
            for t in range(len(sample)):
                up_bound = min(args.prediction_window_size, len(sample) - t)
                out, hidden = model(sample[t].unsqueeze(0), pasthidden, False)
                predictions.append([])
                predictions[t].append(out.data.cpu())
                pasthidden = model.repackage_hidden(hidden)
                for j in range(up_bound - 1):
                    out = torch.cat((out,sample[t+j+1,4:]), axis = 0)
                    out, hidden = model(out.unsqueeze(0), hidden, False)
                    predictions[t].append(out.data.cpu())

                if t >= args.prediction_window_size:
                    organized.append([])
                    errors.append([])
                    for step in range(args.prediction_window_size):
                        organized[-1].append(predictions[step+t-args.prediction_window_size][args.prediction_window_size-1-step])
                    organized[-1]= torch.stack(organized[-1])
                    errors[-1] = organized[-1] - sample[t,:4].data.cpu()
                    errors[-1] = errors[-1].unsqueeze(0)

    errors_tensor = torch.cat(errors, dim = 0)
    mean = errors_tensor.mean(dim=0)
    cov = []
    for i in range(train_dataset[0][1].shape[1]):
        cov.append(errors_tensor[:,:,i].t().mm(errors_tensor[:,:,i])/errors_tensor.size(0) - mean[:,i].unsqueeze(1).mm(mean[:,i].unsqueeze(0)))

    return mean, cov


def anomalyScore(args, model, dataset, mean, cov, channel_idx = 0, score_predictor = None):
    scores = []
    hiddens = []
    predicted_scores = []
    rearranged = []
    errors = []
    predictions = []
    for i in tqdm(range(len(dataset))):
        sample = dataset[i][0]
        hiddens.append([])
        predicted_scores.append([])
        rearranged.append([])
        errors.append([])
        predictions.append([])
        with torch.no_grad():
            # Turn on evaluation mode which disables dropout.
            model.eval()
            pasthidden = model.init_hidden()
            for t in range(len(sample)):
                up_bound = min(args.prediction_window_size, len(sample) - t)
                out, hidden = model(sample[t].unsqueeze(0), pasthidden, False)
                predictions[i].append([])
                rearranged[i].append([])
                errors[i].append([])
                hiddens[i].append(hidden[0][-1].data.cpu())
                if score_predictor is not None:
                    predicted_scores.append(score_predictor.predict(hidden[0][-1].data.cpu().numpy()))

                predictions[i][t].append(out.data.cpu()[channel_idx])
                pasthidden = model.repackage_hidden(hidden)
                for j in range(up_bound - 1):
                    out = torch.cat((out,sample[t+j+1,4:]), axis = 0)
                    out, hidden = model(out.unsqueeze(0), hidden, False)
                    predictions[i][t].append(out.data.cpu()[channel_idx])

                if t >= args.prediction_window_size:
                    for step in range(args.prediction_window_size):
                        rearranged[i][t].append(
                            predictions[i][step + t - args.prediction_window_size][args.prediction_window_size - 1 - step])
                    rearranged[i][t] =torch.FloatTensor(rearranged[i][t]).unsqueeze(0)
                    errors[i][t] = rearranged[i][t] - sample[t][channel_idx].data.cpu()
                else:
                    rearranged[i][t] = torch.zeros(1,args.prediction_window_size)
                    errors[i][t] = torch.zeros(1,args.prediction_window_size)

        predicted_scores[i] = np.array(predicted_scores[i])
        for error in errors[i]:
            mult1 = error-mean.unsqueeze(0) # [ 1 * prediction_window_size ]
            mult2 = torch.inverse(cov) # [ prediction_window_size * prediction_window_size ]
            mult3 = mult1.t() # [ prediction_window_size * 1 ]
            score = torch.mm(mult1,torch.mm(mult2,mult3))
            scores.append(score[0][0])

    scores = torch.stack(scores)
    rearranged = torch.cat(list(itertools.chain.from_iterable(rearranged)))
    errors = torch.cat(list(itertools.chain.from_iterable(errors)))
    return scores, rearranged, errors, hiddens, predicted_scores

def get_precision_recall(args, score, label, num_samples, beta=1.0, sampling='log', predicted_score=None):
    '''
    :param args:
    :param score: anomaly scores
    :param label: anomaly labels
    :param num_samples: the number of threshold samples
    :param beta:
    :param scale:
    :return:
    '''
    if predicted_score is not None:
        score = score - torch.FloatTensor(predicted_score).squeeze().to(args.device)

    maximum = score.max()
    score = score.to(args.device)
    if sampling=='log':
        # Sample thresholds logarithmically
        # The sampled thresholds are logarithmically spaced between: math:`10 ^ {start}` and: math:`10 ^ {end}`.
        th = torch.logspace(0, torch.log10(torch.tensor(maximum)), num_samples).to(args.device)
    else:
        # Sample thresholds equally
        # The sampled thresholds are equally spaced points between: attr:`start` and: attr:`end`
        th = torch.linspace(0, maximum, num_samples).to(args.device)

    precision = []
    recall = []

    for i in range(len(th)):
        anomaly = (score > th[i]).float()
        idx = anomaly * 2 + label
        tn = (idx == 0.0).sum().item()  # tn
        fn = (idx == 1.0).sum().item()  # fn
        fp = (idx == 2.0).sum().item()  # fp
        tp = (idx == 3.0).sum().item()  # tp

        p = tp / (tp + fp + 1e-7)
        r = tp / (tp + fn + 1e-7)

        if p != 0 and r != 0:
            precision.append(p)
            recall.append(r)

    precision = torch.FloatTensor(precision)
    recall = torch.FloatTensor(recall)


    f1 = (1 + beta ** 2) * (precision * recall).div(beta ** 2 * precision + recall + 1e-7)

    return precision, recall, f1