#--unimodals/common_models.py ------------------------------------------------------------ 
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Linear(torch.nn.Module):
    """
    Linear Layer with Xavier Initialization, and 0 Bias.
    """
    def __init__(self, indim, outdim, xavier_init=False):
        super(Linear, self).__init__()
        self.fc = nn.Linear(indim, outdim)
        if xavier_init:
            nn.init.xavier_normal(self.fc.weight)
            self.fc.bias.data.fill_(0.0)

    def forward(self, x):
        return self.fc(x)


class LeNet(nn.Module):
    """
    Adapted from centralnet code https://github.com/slyviacassell/_MFAS/blob/master/models/central/avmnist.py.
    LeNet.
    """
    def __init__(self, in_channels, args_channels, additional_layers, output_each_layer=False, linear=None, squeeze_output=True):
        super(LeNet, self).__init__()
        self.output_each_layer = output_each_layer
        self.convs = [
            nn.Conv2d(in_channels, args_channels, kernel_size=5, padding=2, bias=False)]
        self.bns = [nn.BatchNorm2d(args_channels)]
        self.gps = [GlobalPooling2D()]
        for i in range(additional_layers):
            self.convs.append(nn.Conv2d((2**i)*args_channels, (2**(i+1))
                              * args_channels, kernel_size=3, padding=1, bias=False))
            self.bns.append(nn.BatchNorm2d(args_channels*(2**(i+1))))
            self.gps.append(GlobalPooling2D())
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)
        self.gps = nn.ModuleList(self.gps)
        self.sq_out = squeeze_output
        self.linear = None
        if linear is not None:
            self.linear = nn.Linear(linear[0], linear[1])
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        tempouts = []
        out = x
        for i in range(len(self.convs)):
            out = F.relu(self.bns[i](self.convs[i](out)))
            out = F.max_pool2d(out, 2)
            gp = self.gps[i](out)
            tempouts.append(gp)
            
        if self.linear is not None:
            out = self.linear(out)
        tempouts.append(out)
        if self.output_each_layer:
            if self.sq_out:
                return [t.squeeze() for t in tempouts]
            return tempouts
        if self.sq_out:
            return out.squeeze()
        return out
class GlobalPooling2D(nn.Module):
    def __init__(self):
        super(GlobalPooling2D, self).__init__()

    def forward(self, x):
        # apply global average pooling
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.mean(x, 2)
        x = x.view(x.size(0), -1)

        return x
#--unimodals/common_models.py_END ------------------------------------------------------------

#--utils/helper_modules.py ------------------------------------------------------------ 
class Sequential2(nn.Module):
    def __init__(self, a, b):
        super(Sequential2, self).__init__()
        self.model = nn.Sequential(a, b)

    def forward(self, x):
        return self.model(x)
#--utils/helper_modules.py_END ------------------------------------------------------------

#--fusions/common_fusions.py ------------------------------------------------------------ 

# Simple concatenation on dim 1
class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, modalities):
        flattened = []
        for modality in modalities:
            flattened.append(torch.flatten(modality, start_dim=1))
        return torch.cat(flattened, dim=1)
#--fusions/common_fusions_END ------------------------------------------------------------

#--training_structures/Supervised_Learning.py ------------------------------------------------------------ 
import time
from tqdm import tqdm

softmax = nn.Softmax()


class MMDL(nn.Module):
    def __init__(self, encoders, fusion, head, has_padding=False):
        super(MMDL, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fuse = fusion
        self.head = head
        self.has_padding = has_padding
        self.fuseout = None
        self.reps = []

    def forward(self, inputs):
        outs = []
        if self.has_padding:
            for i in range(len(inputs[0])):
                outs.append(self.encoders[i](
                    [inputs[0][i], inputs[1][i]]))
        else:
            for i in range(len(inputs)):
                outs.append(self.encoders[i](inputs[i]))
        self.reps = outs
        if self.has_padding:
            
            if isinstance(outs[0], torch.Tensor):
                out = self.fuse(outs)
            else:
                out = self.fuse([i[0] for i in outs])
        else:
            out = self.fuse(outs)
        self.fuseout = out
        if type(out) is tuple:
            out = out[0]
        if self.has_padding and not isinstance(outs[0], torch.Tensor):
            return self.head([out, inputs[1][0]])
        return self.head(out)


def deal_with_objective(objective, pred, truth, args):
    if type(objective) == nn.CrossEntropyLoss:
        if len(truth.size()) == len(pred.size()):
            truth1 = truth.squeeze(len(pred.size())-1)
        else:
            truth1 = truth
        return objective(pred, truth1.long().cuda())
    elif type(objective) == nn.MSELoss or type(objective) == nn.modules.loss.BCEWithLogitsLoss or type(objective) == nn.L1Loss:
        return objective(pred, truth.float().cuda())
    else:
        return objective(pred, truth, args)

# encoders: list of modules, unimodal encoders for each input modality in the order of the modality input data.
# fusion: fusion module, takes in outputs of encoders in a list and outputs fused representation
# head: classification or prediction head, takes in output of fusion module and outputs the classification or prediction results that will be sent to the objective function for loss calculation
# total_epochs: maximum number of epochs to train
# additional_optimizing_modules: list of modules, include all modules that you want to be optimized by the optimizer other than those in encoders, fusion, head (for example, decoders in MVAE)
# is_packed: whether the input modalities are packed in one list or not (default is False, which means we expect input of [tensor(20xmodal1_size),(20xmodal2_size),(20xlabel_size)] for batch size 20 and 2 input modalities)
# early_stop: whether to stop early if valid performance does not improve over 7 epochs
# task: type of task, currently support "classification","regression","multilabel"
# optimtype: type of optimizer to use
# lr: learning rate
# weight_decay: weight decay of optimizer
# objective: objective function, which is either one of CrossEntropyLoss, MSELoss or BCEWithLogitsLoss or a custom objective function that takes in three arguments: prediction, ground truth, and an argument dictionary.
# auprc: whether to compute auprc score or not
# save: the name of the saved file for the model with current best validation performance
# validtime: whether to show valid time in seconds or not
# objective_args_dict: the argument dictionary to be passed into objective function. If not None, at every batch the dict's "reps", "fused", "inputs", "training" fields will be updated to the batch's encoder outputs, fusion module output, input tensors, and boolean of whether this is training or validation, respectively.
# input_to_float: whether to convert input to float type or not
# clip_val: grad clipping limit
# track_complexity: whether to track training complexity or not


def train(
        encoders, fusion, head, train_dataloader, valid_dataloader, total_epochs, additional_optimizing_modules=[], is_packed=False,
        early_stop=False, task="classification", optimtype=torch.optim.RMSprop, lr=0.001, weight_decay=0.0,
        objective=nn.CrossEntropyLoss(), auprc=False, save='best.pt', validtime=False, objective_args_dict=None, input_to_float=True, clip_val=8,
        track_complexity=True):
    model = MMDL(encoders, fusion, head, has_padding=is_packed).cuda()

    def trainprocess():
        additional_params = []
        for m in additional_optimizing_modules:
            additional_params.extend(
                [p for p in m.parameters() if p.requires_grad])
        op = optimtype([p for p in model.parameters() if p.requires_grad] +
                       additional_params, lr=lr, weight_decay=weight_decay)
        bestvalloss = 10000
        bestacc = 0
        bestf1 = 0
        patience = 0

        def processinput(inp):
            if input_to_float:
                return inp.float()
            else:
                return inp

        for epoch in range(total_epochs):
            totalloss = 0.0
            totals = 0
            model.train()
            for j in train_dataloader:
                op.zero_grad()
                if is_packed:
                    with torch.backends.cudnn.flags(enabled=False):
                        model.train()
                        out = model([[processinput(i).cuda()
                                    for i in j[0]], j[1]])

                else:
                    model.train()
                    out = model([processinput(i).cuda()
                                for i in j[:-1]])
                if not (objective_args_dict is None):
                    objective_args_dict['reps'] = model.reps
                    objective_args_dict['fused'] = model.fuseout
                    objective_args_dict['inputs'] = j[:-1]
                    objective_args_dict['training'] = True
                    objective_args_dict['model'] = model
                loss = deal_with_objective(
                    objective, out, j[-1], objective_args_dict)

                totalloss += loss * len(j[-1])
                totals += len(j[-1])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                op.step()
            print("Epoch "+str(epoch)+" train loss: "+str(totalloss/totals))
            validstarttime = time.time()
            if validtime:
                print("train total: "+str(totals))
            model.eval()
            with torch.no_grad():
                totalloss = 0.0
                pred = []
                true = []
                pts = []
                for j in valid_dataloader:
                    if is_packed:
                        model.train()
                        out = model([[processinput(i).cuda()
                                    for i in j[0]], j[1]])
                    else:
                        model.train()
                        out = model([processinput(i).cuda()
                                    for i in j[:-1]])

                    if not (objective_args_dict is None):
                        objective_args_dict['reps'] = model.reps
                        objective_args_dict['fused'] = model.fuseout
                        objective_args_dict['inputs'] = j[:-1]
                        objective_args_dict['training'] = False
                    loss = deal_with_objective(
                        objective, out, j[-1], objective_args_dict)
                    totalloss += loss*len(j[-1])
                    
                    if task == "classification":
                        pred.append(torch.argmax(out, 1))
                    elif task == "multilabel":
                        pred.append(torch.sigmoid(out).round())
                    true.append(j[-1])
                    if auprc:
                        # pdb.set_trace()
                        sm = softmax(out)
                        pts += [(sm[i][1].item(), j[-1][i].item())
                                for i in range(j[-1].size(0))]
            if pred:
                pred = torch.cat(pred, 0)
            true = torch.cat(true, 0)
            totals = true.shape[0]
            valloss = totalloss/totals
            if task == "classification":
                acc = accuracy(true, pred)
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss) +
                      " acc: "+str(acc))
                if acc > bestacc:
                    patience = 0
                    bestacc = acc
                    print("Saving Best")
                    torch.save(model, save)
                else:
                    patience += 1
            elif task == "multilabel":
                f1_micro = f1_score(true, pred, average="micro")
                f1_macro = f1_score(true, pred, average="macro")
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss) +
                      " f1_micro: "+str(f1_micro)+" f1_macro: "+str(f1_macro))
                if f1_macro > bestf1:
                    patience = 0
                    bestf1 = f1_macro
                    print("Saving Best")
                    torch.save(model, save)
                else:
                    patience += 1
            elif task == "regression":
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss.item()))
                if valloss < bestvalloss:
                    patience = 0
                    bestvalloss = valloss
                    print("Saving Best")
                    torch.save(model, save)
                else:
                    patience += 1
            if early_stop and patience > 7:
                break
            if auprc:
                print("AUPRC: "+str(AUPRC(pts)))
            validendtime = time.time()
            if validtime:
                print("valid time:  "+str(validendtime-validstarttime))
                print("Valid total: "+str(totals))
    if track_complexity:
        all_in_one_train(trainprocess, [model]+additional_optimizing_modules)
    else:
        trainprocess()


def single_test(
        model, test_dataloader, is_packed=False,
        criterion=nn.CrossEntropyLoss(), task="classification", auprc=False, input_to_float=True):
    def processinput(inp):
        if input_to_float:
            return inp.float()
        else:
            return inp
    with torch.no_grad():
        totalloss = 0.0
        pred = []
        true = []
        pts = []
        for j in test_dataloader:
            model.eval()
            if is_packed:
                out = model([[processinput(i).cuda()
                            for i in j[0]], j[1]])
            else:
                out = model([processinput(i).float().cuda()
                            for i in j[:-1]])
            if type(criterion) == torch.nn.modules.loss.BCEWithLogitsLoss or type(criterion) == torch.nn.MSELoss:
                loss = criterion(out, j[-1].float().cuda())

            # elif type(criterion) == torch.nn.CrossEntropyLoss:
            #     loss=criterion(out, j[-1].long().cuda())

            elif type(criterion) == nn.CrossEntropyLoss:
                if len(j[-1].size()) == len(out.size()):
                    truth1 = j[-1].squeeze(len(out.size())-1)
                else:
                    truth1 = j[-1]
                loss = criterion(out, truth1.long().cuda())
            else:
                loss = criterion(out, j[-1].cuda())
            totalloss += loss*len(j[-1])
            if task == "classification":
                pred.append(torch.argmax(out, 1))
            elif task == "multilabel":
                pred.append(torch.sigmoid(out).round())
            elif task == "posneg-classification":
                prede = []
                oute = out.cpu().numpy().tolist()
                for i in oute:
                    if i[0] > 0:
                        prede.append(1)
                    elif i[0] < 0:
                        prede.append(-1)
                    else:
                        prede.append(0)
                pred.append(torch.LongTensor(prede))
            true.append(j[-1])
            if auprc:
                # pdb.set_trace()
                sm = softmax(out)
                pts += [(sm[i][1].item(), j[-1][i].item())
                        for i in range(j[-1].size(0))]
        if pred:
            pred = torch.cat(pred, 0)
        true = torch.cat(true, 0)
        totals = true.shape[0]
        testloss = totalloss/totals
        if auprc:
            print("AUPRC: "+str(AUPRC(pts)))
        if task == "classification":
            print("acc: "+str(accuracy(true, pred)))
            return {'Accuracy': accuracy(true, pred)}
        elif task == "multilabel":
            print(" f1_micro: "+str(f1_score(true, pred, average="micro")) +
                  " f1_macro: "+str(f1_score(true, pred, average="macro")))
            return {'micro': f1_score(true, pred, average="micro"), 'macro': f1_score(true, pred, average="macro")}
        elif task == "regression":
            print("mse: "+str(testloss.item()))
            return {'MSE': testloss.item()}
        elif task == "posneg-classification":
            trueposneg = true
            accs = eval_affect(trueposneg, pred)
            acc2 = eval_affect(trueposneg, pred, exclude_zero=False)
            print("acc: "+str(accs) + ', ' + str(acc2))
            return {'Accuracy': accs}


# model: saved checkpoint filename from train
# test_dataloaders_all: test data
# dataset: the name of dataset, need to be set for testing effective robustness
# criterion: only needed for regression, put MSELoss there
# all other arguments are same as train
def test(
        model, test_dataloaders_all, dataset='default', method_name='My method', is_packed=False, criterion=nn.CrossEntropyLoss(), task="classification", auprc=False, input_to_float=True, no_robust=False):
    if no_robust:
        def testprocess():
            single_test(model, test_dataloaders_all, is_packed,
                        criterion, task, auprc, input_to_float)
        all_in_one_test(testprocess, [model])
        return

    def testprocess():
        single_test(model, test_dataloaders_all[list(test_dataloaders_all.keys())[
                    0]][0], is_packed, criterion, task, auprc, input_to_float)
    all_in_one_test(testprocess, [model])
    for noisy_modality, test_dataloaders in test_dataloaders_all.items():
        print("Testing on noisy data ({})...".format(noisy_modality))
        robustness_curve = dict()
        for test_dataloader in tqdm(test_dataloaders):
            single_test_result = single_test(
                model, test_dataloader, is_packed, criterion, task, auprc, input_to_float)
            for k, v in single_test_result.items():
                curve = robustness_curve.get(k, [])
                curve.append(v)
                robustness_curve[k] = curve
        for measure, robustness_result in robustness_curve.items():
            robustness_key = '{} {}'.format(dataset, noisy_modality)
            print("relative robustness ({}, {}): {}".format(noisy_modality, measure, str(
                relative_robustness(robustness_result, robustness_key))))
            if len(robustness_curve) != 1:
                robustness_key = '{} {}'.format(robustness_key, measure)
            print("effective robustness ({}, {}): {}".format(noisy_modality, measure, str(
                effective_robustness(robustness_result, robustness_key))))
            fig_name = '{}-{}-{}-{}'.format(method_name,
                                            robustness_key, noisy_modality, measure)
            single_plot(robustness_result, robustness_key, xlabel='Noise level',
                        ylabel=measure, fig_name=fig_name, method=method_name)
            print("Plot saved as "+fig_name)

#--training_structures/Supervised_Learning.py_END ------------------------------------------------------------


#--datasets/avmnist/get_data.py ------------------------------------------------------------ 
import numpy as np
from torch.utils.data import DataLoader

# data dir is the avmnist folder


def get_dataloader(data_dir, batch_size=40, num_workers=8, train_shuffle=True, flatten_audio=False, flatten_image=False, unsqueeze_channel=True, generate_sample=False, normalize_image=True, normalize_audio=True):
    trains = [np.load(data_dir+"/image/train_data.npy"), np.load(data_dir +
                                                                 "/audio/train_data.npy"), np.load(data_dir+"/train_labels.npy")]
    tests = [np.load(data_dir+"/image/test_data.npy"), np.load(data_dir +
                                                               "/audio/test_data.npy"), np.load(data_dir+"/test_labels.npy")]
    if flatten_audio:
        trains[1] = trains[1].reshape(60000, 112*112)
        tests[1] = tests[1].reshape(10000, 112*112)
    if generate_sample:
        saveimg(trains[0][0:100])
        saveaudio(trains[1][0:9].reshape(9, 112*112))
    if normalize_image:
        trains[0] /= 255.0
        tests[0] /= 255.0
    if normalize_audio:
        trains[1] = trains[1]/255.0
        tests[1] = tests[1]/255.0
    if not flatten_image:
        trains[0] = trains[0].reshape(60000, 28, 28)
        tests[0] = tests[0].reshape(10000, 28, 28)
    if unsqueeze_channel:
        trains[0] = np.expand_dims(trains[0], 1)
        tests[0] = np.expand_dims(tests[0], 1)
        trains[1] = np.expand_dims(trains[1], 1)
        tests[1] = np.expand_dims(tests[1], 1)
    trains[2] = trains[2].astype(int)
    tests[2] = tests[2].astype(int)
    trainlist = [[trains[j][i] for j in range(3)] for i in range(60000)]
    testlist = [[tests[j][i] for j in range(3)] for i in range(10000)]
    valids = DataLoader(trainlist[55000:60000], shuffle=False,
                        num_workers=num_workers, batch_size=batch_size)
    tests = DataLoader(testlist, shuffle=False,
                       num_workers=num_workers, batch_size=batch_size)
    trains = DataLoader(trainlist[0:55000], shuffle=train_shuffle,
                        num_workers=num_workers, batch_size=batch_size)
    return trains, valids, tests

# this function creates an image of 100 numbers in avmnist

def saveimg(outa):
    from PIL import Image
    t = np.zeros((300, 300))
    for i in range(0, 100):
        for j in range(0, 784):
            imrow = i // 10
            imcol = i % 10
            pixrow = j // 28
            pixcol = j % 28
            t[imrow*30+pixrow][imcol*30+pixcol] = outa[i][j]
    newimage = Image.new('L', (300, 300))  # type, size
    
    newimage.putdata(t.reshape((90000,)))
    newimage.save("samples.png")


def saveaudio(outa):
    
    from PIL import Image
    t = np.zeros((340, 340))
    for i in range(0, 9):
        for j in range(0, 112*112):
            imrow = i // 3
            imcol = i % 3
            pixrow = j // 112
            pixcol = j % 112
            t[imrow*114+pixrow][imcol*114+pixcol] = outa[i][j]
    newimage = Image.new('L', (340, 340))  # type, size
    
    newimage.putdata(t.reshape((340*340,)))
    newimage.save("samples2.png")
#--datasets/avmnist.get_data.py_END ------------------------------------------------------------

#--objective_functions/objectives_for_supervised_learning.py ------------------------------------------------------------ 
#from objective_functions.recon import recon_weighted_sum, elbo_loss
#from objective_functions.cca import CCALoss

def _criterioning(pred, truth, criterion):
    """Handle criterion ideosyncracies."""
    if isinstance(criterion, torch.nn.CrossEntropyLoss):
        truth = truth.squeeze() if len(truth.shape) == len(pred.shape) else truth
        return criterion(pred, truth.long().cuda())
    if isinstance(criterion, (torch.nn.modules.loss.BCEWithLogitsLoss, torch.nn.MSELoss, torch.nn.L1Loss)):
        return criterion(pred, truth.float().cuda())

def CCA_objective(out_dim, cca_weight=0.001, criterion=torch.nn.CrossEntropyLoss()):
    """
    Define loss function for CCA.
    
    :param out_dim: output dimension
    :param cca_weight: weight of cca loss
    :param criterion: criterion for supervised loss
    """
    lossfunc = CCALoss(out_dim, False, device=torch.device("cuda"))

    def actualfunc(pred, truth, args):
        ce_loss = _criterioning(pred, truth, criterion)
        outs = args['reps']
        cca_loss = lossfunc(outs[0], outs[1])
        return cca_loss * cca_weight + ce_loss
    return actualfunc
#--objective_functions/objectives_for_supervised_learning.py_END ------------------------------------------------------------

#--objective_functions/recon.py ------------------------------------------------------------

def recon_weighted_sum(modal_loss_funcs, weights):
    def actualfunc(recons, origs):
        totalloss = 0.0
        for i in range(len(recons)):
            trg = origs[i].view(recons[i].shape[0], recons[i].shape[1]) if len(
                recons[i].shape) != len(origs[i].shape) else origs[i]
            totalloss += modal_loss_funcs[i](recons[i], trg)*weights[i]
        return torch.mean(totalloss)
    return actualfunc

#--objective_functions/recon.py_END ------------------------------------------------------------

#--objective_functions/cca.py ------------------------------------------------------------
class CCALoss(nn.Module):
    def __init__(self, outdim_size, use_all_singular_values, device):
        super(CCALoss, self).__init__()
        self.outdim_size = outdim_size
        self.use_all_singular_values = use_all_singular_values
        self.device = device        

    def forward(self, H1, H2):
        """
        It is the loss function of CCA as introduced in the original paper. There can be other formulations.
        """

        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-9

        H1, H2 = H1.t(), H2.t()
        
        
        assert torch.isnan(H1).sum().item() == 0
        assert torch.isnan(H2).sum().item() == 0

        o1 = o2 = H1.size(0)

        m = H1.size(1)


        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)
        # assert torch.isnan(H1bar).sum().item() == 0
        # assert torch.isnan(H2bar).sum().item() == 0

        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar,
                                                    H1bar.t()) + r1 * torch.eye(o1, device=self.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar,
                                                    H2bar.t()) + r2 * torch.eye(o2, device=self.device)
        # assert torch.isnan(SigmaHat11).sum().item() == 0
        # assert torch.isnan(SigmaHat12).sum().item() == 0
        # assert torch.isnan(SigmaHat22).sum().item() == 0

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        [D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)
        [D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)
        # assert torch.isnan(D1).sum().item() == 0
        # assert torch.isnan(D2).sum().item() == 0
        # assert torch.isnan(V1).sum().item() == 0
        # assert torch.isnan(V2).sum().item() == 0

        # Added to increase stability
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]        
        
        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)


        if self.use_all_singular_values:
            # all singular values are used to calculate the correlation
            tmp = torch.matmul(Tval.t(), Tval)
            corr = torch.trace(torch.sqrt(tmp))
            # assert torch.isnan(corr).item() == 0
        else:
            # just the top self.outdim_size singular values are used
            trace_TT = torch.matmul(Tval.t(), Tval)
            # regularization for more stability
            trace_TT = torch.add(trace_TT, (torch.eye(
                trace_TT.shape[0])*r1).to(self.device))
            U, V = torch.symeig(trace_TT, eigenvectors=True)
            U = torch.where(U > eps, U, (torch.ones(
                U.shape).float()*eps).to(self.device))
            U = U.topk(self.outdim_size)[0]
            corr = torch.sum(torch.sqrt(U))
        
        return -corr
#--objective_functions/cca.py_END ------------------------------------------------------------

#--eval_scripts/performance.py ------------------------------------------------------------ 
import sklearn.metrics

def ptsort(tu):
    return tu[0]

def AUPRC(pts):
    true_labels = [int(x[1]) for x in pts]
    predicted_probs = [x[0] for x in pts]
    return sklearn.metrics.average_precision_score(true_labels, predicted_probs)


def f1_score(truth, pred, average):
    return sklearn.metrics.f1_score(truth.cpu().numpy(), pred.cpu().numpy(), average=average)


def accuracy(truth, pred):
    return sklearn.metrics.accuracy_score(truth.cpu().numpy(), pred.cpu().numpy())


def eval_affect(truths, results, exclude_zero=True):
    if type(results) is np.ndarray:
        test_preds = results
        test_truth = truths
    else:
        test_preds = results.cpu().numpy()
        test_truth = truths.cpu().numpy()

    non_zeros = np.array([i for i, e in enumerate(
        test_truth) if e != 0 or (not exclude_zero)])

    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)

    return sklearn.metrics.accuracy_score(binary_truth, binary_preds)

#--eval_scripts/performance.py_END ------------------------------------------------------------


#--eval_scripts/complexity.py ------------------------------------------------------------ 
from memory_profiler import memory_usage

def getallparams(li):
    params = 0
    for module in li:
        for param in module.parameters():
            params += param.numel()
    return params


def all_in_one_train(trainprocess, trainmodules):
    starttime = time.time()
    mem = max(memory_usage(proc=trainprocess))
    endtime = time.time()

    print("Training Time: "+str(endtime-starttime))
    print("Training Peak Mem: "+str(mem))
    print("Training Params: "+str(getallparams(trainmodules)))


def all_in_one_test(testprocess, testmodules):
    teststart = time.time()
    testprocess()
    testend = time.time()
    print("Inference Time: "+str(testend-teststart))
    print("Inference Params: "+str(getallparams(testmodules)))

#--eval_scripts/complexity.py_END ------------------------------------------------------------


#--eval_scripts/robustness.py ------------------------------------------------------------ 
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def relative_robustness(robustness_result, task):
    return get_robustness_metric(robustness_result, task, 'relative')


def effective_robustness(robustness_result, task):
    return get_robustness_metric(robustness_result, task, 'effective')


def get_robustness_metric(robustness_result, task, metric):
    if metric == 'effective' and task not in robustness['LF']:
        return "Invalid example name!"
    else:
        result = dict()
        if metric == 'relative':
            helper = relative_robustness_helper
        elif metric == 'effective':
            helper = effective_robustness_helper
        my_method = helper(robustness_result, task)
        for method in list(robustness.keys()):
            if not method.endswith('Transformer'):
                for t in list(robustness[method].keys()):
                    if t == task:
                        if (method == 'EF' or method == 'LF') and task in robustness[method+'-Transformer']:
                            result[method] = helper((np.array(
                                robustness[method][task])+np.array(robustness[method+'-Transformer'][task]))/2, task)
                        else:
                            result[method] = helper(
                                robustness[method][task], task)
        result['my method'] = my_method
        return maxmin_normalize(result, task)


def relative_robustness_helper(robustness_result, task):
    area = 0
    for i in range(len(robustness_result)-1):
        area += (robustness_result[i] + robustness_result[i+1]) * 0.1 / 2
    return area


def effective_robustness_helper(robustness_result, task):
    f = np.array(robustness_result)
    lf = np.array(robustness['LF'][task])
    beta_f = lf + (f[0] - lf[0])
    return np.sum(f - beta_f)


def maxmin_normalize(result, task):
    tmp = []
    method2idx = dict()
    for i, method in enumerate(list(result.keys())):
        method2idx[method] = i
        tmp.append(result[method])
    tmp = np.array(tmp)
    if task.startswith('finance'):
        tmp = -1 * tmp
    tmp = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp))
    return tmp[method2idx['my method']]


def single_plot(robustness_result, task, xlabel, ylabel, fig_name, method):
    fig, axs = plt.subplots()
    if task.startswith('gentle push') or task.startswith('robotics image') or task.startswith('robotics force'):
        robustness_result = list(np.log(np.array(robustness_result)))
        plt.ylabel('log '+ylabel, fontsize=20)
    axs.plot(np.arange(len(robustness_result)) / 10,
             robustness_result, label=method, linewidth=2.5)
    plt.xlabel(xlabel, fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # Uncomment the line below to show legends
    # fig.legend(fontsize=15, loc='upper left', bbox_to_anchor=(0.92, 0.94))
    axs.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close(fig)
#--eval_scripts/robustness.py_END ------------------------------------------------------------


#--avmnist_cca.py ------------------------------------------------------------ 
import sys
import os
sys.path.append(os.getcwd())

#from unimodals.common_models import LeNet
#from unimodals.common_models import Linear
#from utils.helper_modules import Sequential2
#from fusions.common_fusions import Concat
#from training_structures.Supervised_Learning import train, test
#from datasets.avmnist.get_data import get_dataloader
#from objective_functions.objectives_for_supervised_learning import CCA_objective


traindata, validdata, testdata = get_dataloader(
    '/home/hugh/dev/datasets/avmnist', batch_size=800)
channels = 6
encoders = [LeNet(1, channels, 3).cuda(), Sequential2(
    LeNet(1, channels, 5), Linear(192, 48, xavier_init=True)).cuda()]
#encoders=[MLP(300,512,outdim), MLP(4096,1024,outdim)]
#encoders=[MLP(300, 512, 512), VGG16(512)]
#encoders=[Linear(300, 512), Linear(4096,512)]
# head=MLP(2*outdim,2*outdim,23).cuda()
head = Linear(96, 10, xavier_init=True).cuda()
fusion = Concat().cuda()

train(encoders, fusion, head, traindata, validdata, 25,
      save="best_cca.pt", optimtype=torch.optim.AdamW, lr=1e-2, objective=CCA_objective(48), objective_args_dict={})
# ,weight_decay=0.01)

print("Testing:")
model = torch.load('best_cca.pt').cuda()
test(model, testdata, no_robust=True)
#--avmnist_cca.py_END ------------------------------------------------------------ 