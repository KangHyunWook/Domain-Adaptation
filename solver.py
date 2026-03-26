from sklearn.metrics import classification_report, accuracy_score

import torch
import models
import numpy as np
import os

import torch.nn as nn



class Solver(object):
    def __init__(self, source_config, target_config, source_dataloader, target_dataloader, device, is_train=True, model=None):
        self.source_config = source_config
        self.source_dataloader = source_dataloader
        self.target_dataloader = target_dataloader
        self.model=model
        self.is_train=is_train
        self.device = device
        self.source_config.device=device
        self.i=0
        
    def build(self, cuda=True):
        if self.model is None:
            self.model = getattr(models, self.source_config.model_name)(self.source_config)
        
        if torch.cuda.is_available() and cuda:
            self.model.to(self.device)
            
        if self.is_train:
            self.optimizer = self.source_config.optimizer(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.source_config.learning_rate)

        self.loss_domain = torch.nn.NLLLoss()             

    def train(self):
        curr_patience = patience = self.source_config.patience
        
        
        self.criterion = criterion = nn.CrossEntropyLoss(reduction="mean")
        self.domain_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        

        best_valid_loss = float('inf')
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)
        
        train_losses = []
        valid_losses = []

        len_dataloader = len(self.target_dataloader)
        

        for e in range(self.source_config.num_epochs):
            len_dataloader = min(len(self.source_dataloader), len(self.target_dataloader))
            data_source_iter = iter(self.source_dataloader)
            data_target_iter = iter(self.target_dataloader)

            i = 0
            while i < len_dataloader:
                p = float(i + e * len_dataloader) / self.source_config.num_epochs / len_dataloader
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                # training model using source data
                data_source = next(data_source_iter)
                s_img, s_label = data_source
                
                s_label=s_label[self.source_config.label_type]

                self.model.zero_grad()
                batch_size = len(s_label)

                input_img = torch.FloatTensor(batch_size, self.source_config.num_electrodes, s_img.shape[2], s_img.shape[3])
                
                class_label = torch.LongTensor(batch_size)
                domain_label = torch.zeros(batch_size)
                domain_label = domain_label.long()


                s_img = s_img.to(self.device)
                s_label = s_label.to(self.device)
                input_img = input_img.to(self.device)
                class_label = class_label.to(self.device)
                domain_label = domain_label.to(self.device)

                input_img.resize_as_(s_img).copy_(s_img)
                class_label.resize_as_(s_label).copy_(s_label)

                class_output, domain_output = self.model(input_img, alpha=alpha)
                err_s_label = self.criterion(class_output, class_label)
                err_s_domain = self.domain_loss_criterion(domain_output, domain_label)

                # training model using target data
                data_target = next(data_target_iter)
                t_img, _ = data_target

                batch_size = len(t_img)

                input_img = torch.FloatTensor(batch_size, self.source_config.num_electrodes, s_img.shape[2], s_img.shape[3])
                domain_label = torch.ones(batch_size)
                domain_label = domain_label.long()


                t_img = t_img.to(self.device)
                input_img = input_img.to(self.device)
                domain_label = domain_label.to(self.device)

                input_img.resize_as_(t_img).copy_(t_img)

                _, domain_output = self.model(input_img, alpha=alpha)
                err_t_domain = self.domain_loss_criterion(domain_output, domain_label)
                err = err_t_domain + err_s_domain + err_s_label
                err.backward()
                self.optimizer.step()

                i += 1
                # print('err:', err.item())
                # print ('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f')\
                #     % (e, i, len_dataloader, err_s_label.cpu().data.numpy(),
                #         err_s_domain.cpu().data.numpy(), err_t_domain.cpu().data.numpy())


        if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
        
        torch.save(self.model.state_dict(), f'./checkpoints/model_{self.source_config.target_sub}.std')
        torch.save(self.optimizer.state_dict(), f'./checkpoints/optim_{self.source_config.target_sub}.std')


       
        # write loss
        # write_losses(epoch_train_diff_losses,epoch_train_sim_losses,epoch_train_recon_losses, 'train_losses.csv')
        # write_losses(epoch_valid_diff_losses,epoch_valid_sim_losses,epoch_valid_recon_losses, 'valid_losses.csv')
        # exit()
       
        
    

    def eval(self, mode=None, to_print=False):
        assert(mode is not None)
        self.model.eval()

        y_true, y_pred = [], []
        eval_loss, eval_loss_diff = [], []

        dataloader = self.target_dataloader

        if to_print:
            self.model.load_state_dict(torch.load(
                f'checkpoints/model_{self.source_config.target_sub}.std'))


        with torch.no_grad():
            for features, labels in dataloader:
                    
                labels=labels[self.source_config.label_type].type(torch.LongTensor)
                


                self.model.zero_grad()
                  





                features = features.float().to(self.device)
                labels = labels.to(self.device)

                cls_output, domain_output = self.model(features, alpha=0)

                # if self.train_config.model_name=='SC_EEST':
                    # diff_loss = self.get_diff_loss()
                    # domain_loss = self.get_domain_loss()
                    # recon_loss = self.get_recon_loss()
                    # cmd_loss = self.get_cmd_loss()

                    # if self.train_config.use_cmd_sim:
                        # sim_loss = cmd_loss
                    # else:
                        # sim_loss = domain_loss
                
                # valid_diff_losses.append(diff_loss.item())
                # valid_sim_losses.append(sim_loss.item())
                # valid_recon_losses.append(recon_loss.item())
                
                cls_loss = self.criterion(cls_output, labels)
                loss = cls_loss
                
                eval_loss.append(loss.item())

                y_pred.append(cls_output.detach().cpu().numpy())
                y_true.append(labels.detach().cpu().numpy())

        eval_loss = np.mean(eval_loss)

        y_true = np.concatenate(y_true, axis=0).squeeze()
        y_pred = np.concatenate(y_pred, axis=0).squeeze()
        #todo: if exper_setting=='indep' return y_true and y_pred
        
        y_pred = np.argmax(y_pred, 1)


        # accuracy = self.calc_metrics(y_true, y_pred, mode, to_print)

        return eval_loss, y_true, y_pred#, round(np.mean(valid_diff_losses),4), round(np.mean(valid_sim_losses),4), round(np.mean(valid_recon_losses),4)
        
        
    def multiclass_acc(self, preds, truths):
        """
        Compute the multiclass accuracy w.r.t. groundtruth
        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

