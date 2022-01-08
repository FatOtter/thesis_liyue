import itertools
import torch
import pandas as pd
from constants import *


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CGD_model(torch.nn.Module):
    def __init__(self, Pv, out_channel, reso=28):
        super().__init__()
        self.Pv = Pv
        self.reso = reso
        self.out_channel = out_channel
        self.input = torch.nn.Sequential(
            torch.nn.Conv2d(Pv ** 2, out_channel, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.hidden = torch.nn.Sequential(
            torch.nn.Linear((reso//(Pv*2))**2 * out_channel, 10),
            torch.nn.Softmax(dim=1)
        )
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, x):
        x = x.view(x.size(0), self.Pv ** 2, self.reso // self.Pv, -1)
        h_input = self.input(x)
        h_input = h_input.view(x.size(0), -1)
        out = self.hidden(h_input)
        return out

    def step(self):
        self.optimizer.step()

    def get_flatten_parameters(self):
        """
        Return the flatten parameter of the current model
        :return: the flatten parameters as tensor
        """
        out = torch.zeros(0)
        with torch.no_grad():
            for parameter in self.parameters():
                out = torch.cat([out, parameter.flatten()])
        return out

    def load_parameters(self, parameters: torch.Tensor):
        """
        Load parameters to the current model using the given flatten parameters
        :param parameters: The flatten parameter to load
        :return: None
        """
        start_index = 0
        for param in self.parameters():
            with torch.no_grad():
                length = len(param.flatten())
                to_load = parameters[start_index: start_index + length]
                to_load = to_load.reshape(param.size())
                param.copy_(to_load)
                start_index += length

class CGD_torch:
    def __init__(self,
                 num_iter,
                 train_imgs,
                 train_labels,
                 test_imgs,
                 test_labels,
                 Ph,
                 Pv,
                 n_H,
                 init_lambda,
                 batch,
                 dataset,
                 sampling_prob,
                 max_grad_norm,
                 sigma,
                 sub_batch=8):
        self.num_iter = num_iter
        self.train_imgs = train_imgs
        self.train_labels = train_labels
        self.test_imgs = test_imgs
        self.test_labels = test_labels
        self.Ph = Ph
        self.Pv = Pv
        self.n_H = n_H
        self.init_lambda = init_lambda
        self.batch_size = train_imgs.size(0) // batch
        self.batch = batch
        self.dataset = dataset
        self.sampling_prob = sampling_prob
        self.max_grad_norm = max_grad_norm
        self.sigma = sigma
        self.sub_batch = sub_batch
        self.models = []
        self.loss = torch.nn.CrossEntropyLoss()
        self.sum_grad = None

    def confined_init(self, random_lambda=False):
        for i in range(self.Ph):
            model = CGD_model(Pv=self.Pv, out_channel=self.n_H)
            bound = self.init_lambda
            if random_lambda:
                bound = self.init_lambda * torch.randn(1).item()
            param = torch.rand(model.get_flatten_parameters().size()) * bound
            model.load_parameters(param)
            self.models.append(model)

    def shuffle_data(self):
        shuffled_index = torch.randperm(self.train_imgs.size(0))
        self.train_imgs = self.train_imgs[shuffled_index]
        self.train_labels = self.train_labels[shuffled_index]

    def RMSprop(self, model: torch.nn.Module, prev_g, alpha=0.99, eps=1e-3, eta=5e-1):
        if prev_g.__class__ != torch.Tensor.__class__:
            return
        current_idx = 0
        for param in model.parameters():
            with torch.no_grad():
                length = param.flatten().size(0)
                grad = prev_g[current_idx: current_idx+length]
                grad = alpha * grad + (1-alpha) * torch.sum(grad ** 2)
                grad_eta = eta / torch.sqrt(grad + eps)
                to_load = param - grad * grad_eta
                param.copy_(to_load)
                current_idx += length

    def one_epoch(self, prev_g, step_mean=True, restore=False, drop_sample=None):
        gradients = []
        sum_acc = 0
        sum_loss = 0
        sample_per_participant = self.train_imgs.size(0) // self.Ph
        for i in range(self.Ph):
            model = self.models[i]
            # self.RMSprop(model, prev_g=prev_g)
            cache_i = model.get_flatten_parameters()
            cache_i += prev_g
            self.models[i].load_parameters(cache_i)
            lower = sample_per_participant * i
            upper = sample_per_participant * (i + 1)
            x_i = self.train_imgs[lower: upper]
            y_i = self.train_labels[lower: upper].flatten()
            if drop_sample is not None and lower <= drop_sample < upper:
                mask = torch.ones(upper-lower, dtype=torch.bool)
                mask[drop_sample-lower] = 0
                x_i = x_i[mask]
                y_i = y_i[mask]
            x_i = x_i.reshape(x_i.size(0), self.train_imgs.size(1) // self.Pv, self.Pv)
            j = 0
            acc_i = 0
            loss_i = 0
            while j * self.batch_size < x_i.size(0):
                x_i_j = x_i[j * self.batch_size: (j + 1) * self.batch_size]
                y_i_j = y_i[j * self.batch_size: (j + 1) * self.batch_size]
                self.models[i].optimizer.zero_grad()
                out = self.models[i](x_i_j)
                y_pred = torch.max(out, dim=1).indices
                j_loss = self.loss(out, y_i_j)
                j_loss.backward()
                self.models[i].optimizer.step()
                acc_i += (y_pred == y_i_j).sum()
                loss_i += j_loss.item()
                j += 1
            acc_i = acc_i / y_i.size(0)
            # loss_i = loss_i / y_i.size(0)
            sum_acc += acc_i
            sum_loss += loss_i
            grad = self.models[i].get_flatten_parameters() - cache_i
            if grad.norm() > self.max_grad_norm:
                grad = grad * self.max_grad_norm / grad.norm()
            grad += torch.randn(grad.size()) * (self.sigma * self.max_grad_norm)
            gradients.append(grad)
            if restore:
                model.load_parameters(cache_i)
        gradients = torch.vstack(gradients)
        if step_mean:
            gradients = torch.mean(gradients, dim=0)
        else:
            gradients = torch.sum(gradients, dim=0)
        sum_acc = sum_acc / self.Ph
        sum_loss = sum_loss / self.Ph
        return gradients, sum_acc, sum_loss

    def grad_reset(self):
        if self.sum_grad is None:
            sum_grad = []
            for param in self.models[0].parameters():
                g_l = torch.zeros(param.size())
                sum_grad.append(g_l)
            self.sum_grad = sum_grad
        else:
            for item in self.sum_grad:
                item.zero_()

    def back_prop(self, X, Y, drop_idx=None):
        self.grad_reset()
        sum_acc = 0
        sum_loss = 0
        sample_per_participant = X.size(0) // self.Ph
        for i in range(self.Ph):
            model = self.models[i]
            lower = sample_per_participant * i
            upper = sample_per_participant * (i + 1)
            x_i = X[lower: upper]
            y_i = Y[lower: upper]
            if drop_idx is not None and lower <= drop_idx < upper:
                mask = torch.ones(x_i.size(0), dtype=torch.bool)
                mask[drop_idx-lower] = 0
                x_i = x_i[mask]
                y_i = y_i[mask]
            y_i = y_i.flatten()
            acc_i = 0
            loss_i = 0
            out = model(x_i)
            y_pred = torch.max(out, dim=1).indices
            j_loss = self.loss(out, y_i)
            model.zero_grad()
            j_loss.backward()
            self.collect_grad(model)
            acc_i += (y_pred == y_i).sum()
            loss_i += j_loss.item()
            acc_i = acc_i / y_i.size(0)
            # loss_i = loss_i / y_i.size(0)
            sum_acc += acc_i
            sum_loss += loss_i
        sum_acc = sum_acc / self.Ph
        sum_loss = sum_loss / self.Ph
        return sum_acc, sum_loss

    def collect_grad(self, model: CGD_model):
        grad_iter = iter(self.sum_grad)
        for param in model.parameters():
            grad = param.grad
            grad = self.grad_makeup(grad, norm_clip=True)
            collector = next(grad_iter, None)
            collector += grad

    def grad_makeup(self, grad, norm_clip=True, noisy_update=False, sparsify_update=True):
        with torch.no_grad():
            if norm_clip and grad.norm() > self.max_grad_norm:
                grad = grad * self.max_grad_norm / grad.norm()
            if noisy_update:
                grad += torch.randn(grad.size()) * (self.sigma * self.max_grad_norm)
            if sparsify_update:
                grad = self.sparsify_update(grad)
        return grad

    def apply_grad(self, grad_mean=True):
        if grad_mean:
            for grad in self.sum_grad:
                grad /= self.Ph
        for i in range(self.Ph):
            model = self.models[i]
            grad_iter = iter(self.sum_grad)
            for param in model.parameters():
                grad = next(grad_iter)
                param.grad.copy_(grad)
            model.step()

    def sparsify_update(self, gradient: torch.Tensor, p=None):
        if p is None:
            p = self.sampling_prob
        sampling_idx = torch.zeros(gradient.size(), dtype=torch.bool)
        sampling_idx.bernoulli_(1-p)
        gradient[sampling_idx] = 0
        return gradient

    def evaluate(self):
        outs = torch.zeros(self.test_labels.size(0), 10)
        loss = 0
        test_x = self.test_imgs
        test_y = self.test_labels.flatten()
        for i in range(self.Ph):
            model = self.models[i]
            with torch.no_grad():
                out = model(test_x)
            outs += out
            loss_i = self.loss(out, test_y)
            loss += loss_i.item()
        pred_y = torch.max(outs, dim=1).indices
        loss = loss / self.Ph
        acc = (pred_y == test_y).sum()
        acc = acc/self.test_labels.size(0)
        return acc, loss

    def accumulate(self, accountant, sample_grad):
        pairs = list(zip(*itertools.combinations(sample_grad, 2)))
        accountant.accumulate(
            ldistr=(torch.stack(pairs[0]), self.sigma * self.max_grad_norm),
            rdistr=(torch.stack(pairs[1]), self.sigma * self.max_grad_norm),
            q=1/self.batch,
            steps=1,
        )

    # def train(self, accountant=None):
    #     prev_g = 0
    #     for epoch in range(self.num_iter):
    #         prev_g, acc, loss = self.one_epoch(prev_g)
    #         prev_g = self.sparsify_update(prev_g)
    #         if epoch % 10 == 1:
    #             test_acc, test_loss = self.evaluate()
    #             print(f'Model performance - test acc {test_acc:6.4f}, test loss: {test_loss:6.4f}')
    #         if accountant:
    #             running_eps = self.accountant(accountant)
    #             print(f'Epoch {epoch} - train acc: {acc:6.4f}, train loss: {loss:6.4f}, Privacy (𝜀,𝛿): {running_eps}')
    #         else:
    #             print(f'Epoch {epoch} - train acc: {acc:6.4f}, train loss: {loss:6.4f}')

    def eq_train(self, accountant=None):
        train_acc_col = []
        train_loss_col = []
        test_acc_col = []
        test_loss_col = []
        eps_col = []
        for epoch in range(self.num_iter):
            batch_idx = 0
            acc = 0
            loss = 0
            while batch_idx * self.batch_size < self.train_imgs.size(0):
                lower = batch_idx * self.batch_size
                upper = (batch_idx+1) * self.batch_size
                batch_X = self.train_imgs[lower: upper]
                batch_Y = self.train_labels[lower: upper]
                batch_idx += 1
                if accountant:
                    drop_samples = torch.randint(low=0, high=batch_X.size(0), size=(3, ))
                    sample_grad = []
                    for sample_idx in drop_samples:
                        self.back_prop(batch_X, batch_Y, sample_idx)
                        grad_est = None
                        for grad in self.sum_grad:
                            flat = grad.flatten()
                            if grad_est is None:
                                grad_est = flat
                            else:
                                grad_est = torch.cat([grad_est, flat])
                        sample_grad.append(grad_est)
                    self.accumulate(accountant, sample_grad)
                batch_acc, batch_loss = self.back_prop(batch_X, batch_Y)
                acc += batch_acc
                loss += batch_loss
                self.apply_grad()
            acc /= batch_idx
            loss /= batch_idx
            print(f'Epoch {epoch} - train acc: {acc:6.4f}, train loss: {loss:6.4f}')
            if epoch % 10 == 0:
                test_acc, test_loss = self.evaluate()
                test_acc_col.append(test_acc.item())
                test_loss_col.append(test_loss)
                train_acc_col.append(acc.item())
                train_loss_col.append(loss)
                running_eps = accountant.get_privacy(target_delta=1e-5) if accountant else None
                if accountant:
                    print(f'Model performance - test acc {test_acc:6.4f}, test loss: {test_loss:6.4f}, Privacy (𝜀,𝛿): '
                          f'{running_eps}')
                    eps_col.append(running_eps[0])
                else:
                    print(f'Model performance - test acc {test_acc:6.4f}, test loss: {test_loss:6.4f}')
                    eps_col.append(0)
        recorder = pd.DataFrame({"test_acc":test_acc_col, "test_loss":test_loss_col, "train_acc":train_acc_col,
                                 "train_loss":train_loss_col, "epsilon":eps_col})
        recorder.to_csv(RECORDING_PATH+f"CGD_BDP_{self.dataset}_epoch_{self.num_iter}_Pv_{self.Pv}_Ph_{self.Ph}_Sigma_"
                                       f"{self.sigma}_lambda_{self.init_lambda}_"+time_str+".csv")
