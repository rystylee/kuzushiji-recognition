import os
from collections import OrderedDict
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from models import Net


class Trainer(object):
    def __init__(self, train_loader, test_loader, config):
        self.config = config
        self.device = config.device

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.n_epoch = config.n_epoch
        self.lr = config.lr
        self.gamma = config.gamma
        self.device = config.device

        # self.start_epoch = 1
        self.start_itr = 1

        n_classes = len(self.train_loader.dataset.classes)
        self.model = Net(n_classes=n_classes).to(self.device)
        print(self.model)
        print('Initialized model...\n')

        self.optim = torch.optim.Adadelta(self.model.parameters(), self.lr)
        self.scheduler = StepLR(self.optim, step_size=1, gamma=self.gamma)

        # if not self.config.model_state_path == '':
        #     self._load_models(self.config.model_state_path)

        self.writer = SummaryWriter(log_dir=self.config.log_dir)

    def train(self):
        self.model.train()

        n_itr = self.start_itr
        print('Start training...!')
        for epoch in range(1, self.n_epoch + 1):
            with tqdm(total=len(self.train_loader)) as pbar:
                for idx, (img, label) in enumerate(self.train_loader):
                    pbar.set_description(f'Epoch[{epoch}/{self.n_epoch}], iteration[{idx}/{len(self.train_loader)}]')

                    img, label = img.to(self.device), label.to(self.device)

                    self.optim.zero_grad()
                    output = self.model(img)
                    loss = F.nll_loss(output, label)
                    loss.backward()
                    self.optim.step()

                    if n_itr % self.config.log_interval == 0:
                        pbar.set_postfix(OrderedDict(loss=loss.item()))
                        tqdm.write(f'Epoch[{epoch}], iteration[{idx}/{len(self.train_loader)}], loss: {loss.item()}')
                        self.writer.add_scalar('loss/loss', loss.item(), n_itr)

                    if n_itr % self.config.checkpoint_interval == 0:
                        self._save_models(epoch, n_itr)

                    n_itr += 1
                    pbar.update()
            self.scheduler.step()
            self.test(n_itr)

        self.writer.close()

    def test(self, n_itr):
        self.model.eval()
        test_loss = 0
        correct = 0
        print('Start testing...!')
        with torch.no_grad():
            for _, (img, label) in enumerate(self.test_loader):
                img, label = img.to(self.device), label.to(self.device)
                output = self.model(img)
                test_loss += F.nll_loss(output, label, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        accuracy = 100.0 * (correct / len(self.test_loader.dataset))
        self.writer.add_scalar('accuracy/test_accuracy', n_itr)
        tqdm.write(f'Test: Average loss: {test_loss}, Accuracy: {accuracy}%')
        self.model.train()

    def _save_models(self, epoch, n_itr):
        checkpoint_name = f'{self.config.dataset_name}_model_ckpt_{n_itr}.pt'
        checkpoint_path = os.path.join(self.config.checkpoint_dir, checkpoint_name)
        torch.save({
            # 'epoch': epoch,
            'n_itr': n_itr,
            'model': self.model.state_dict(),
            'optim': self.optim.state_dict(),
        }, checkpoint_path)
        tqdm.write(f'Saved models state_dict: n_itr_{n_itr}')

    def _load_models(self, model_state_path):
        checkpoint = torch.load(model_state_path)
        # self.start_epoch = checkpoint['epoch']
        self.start_itr = checkpoint['n_itr'] + 1
        self.model.load_state_dict(checkpoint['model'])
        self.optim.load_state_dict(checkpoint['optim'])
        print(f'start_itr: {self.start_itr}')
        print('Loaded pretrained models...\n')
