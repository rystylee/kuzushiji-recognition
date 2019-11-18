import torch

from config import get_config
from data_loader import DataLoader
from trainer import Trainer


def main():
    config = get_config()
    print(config)
    print()

    dataloader = DataLoader(config.data_root, config.dataset_name, config.batch_size)
    train_loader, test_loader = dataloader.get_loader()

    torch.backends.cudnn.benchmark = True

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    config.device = device

    trainer = Trainer(train_loader, test_loader, config)
    if config.mode == 'train':
        trainer.train()
    else:
        trainer.test()


if __name__ == "__main__":
    main()
