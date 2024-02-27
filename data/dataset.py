from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


def get_loader(config):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(config['img_size'], scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(config['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    trainset = datasets.CIFAR10(root="./data", train=True, transform=transform_train)
    testset = datasets.CIFAR10(root="./data", train=False, transform=transform_test)

    train_sampler = RandomSampler(trainset)
    test_sampler = SequentialSampler(testset)

    train_loader = DataLoader(trainset, sampler=train_sampler, batch_size=config['train_batch_size'], num_workers=4, pin_memory=True)
    test_loader = DataLoader(testset, sampler=test_sampler, batch_size=config['test_batch_size'], num_workers=4, pin_memory=True)

    return train_loader, test_loader