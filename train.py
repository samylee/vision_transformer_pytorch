import torch
import numpy as np
from tqdm import tqdm
from data.dataset import get_loader
from models.vision_transformer import VisionTransformer
from utils.utils import set_seed, AverageMeter, WarmupCosineSchedule


def valid(model, test_loader, device):
    # Validation!
    model.eval()
    all_preds, all_label = [], []
    pbar = tqdm(test_loader, desc='Validating', bar_format='{l_bar}{r_bar}', dynamic_ncols=True)
    for step, batch in enumerate(pbar):
        batch = tuple(t.to(device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = (all_preds == all_label).mean()
    print('accuracy:', accuracy)

    return accuracy


if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    config = {
        'seed': 42,
        'img_size': [224, 224],
        'num_classes': 10,
        'train_batch_size': 32,
        'test_batch_size': 16,

        'learning_rate': 3e-2,
        'weight_decay': 0,
        'max_grad_norm': 1.0,

        'warmup_steps': 500,
        'num_steps': 10000,
        'test_frequency': 100,

        'patch_size': [16, 16],
        'hidden_size': 768,
        'mlp_dim': 3072,
        'num_heads': 12,
        'num_layers': 12,
        'dropout_rate': 0.1,
        'attention_dropout_rate': 0.0
    }

    train_loader, test_loader = get_loader(config)

    model = VisionTransformer(config)
    # pretrain
    model.load_state_dict(torch.load('pretrain/vit_b_16_224.pth', map_location='cpu'))

    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9, weight_decay=config['weight_decay'])
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=config['warmup_steps'], t_total=config['num_steps'])

    model.zero_grad()
    set_seed(config['seed'])
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    while True:
        model.train()
        pbar = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", bar_format="{l_bar}{r_bar}", dynamic_ncols=True)
        for step, batch in enumerate(pbar):
            batch = tuple(t.to(device) for t in batch)
            x, y = batch
            loss = model(x, y)

            loss.backward()

            losses.update(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

            pbar.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, config['num_steps'], losses.val)
            )
            if global_step % config['test_frequency'] == 0:
                accuracy = valid(model, test_loader, device)
                if best_acc < accuracy:
                    best_acc = accuracy
                    torch.save(model.state_dict(), 'checkpoints/vit_b_16_' + str(global_step) + '_' + str(best_acc) + '.pth')
                model.train()

            if global_step % config['num_steps'] == 0:
                break
        losses.reset()
        if global_step % config['num_steps'] == 0:
            break
