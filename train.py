import argparse
import arrow
import json
import logging
import numpy as np
import shutil
import torch
import os
import yaml

from sklearn.metrics import accuracy_score, roc_auc_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizerFast, BertConfig, BertForSequenceClassification
from transformers.optimization import AdamW


device = torch.device('cuda')


class TextDataset(Dataset):
    def __init__(self, start_date, end_date, tokenizer, board='joke', max_len=256):
        self.feature = []
        self.label = []
        path = 'data/%s/line.json' % board
        # 讀取資料，每篇文章是一行 JSON
        for line in open(path):
            d = json.loads(line)
            atime = arrow.get(d['time']).shift(hours=8)
            if start_date <= atime < end_date:
                # 計算七天內推噓相減
                push = 0
                for r in d['Responses']:
                    if r['ResponseTime'] - d['time'] > 86400 * 7:
                        continue
                    if r['Vote'] == '推':
                        push += 1
                    elif r['Vote'] == '噓':
                        push -= 1

                # 使用標題 + 內文當作 feature
                # 將標題與內文切成 token list
                tokens = ['CLS'] + tokenizer.tokenize(d['Title']) + ['SEP'] + tokenizer.tokenize(d['Content']) + ['SEP']
                # 將文字轉成 id
                token_ids = tokenizer.convert_tokens_to_ids(tokens)

                # 將 feature 固定在長度為 max_len，太長切掉，太短補 0
                if len(token_ids) > max_len:
                    token_ids = token_ids[:max_len]
                else:
                    token_ids += [0] * (max_len - len(token_ids))
                self.feature.append(token_ids)
                self.label.append(int((push >= 30)))
        self.feature = np.array(self.feature)
        self.label = np.array(self.label)
        logging.info('#pos %d #neg %d', np.sum(self.label), len(self.label) - np.sum(self.label))

    # 取得第 idx 筆資料
    def __getitem__(self, idx):
        return self.feature[idx], self.label[idx]

    # dataset 的長度
    def __len__(self):
        return len(self.label)


def train_epoch(train_loader, model, optimizer, criterion):
    running_loss = 0.0
    y_pred = []
    y_true = []
    # 訓練模式
    model.train()
    # 逐 batch 讀出 feature 及 label
    for feature, label in tqdm(train_loader):
        # batch 中最長的長度
        length = feature.numpy().max(axis=0).nonzero()[0][-1] + 1
        feature = feature[:, :length]
        # mask 指出有 token 的部份，feature 補 0 的部份在 mask 也為 0
        mask = torch.tensor(feature.numpy() > 0, dtype=torch.int64).to(device)
        # 將資料搬到 GPU
        feature = feature.to(device)
        label = label.to(device)

        # optimizer 的 gradient 歸零
        optimizer.zero_grad()
        output = model(input_ids=feature, attention_mask=mask)[0]
        # 計算 loss
        loss = criterion(output, label)
        # 計算 gradient
        loss.backward()
        # 更新 model weight
        optimizer.step()

        running_loss += loss.item()
        y_pred.append(torch.argmax(output, axis=1).cpu().numpy())
        y_true.append(label.cpu().numpy())
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    train_loss = running_loss / len(train_loader.dataset)
    return train_loss, accuracy_score(y_true, y_pred), roc_auc_score(y_true, y_pred)


def eval_epoch(valid_loader, model, criterion):
    running_loss = 0.0
    y_pred = []
    y_true = []
    # 預測模式
    model.eval()
    # 逐 batch 讀出 feature 及 label
    for feature, label in tqdm(valid_loader):
        # batch 中最長的長度
        length = feature.numpy().max(axis=0).nonzero()[0][-1] + 1
        feature = feature[:, :length]
        # mask 指出有 token 的部份，feature 補 0 的部份在 mask 也為 0
        mask = torch.tensor(feature.numpy() > 0, dtype=torch.int64).to(device)
        # 將資料搬到 GPU
        feature = feature.to(device)
        label = label.to(device)

        # 告訴 PyTorch 我們不計算 gradient，可以節省 GPU 記憶體
        with torch.no_grad():
            output = model(input_ids=feature, attention_mask=mask)[0]
            # 計算 loss
            loss = criterion(output, label)

        running_loss += loss.item()
        y_pred.append(torch.argmax(output, axis=1).cpu().numpy())
        y_true.append(label.cpu().numpy())
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    valid_loss = running_loss / len(valid_loader.dataset)
    return valid_loss, accuracy_score(y_true, y_pred), roc_auc_score(y_true, y_pred)


def load_checkpoint(name, model, optimizer):
    best_loss = np.inf
    best_epoch = -1
    start_epoch = 0
    scores = {
        'train_loss': [],
        'train_acc': [],
        'train_auc': [],
        'valid_loss': [],
        'valid_acc': [],
        'valid_auc': [],
    }
    # 如果有之前訓練到一半的，從最好的那個接續訓練
    path = os.path.join('checkpoints', name, 'best.bin')
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scores = checkpoint['scores']
        best_epoch = len(scores['valid_loss']) - 1
        best_loss = scores['valid_loss'][-1]
        start_epoch = best_epoch + 1
    return model, optimizer, start_epoch, best_epoch, best_loss, scores


def main():
    args = parse_args()
    conf = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    pretrained = conf['pretrained_weight']
    train_batch_size = conf['train_batch_size']
    eval_batch_size = conf['eval_batch_size']
    epochs = conf['epochs']
    patient = conf['patient']
    lr = conf['lr']
    checkpoint_dir = os.path.join('checkpoints', conf['name'])
    os.makedirs(checkpoint_dir, exist_ok=True)

    # prepare data
    tokenizer = BertTokenizerFast.from_pretrained(pretrained)
    # train
    config = BertConfig.from_pretrained(pretrained, num_labels=2)
    model = BertForSequenceClassification.from_pretrained(pretrained, config=config)
    model.cuda()
    optimizer = AdamW(model.parameters(), lr=lr)
    model, optimizer, start_epoch, best_epoch, best_loss, scores = load_checkpoint(conf['name'], model, optimizer)

    criterion = CrossEntropyLoss()
    train_loader = DataLoader(
        TextDataset(arrow.get('20190101', 'YYYYMMDD'), arrow.get('20200301', 'YYYYMMDD'), tokenizer),
        batch_size=train_batch_size, shuffle=True)
    valid_loader = DataLoader(
        TextDataset(arrow.get('20200301', 'YYYYMMDD'), arrow.get('20200501', 'YYYYMMDD'), tokenizer),
        batch_size=eval_batch_size, shuffle=False)

    for epoch in range(start_epoch, epochs):
        logging.info('training epoch %d', epoch)
        # 訓練模型以及在 validation set 上計算分數
        train_loss, train_acc, train_auc = train_epoch(train_loader, model, optimizer, criterion)
        valid_loss, valid_acc, valid_auc = eval_epoch(valid_loader, model, criterion)
        logging.info('Train loss %f Train acc %f Train auc %f', train_loss, train_acc, train_auc)
        logging.info('Validation loss %f Validation acc %f Validation auc %f', valid_loss, valid_acc, valid_auc)
        scores['train_loss'].append(train_loss)
        scores['train_acc'].append(train_acc)
        scores['train_auc'].append(train_auc)
        scores['valid_loss'].append(valid_loss)
        scores['valid_acc'].append(valid_acc)
        scores['valid_auc'].append(valid_auc)
        # 紀錄這個 epoch 的分數及模型
        torch.save({
            'epoch': epoch,
            'best_loss': best_loss,
            'scores': scores,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }, os.path.join(checkpoint_dir, '%d.bin' % epoch))
        if valid_loss < best_loss:
            # validation loss 最好的 epoch 就是我們要使用的最好的模型
            logging.info('save model with best loss at epoch %d', epoch)
            best_loss = valid_loss
            best_epoch = epoch
            shutil.copy2(checkpoint_dir + '%d.bin' % epoch, checkpoint_dir + 'best.bin')
        if epoch - best_epoch >= patient:
            logging.info('Epoch %d not improving from best_epoch %d beark!', epoch, best_epoch)
            break

    # 讀入 validation loss 最好的模型
    checkpoint = torch.load(checkpoint_dir + 'best.bin')
    model.load_state_dict(checkpoint['model_state'])

    # evaluate on test
    test_loader = DataLoader(
        TextDataset(arrow.get('20200501', 'YYYYMMDD'), arrow.get('20200701', 'YYYYMMDD'), tokenizer),
        batch_size=eval_batch_size, shuffle=False)
    test_loss, test_acc, test_auc = eval_epoch(test_loader, model, criterion)
    logging.info('Test loss %f Test acc %f Test auc %f', test_loss, test_acc, test_auc)
    logging.info('Done')


def parse_args():
    parser = argparse.ArgumentParser(description='Training and offline evaluation')
    parser.add_argument('-c', '--config', type=str, default='config.yaml', help='Model config file')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-2s [%(module)s:%(funcName)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
    assert torch.cuda.is_available()
    os.makedirs('checkpoints', exist_ok=True)
    main()
