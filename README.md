# PTT 推文預測

爬取 PTT Joke 版文章，並且使用 PyTorch BERT 預測文章是否 (推噓相減後) 七天內會大於等於 30 推。

介紹請看我的部落格文章：[如何訓練一個合理的文字分類模型](https://writings.jigfopsda.com/zh/posts/2020/reasonable_modeling/)

## 環境設定

建議使用 virtualenv 安裝套件。

```bash
virtualenv __
source __/bin/activate
pip install -r requirements.txt
```

## 準備資料

使用以下指令爬取資料：

```bash
./scripts/crawling.sh
```

或者執行 `crawler.py`：

```bash
python crawler.py --board {版名} --date {開始日期} --length {爬取天數}
```

## 訓練模型

請先將參數寫進 `config.yaml`：

```yaml
pretrained_weight: bert-base-multilingual-cased
train_batch_size: 12
eval_batch_size: 24
epochs: 30
patient: 3
lr: 0.000001
name: checkpoint/
```

訓練及預測：

```bash
python train.py
```

最後會印出 test set 分數：

```bash
2020-08-16 19:43:06,477 INFO [train:main:196] Test loss 0.008711 Test acc 0.935286 Test auc 0.769546
2020-08-16 19:43:06,477 INFO [train:main:197] Done
```

## PTT 推文預測 授權條款

MIT License
