import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import DataCollatorWithPadding

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# 指定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 加载本地 LLaMA 模型和 Tokenizer
model_path = "../models/llama"  # 替换为本地 LLaMA 模型路径
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token  # 确保有 pad_token
if tokenizer.pad_token is None:  # 如果没有设置，则使用 eos_token
    tokenizer.pad_token = tokenizer.eos_token

# 加载分类模型
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=2,  # 二分类
    torch_dtype=torch.float32,
    device_map="auto",  # 自动分配 GPU/CPU
)
model.config.pad_token_id = tokenizer.pad_token_id

# 数据预处理
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

# 加载数据
def load_data(csv_path):
    import pandas as pd
    data = pd.read_csv(csv_path)
    texts = (data["summary"] + " " + data["description"]).fillna("none").tolist()
    labels = data["valid"].tolist()
    return texts, labels

csv_path = "../gpt_data_res/netbeans/netbeans_after_nltk.csv"
texts, labels = load_data(csv_path)

# 数据划分
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.1, random_state=42
)

# 数据集准备
max_len = 256
train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_len)
val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_len)

# 自定义 collate_fn
def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    input_ids_padded = pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return {"input_ids": input_ids_padded, "labels": labels_tensor}

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)


# 检查点保存路径
checkpoint_path = "../llama_data_res/netbeans/checkpoint"

# 训练参数
training_args = TrainingArguments(
    learning_rate=1e-5,
    output_dir="../llama_data_res/netbeans/checkpoint",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="../llama_data_res/netbeans/logs",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    remove_unused_columns=False,
    save_strategy="steps",
)

# 自定义评价函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    probabilities = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)
    roc_auc = roc_auc_score(labels, probabilities)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "roc_auc": roc_auc}

if os.path.exists(checkpoint_path) and os.listdir(checkpoint_path):
    print("Loading checkpoint from:", checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_path,
        num_labels=2,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    model.config.pad_token_id = tokenizer.pad_token_id
else:
    print("No checkpoint found. Starting training from scratch.")
    # 如果没有 checkpoint，直接加载模型
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        torch_dtype=torch.float32,
        device_map="auto"
    )
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id

# 使用Trainer进行微调
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator,  # 使用正确的 collator
)



# 开始训练，如果没有 checkpoint，则直接开始
trainer.train()  # 去掉了 `resume_from_checkpoint` 参数

# 保存最终微调模型
final_model_path = "../llama_data_res/netbeans/fine_tuned_llama/"
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)

# 评估模型
eval_results = trainer.evaluate()
print(eval_results)

# 保存评估结果到 txt 文件
output_file = "../llama_data_res/netbeans/eval_results.txt"
with open(output_file, "w") as file:
    for key, value in eval_results.items():
        file.write(key + ": " + str(value) + "\n")
