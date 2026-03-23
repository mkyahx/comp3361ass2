from utils.data_loader import DataLoader,NERDataset,collate
from models.lstm import BiLSTM_Tagger
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm



label_url = "./ontonotes5/dataset/label.json"
sentences_url = "./ontonotes5/dataset/train00.json"
my_dataset = NERDataset(label_url,sentences_url,"lstm")
device = torch.device("cpu")
model = BiLSTM_Tagger(

    vocab_size=len(my_dataset.token2idx), 
    tag_size=len(my_dataset.t2i), # 对应 label.json 里的数量，通常是 18
    emb_dim=256, 
    hidden_dim=512
    
    )



optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=-100)
train_loader = DataLoader(
    my_dataset,
    batch_size = 4,
    shuffle = True,
    collate_fn = collate
)
model.train()
num_epoch = 1
history = []
for  epoch in range(num_epoch):
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epoch}", leave=True)
    for batch_idx, (sentences, tags) in enumerate(pbar):
        # 1. 搬运数据到 GPU/CPU
        sentences = sentences.to(device)
        tags = tags.to(device)
        
        # 2. 梯度清零 (非常重要！否则梯度会累加)
        optimizer.zero_grad()
        
        # 3. 前向传播
        outputs = model(sentences) # shape: [batch, seq_len, num_tags]
        
        # 4. 计算损失
        # CrossEntropyLoss 期望输入是 [N, C, L] 或者把前两维展平
        # 我们把 outputs 展平为 [batch * seq_len, num_tags]
        # 把 tags 展平为 [batch * seq_len]
        loss = criterion(outputs.view(-1, 37), tags.view(-1))
        
        # 5. 反向传播 & 更新参数
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        total_loss += current_loss
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        pbar.set_postfix({"loss": f"{current_loss:.4f}", "avg_loss": f"{total_loss/(batch_idx+1):.4f}"})
        history.append(total_loss / len(train_loader))


# 定义保存路径
save_path = "parameters/bilstm_1.pth"

# 保存模型参数 (State Dict)
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")
with open("log/lstm_loss_history.txt", "w") as f:
    for l in history:
        f.write(f"{l}\n")