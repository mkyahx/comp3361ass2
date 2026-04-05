from utils.data_loader import DataLoader,NERDataset,collate
from models.lstm import BiLSTM_Tagger
from models.transformer import Trans_Tagger
from models.bert import DistilBert_Tagger
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
import json
from seqeval.metrics import f1_score, classification_report

mode = "bert" # options: ['trans'.'lstm','bert']
def train(config,store=False):
    lr = config['lr']
    hd = config.get("hidden_dim",768)
    do = config["dropout"]
    w0 = config["weight_0"]
    bs = config["batch_size"]
    ed = config.get("emb_dim",768)
    nh = config.get("n_head",8)
    nl = config.get("num_layers",3)
    num_epoch = config["testing_epoch"] if store ==False else config["epoch"]
    mode = config["name"]
    wd = config.get("weight_decay",0)
    ls = config.get("label_smoothing", 0)


    eval_dataset = NERDataset(label_url,eval_data,mode,word2idx = train_dataset.word2idx)
    
    print(f"当前使用的设备: {device}")

    if mode == "lstm":
        model = BiLSTM_Tagger(

            vocab_size=len(train_dataset.token2idx), 
            tag_size=len(train_dataset.t2i),
            emb_dim=ed, 
            hidden_dim=hd,
            dropout=do
            )
    elif mode == "trans":
        model = Trans_Tagger(
            vocab_size=len(train_dataset.token2idx),
            tag_size=len(train_dataset.t2i),
            emb_dim=ed,
            nhead=nh, 
            num_layers=nl,
            hidden_dim=hd,
            dropout=do
        )
    elif mode == "bert":
        model = DistilBert_Tagger(
            tag_size = len(train_dataset.t2i),
            dropout=do
        )

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd) if mode =="bert" else optim.Adam(model.parameters(), lr=lr,weight_decay = wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    weights = torch.ones(37)
    weights[0] = w0
    weights = weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-100,label_smoothing = ls)
    train_loader = DataLoader(
        train_dataset,
        batch_size = bs,
        shuffle = True,
        collate_fn = collate
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size = bs,
        shuffle = True,
        collate_fn = collate
    )
    model.train()

    history = {"loss_history" : [],"f1_history" : []}
    best_f1 = 0
    for  epoch in range(num_epoch):
        total_loss = 0
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epoch}", leave=True)
        for batch_idx, (sentences, tags,_) in enumerate(pbar):
            sentences = sentences.to(device)
            tags = tags.to(device)
            
            if mode == "trans":
                key_padding_mask = (sentences == 0) 
            
            
            optimizer.zero_grad()
            
            
            if mode =="trans":
                outputs = model(sentences, mask=key_padding_mask) 
            elif mode =="lstm":
                outputs = model(sentences)
            elif mode =="bert":
                attn_mask = (sentences!=0).long()
                outputs = model(sentences,attention_mask = attn_mask)
            
            
            loss = criterion(outputs.view(-1, 37), tags.view(-1))
            
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            current_loss = loss.item()
            total_loss += current_loss
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

            pbar.set_postfix({"loss": f"{current_loss:.4f}", "avg_loss": f"{total_loss/(batch_idx+1):.4f}"})
        history["loss_history"].append(total_loss / len(train_loader))
        dev_f1, dev_report = evaluate(model, eval_loader, device, train_dataset.i2t)
        scheduler.step(dev_f1)
        print(f"Epoch {epoch} Dev Micro-F1: {dev_f1:.4f}")
        history["f1_history"].append(dev_f1)
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            torch.save(model.state_dict(), f"parameters/best_{mode}_{num_epoch}_ner.pth")
    

    
    if store:
        save_path = f"parameters/{mode}_{num_epoch}_model.pth"

        
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        with open(f"log/{mode}_{num_epoch}_history.txt", "w") as f:
            f.write(f"{'Epoch':<10}{'Loss':<15}{'F1-Score':<15}\n")
            f.write("-" * 40 + "\n")
            
            
            for i, (l, f1) in enumerate(zip(history["loss_history"], history["f1_history"])):
                f.write(f"{i:<10}{l:<15.4f}{f1:<15.4f}\n")
    return best_f1

def evaluate(model, data_loader, device, i2t, export=False):
    model.eval()
    all_true_tags = []
    all_pred_tags = []
    all_pred_indices = []
    
    with torch.no_grad():
        for sentences, tags,_ in data_loader:
            sentences = sentences.to(device)
            
            if mode=='lstm':
                outputs = model(sentences) 
            elif mode == "trans":
                key_padding_mask = (sentences == 0)
                outputs = model(sentences, mask=key_padding_mask)
            elif mode == "bert":
                attn_mask = (sentences != 0).long().to(device)
                outputs = model(sentences, attention_mask=attn_mask)
            predictions = torch.argmax(outputs, dim=-1) 
            
            
            for i in range(len(tags)):
                true_labels = []
                pred_labels = []
                pred_indices = []
                for j in range(len(tags[i])):
                    if tags[i][j] != -100:
                        true_labels.append(i2t[tags[i][j].item()])
                        pred_labels.append(i2t[predictions[i][j].item()])
                        if export:
                            pred_indices.append(int(predictions[i][j].item()))
                
                all_true_tags.append(true_labels)
                all_pred_tags.append(pred_labels)
                if export:
                    all_pred_indices.append(pred_indices)
    
    micro_f1 = f1_score(all_true_tags, all_pred_tags, average='micro')
    report = classification_report(all_true_tags, all_pred_tags)
    if export:
        return micro_f1,report,all_pred_indices
    return micro_f1, report

label_url = "./ontonotes5/dataset/label.json"
train_data = ["./ontonotes5/dataset/train00.json","./ontonotes5/dataset/train01.json","./ontonotes5/dataset/train02.json","./ontonotes5/dataset/train03.json"]
eval_data = ["./ontonotes5/dataset/valid.json"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = NERDataset(label_url,train_data,mode)

lstm_grid={
    "lr": [0.001,0.002],#[0.001,0.002]
    "hidden_dim" : [512,1024],#[512,1024]
    'dropout': [0.3,0.5],#[0.3,0.5]
    "weight_0" : [0.1,0.3],#[0.1,0.3]
    "batch_size" : [64],
    "emb_dim" : [512],
    "epoch" : [20],
    "testing_epoch" : [5],
    "name" : ["lstm"]
}

trans_grid = {
    'lr': [1e-4,2e-4],  #[1e-4,2e-4]        
    'hidden_dim': [1024,2048],      
    'dropout': [0.1],      
    'weight_0': [0.4,0.5], #0.4,0.5
    'batch_size': [32],    
    'emb_dim': [512],
    "n_head" : [8],
    "num_layers": [4,6], #[4,6]
    "epoch" : [60], #60
    "testing_epoch": [5], #5
    "name" : ["trans"],
    "weight_decay":[1e-4],
    "label_smoothing" : [0.1]
}

bert_grid={
    'lr':[2e-5,3e-5,5e-5],
    'weight_decay':[0.01],
    "dropout":[0.1],
    'name':['bert'],
    'testing_epoch':[3],
    'epoch':[5],
    'weight_0':[1],
    'batch_size':[32]
}


def test(config):
    lr = config['lr']
    hd = config.get("hidden_dim",768)
    do = config["dropout"]
    w0 = config["weight_0"]
    bs = config["batch_size"]
    ed = config.get("emb_dim",768)
    nh = config.get("n_head",8)
    nl = config.get("num_layers",3)
    num_epoch = config["epoch"]
    mode = config["name"]
    wd = config.get("weight_decay",0)



    test_data = ["./ontonotes5/dataset/test.json"]
    test_dataset = NERDataset(label_url, test_data, mode, word2idx=train_dataset.word2idx)
    test_loader = DataLoader(
        test_dataset,
        batch_size=bs, 
        shuffle=False, 
        collate_fn=collate
    )

    
    if mode == "lstm":
        model = BiLSTM_Tagger(
            vocab_size=len(train_dataset.token2idx), 
            tag_size=len(train_dataset.t2i), 
            emb_dim=ed, 
            hidden_dim=hd,
            dropout=do
        ) 
    elif mode == "trans":
        model = Trans_Tagger(
        vocab_size=len(train_dataset.token2idx),
        tag_size=len(train_dataset.t2i),
        emb_dim=ed,
        nhead=nh, 
        num_layers=nl, 
        hidden_dim=hd,
        dropout=do
    )
    elif mode=="bert":
        model = DistilBert_Tagger(
            tag_size = len(train_dataset.t2i),
            dropout=do
        )
    model.to(device)

    
    model_path = f"parameters/best_{mode}_{num_epoch}_ner.pth" 
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Successfully loaded model from {model_path}")

    
    model.eval() 
    with torch.no_grad(): 
        test_f1, test_report, results = evaluate(model, test_loader, device, train_dataset.i2t, export=True)
    uid = 3036128157
    if mode =="lstm":
        mode_name = "lstm"
    elif mode == "trans":
        mode_name = "transformer"
    elif mode =="bert":
        mode_name = "distilbert"
    result_path = f"{uid}.{mode_name}.test.txt"
    with open(result_path, "w", encoding="utf-8") as f:
        for idx, pred_tags_indices in enumerate(results):
            # 获取对应的原始 tokens
            original_tokens = test_dataset.sentences[idx]['tokens']
            
            # 遍历当前句子的每个 token
            for i in range(len(original_tokens)):
                token = original_tokens[i]
                # 从索引转回字符串标签 (如 'B-PERSON')
                tag = train_dataset.i2t[pred_tags_indices[i]]
                f.write(f"{token} {tag}\n")
            
            # 句子之间添加一个空行
            f.write("\n")

    print(f"{mode} predictions saved to: {result_path}")


    report_path = f"log/{mode}_test_final_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Evaluation on Test Set\n")
        f.write("="*30 + "\n")
        f.write(f"Overall Micro-F1: {test_f1:.4f}\n\n")
        f.write("Detailed Entity-level Report:\n")
        f.write(test_report)
        config_str = json.dumps(config, indent=4, sort_keys=True)
        f.write(config_str)

    print(f"Test report has been saved to: {report_path}")


def grid_search(param_grid):
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"in total {len(experiments)} configs")

    best_overall_f1 = 0.0
    best_config = {}

    for i, config in enumerate(experiments):
        print(f"\n" + "="*50)
        print(f"experiment {i+1}/{len(experiments)} | config: {config}")
        print("="*50)


        current_best_f1 = train(config) 


        if current_best_f1 > best_overall_f1:
            best_overall_f1 = current_best_f1
            best_config = config
            
            
            with open("log/best_hyperparameters_lstm.txt", "w") as f:
                f.write(f"Best F1 Score: {best_overall_f1:.4f}\n")
                f.write(f"Best Config: {best_config}\n")
                
    print(f"best F1: {best_overall_f1:.4f}")
    print(f"best config: {best_config}")
    train(best_config, store=True)



    #run on test with the best parameters stored

    test(best_config)
    return best_config




# train(trans_config,store=True)


if mode == "lstm":
    grid=lstm_grid
elif mode=="trans":
    grid = trans_grid
elif mode=="bert":
    grid = bert_grid


best_config=grid_search(grid)
test(best_config)