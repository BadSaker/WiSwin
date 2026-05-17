import torch
import torch.nn as nn
import model
import data_loader
import matplotlib.pyplot as plt
import numpy as np
import CNN_baseline
# --- 新增导入 ---
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from sklearn.metrics import confusion_matrix


def train(model,train_loader,optimizer,criterion,device):

    model.train()
    total_loss,total_correct=0,0
    for bvps,labels,bvp_len in train_loader: #B T 20 20
        #assert (bvp_len > 0).all(), f'zero-length found: {bvp_len.tolist()}'
        bvps,labels=bvps.to(device),labels.to(device)
        
        optimizer.zero_grad()
        output=model(bvps,bvp_len)
        loss=criterion(output,labels)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()*bvps.size(0)
        total_correct+=(output.argmax(1)==labels).sum().item()
    return total_loss/len(train_loader.dataset),total_correct/len(train_loader.dataset)

def validate(model,test_loader,criterion,device):
    model.eval()
    total_loss,total_correct=0,0
    pred_labels=[]
    true_labels=[]
    total_samples = 0  # 新增：用来记录实际跑了多少个样本

    with torch.no_grad():
        for bvps,labels,bvp_len in test_loader:
            bvps,labels=bvps.to(device),labels.to(device)
            output=model(bvps,bvp_len)
            loss=criterion(output,labels)

            # 动态累加实际处理的样本数
            current_batch_size = bvps.size(0)
            total_samples += current_batch_size

            total_loss+=loss.item()*bvps.size(0)
            total_correct+=(output.argmax(1)==labels).sum().item()
            pred_labels.extend(output.argmax(1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return total_loss/total_samples,total_correct/total_samples,true_labels, pred_labels

def plot_confusion_matrix(cm,class_names,save_path):
    plt.figure(figsize=(10,8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], '.3f'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path,bbox_inches='tight',dpi=300)
    plt.show()

def main():
    model_pth='C:/Users/林鹏/Desktop/Files/WiSwin/checkpoints/model.pth'
    cm_pth='C:/Users/林鹏/Desktop/Files/WiSwin/confusion_matrix.png'
    setting=2  # 选择设置 1 或 2
    if setting==1:
        class_names=['Push&Pull','Sweep','Clap','Slide','Draw-N','Draw-Z']
    elif setting==2:
        class_names=['Push&Pull','Sweep','Clap','Slide','Draw-O','Draw-Z']

    best_val_acc = 0.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, test_loader = data_loader.build_dataloader(
        batch_size=32,
        num_workers=4,
        setting=setting,  # 选择设置 1 或 2
        mode='cross_room',  # 选择 'in_domain', 'cross_user'或 'cross_location'
        target_domain=1
    )

    epochs = 30
    warmup_epochs = 6  # 设为 10 轮预热（根据你的总轮数调整）

    train_model = model.WiSwin().to(device)
    total_params = sum(p.numel() for p in train_model.parameters())
    print("=" * 30)
    print(f"模型总参数量: {total_params:,} ({total_params / 1e6:.2f} M)")
    print("=" * 30)
    
    # --- 修改 1: 使用 AdamW 优化器 ---
    # 根据论文建议：设置权重衰减（Weight Decay）防止过拟合
    optimizer = AdamW(train_model.parameters(), lr=1e-3, weight_decay=0.05)

    # --- 修改 2: 设置学习率调度器 ---
    # 1. 余弦退火调度器
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)

    # 2. 线性预热函数：在前 warmup_epochs 轮让学习率从很小线性增加到设定值
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []
    lr_history = []  # 记录学习率变化
    best_labels=[]
    best_preds=[]

    print(f"开始训练，总轮数: {epochs}, 预热轮数: {warmup_epochs}")

    for epoch in range(epochs):
        # 记录当前学习率以便观察
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)

        train_loss, train_acc = train(train_model, train_loader, optimizer, nn.CrossEntropyLoss(), device)
        val_loss, val_acc, val_labels, val_preds = validate(train_model, test_loader, nn.CrossEntropyLoss(), device)

        # --- 修改 3: 更新学习率 ---
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        print(
            f'Epoch {epoch + 1}/{epochs} | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} |Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(train_model.state_dict(), model_pth)

            best_labels = val_labels
            best_preds = val_preds
            print(f"🎉 最佳准确率更新: {best_val_acc:.4f}，模型参数已保存")

    # --- 修改 4: 画出三张图（增加学习率曲线） ---
    plt.figure(figsize=(15, 5))

    # 1. Loss 曲线
    plt.subplot(1, 3, 1)
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.title('Loss')
    plt.legend()

    # 2. Acc 曲线
    plt.subplot(1, 3, 2)
    plt.plot(train_acc_history, label='Train Acc')
    plt.plot(val_acc_history, label='Val Acc')
    plt.title('Accuracy')
    plt.legend()

    # 3. 学习率曲线（非常重要，看看预热和余弦衰减是否生效）
    plt.subplot(1, 3, 3)
    plt.plot(lr_history, color='green')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epochs')

    plt.tight_layout()
    plt.savefig('C:/Users/林鹏/Desktop/Files/WiSwin/training_curves.png',bbox_inches='tight')  # 保存图像
    plt.show()

    cm=confusion_matrix(best_labels, best_preds,normalize='true')
    plot_confusion_matrix(cm,class_names,save_path=cm_pth)


if __name__ == '__main__':
    main()