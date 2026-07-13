import matplotlib.pyplot as plt

# --- 新しい学習ログデータ ---
epochs = list(range(1, 11))

# 誤差 (Loss) の推移
train_loss = [0.3662, 0.0940, 0.0498, 0.0211, 0.0148, 0.0163, 0.0058, 0.0062, 0.0096, 0.0160]
val_loss = [0.2550, 0.1744, 0.1448, 0.1059, 0.1556, 0.1274, 0.1093, 0.1394, 0.1394, 0.0947]

# 正解率 (Accuracy) の推移
train_acc = [0.8195, 0.9749, 0.9850, 0.9975, 0.9962, 0.9950, 1.0000, 0.9987, 0.9950, 0.9950]
val_acc = [0.9045, 0.9397, 0.9598, 0.9648, 0.9397, 0.9598, 0.9749, 0.9548, 0.9648, 0.9648]
# ----------------------------------------

def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左側のグラフ：正解率 (Accuracy)
    ax1.plot(epochs, train_acc, 'b-o', label='Train Accuracy')
    ax1.plot(epochs, val_acc, 'r-o', label='Val Accuracy')
    
    # ベストモデル（Epoch 7）を強調
    best_epoch = 7
    ax1.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5, label='Best Model (Epoch 7)')
    ax1.scatter(best_epoch, val_acc[best_epoch-1], color='green', s=100, zorder=5)
    
    ax1.set_title('Accuracy over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_xticks(epochs)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # 右側のグラフ：誤差 (Loss)
    ax2.plot(epochs, train_loss, 'b-o', label='Train Loss')
    ax2.plot(epochs, val_loss, 'r-o', label='Val Loss')
    
    # 今回は異常なスパイクがないため、Y軸のクリップを解除して自然に表示します
    ax2.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5, label='Best Model (Epoch 7)')
    ax2.scatter(best_epoch, val_loss[best_epoch-1], color='green', s=100, zorder=5)
    
    ax2.set_title('Loss over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_xticks(epochs)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    save_path = 'learning_curve_perfect.png'
    plt.savefig(save_path, dpi=300)
    print(f"グラフを '{save_path}' として保存しました！")
    plt.show()

if __name__ == "__main__":
    main()