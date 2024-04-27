from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Khởi tạo EventAccumulator với đường dẫn đến file sự kiện
event_acc = EventAccumulator('./log/train/events.out.tfevents.1713949467.LAPTOP-T6KSEJC7.3312.0.v2')
event_acc.Reload()

# Lấy dữ liệu từ file sự kiện
train_loss = []
train_accuracy = []
val_loss = []
val_accuracy = []

for tag in event_acc.Tags()['scalars']:
    if 'loss' in tag:
        train_loss.extend(event_acc.Scalars(tag))
    elif 'acc' in tag:
        train_accuracy.extend(event_acc.Scalars(tag))
    elif 'val_loss' in tag:
        val_loss.extend(event_acc.Scalars(tag))
    elif 'val_acc' in tag:
        val_accuracy.extend(event_acc.Scalars(tag))

# Chuyển đổi dữ liệu sang dạng DataFrame
import pandas as pd
train_loss_df = pd.DataFrame(train_loss, columns=['step', 'train_loss'])
train_accuracy_df = pd.DataFrame(train_accuracy, columns=['step', 'train_accuracy'])
val_loss_df = pd.DataFrame(val_loss, columns=['step', 'val_loss'])
val_accuracy_df = pd.DataFrame(val_accuracy, columns=['step', 'val_accuracy'])

# Vẽ biểu đồ
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(train_accuracy_df['step'], train_accuracy_df['train_accuracy'], label='Training accuracy')
plt.plot(val_accuracy_df['step'], val_accuracy_df['val_accuracy'], label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(train_loss_df['step'], train_loss_df['train_loss'], label='Training loss')
plt.plot(val_loss_df['step'], val_loss_df['val_loss'], label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()

plt.show()
