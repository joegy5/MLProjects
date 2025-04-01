import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from layers import *
from model import *
from tensorflow.keras.datasets import mnist # type: ignore

class MNISTDataset(Dataset):
    def __init__(self, x, y):
        super(MNISTDataset, self).__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.x.shape[0]


def cross_entropy_loss(A, Y):
    loss =  (-1. / A.shape[0]) * np.sum(Y * np.log(np.clip(A, 1e-7, 1-1e-7) + 1e-7), axis=1)
    return np.mean(loss)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = torch.tensor(x_train[..., np.newaxis].astype('float32') / 255.0)
x_test = torch.tensor(x_test[..., np.newaxis].astype('float32') / 255.0)
y_train = torch.tensor(torch.nn.functional.one_hot(torch.LongTensor(y_train.astype('int'))))
y_test = torch.tensor(torch.nn.functional.one_hot(torch.LongTensor(y_test.astype('int'))))

train_loader = DataLoader(MNISTDataset(x_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(MNISTDataset(x_test, y_test), batch_size=1, shuffle=False)

new_model = CNN()
NUM_EPOCHS = 50
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0
    for images, labels in train_loader:
        np_images = np.array(images)
        np_labels = np.array(labels)
        out_tuples = new_model.forward(np_images)
        epoch_loss += cross_entropy_loss(out_tuples[-1][0], np_labels) * images.shape[0]
        new_model.backward(out_tuples, np_images, np_labels)
    average_loss_for_epoch = epoch_loss / len(train_loader.dataset)
    print(f"EPOCH {epoch} TRAINING LOSS: {average_loss_for_epoch}")
    for test_images, test_labels in test_loader:
        np_t_im, np_t_l = np.array(test_images), np.array(test_labels)
        out = new_model.forward(np_t_im)[-1][0]
        predicted_idx = np.argmax(out, axis=1)
        truth_idx = np.argmax(np_t_l, axis=1)
        num_correct += 1 if predicted_idx[0] == truth_idx[0] else 0
    print(f"EPOCH {epoch} TEST ACCURACY: {num_correct / len(test_loader.dataset)}")

