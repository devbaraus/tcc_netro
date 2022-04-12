# %%
from calendar import c
import math
from matplotlib import pyplot as plt
import numpy as np
from praudio import utils
import seaborn as sns


def plot_confusion_matrix(confusion_matrix, size, save_path=''):
    fig, ax = plt.subplots(figsize=(max(size/2, 16), max(size/2, 16)))
    ax.matshow(confusion_matrix, cmap=plt.cm.Blues, alpha=0.3)

    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(x=j, y=i, s=confusion_matrix[i, j],
                    va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)

    fig.tight_layout()
    plt.show()
    plt.draw()

    if save_path:
        utils.create_dir_hierarchy(save_path)
        fig.savefig(f'{save_path}/confusion_matrix.jpg',
                    dpi=300)

    plt.close()


def plot_history(history, save_path=''):
    """Plots accuracy/loss for training/validation set as a function of the epochs

    :param history: Training history of model
    :return:
    """

    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="accuracy")
    axs[0].plot(history.history['val_accuracy'], label="val_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation")
    axs[0].margins(x=0)
    axs[0].grid()

    # create loss subplot
    axs[1].plot(history.history["loss"], label="loss")
    axs[1].plot(history.history['val_loss'], label="val_loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss evaluation")
    axs[1].margins(x=0)
    axs[1].grid()

    fig.tight_layout()
    plt.show()
    plt.draw()

    if save_path:
        utils.create_dir_hierarchy(save_path)
        fig.savefig(f'{save_path}/train_history.jpg',
                    dpi=300)

    plt.close()

# %%


def plot_class_distribution(labels, count, save_path='', filename='class_distribution.jpg'):
    labels = [str(x) for x in labels]

    fig = plt.figure(figsize=(16, 10))

    c1 = plt.bar(labels, count)
    plt.bar_label(c1, label_type='center')
    plt.grid()

    plt.title('Class Distribution')
    plt.xlabel('Classes')
    plt.ylabel('Documents')
    plt.margins(x=0)

    fig.tight_layout()
    plt.show()
    plt.draw()

    if save_path:
        utils.create_dir_hierarchy(save_path)
        fig.savefig(f'{save_path}/{filename}',
                    dpi=300)

    plt.close()


# %%
if __name__ == '__main__':
    plot_class_distribution(['1', '2', '3', '4'],
                            [5, 10, 5, 10],
                            save_path='/src/tcc')
