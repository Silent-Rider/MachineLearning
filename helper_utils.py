from matplotlib import pyplot as plt
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

def draw_plots(history_dict):
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'o', color='orange', label='Training loss')
    plt.plot(epochs, val_loss_values, 'g', label='Validation loss')
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    accuracy_values = history_dict['accuracy']
    val_accuracy_values = history_dict['val_accuracy']
    plt.clf()
    plt.plot(epochs, accuracy_values, 'o', color='orange', label='Training accuracy')
    plt.plot(epochs, val_accuracy_values, 'g', label='Validation accuracy')
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()