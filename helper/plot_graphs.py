from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd 

def plot_confusion_matrix(y_actual, y_prediction, classes, title, file_path=None):
    cm = confusion_matrix(y_actual, y_prediction)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    if(file_path != None):
        plt.savefig(file_path)
    plt.show()


def plot_train_and_test_set(train_data, test_data, test_points, label_train_data, label_test_data, title='Loss of CNN', file_path=None):
    plt.figure(figsize=(10, 8))
    plt.plot(train_data, label=label_train_data)
    plt.plot(test_points, test_data, label=label_test_data)
    plt.xlabel("Epoch Number")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if(file_path != None):
        plt.savefig(file_path)
    plt.show()

def pca_and_plot(data, labels, label_names, title='', file_path=None):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)

    pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
    pca_df['Label'] = labels['Labels'].values

    plt.figure(figsize=(10, 8))

    colors = plt.cm.get_cmap('tab20', 17)
    unique_labels = sorted(labels['Labels'].unique())
    for i, label in enumerate(unique_labels):
        plt.scatter(
            pca_df[pca_df['Label'] == label]['PCA1'],
            pca_df[pca_df['Label'] == label]['PCA2'],
            color=colors(i),
            label=label_names[i],
            alpha=0.6,
            edgecolors='w',
            s=100
        )

    explained_variance_ratio = pca.explained_variance_ratio_

    plt.title(title)
    plt.xlabel(f'PCA1 ({explained_variance_ratio[0]*100:.2f}%)')
    plt.ylabel(f'PCA2 ({explained_variance_ratio[1]*100:.2f}%)')

    plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    if(file_path != None):
        plt.savefig(file_path)
    plt.show()
