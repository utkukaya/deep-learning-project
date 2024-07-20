import numpy as np
import torch
from sklearn.model_selection import KFold

def precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) != 0 else 0

def recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) != 0 else 0

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * prec * rec / (prec + rec) if (prec + rec) != 0 else 0

def test_model(model, data):
    X_train_tensor = torch.tensor(data.values.reshape(int(len(data)), 1, data.shape[1]), dtype=torch.float32)
    outputs = model.forward(X_train_tensor)
    predictions = torch.argmax(outputs, axis=1)
    return predictions


def create_permutation(x, y):
    perm = np.random.permutation(len(x))
    return x[perm], y[perm]

def train_test_split(X, y, ratio=.2):
    X, y = create_permutation(X, y)
    split_index =  int(len(X) * (1-ratio))
    X_train, y_train = X[:split_index], y[:split_index]
    X_test, y_test = X[split_index:], y[split_index:]
    return X_train, y_train, X_test, y_test
    
def train_model(X, y, model, criterion, optimizer, n_epochs=200, print_every=10, test_every=10, shuffle_on_each_epoch=True, reshape_size=4172):
    train_accs = []
    test_accs = []
    train_losses = []
    test_losses = []
    test_points = []
    train_points = []

    X_train, y_train, X_test, y_test = train_test_split(X, y)
    print(X_train.shape)
    X_train_tensor = torch.tensor(X_train.reshape(int(4*len(X) // 5), 1, reshape_size), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train)
    X_test_tensor = torch.tensor(X_test.reshape(len(X) - int(4*len(X) // 5), 1, reshape_size), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test)
    
    for epoch in range(n_epochs):
        if shuffle_on_each_epoch:
            X_train, y_train, X_test, y_test = train_test_split(X, y)

            X_train_tensor = torch.tensor(X_train.reshape(int(4*len(X) // 5), 1, reshape_size), dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train)
            X_test_tensor = torch.tensor(X_test.reshape(len(X) - int(4*len(X) // 5), 1, reshape_size), dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test)

        optimizer.zero_grad()
        outputs = model.forward(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        predictions = torch.argmax(outputs, axis=1)

        train_acc = (predictions == y_train_tensor).detach().numpy().astype(int).mean() * 100
        train_accs.append(train_acc)
        train_losses.append(loss.detach().numpy())
        train_points.append(epoch)
        if epoch % print_every == 0:
            print(f"TRAIN:\tEpoch: {epoch:3d}, Loss: {loss:.5f}, Accuracy: {train_acc:.5f}")

        if epoch % test_every == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                outputs = model.forward(X_test_tensor)
                predictions = torch.argmax(outputs, axis=1)
                test_acc = (predictions == y_test_tensor).detach().numpy().astype(int).mean() * 100
                test_losses.append(loss.detach().numpy())
                test_points.append(epoch)
                test_accs.append(test_acc)
                print(f"TEST:\tEpoch: {epoch:3d}, Loss: {loss:.5f}, Accuracy: {test_acc:.5f}")

    return model, train_accs, train_losses, train_points, test_accs, test_losses, test_points


def train_model_k_fold(X, y, model, criterion, optimizer, n_epochs=10, print_every=10, test_every=10, reshape_size=4172):
    train_accs = []
    test_accs = []
    train_losses = []
    test_losses = []
    test_points = []
    train_points = []
    kfold = KFold(n_splits=5, shuffle=True)

    X_train, y_train, X_test, y_test = train_test_split(X, y)
    print(X_train.shape)

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        X_train_tensor = torch.tensor(X_train.reshape(int(len(train_idx)), 1, reshape_size), dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train)
        X_test_tensor = torch.tensor(X_test.reshape(len(test_idx), 1, reshape_size), dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test)
        for epoch in range(n_epochs):

            optimizer.zero_grad()
            outputs = model.forward(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            predictions = torch.argmax(outputs, axis=1)

            train_acc = (predictions == y_train_tensor).detach().numpy().astype(int).mean() * 100
            train_accs.append(train_acc)
            train_losses.append(loss.detach().numpy())
            train_points.append(epoch)
            if epoch % print_every == 0:
                print(f"TRAIN:\tEpoch: {epoch + n_epochs*fold:3d}, Loss: {loss:.5f}, Accuracy: {train_acc:.5f}")

            if (epoch % test_every == 0 or epoch + n_epochs * fold == n_epochs*5-1):
                with torch.no_grad():
                    outputs = model.forward(X_test_tensor)
                    predictions = torch.argmax(outputs, axis=1)
                    test_acc = (predictions == y_test_tensor).detach().numpy().astype(int).mean() * 100
                    test_losses.append(loss.detach().numpy())
                    test_points.append(epoch + n_epochs*fold)
                    test_accs.append(test_acc)
                    print(f"TEST:\tEpoch: {epoch + n_epochs*fold:3d}, Loss: {loss:.5f}, Accuracy: {test_acc:.5f}")

    return model, train_accs, train_losses, train_points, test_accs, test_losses, test_points




def precision_recall_multiclass(y_actual, y_prediction, average='macro'):
    classes = np.unique(y_actual)

    precisions = []
    recalls = []
    f1_scores = []
    for cls in classes:
        true_positive = ((y_actual == cls) & (y_prediction == cls)).sum()
        false_positive = ((y_actual != cls) & (y_prediction == cls)).sum()
        false_negative = ((y_actual == cls) & (y_prediction != cls)).sum()
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
        f1_score = (2*precision*recall) / (precision + recall) if (precision + recall) != 0 else 0
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    if average == 'macro':
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1_score = np.mean(f1_scores)
    elif average == 'weighted':
        weights = [(y_actual == cls).sum() for cls in classes]
        precision = np.average(precisions, weights=weights)
        recall = np.average(recalls, weights=weights)
        f1_score = np.average(f1_scores, weights=weights)
    else:
        raise ValueError("Average must be one of ['macro', 'weighted']")

    return precision, recall, f1_score


def prepare_train(model, X, y, reshape_size=4172, n_epochs=200):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return train_model(X, y, model, criterion, optimizer, n_epochs=n_epochs, print_every=10, test_every=10, shuffle_on_each_epoch=False, reshape_size=reshape_size)


def prepare_train_k_fold(model, X, y, reshape_size=4172, n_epochs=10):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return train_model_k_fold(X, y, model, criterion, optimizer, n_epochs=n_epochs, print_every=10, test_every=10, reshape_size=reshape_size)
