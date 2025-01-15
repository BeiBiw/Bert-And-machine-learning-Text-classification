from sklearn.feature_extraction.text import TfidfVectorizer
from model.config import set_args
from model.dataset import load_data

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,auc
import os
from tqdm import tqdm
import numpy as np
import random

if __name__ == "__main__":
    args = set_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    print("Loading data......")
    train_data,train_label = load_data(args.train_path)
    train_data = train_data.tolist()
    train_label = train_label.tolist()

    test_data,test_label = load_data(args.test_path)
    test_data = test_data.tolist()
    test_label = test_label.tolist()
    print("Processing data......")
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_data)
    X_test = vectorizer.transform(test_data)
    print("Train......")
    # RF
    # model = RandomForestClassifier(random_state=args.seed)
    # DT
    # model = DecisionTreeClassifier(random_state=args.seed)
    # AB
    # model = AdaBoostClassifier(random_state=args.seed)
    # SVM
    model = SVC(random_state=args.seed)

    model.fit(X_train, train_label)
    print("Predict......")

    predictions = model.predict(X_test)

    # 计算评估指标
    accuracy = accuracy_score(test_label, predictions)
    precision = precision_score(test_label, predictions, average='weighted')  # 使用加权平均
    recall = recall_score(test_label, predictions, average='weighted')  # 使用加权平均
    f1 = f1_score(test_label, predictions, average='weighted')

    # 打印结果
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1 值 (F1 Score): {f1:.4f}")
