import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(conf_matrix):
    # 绘制混淆矩阵图表
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    classes = ['Class 0', 'Class 1']  # 你的类别标签
    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center')

    return plt

# 在你的训练循环中使用这个函数
conf_matrix = confusion_matrix(true_labels, predicted_labels)
plot_confusion_matrix(conf_matrix).show()