import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import gc
from scipy.io import loadmat, savemat


# 这里是故障标签列表，方便混淆矩阵拿到坐标的汉字编码
fault_list_label = ['O', 'I', 'B', 'R', 'OI', 'OB', 'OR', 'IB', 'IR', 'BR', 'N']


def get_label(array, fault_label):
    # 按照混淆矩阵的横纵坐标的文字标签形式，将样本的测试结果和真实结果的标签从数字变成汉字
    return [fault_label[i] for i in np.argmax(array, axis=1)]


def plot_with_labels(path_tsne, layers, y_test, dimension, title):
    '''
    画tsne图函数
    :param path_tsne: tsne文件保存地址
    :param layers: 解析的是哪一个特征层
    :param y_test: 测试样本的标签，用来指定散点的颜色
    :param dimension: tsne图是2维还是3维
    :param title: 文件名
    :return: None
    '''
    plt.close()
    plt.ion()
    # tsne = TSNE(perplexity=30, n_components=dimension, init='random', n_iter=1000, random_state=1000)
    tsne = TSNE(perplexity=40, n_components=dimension, init='pca', n_iter=300)
    lowDWeights = tsne.fit_transform(layers)
    savemat(path_tsne + 'tsne_mats/' + title+'.mat', {title: lowDWeights})

    labels = np.argmax(y_test, axis=1)
    shapes = ['o' for i in range(4)]+['s' for i in range(6)]+['o']
    colors = ['red', 'gold', 'yellow', 'lime', 'mediumpurple', 'dodgerblue', 'fuchsia', 'darkgoldenrod', 'cyan', 'gray', 'black']
    makers = []
    for shape, color in zip(shapes, colors):
        makers.append([shape, color])
    # shape = ['s', 'o', '^']
    # color_list = ['red', 'gold', 'cyan']

    # makers = []
    # for i in shape:
    #     for j in color_list:
    #         makers.append([i, j])
    if dimension == 2:
        plt.figure()
        X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
        for x, y, s in zip(X, Y, labels):
            # plt.scatter(x, y, c=color_list[s], s=10, label=fault_list_label[s])

            # plt.scatter(x, y, c=makers[s][1], marker=makers[s][0], label=fault_list_label[s], s=12, linewidths=1)
            plt.scatter(x, y, c='', marker=makers[s][0], edgecolors=makers[s][1], label=fault_list_label[s], s=12, linewidths=1)

        plt.xlim(X.min(), X.max());
        plt.ylim(Y.min(), Y.max());
        plt.xlabel('TS1')
        plt.ylabel('TS2')
    else:
        plt.figure()
        from mpl_toolkits.mplot3d import Axes3D
        X, Y, Z = lowDWeights[:, 0], lowDWeights[:, 1], lowDWeights[:, 2]
        ax = plt.subplot(111, projection='3d', facecolor='w')

        ax.grid(linestyle='--', alpha=0.1, which='major', linewidth=1)
        for x, y, z, s in zip(X, Y, Z, labels):
            # ax.scatter(x, y, z, c=color_list[s], s=10, label=fault_list_label[s])

            # ax.scatter(x, y, z, c=makers[s][1], marker=makers[s][0],  label=fault_list_label[s], s=12, linewidths=1)
            ax.scatter(x, y, z, c='', marker=makers[s][0], edgecolors=makers[s][1], label=fault_list_label[s], s=12, linewidths=1)

        ax.set_zlabel('TS1')
        ax.set_ylabel('TS2')
        ax.set_xlabel('TS3')
        # ax.grid(False)

    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = OrderedDict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys(), loc='2')

    plt.savefig(path_tsne+'tsne_pictures/'+title+".tif", dpi=300)
    plt.ioff()




def plot_confusion_matrix(font_size_set_times, path, file_name, y_true, y_pred, labels):
    cmap = plt.cm.get_cmap('cool')
    # font_size_set = font_size_set_times*25
    font_size_set = 25
    plt.close()
    cm = confusion_matrix(y_true, y_pred)
    tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    # print(cm)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis].astype('float')

    plt.figure(figsize=(10*font_size_set_times, 8*font_size_set_times), dpi=300)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    intFlag = 0
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        if (intFlag):
            c = cm[y_val][x_val]
            plt.text(x_val, y_val, "%.2f" % (c,), color='black', fontsize=font_size_set, va='center', ha='center')

        else:
            c = cm_normalized[y_val][x_val]
            # print(c)
            if c>0.60:
                plt.text(x_val, y_val, "%0.2f" % (c,), color='white', fontsize=font_size_set, va='center', ha='center')
            elif (c >= 0.01):

                plt.text(x_val, y_val, "%0.2f" % (c,), color='black', fontsize=font_size_set, va='center', ha='center')
            else:
                plt.text(x_val, y_val, "%.2f" % (0,), color='black', fontsize=font_size_set, va='center', ha='center')
    if(intFlag):
        plt.imshow(cm, interpolation='gaussian', cmap=cmap)
    else:
        plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.title('')
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, fontsize=font_size_set*0.8)
    plt.yticks(xlocations, labels, fontsize=font_size_set*0.8)
    plt.ylabel('Actual label', labelpad=font_size_set*0.6, fontsize=font_size_set)
    plt.xlabel('Predict label', labelpad=font_size_set*0.6, fontsize=font_size_set)
    plt.savefig(path+'picture/'+file_name+'_confusion_matrix.tif', dpi=300)
    del cm, cm_normalized
    gc.collect()