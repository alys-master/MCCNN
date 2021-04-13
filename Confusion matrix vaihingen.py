import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,precision_score,recall_score

cls_num = 9
labels = np.loadtxt('E:\\RK\\Toronto_3D\\test\\L001 - Cloud.labels', dtype=int)
predicted = np.loadtxt('E:\\RK\\Toronto_3D\\test\\results\\L001_pred.labels', dtype=int)


# 第一步：创建混淆矩阵
# 获取类别数，创建 N*N 的零矩阵
conf_mat = np.zeros([cls_num, cls_num])
# 第二步：获取真实标签和预测标签
# labels 为真实标签，通常为一个 batch 的标签
# predicted 为预测类别，与 labels 同长度
# 第三步：依据标签为混淆矩阵计数
for i in range(len(labels)):
    true_i = np.array(labels[i])
    pre_i = np.array(predicted[i])
    conf_mat[true_i, pre_i] += 1.0
# print(conf_mat)
# ----------------RuntimeWarning: invalid value encountered in true_divide----------#
np.seterr(divide='ignore', invalid='ignore')

#计算每类的F1分数
f1_1 = f1_score(labels==0, predicted==0,labels=True)
f1_2 = f1_score(labels==1, predicted==1,labels=True)
f1_3 = f1_score(labels==2, predicted==2,labels=True)
f1_4 = f1_score(labels==3, predicted==3,labels=True)
f1_5 = f1_score(labels==4, predicted==4,labels=True)
f1_6 = f1_score(labels==5, predicted==5,labels=True)
f1_7 = f1_score(labels==6, predicted==6,labels=True)
f1_8 = f1_score(labels==7, predicted==7,labels=True)
f1_9 = f1_score(labels==8, predicted==8,labels=True)
print("第一类f1：" ,f1_1)
print("第二类f1：" ,f1_2)
print("第三类f1：" ,f1_3)
print("第四类f1：" ,f1_4)
print("第五类f1：" ,f1_5)
print("第六类f1：" ,f1_6)
print("第七类f1：" ,f1_7)
print("第八类f1：" ,f1_8)
print("第九类f1：" ,f1_9)
# Average_F1 = (f1_2 + f1_3 + f1_4 + f1_5 + f1_6 + f1_7 + f1_8 + f1_9)/ 9
# print("Average F1:",Average_F1)

# def show_confMat(confusion_mat, classes_name, set_name, out_dir):
#     """
#     可视化混淆矩阵，保存png格式
#     :param confusion_mat: nd-array
#     :param classes_name: list,各类别名称
#     :param set_name: str, eg: 'valid', 'train'
#     :param out_dir: str, png输出的文件夹
#     :return:
#     """
#     # 归一化
#     confusion_mat_N = confusion_mat.copy()
#     for i in range(len(classes_name)):
#         confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()
#
#     # 获取颜色
#     cmap = plt.cm.get_cmap('Blues')  # 更多颜色  http://matplotlib.org/examples/color/colormaps_reference.html
#     plt.gcf().subplots_adjust(bottom=0.3)
#     plt.imshow(confusion_mat_N, cmap=cmap)
#     plt.colorbar()
#
#     # 设置文字
#     xlocations = np.array(range(len(classes_name)))
#     plt.xticks(xlocations, classes_name, rotation=45)
#     plt.yticks(xlocations, classes_name)
#     plt.xlabel('Predict label')
#     plt.ylabel('True label')
#     plt.title('CVI-PointCNN' + set_name)
#
#     # 打印数字
#     for i in range(confusion_mat_N.shape[0]):
#         for j in range(confusion_mat_N.shape[1]):
#             plt.text(x=j, y=i, s=round(confusion_mat_N[i, j], 2), va='center', ha='center',color='black', fontsize=7)
#             # plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=7)
#     # 保存
#     plt.savefig(os.path.join(out_dir, 'CVI-PointCNN' + '.png'))
#     plt.close()
#
#
# # 函数调用示例
# show_confMat(conf_mat,['power', 'low_veg', 'imp_surf', 'car', 'fence', 'roof', 'fac', 'shurb', 'tree'], "valid", "./")


