import numpy as np

gt_label = np.loadtxt('E:\\RK\\Toronto_3D\\test\\L001 - Cloud.labels', dtype=int)
indices = (gt_label != 0)
gt_label = gt_label[indices]
#print(gt_label)

pred_label = np.loadtxt('E:\\RK\\Toronto_3D\\test\\results\\L001_pred.labels', dtype=int)
pred_label = pred_label[indices]
#print(pred_label)


gt_classes = [0] * 9
positive_classes = [0] * 9
true_positive_classes = [0] * 9
for i in range(gt_label.shape[0]):
    gt_l = int(gt_label[i])
    pred_l = int(pred_label[i])
    gt_classes[gt_l] += 1
    positive_classes[pred_l] += 1
    true_positive_classes[gt_l] += int(gt_l==pred_l)

# print("Classes:\t{}".format("\t".join(map(str, gt_classes))))
# print("Positive:\t{}".format("\t".join(map(str, positive_classes))))
# print("True positive:\t{}".format("\t".join(map(str, true_positive_classes))))

print("Overall accuracy: {0}".format(sum(true_positive_classes)/float(sum(positive_classes))))


print("Class IoU:")
iou_list = []
for i in range(9):
    # iou = true_positive_classes[i] / float(gt_classes[i] + positive_classes[i] + true_positive_classes[i])
    iou = true_positive_classes[i+1]/float(gt_classes[i+1]+positive_classes[i+1]- true_positive_classes[i+1])
    print("  {}: {}".format(i, iou))
    iou_list.append(iou)


print("Average IoU: {}".format(sum(iou_list)/8.0))
print("mIoU: {}".format(np.nanmean(iou)))