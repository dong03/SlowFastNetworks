import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, \
    log_loss

def evaluate(gt_labels, pred_labels, scores):
    # TODO one2multi 多分类和二分类的指标不同
    n = gt_labels.shape[0]
    fake_idx = gt_labels > 0.5
    real_idx = gt_labels < 0.5
    real_loss = 0
    fake_loss = 0
    if np.sum(real_idx * 1) > 0:
        real_loss = log_loss(fake_idx[real_idx], scores[real_idx], labels=[0, 1])
    if np.sum(fake_idx * 1) > 0:
        fake_loss = log_loss(fake_idx[fake_idx], scores[fake_idx], labels=[0, 1])

    print("{}fake_loss".format(""), fake_loss)
    print("{}real_loss".format(""), real_loss)

    bce = (fake_loss + real_loss) / 2
    if fake_loss * real_loss == 0:
        n += 1
        temp = [gt_labels,pred_labels,scores]

        for i in range(3):
            temp[i] = temp[i].tolist()
            temp[i].append((fake_loss==0)*1)
            temp[i] = np.array(temp[i])
        gt_labels, pred_labels, scores = temp

    tn, fp, fn, tp = confusion_matrix(gt_labels, pred_labels).reshape(-1)
    assert ((tn + fp + fn + tp) == n)
    sen = float(tp) / (tp + fn + 1e-8)
    spe = float(tn) / (tn + fp + 1e-8)
    f1 = 2.0 * sen * spe / (sen + spe)
    acc = float(tn + tp) / n

    auc = roc_auc_score(gt_labels, scores)
    ap = average_precision_score(gt_labels, scores)

    return {'bce':bce,'auc': auc, 'ap': ap, 'sen': sen, 'spe': spe, 'f1': f1, 'acc': acc}
