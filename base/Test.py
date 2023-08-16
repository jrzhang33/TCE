import torch
import torch.utils.data
import torch
loss_function=torch.nn.CrossEntropyLoss()
def test(valid_queue,net_out,n_classes,val_steps):
    net_out.eval()
    test_loss = 0 
    target_num = torch.zeros((1, n_classes)) 
    predict_num = torch.zeros((1, n_classes))
    acc_num = torch.zeros((1, n_classes))
    running_loss2=0.0


    with torch.no_grad():
        for step, data in enumerate(valid_queue):
            x_train1,x_train2,labels,targets,y_train2= data
            try:
                outputs = net_out(x_train1.squeeze(3))
            except:
                outputs = net_out(x_train1)
            if len(outputs.shape)==1:
                outputs=outputs.unsqueeze(0)
            
            loss2 = loss_function(outputs, targets.long())
            running_loss2 += loss2.item()

            predicted = outputs.max(1).indices 
            
            pre_mask = torch.zeros(outputs.size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
            predict_num += pre_mask.sum(0) 
            tar_mask = torch.zeros(outputs.size()).scatter_(1, targets.data.long().cpu().view(-1, 1), 1.)
            target_num += tar_mask.sum(0) 
            acc_mask = pre_mask * tar_mask 
            acc_num += acc_mask.sum(0) 
        loss=running_loss2/val_steps

        recall = (acc_num / target_num).mean()  
        precision =(acc_num / predict_num) .mean() 
        F1 =(2 * recall * precision / (recall + precision)) .mean()
        accuracy = 100. * acc_num.sum(1) / target_num.sum(1)

        print('Test Acc {}, recal {}, precision {}, F1-score {}'.format(accuracy, recall, precision, F1))
    return accuracy.item(),loss