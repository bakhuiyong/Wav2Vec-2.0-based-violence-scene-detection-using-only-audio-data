from tqdm import tqdm
import logging
logging.basicConfig(filename='log.log',level=logging.INFO)
import torch
import torch.nn as nn
import os

def train(model, criterion, optimizer, train_dataloader,validation_dataloader, epochs, save_epoch, save_path):

    def save(name, model, epoch, optimizer, loss, acc):
        path = os.path.join(save_path,name)
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'acc': acc,
                    }, path)


    best_acc = 0
    for epoch in range(epochs):
        pbar = tqdm(train_dataloader)
        total_loss = 0
        total_sample = 0
        correct = 0
        model.train()
        cur_step = 0
        logging.info("[training]training start")
        for batch in pbar:
            optimizer.zero_grad()
            x, y = batch['net_input'], batch['target']
            for k in x.keys():
                x[k] = x[k].cuda()
            y = y.cuda()
            
            
            logits = model(x)
            loss = criterion(logits,y)

            total_loss += loss.item()
            total_sample += y.size(0)
            
            pred = logits.data.max(1, keepdim=True)[1].squeeze()

            correct += pred.eq(y.data.view_as(pred)).sum().item()

            loss.backward()
            optimizer.step()
            cur_step += 1
            acc = (correct/total_sample)*100
            pbar.set_description("loss: {} acc:{:.2f}".format(total_loss/cur_step, acc))
        
        logging.info("[test]test start")
        with torch.no_grad():
            pbar = tqdm(validation_dataloader)
            total_loss = 0
            total_sample = 0
            correct = 0
            cur_step = 0
            model.eval()
            for batch in pbar:
                x, y = batch['net_input'], batch['target']
                
                for k in x.keys():
                    x[k] = x[k].cuda()
                y = y.cuda()

                logits = model(x)
                loss = criterion(logits,y)

                total_loss += loss.item()
                total_sample += y.size(0)

                pred = logits.data.max(1, keepdim=True)[1].squeeze()
                cur_step += 1
                correct += pred.eq(y.data.view_as(pred)).sum().item()
                acc = (correct/total_sample)*100
                pbar.set_description("[test]epoch {} loss: {} acc:{:.2f}".format(epoch,total_loss/cur_step, acc))
            if acc>best_acc:
                best_acc = acc
                save('best_model.pth', model, epoch, optimizer, loss, acc)
                
            
            if epoch%save_epoch==0:
                save('{}epoch-{:.2f}acc.pth'.format(epoch, acc)
                    , model, epoch, optimizer, loss, acc)
            save('latest_model.pth', model, epoch, optimizer, loss, acc)
        logging.info("best acc: {:.2f}".format(best_acc))