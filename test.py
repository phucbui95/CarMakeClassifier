from models import BaseModel
from options.base_options import BaseOptions
from dataset import get_cars_datasets
import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd

from tqdm import tqdm as tqdm

def inference(model, dataloader, device):
    model.eval()
    model.to(device)
    with torch.no_grad():
        ids = []
        prediction = []
        i = 0
        data_iter = iter(dataloader)
        for batch_idx, batch_data in tqdm(enumerate(data_iter)):
            data, id = batch_data['data'], batch_data['id']
            data = Variable(data)
            if device.type != 'cpu':
                data = data.type(torch.cuda.FloatTensor)
            else:
                data = data.type(torch.FloatTensor)

            out = model(data)
            pred = out.cpu().numpy()

            ids.append(id)
            prediction.append(pred)
            i += 1
            if i % 2 == 0: break

    id_col = np.hstack(ids).reshape(-1, 1)
    pred_col = np.vstack(prediction)
    print(id_col.shape)
    print(pred_col.shape)
    n_classes = pred_col.shape[1]
    column_names = ['class_' + str(i) for i in range(n_classes)]
    column_names = ['id'] + column_names
    df =  pd.DataFrame(np.hstack([id_col, pred_col]), columns=column_names)
    return df

if __name__ == '__main__':
    option_parser = BaseOptions()
    opt = option_parser.parse()

    dataloaders = get_cars_datasets(opt)
    testloader = dataloaders['test']
    n_classes = 196
    clf = BaseModel(n_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df_submission = inference(clf, testloader, device=device)
    df_submission.to_csv('outputs/submission.csv', index=False)






