import os
from create_groundtruth import create_gt
from create_predict import create_prediction
for root, dirs, files in os.walk("data", topdown=True):
    for name in dirs:
        try:
            os.mkdir(os.path.join('output','groundtruth',name))
        except FileExistsError:
            print(name+' groundtruth folder already exists')
        try:
            os.mkdir(os.path.join('output','predictions',name))
        except FileExistsError:
            print(name+' prediction folder already exists')


    for name in files:
        folder = root.split('\\')[-1]
        print(folder)
        path_gt = os.path.join('output','groundtruth', folder, name[:-4])
        path_pred = os.path.join('output','predictions', folder, name[:-4])
        try:
            os.mkdir(path_gt)
        except FileExistsError:
            print(path_gt+'folder already exists')
        try:
            os.mkdir(path_pred)
        except FileExistsError:
            print(path_pred+'folder already exists')

        create_gt(os.path.join(root,name),path_gt)
        create_prediction(os.path.join(root,name),path_pred)







