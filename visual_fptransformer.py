import torch
import numpy as np
from pathlib import Path
import os


g_classes = [x.rstrip() for x in open('/home/hnu/hy-data/nextpaper/sem/s3dis/PT_test1/data/s3dis/s3dis_names.txt')]
g_class2lable = {cls: i for i, cls in enumerate(g_classes)}
g_class2color = {'ceiling':[0,255,0],
                                    'floor':[10,200,100],
                                    'wall':[255,200,100],
                                    'beam':[255,255,0],
                                    'column':[255, 65 , 125],
                                    'window':[100,100,255],
                                    'door':[255,0,0],
                                    'chair':[140,0,180],
                                    'table':[0,255,255],
                                    'bookcase':[200,100,100],
                                    'sofa':[0,130,255],
                                    'board':[200,200,200],
                                    'clutter':[50,50,50]}
g_easy_view_labels = [7,8,9,10,11,1]
g_label2color = {g_classes.index(cls): g_class2color[cls] for cls in g_classes}


######################
visual_dir = '/home/hnu/hy-data/nextpaper/sem/s3dis/PT_test2_1/exp/s3dis/pointtransformer_repro/result/best/visual'
visual_dir = Path(visual_dir)
visual_dir.mkdir(exist_ok=True)
print(visual_dir )

path = '/home/hnu/hy-data/data/s3dis/trainval_fullarea'
data_list = sorted(os.listdir(path))
data_list = [item[:-4] for item in data_list if 'Area_5' in item]
fout_error_ratio = open(os.path.join(visual_dir, 'area5_error_ratio.txt'),'w')
for item in data_list:
    data_path = os.path.join(path, item + '.npy')
    data = np.load(data_path)  # xyzrgbl, N*7
    data_input_scene = data[:,0:6]
    np.savetxt('%s/%s_input.txt' %(visual_dir, item ),data_input_scene)
    print(item,'input completed')

    path1 = '/home/hnu/hy-data/nextpaper/sem/s3dis/PT_test2_1/exp/s3dis/pointtransformer_repro/result/best'
    pred = np.load(os.path.join(path1, item + '_64_pred.npy'))
    gt = np.load(os.path.join(path1, item + '_64_label.npy'))


    assert len(data_input_scene) == len(pred) == len(gt)
    #############
    # item = item
    fout_pred = open(os.path.join(visual_dir, item + '_pred.txt'), 'w')
    fout_gt = open(os.path.join(visual_dir, item + '_gt.txt'), 'w')
    fout_error = open(os.path.join(visual_dir,item + '_error.txt'),'w')
    a = 0
    ###################
    for i in range(data.shape[0]):
        color = g_label2color[pred[i]]
        color_gt = g_label2color[gt[i]]
        fout_pred.write('%f %f %f %d %d %d \n' % (data[i,0],data[i,1],data[i,2], color[0],color[1],color[2]))
        fout_gt.write('%f %f %f %d %d %d \n' % (data[i,0],data[i,1],data[i,2], color_gt[0],color_gt[1],color_gt[2]))
        if pred[i] == gt[i]:
            fout_error.write('%f %f %f %d %d %d \n' % (data[i,0],data[i,1],data[i,2], 0, 0, 255))
            a = a + 1
        else:
            fout_error.write('%f %f %f %d %d %d \n' % (data[i,0],data[i,1],data[i,2], 255, 0, 255))
    
    fout_error_ratio.write('The accuracy of %s is %f  \n' % (item, a/(data.shape[0])))
    fout_pred.close()
    fout_gt.close()
    fout_error.close()
    print(item,'prediction completed')
    print(item,'groundtruth completed')
    print(item,'error completed')
fout_error_ratio.close()
print('error ratio completed!!!')