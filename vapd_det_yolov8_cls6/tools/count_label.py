import os

img_folder = '/app/gwc/VAPD/vapd_det_yolov8_cls6/outputs/vapd'

img_list = os.listdir(img_folder)
img_list.sort() 
img_info_dict = {}
for img in img_list:
    img_label=img.split("_")[2]
    if img_label not in img_info_dict.keys():
        img_info_dict[img_label]=1
    else:
        img_info_dict[img_label]+=1
        
print(img_info_dict)