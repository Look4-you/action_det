import os 
import shutil

img_floder="/app/gwc/VAPD/vapd_det_yolov8_cls6/outputs/vapd"
save_floder="/app/gwc/VAPD/vapd_det_yolov8_cls6/outputs/vapd"

imgs_list=os.listdir(img_floder)
imgs_list.sort()
img_info_dict={}

for img in imgs_list:
    if not img.endswith(".jpg"): continue
    img_path=os.path.join(img_floder,img)
    sub_dir=os.path.join(save_floder,img[:-6])
    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)
    save_img=os.path.join(sub_dir,img)
    print(save_img)
    print(save_img)
    shutil.move(img_path,save_img)

