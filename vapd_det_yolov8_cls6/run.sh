rm -rf outputs/vapd/*.jpg
rm -rf outputs/det_resutl/*/*
rm -rf outputs/vapd_video/*


cd build 
cmake ..
make 
CUDA_VISIBLE_DEVICES=1 ./vapd_yolov8_det 


# rm outputs/vapd-s/*.jpg
# cd build 
# cmake ..
# make 
# CUDA_VISIBLE_DEVICES=0 ./vapd_yolov8_det 