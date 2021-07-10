# MM3DFace

- 使用MMCV对3D人脸重建项目进行封装，规范化训练流程，旨在搭建一个更高的baseline
- 使用方法和mmdet类似，目录结构有轻微区别，目前只集成了PRNet 和 3DDFA_v1

----
##  Pre-Requirements 

```shell
pip install -r requirements.txt
```

具体每个项目的启用方法在对应config文件中

## Train

具体数据集准备步骤见config/PRNet

**Single GPU:**

```shell
cd ${PROJECT_PATH}/tools
python train.py config/PRNet/ori_config.py
```

**Multi GPU**

```shell
cd ${PROJECT_PATH}/tools
./dist_train.sh config/PRNet/ori_config.py 2
```

## Test

目前仅支持单卡测试多张图片，保存关键点图片

```shell
cd ${PROJECT_PATH}/tools
python test_image.py \
../config/PRNet/ori_config.py \
../tools/${YOUR WORK_DIRS}/latest.pth \
--out_dir ../sample/results
```

## Onnx

导出PRNet的Onnx模型

```shell
cd ${PROJECT_PATH}/tools
python export_onnx.py \
../config/PRNet/ori_config.py \
../tools/${YOUR WORK_DIRS}/latest.pth \
--output-file \
OnnxDir/PRNet.onnx \
```

