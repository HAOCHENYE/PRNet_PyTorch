# 准备数据集：

- ## 步骤一：

  进入网页：

  [300WLP]: http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm

  下载300WLP数据集，并解压

  

  下载Basel model的模型数据：其中BFM_UV.mat已经在util中，还需要下载BFM.mat文件：

  谷歌网盘链接：

  [BFM.mat]: https://drive.google.com/file/d/1Bl21HtvjHNFguEy_i1W5g0QOL8ybPzxw/view

  百度网盘链接：

  **待放出**

- ## 步骤二：

  对数据进行预处理(300WLP文件下有多个目录，目前脚本只处理单个目录，例如IBUG,HELEN， 训练需要几个数据就处理几个数据，将处理好的数据放在同一根目录下)：

  ```shell
  cd utils
  python generate_posmap_300WLP.py \
  ${SUB 300WLP DATASET} \
  ${SAVE DIR} \
  --uv_path ${BFM_UV PATH} \
  --bfm_path ${BFM PATH} \
```
  
  生成所需数据后，进一步生成json文件：
  
  ```shell
  python generate_json.py ${PROCESSED DATASET DIR}
  ```
  
  label.json会生成在数据根目录下。