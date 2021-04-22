# face-age-estimator
implement human age train and predict
using method based on [Rank consistent ordinal regression for neural networks with application to age estimation](https://arxiv.org/pdf/1901.07884v7.pdf)
and implement label salient visualize using [smooth grad](https://arxiv.org/pdf/1706.03825.pdf) and [guided backpropagation](https://arxiv.org/pdf/1412.6806.pdf)

## usage
### pretrained model(optional)
Here is a pretrained model of efficientnet-b5 based on a mix subset of [IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/), [megaage](http://mmlab.ie.cuhk.edu.hk/projects/MegaAge/), [UTKFace](https://susanqq.github.io/UTKFace/), [AFAD](https://afad-dataset.github.io/), and [AAF](https://github.com/JingchunCheng/All-Age-Faces-Dataset), which can provide age predict from 1 to 90
**Download:** 
[Google Drive](https://drive.google.com/file/d/1uQXzXK8blsp8nNu5i7Dj9IcSyuTr9BcG/view?usp=sharing)
[Baidu Netdisk](https://pan.baidu.com/s/1uZHZv8JXBzWqwPwYGEyU9Q) p4kt

### training
1. prepare original dataset, and put all image paths into one file (optional)
2. extract bounding box and face 5 keypoint and make a json file like in **image_crop_info.json** (optional)
3. config **parameters.yml**
4. run 
```python
python main.py ./parameters.yml
```

### resume
```python
python main.py ./{model path you save}/parameters.yml
```

### test
after train your model, configure the file in **./test/test_parameters.yml**, then
```python
python ./test/image_predict.py ./test/test_parameters.yml
```

### visualize
```python
python ./visualize/visualize.py ./visualize/visualize_parameters.yml {path of image you want to visualize}
```

