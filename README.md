# face-age-estimator
implement human age train and predict
using method based on [Rank consistent ordinal regression for neural networks with application to age estimation](https://arxiv.org/pdf/1901.07884v7.pdf)
and implement label salient visualize using [smooth grad](https://arxiv.org/pdf/1706.03825.pdf) and [guided backpropagation](https://arxiv.org/pdf/1412.6806.pdf)

## usage
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
