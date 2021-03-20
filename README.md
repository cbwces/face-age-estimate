# face-age-estimator
implement human age train and predict
using method based on [Rank consistent ordinal regression for neural networks with application to age estimation](https://arxiv.org/pdf/1901.07884v7.pdf)
and one implement of [smooth grad](https://arxiv.org/pdf/1706.03825.pdf)

## usage
1. prepare original dataset, and put all image paths into one file (optional)
2. extract bounding box and face 5 keypoint and make a json file like in **image_crop_info.json** (optional)
3. config **parameters.yml**
4. run 
```python
python main.py ./parameters.yml
```

## resume
```python
python ./{model path you save}/parameters.yml
```

## visualize
```python
python ./visualize/visualize.py ./visualize/visualize_parameters.yml {path of image you want to visualize}
```

