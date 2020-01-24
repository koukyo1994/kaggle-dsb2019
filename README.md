# kaggle-dsb2019

This is a repository for the Kaggle competition "Data Science Bowl 2019"

## Writeup

We(@yasufuminakama, @currypurin, @hidehisaarai1213) would like to thank Booz Allen Hamilton for the very interesting competition and to all the participants for giving us a lot of ideas.

### Features

#### Nakama Feature
- Nunique features of ['event_id', 'game_session', ...and so on]
- Decayed Count features & Ratio features of ['title_event_code', 'title', ...and so on]
Count features decayed by elapsed time from previous assessment and their Ratio features.
Ratio features are better than Count features.
Below is an example of decay function.
```
def decaying_counter(x, days):
    return max(0.xx, 1-days/30) * x
```
- Misclicking features
As mentioned [here](https://www.kaggle.com/zgzjnbzl/visualizing-distraction-and-misclicking), event_code 4070 are clicks on invalid places on the screen.
So by using kmeans clustering of 4070 (x, y), we expect cluster as object or something on the screen, then calculating distance from it's own cluster, the distance can represent
"Operation is rougher or unfamiliar as the distance is larger?".
- Past assessment features
Statistical features of past assessment of all time & last 7 days for overall & each assessment title.
- What didn't work
TfIdf & w2v feature on sequence of titles before assessment. I should've tried more...

#### アライさん's features

Feature generation code is made public at [https://github.com/koukyo1994/kaggle-dsb2019/blob/master/src/features/past_summary3_decay.py](https://github.com/koukyo1994/kaggle-dsb2019/blob/master/src/features/past_summary3_decay.py).

* Features based on public kernels

Although it worked well, it can also be a cause of overfitting since the distribution of count based features differs between train and test. Therefore, I also applied decaying factor to count them or get average of them, which had already been proved to be effective to avoid overfitting in Y.Nakama's experiment.

Decaying of count features in Y.Nakama's and mine is slightly different, since Y.Nakama applied decaying to every assessments while I applied decaying to every sessions.

Note that some of those features which appeared to be not useful so much based on LightGBM importance or showed high correlation with other features were deleted from feature generation code.

* Past Assessment features
  * {mean, var, last} of {success_ratio, n_failure, accuracy_group} of the same assessment type in the past
  * time to get success
  * {mean, var} of interval of actions (event_code `4020`, `4025`)

* Past Game features
  * {mean, var, last} of {correct, incorrect} (decayed) count of each Game
  * {mean, var, last} of success ratio of each Game
  * {mean, var} of interval of actions in each Game

* Past Activity features

Few handcrafted features of some of the activities.

### data augmentation idea
As we apply decay function to Count features, we could augment data by using different decay functions.
The aim is that oblivion curve is different from each people by age or something.

### Feature selection

Feature selection using LightGBM / CatBoost importance were applied before training. About 80 - 90% of the features were deleted at this step and the resulting number of features are around 400. Feature selection was effective especially to NN model (probably because of high dropout rate in NN model) and bumped up the oof score around 0.005 for GBDT model and 0.01 for NN model.

### Model

#### Tree based models
My team tried several objectives. Cross entropy and multiclass worked, used it for the final model.
* Final model used three models
  * Lightgbm: cross entropy
  * Lightgbm: multiclass
  * Catboost: multiclass
* cross entropy
  * Divide the target by 3 and convert from 0 to 1, then learn with cross entropy (objective: xentropy). In the final model, this model's weight was the largest.
* multiclass
  * In multiclass, after calculating the probabilities of the target class from 0 to 3, the following calculation is performed to make continuous values.
  * `preds @ np.arange(4)`
#### Tree based models didn't work
* CatBoost
  * regression, CrossEntropy
* Lightgbm
  * regression, multiclassova（One-vs-All)
* Xgboost
  * regression, reg:logistic

#### NN model

Applying certain transformation to the output of multiclass classification gives us better result
compared to regression. The transformation is as follows.

```python
prediction @ np.arange(4)  # the format of prediction should be (n_samples, 4)
```

Our NN model is simple 3 layer MLP. The implementation is [here](https://github.com/koukyo1994/kaggle-dsb2019/blob/master/src/models/neural_network/model.py) (`DSBOvR` is the model we used).

We used training of one-vs-rest fashion, so the output of the model is a (n_batch, 4) shape tensor and each column represents the probability of each class. `torch.nn.BCELoss` was used for loss function and after getting the output tensor, following transformation is applied to get (pseudo-)regression value.

```python
valid_preds = valid_preds / np.repeat(
    valid_preds.sum(axis=1), 4).reshape(-1, 4)  # normalization
valid_preds = valid_preds @ np.arange(4) / 3  # conversion to get pseudo-regression value
```

this pseudo-regression value can be used for threshold optimization. Note that we normalized this value to be in the range of (0.0, 1.0) while training.

Before training, feature selection using LightGBM importance (about 80-90% of the features were deleted), preprocessing (fillna, log transformation for those feature which showed high skewness, feature scaling with `StandardScaler`) was applied. When training, Adam optimizer is used with CosineAnnealing lr scheduler and for each fold we trained the model 100 epochs. At the end of each epoch we calculate QWK using threshold optimization to pseudo-regression value and saved the weights if the best score is achieved. Final oof and prediction to test data was made with the weights which achived the best QWK score at each fold.

We've also prepared NN only kernel [here](https://www.kaggle.com/hidehisaarai1213/dsb2019-nn-ovr-reduce-90-val-60-percentile).

### validation strategy
- validation selected by number of Assessments
If validation is performed using all data, model fits strongly to the data which has many previous assessments and thus easy to predict.
Therefore, the 95% quantile of the distribution of the Assessment number of the test that is truncated is used as a threshold, then removed the data that exceeds the threshold from validation. That one also raised all oof CV.

### Ensemble and QWK threshold
Ensemble using all oof is not appropriate to maximize truncated CV.
Therefore, We sampled the training data at the same ratio as when truncate.
In particular, sampling weight is 1/(Assessment Count) for each installation_id.
Blend is performed based on this sampled data. We also tried stacking by Ridge regression, but we don't think there is a big difference from blending.
The threshold is also determined so that the truncated cv of this sampled data is maximized.

### Metric used for validation

Both public LB score and oof score was not very helpful to judge if a change in our submission is effective or not. Therefore we used truncation to train data to mimic the generation process of test data. This truncation is mostly the same as that shared in common in discussion (select 1 assessment from each installation\_id). Since this score is a bit unstable we repeated the sampling & scoring process 1000 times and calulated the mean of the score.

### Final result

truncated score: 0.5818, public score: 0.565, private score: 0.557
