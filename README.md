# kaggle-dsb2019

This is a repository for the Kaggle competition "Data Science Bowl 2019"

## Writeup - kaggle masterのアライさん part

### Features - アライさん's features

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

### Model
