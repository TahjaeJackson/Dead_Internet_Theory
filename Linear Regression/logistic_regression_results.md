# Logistic Regression Results

## Model Performance
- Accuracy: 0.9900
- Precision: 1.0000
- Recall: 0.9773
- ROC-AUC: 1.0000

## Dataset Sizes
- Training samples: 400
- Test samples: 100
- Number of features: 12

## Classification Report
```
              precision    recall  f1-score   support

           0       0.98      1.00      0.99        56
           1       1.00      0.98      0.99        44

    accuracy                           0.99       100
   macro avg       0.99      0.99      0.99       100
weighted avg       0.99      0.99      0.99       100

```

## Feature Importance
- avg_word_length: 2.6164
- contains_links: 1.3943
- sentiment_score: 0.3002
- subreddit_gaming: 0.2361
- user_karma: 0.1040
- subreddit_politics: 0.0557
- subreddit_funny: -0.0041
- subreddit_pics: -0.0676
- subreddit_worldnews: -0.0978
- subreddit_technology: -0.1436
- account_age_days: -0.3861
- reply_delay_seconds: -3.7883

## Discussion Notes

- The logistic regression model performed extremely well on this dataset, reaching 99% accuracy and a ROC-AUC of 1.00. This suggests that the features we used are very effective at separating bot activity from human behavior.

- One of the strongest signals in the model was reply delay. Bots in the dataset tend to respond much faster than humans, which the model picked up very clearly.

- Features related to writing style also played a role. In particular, average word length and whether a comment contains links appear to help distinguish bot comments from human ones.

- Account behavior features such as account age and user karma were also useful,since bot accounts are often newer or have different activity patterns.

- The subreddit features had a much smaller impact compared to behavioral features, suggesting that how an account behaves is more informative than where it posts.

- While the results are very strong, the dataset itself is relatively small (400 training samples and 100 test samples). Testing the model on a larger dataset would help confirm whether the performance remains this high.

- Overall, logistic regression worked well for this task and also allowed us to easily interpret which features were most important for detecting bot behavior.