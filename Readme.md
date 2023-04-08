# Want to instantly categorize financial articles?
_If yes, then this model is for you!_


# __What does this ML model do?__

This script trains Naive Bayes, SVM, and Random Forest classifiers on the same dataset of articles, and then uses a voting classifier to combine their predictions and make the final prediction. The voting classifier uses the "soft" voting strategy, which means it predicts the class with the highest average probability across all the individual classifiers.

You can modify this script to use different machine learning algorithms, or add more classifiers to the voting classifier to improve its accuracy.


**Why try it ?**
---

- _Learn how to create & combine different classifier models._
- _Basic preprocessing for text inputs._
- _Familiarize with saving & reusing trained ML models._

**High level overview**
---
This diagram shows the main steps of the script, starting with the input data (articles), followed by data preprocessing, machine learning algorithms, and a voting classifier that combines the predictions of the three classifiers (Naive Bayes, SVM, and Random Forest) to make the final prediction of the article's category. The predicted categories are the output of the script.

```
+-----------------------+
|                       |
|  Input Data (Articles)|              
|                       |
+-----------+-----------+
            |
            |
            v
+-----------+-----------+
|                       |
|    Data Preprocessing |              
|                       |
+-----------+-----------+
            |
            |
            v
+-----------+-----------+
|                       |
|     Machine Learning  |              
|       Algorithms      |
|                       |
+-----------+-----------+
            |
            |
            v
+-----------+-----------+
|                       |
|  Voting Classifier    |              
|                       |
+-----------+-----------+
            |
            |
            v
+-----------+-----------+
|                       |
|  Predicted Categories |              
|                       |
+-----------------------+
```

**Install required dependencies, with pip.**
---

```
pandas==1.2.4
scikit-learn==0.24.2
numpy==1.20.2
```

**Run the Supermodel core.**
---

```
python Supermodel_core.py
```


**Run the application.**
---

```
python Supermodel.py 
```

**Inspiration drawn:**
---

https://www.youtube.com/watch?v=i6qL3NqFjs4