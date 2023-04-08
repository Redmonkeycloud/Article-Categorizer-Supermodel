# Want to instantly categorize financial articles?
_If yes, then this model is for you!_


# __What does this ML model do?__

This script trains Naive Bayes, SVM, and Random Forest classifiers on the same dataset of articles, and then uses a voting classifier to combine their predictions and make the final prediction. The voting classifier uses the "soft" voting strategy, which means it predicts the class with the highest average probability across all the individual classifiers.

You can modify this script to use different machine learning algorithms, or add more classifiers to the voting classifier to improve its accuracy.


**Why try it ?**
- _Introduction to the web design world._
- _Get to know Lottie._
- _Familiarize with Streamlit._
- _Just a tiny bit of CSS._

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
python Supermodel_core.py.py 
```


**Run the application.**
---

```
python Supermodel.py 
```

**Inspiration drawn:**
---

https://www.youtube.com/watch?v=i6qL3NqFjs4