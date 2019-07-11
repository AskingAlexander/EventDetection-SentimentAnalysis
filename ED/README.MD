# ED 
For this part we are going to use 2 approaches:

 - MABED ([Mention-Anomaly-Based Event Detection](https://github.com/AdrienGuille/pyMABED))
 - OLDA ([Latent Dirichlet Allocation with online variational Bayes algorithm]([https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html)))

For both approaches I am attaching python files in order to use them. In order for this to work with both OLDA and MABED you must put the content of  ([this repo](https://github.com/AdrienGuille/pyMABED)) inside the ED folder (the whole pyMABED folder).

In order to run the sample code position yourself at the root of the repository and use ``python ED\EDMethods.py``.

## Requirements
```
pip install scipy
pip install numpy
pip install networkx
pip install matplotlib
pip install Frozen-Flask
pip install sklearn
pip install seaborn
pip install pandas
pip install nltk

import nltk
nltk.download()

```
