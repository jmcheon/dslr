#  Datascience X Logistic Regression - Harry Potter and a Data Scientist
>*_Summary: Write a classifier and save Hogwarts!_*

| Requirements | Skills |
|--------------|--------|
| - `python3.7`<br> - `numpy`<br> - `pandas`<br> - `matplotlib`<br> - `scikit-learn`<br> - `seaborn` | - `DB & Data`<br> - `Algorithms & AI` |


## 1. Data Analysis
### Description
| Column name                   | data type    |
|:------------------------------|:------------:|
| Index                         | int          |
| Hogwarts House                | object (str) |
| First Name                    | object (str) |
| Last Name                     | object (str) |
| Birthday                      | object (str) |
| Best Hand                     | object (str) |
| Arithmancy                    | float        |
| Astronomy                     | float        |
| Herbology                     | float        |
| Defense Against the Dark Arts | float        |
| Divination                    | float        |
| Muggle Studies                | float        |
| Ancient Runes                 | float        |
| History of Magic              | float        |
| Transfiguration               | float        |
| Potions                       | float        |
| Care of Magical Creatures     | float        |
| Charms                        | float        |
| Flying                        | float        |


##### Categorical data
```
Index                 0           1           2           3     ...        1596         1597        1598        1599
Hogwarts House   Ravenclaw   Slytherin   Ravenclaw  Gryffindor  ...   Slytherin   Gryffindor  Hufflepuff  Hufflepuff
First Name          Tamara       Erich    Stephany       Vesta  ...      Shelli     Benjamin   Charlotte       Kylie
Last Name              Hsu     Paredes       Braun   Mcmichael  ...        Lock  Christensen      Dillon       Nowak
Birthday        2000-03-30  1999-10-14  1999-11-03  2000-08-19  ...  1998-03-12   1999-10-24  2001-09-21  2000-08-21
Best Hand             Left       Right        Left        Left  ...        Left        Right        Left        Left
```
##### Numerical data

```
                                count          mean           std           min           25%           50%           75%            max                     
Arithmancy                     1566.0  49634.570243  16679.806036 -24370.000000  38511.500000  49013.500000  60811.250000  104956.000000
Astronomy                      1568.0     39.797131    520.298268   -966.740546   -489.551387    260.289446    524.771949    1016.211940
Herbology                      1567.0      1.141020      5.219682    -10.295663     -4.308182      3.469012      5.419183      11.612895
Defense Against the Dark Arts  1569.0     -0.387863      5.212794    -10.162119     -5.259095     -2.589342      4.904680       9.667405
Divination                     1561.0      3.153910      4.155301     -8.727000      3.099000      4.624000      5.667000      10.032000
Muggle Studies                 1565.0   -224.589915    486.344840  -1086.496835   -577.580096   -419.164294    254.994857    1092.388611
Ancient Runes                  1565.0    495.747970    106.285165    283.869609    397.511047    463.918305    597.492230     745.396220
History of Magic               1557.0      2.963095      4.425775     -8.858993      2.218653      4.378176      5.825242      11.889713
Transfiguration                1566.0   1030.096946     44.125116    906.627320   1026.209993   1045.506996   1058.436410    1098.958201
Potions                        1570.0      5.950373      3.147854     -4.697484      3.646785      5.874837      8.248173      13.536762
Care of Magical Creatures      1560.0     -0.053427      0.971457     -3.313676     -0.671606     -0.044811      0.589919       3.056546
Charms                         1600.0   -243.374409      8.783640   -261.048920   -250.652600   -244.867765   -232.552305    -225.428140
Flying                         1600.0     21.958012     97.631602   -181.470000    -41.870000     -2.515000     50.560000     279.070000
```

## 2. Data Visualization
|[histogram.py](./histogram.py)|[scatter.py](./scatter.py)    |
|---------------------------------------------|-------------------------------------------------------|
|![histogram](https://github.com/jmcheon/dslr/assets/40683323/37f1aff8-fa15-4786-849c-dca507659868)|![scatter](https://github.com/jmcheon/dslr/assets/40683323/d0291802-b765-47ab-b4af-fd1293ee49b3)|

### 2.1 Histogram
Make a script called histogram.[extension] which displays a histogram answering the next question : 

Which Hogwarts course has a homogeneous score distribution between all four houses?
```
variances:
Arithmancy                       2.782159e+08
Astronomy                        2.707103e+05
Herbology                        2.724508e+01
Defense Against the Dark Arts    2.717322e+01
Divination                       1.726653e+01
Muggle Studies                   2.365313e+05
Ancient Runes                    1.129654e+04
History of Magic                 1.958748e+01
Transfiguration                  1.947026e+03
Potions                          9.908986e+00
Care of Magical Creatures        9.437286e-01
Charms                           7.715233e+01
Flying                           9.531930e+03
dtype: float64
```

### 2.2 Scatter plot
Make a script called scatter_plot.[extension] which displays a scatter plot answering the next question : 

What are the two features that are similar?
```
Feature 1                     Feature 2                     Threshold
Astronomy                     Defense Against the Dark Arts 0.9999999999999984
Muggle Studies                Charms                        0.8476070313934801
History of Magic              Transfiguration               0.8492027176461879
History of Magic              Flying                        0.8962834248882747
Transfiguration               Flying                        0.8736726050021425
```

### 2.3 Pair plot
Make a script called pair_plot.[extension] which displays a pair plot or scatter plot matrix (according to the library that you are using). 
![pair plot](https://github.com/jmcheon/dslr/assets/40683323/188ab916-fa6f-4436-823a-46d0859de23a)

From this visualization, what features are you going to use for your logistic regression?


## 3. Logistic Regression

#### Logistic regression

Logistic regression is a supervised machine learning algorithm used primarily for binary classification problems. Its goal is to predict one of two possible outcomes (usually represented as 0 and 1), based on the given input features.

It is a type of generalized linear model (GLM) that uses a logistic function (sigmoid function) to model the probability of an instance belonging to the positive class, given its features.

- [logreg_train.py](./logreg_train.py) saves ./weights.csv
```
Usage:  python logreg_train.py [data path] (for batch gradient descent)
        python logreg_train.py [data path] [batch option]
        three batch options: batch, sgd (for stochastic), mini (for mini-batch)
```
- [logreg_predict.py](./logreg_predict.py) takes ./weights.csv and saves ./houses.csv
- [evaluate.py](./evaluate.py) - evaluates on dataset_truth.csv with houses.csv

![optimizers0](https://github.com/jmcheon/dslr/assets/40683323/c6221d34-d6e6-4edc-a906-733046cccced)
