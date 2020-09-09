# NBA Player Salary Prediction

This is NBA Salaries Prediction; feel free to use any part of this code you found useful for your projects, and if my code has a problem or you think you have ways to improve it, let me know or make changes and discuss your idea with me. Please upvote if you found it useful. Thank you!

## Introduction
This model helps predict players’ salaries in the NBA based on statistics of players who have played before them. Changes in the salary cap and new tv deals have caused significant changes to player salaries since 1985. This means that the player managers and coaches will predict how much a player is a worth based on this model.
Therefore, a model which can predict players’ salaries according to their performance data is essential, and this can also provide valuable information to a player
on how much he is worth.

There are two datasets: 

- Dataset 1: Contains salaries of all players from 1985-2018 
(https://data.world/datadavis/nba-salaries/workspace/file?filename=salaries_1985to2018.csv)

- Dataset 2: Has information on player statistics. 
(https://data.world/datadavis/nba-salaries/workspace/file?filename=players.csv)



Through my analysis, we can provide a better understanding of:

Essential features that influence the salary.
The most suitable regression models predict players’ salaries and the parameters used to achieve the score best. We have compared data with the following models
- Linear Regression
- Lasso Regression
- Polynomial Regression
- Knn
- Decision Tree
- Random Forest

These are the followed on the code:
1. Data Wrangling and preprocessing
2. Descriptive Statistics and Data Visualization
3. Feature Engineering
4. Algorithms
5. Hyper Parameter Tuning

## Results
THe following are the results obtained from my model:

### Descriptive Statistics and Data Visualization
The following images shows correlation of various features from my code :

![](/NBA%20Player%20Salary%20Prediction/Images/heat_map.png)

![](/NBA%20Player%20Salary%20Prediction/Images/correlation.png)


### Model Evalution
 
 ![](/NBA%20Player%20Salary%20Prediction/Images/Model%20Evaluation.png)


### Prediction

 ![](/NBA%20Player%20Salary%20Prediction/Images/Prediction.png)

### Feature importance

 ![](/NBA%20Player%20Salary%20Prediction/Images/RFR%20feature%20importance.png)
 
 
The evalutaion metrics used are R2 Score, Best Parameter ,RMSE.
It is clear from the results of prediction that random forest model has done better that others.
 
 





