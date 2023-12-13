MAI_for_stock_return_predictability
==============================

In this project we are examining whether the macroeconomic attention indices proposed by Fischer et al. (2022) have the ability to predict stock returns based on some statistical and machine learning algorithms. To evaluate that we compare their predictability efficiency with that of other popular macroeconomic variables.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebooks
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    │
    ├── Dockerfile
    │
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to turn raw data into features for modeling
        │   └── make_dataset.py
        │
        ├── models         <- Scripts to train models and make predictions
        │   │                 
        │   ├── linear_model.py
        │   │
        │   └── NN_model.py
        │
        └── shiny  <- Code and data to create R shiny app for visualizations
            │
            ├── shiny_data
            │
            └── stock_return_prediction_app.R
    
Description of Steps 
------------

- Preparation of  a folder structure, using the cookiecutter Data Science template

- Downloaded of raw data [see below]

- Create upstream and downstream functions

- Generation of tables which can be directly imported to Latex 

- Connect Overleaf to Github (all Overleaf Files can be directly accessed [here](https://www.overleaf.com/read/yqkhbqjwvtbs#b7cd7c))

- Use R Shiny and Plotly to create interactive graphs in order to do robustness checks - data frequency, data features, date range, visualization - (app can be directly accessed [here](https://baumender11.shinyapps.io/Alpha/))

- Ensure reproducibility by adding Dockerfile 

- Analysis and interpretation of findings

- Compare results with current research

Data 
------------

Macroeconomic Attention Indices (MAI) data are obtained from [charlesmartineau/mai_rfs](https://github.com/charlesmartineau/mai_rfs) Github repository.

MEF data are obtained from the [powder197/Goyal-and-Welch-2008-](https://github.com/powder197/Goyal-and-Welch-2008-/tree/master) Github repository.

S&P500 proces are obtained from [Yahoo Finance](https://finance.yahoo.com).
  
Resources
------------

[Eugene F. Fama, Kenneth R. French, Dividend yields and expected stock returns, Journal of Financial Economics, Volume 22, Issue 1, 1988, Pages 3-25, ISSN 0304-405X](https://www.sciencedirect.com/science/article/pii/0304405X88900207)

[Goyal, Amit, and Ivo Welch. “Predicting the Equity Premium with Dividend Ratios.” Management Science, vol. 49, no. 5, 2003, pp. 639–54. JSTOR](https://www.jstor.org/stable/4133989)

[Andrew W. Lo & Manish Singh (2023) Deep-learning models for forecasting financial risk premia and their interpretations, Quantitative Finance, 23:6, 917-929](https://www.tandfonline.com/doi/full/10.1080/14697688.2023.2203844)

[Shihao Gu, Bryan Kelly, Dacheng Xiu, Empirical Asset Pricing via Machine Learning, The Review of Financial Studies, Volume 33, Issue 5, May 2020, Pages 2223–2273](https://dachxiu.chicagobooth.edu/download/ML.pdf)

[Andrei, Daniel and Hasler, Michael, Investor Attention and Stock Market Volatility (June 3, 2013). The Review of Financial Studies, 2015](https://www.epfl.ch/labs/cfi/wp-content/uploads/2018/08/WP757_A2.pdf)

[Jussi Nikkinen, Mohammed Omran, Petri Sahlström, Janne Äijö, Global stock market reactions to scheduled U.S. macroeconomic news announcements, Global Finance Journal, Volume 17, Issue 1, 2006, Pages 92-104, ISSN 1044-0283](https://www.sciencedirect.com/science/article/pii/S104402830600024X)

[Feng Ma, Xinjie Lu, Jia Liu, Dengshi Huang, Macroeconomic attention and stock market return predictability, Journal of International Financial Markets, Institutions and Money, Volume 79, 2022, 101603, ISSN 1042-4431](https://doi.org/10.1016/j.intfin.2022.101603)


Contributors
------------

Aaron Arauz Baumender (@Baumender11), Michael Geiser (@mikegeiser), Andreas Stavropoulos (@astavrop0), Zhiyi Tang (@sukotzy)

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
