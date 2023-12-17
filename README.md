MAI_for_stock_return_predictability
==============================

We conduct forecasts for equity risk premia utilizing regression and neural network models trained on sets of macroeconomic factors (MEF) and macroeconomic attention indices (MAI). The macroeconomic factors set comprises the 14 features recommended in previous works by Goyal and Welch (2008). The set of macro attention indices includes the eight features constructed by Fisher et al. (2022). Equity risk premia represent the annualized excess return of the one-month S&P500 index over the prevailing risk-free rate, approximated by the yield on short-term Treasury Bills. 

Our analysis is based on datasets published in other Github repositories and Yahoo Finance (see below), with focus on the period between 1985 and 2018, given the availability of the combined data. We provide an [interactive app](https://baumender11.shinyapps.io/Alpha/) for the user to undertake robustness checks and visualizations of the data.

The outcomes of our prediction models deviate from previous research, suggesting the need for further investigation. More specifically, the results of our predictive modeling efforts showed a noticeable divergence from established research, revealing significant prediction errors. This indicates that the models were unable to adequately capture the complexity of the financial data. Potential reasons for the poor performance of the models include:

• Inadequate feature selection (i.e., features without a strong relationship with the target variable).

• Insufficient data preprocessing, such as mishandling missing values or neglecting outliers.

• Inappropriate hyperparameter tuning or insufficient model complexity.

• Unsuitable assumptions about the stationarity properties of the data.

Aligned with the pitfalls our analysis indicated, further steps for researching the predictive power of MEF and MAI data include:

• Experimenting with combinations of MAI and MEF features as inputs to the predictive models.

• Exploring advanced preprocessing techniques, outlier detection methods, and handling missing data to improve model inputs.

• Conducting a more extensive hyperparameter search and exploring more sophisticated modeling approaches.

• Considering shorter time spans or applying advanced time series analysis techniques to address non-stationarity.

Project Organization
------------
```
    ├── LICENSE
    ├── README.md                <- The top-level README for developers using this project
    ├── data
    │   ├── interim              <- Intermediate data (filled missing values, droped useless features)
    │   ├── processed            <- The final, canonical data sets for modeling (after transformations)
    │   └── raw                  <- The original, immutable data dump
    │
    ├── notebooks                <- Jupyter notebooks with codes to transform data and implement models
    │   └── Data_raw_to_interim.ipynb
    │   └── Data_interim_to_processed.ipynb
    │   └── RegressionModel.ipynb 
    │   └── NeuralNetworkModel.ipynb 
    │
    ├── reports                  <- Generated analysis as PDF and LaTeX report and beamer presentation
    │   └── figures              <- Generated graphics and figures to be used in reporting as .png
    │   └── tables               <- Generated tables to be used in reporting as .tex
    │
    ├── requirements.txt         <- The requirements file for reproducing the analysis environment
    │
    ├── Dockerfile               <- Defines the Docker container configuration to run the analysis
    │
    ├── src                      <- Source code for use in this project
    │   ├── __init__.py          <- Makes src a Python module
    │   │
    │   ├── data                 <- Scripts to turn raw data into features for modeling
    │   │   └── make_dataset.py
    │   │
    │   └── models               <- Scripts to train and evaluate models             
    │       ├── regression_model.py
    │       └── NN_model.py
    │
    └── shiny                    <- Code and data to create R shiny app for visualizations
        ├── shiny_data
        └── app.R
 ``` 
  
Description of Steps 
------------

- Prepare folder structure, using the cookiecutter Data Science template

- Download raw data (see below)

- Preprocess and save data as features ready to be used for modelling

- Create an interactive app in Shiny For R to perform robustness checks - data frequency, data features, date range, visualization - (app can be directly accessed [here](https://baumender11.shinyapps.io/Alpha/))

- Build models (ridge regression, neural network), train and evaluate them 

- Plot graphs to analyze results

- Ensure reproducibility by adding Dockerfile

- Connect Overleaf to Github (report can be accessed [here](https://www.overleaf.com/read/yqkhbqjwvtbs#b7cd7c), beamer presentation can be accessed [here](https://www.overleaf.com/read/hvngdthxhprd#c75379))

- Analyse and interpret findings

- Compare results with current research

- Propose steps for further research

Data 
------------

### Sources

Macroeconomic Factors (MEF) data is obtained from the [powder197/Goyal-and-Welch-2008-](https://github.com/powder197/Goyal-and-Welch-2008-/tree/master) Github repository.

Macro Attention Indices (MAI) data is obtained from [charlesmartineau/mai_rfs](https://github.com/charlesmartineau/mai_rfs) Github repository.

S&P500 data is obtained from [Yahoo Finance](https://finance.yahoo.com).

### Description

In the file `data/processed` you can find the processed/final datasets we use for our analysis (for detailed description of how we construct them from the raw data look at the report and notebooks `Data_raw_to_interim.ipynb` and `Data_interim_to_processed.ipynb`). All the datasets include data for the time period 1985-2018 and can be split into two groups according to their frequency:

- __Monthly__ (408 data points): `mai_monthly_data_processed.csv`, `mef_monthly_data_processed.csv`, `mkt_monthly_data_processed.csv`. These datasets include monthly data for indices, factors and stock returns (see below)

- __Daily__ (8523 data ponts): `mai_daily_data_processed.csv`, `mef_daily_data_processed.csv`, `mkt_dail_data_processed.csv`. These datasets include daily data for the same indices, factors and stock returns (see below)

The __MAI (Macroeconomic Attention Indices)__ datasets (monthly and daily) consist of the following columns:

- `date`: Date
- `credit_rating`: Credit Rating
- `gdp`: Gross Domestic Product
- `house_mkt`: House Market
- `inflation`: Inflation
- `monetary`: Monetary
- `oil`: Oil
- `unemp`: Unemployment Rate
- `usd`: US Dollar

The __MEF (Macroeconomic Factors)__ datasets (monthly and daily) consist of the following columns:

- `date`: Date
- `dp`: Log Dividend Price
- `dy`: Log Dividend-Yield
- `ep`: Log Earnings-Price
- `de`: Log Dividend-Payout
- `rvol`: Equity Premium Vol
- `bm`: Book-to-Market
- `ntis`: Net Equity Expansion
- `tbl`: Treasury Bill Rate
- `lty`: Long-Term Yield
- `ltr`: Long-Term Return
- `tms`: Term Spread
- `dfy`: Default Yield Spread
- `dfr`: Default Return Spread
- `infl`: Inflation

The __MKT (annualized excess returns)__ datasets (monthly and daily) consist of the following columns:

- `date`: Date
- `GSPCprem`: Equity Risk Premia

A detailed description of all the features can be found in the report.

Installation
------------

To install the required packages and replicate this project, follow these steps:

1. Clone the repository to your local machine:
```
  git clone https://github.com/astavrop0/MAI_for_stock_return_predictability.git
```
2. Navigate to the cloned repository directory:
```
  cd MAI_for_stock_return_predictability
```
3. Install the required packages using pip:
```
  pip install -r requirements.txt
```
  This will install all the Python packages listed in `requirements.txt`.

Running with Docker
------------

This project can be easily set up and run using Docker. Follow the steps below to build and run the project inside a Docker container.

### Prerequisites
1. Ensure you have [Docker installed](https://docs.docker.com/get-docker/) on your machine.

2. Navigate to the cloned repository's directory.

3. Build the Docker image. Replace `your-image-name` with a name of your choice for the Docker image:
```
docker build -t your-image-name .
```

### Running the Docker Container

Once the image is built, you can run it as a container. To do it, execute the following command:
```
docker run your-image-name
```

Resources
------------

[[1]](https://www.sciencedirect.com/science/article/pii/0304405X88900207) Eugene F. Fama, Kenneth R. French, Dividend yields and expected stock returns, Journal of Financial Economics, Volume 22, Issue 1, 1988, Pages 3-25, ISSN 0304-405X

[[2]](https://www.jstor.org/stable/4133989) Goyal, Amit, and Ivo Welch. “Predicting the Equity Premium with Dividend Ratios.” Management Science, vol. 49, no. 5, 2003, pp. 639–54. JSTOR

[[3]](https://www.tandfonline.com/doi/full/10.1080/14697688.2023.2203844) Andrew W. Lo & Manish Singh (2023) Deep-learning models for forecasting financial risk premia and their interpretations, Quantitative Finance, 23:6, 917-929

[[4]](https://dachxiu.chicagobooth.edu/download/ML.pdf) Shihao Gu, Bryan Kelly, Dacheng Xiu, Empirical Asset Pricing via Machine Learning, The Review of Financial Studies, Volume 33, Issue 5, May 2020, Pages 2223–2273

[[5]](https://www.epfl.ch/labs/cfi/wp-content/uploads/2018/08/WP757_A2.pdf) Andrei, Daniel and Hasler, Michael, Investor Attention and Stock Market Volatility (June 3, 2013). The Review of Financial Studies, 2015

[[6]](https://www.sciencedirect.com/science/article/pii/S104402830600024X) Jussi Nikkinen, Mohammed Omran, Petri Sahlström, Janne Äijö, Global stock market reactions to scheduled U.S. macroeconomic news announcements, Global Finance Journal, Volume 17, Issue 1, 2006, Pages 92-104, ISSN 1044-0283

[[7]](https://doi.org/10.1016/j.intfin.2022.101603) Feng Ma, Xinjie Lu, Jia Liu, Dengshi Huang, Macroeconomic attention and stock market return predictability, Journal of International Financial Markets, Institutions and Money, Volume 79, 2022, 101603, ISSN 1042-4431


Contributors
------------

Aaron Arauz Baumender (@Baumender11), Michael Geiser (@mikegeiser), Andreas Stavropoulos (@astavrop0), Zhiyi Tang (@sukotzy)

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
