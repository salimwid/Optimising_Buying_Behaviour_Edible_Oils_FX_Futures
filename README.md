# Optimising_Buying_Behaviour_Edible_Oils_FX_Futures

#### Models Deployed
Non-linear and Linear Regression, KMeans Clustering and Classification Models (e.g: XGBoost, Light GBM, etc.)

#### Techniques Employed
Data Mining, Exploratory Data Analysis, Feature Engineering, Text Mining, Data Visualization, Machine Learning, Time Series Regression, Clustering

#### Tools Employed
Pickle, pyplot, statsmodels, Pandas, NLTK, Google News API, Twitter API, BarChart getNews API

### Context
The objective of this project is to minimize the risks of Edible Oils and FX Futures trading through three different approaches: <br>
(i) Segmenting customers into different groups based on their past behaviours <br>
(ii) Predicting how certain customer groups will transact <br>
(iii) Predicting buying decision and <br>

### Datasets
There were internal data which was given at the beginning of the projects, this consisted of edible oils pricing, customer orders and FX trading pairs datasets. These datasets were enriched with external data, such as news sentiment, commodity price and contract features collected through different APIs. 

Overall, 227 features were explored consisted of 143 raw features and 84 engineered features.<br>

### Chosen Model & Performance
(i) For customer segmentation, the best performing model was K-means Clustering. Performance was evaluated based on K-Elbow method to find the optimal K and Silhouette score for each edible oil. The results showed that there were clear groups of short, mid and long term buyers - these segmentation held across all oils. <br><br>
(ii) For customer transaction behaviour prediction, bagged/boosted models (e.g: Light GBM and Random Forest) generally outperformed linear classifier. To measure performance, F1 and CV F1 score were adopted. The results indicated that recent periods were the most relevant features and that the models were stable across all time periods (short, medium and long term) on unseen data. <br><br>
(iii) For customer transaction forecasting, the best performing model was the one with shorter time-lag. Performance was evaluated based on adjusted R-squared. Insights on customer past behaviour, sentiment and futures could be derived from significant coefficients based on p-value. <br>
<br>

### Collaborators
Wong Cheng An <br>
Gino Tiu <br>
Widya Salim <br>
Susan Koruthu <br>
Felipe Chapa <br>
Rachel Sng
