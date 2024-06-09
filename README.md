# 🟣 Exploring the Relationship between Characteristics and Diagnosis of Alzheimer's Disease
<img width="949" alt="image" src="https://github.com/MuhammadAhsanBughio/Alzheimer-Disease-Diagnosis/assets/139073097/a84c9d12-bec6-437b-bbd8-b4f8dc52c749">

## Project Overview

This report presents a comprehensive analysis of a dataset related to Alzheimer's disease, aiming to investigate the relationship between various characteristics and the diagnosis status (Demented vs. Nondemented). Statistical methods and machine learning algorithms were employed in R to uncover significant patterns and predictors of Alzheimer's disease.

## Tools and Software Used
- **R**: [Statistical programming language](https://www.r-project.org/) used for data analysis and visualization.
- **Libraries**:
  - **ggplot2**: For creating visualizations such as scatter plots and box plots. [Link](https://ggplot2.tidyverse.org/)
  - **dplyr**: For data manipulation tasks like filtering and summarizing. [Link](https://dplyr.tidyverse.org/)
  - **tidyr**: For reshaping data frames. [Link](https://tidyr.tidyverse.org/)
  - **gridExtra**: For arranging multiple plots on a single page. [Link](https://cran.r-project.org/web/packages/gridExtra/index.html)
  - **GGally**: For creating scatterplot matrices. [Link](https://cran.r-project.org/web/packages/GGally/index.html)
  - **corrplot**: For visualizing correlation matrices. [Link](https://cran.r-project.org/web/packages/corrplot/index.html)
  - **factoextra**: For visualizing clustering results. [Link](https://cran.r-project.org/web/packages/factoextra/index.html)
  - **Boruta**: For feature selection using the Boruta algorithm. [Link](https://cran.r-project.org/web/packages/Boruta/index.html)
  - **caret**: For machine learning model training and evaluation. [Link](https://cran.r-project.org/web/packages/caret/index.html)
  - **glmnet**: For fitting logistic regression models with regularization. [Link](https://cran.r-project.org/web/packages/glmnet/index.html)
  - **knitr**: For generating formatted tables. [Link](https://cran.r-project.org/web/packages/knitr/index.html)


## Dataset Overview

The dataset comprises demographic information, cognitive assessments, and brain measurements of individuals, along with their diagnosis status (Demented or Nondemented). Before proceeding with the analysis, we standardized the numerical variables and encoded the target variable, assigning a value of 1 for Demented and 0 for Nondemented.

## Methodology

### Descriptive Statistics

We started the analysis by computing descriptive statistics to summarize the dataset. Visual representations such as boxplots and histograms were created to gain insights into the distribution and potential outliers of each variable.

### Clustering Algorithms

Two clustering algorithms, namely K-means clustering and hierarchical clustering, were employed to explore the underlying structure of the data. K-means clustering revealed two distinct clusters, while hierarchical clustering provided additional insights into the hierarchical relationships between the data points.

<img width="600" alt="Screenshot 2024-06-09 at 12 53 26 am" src="https://github.com/MuhammadAhsanBughio/Alzheimer-Disease-Diagnosis/assets/139073097/195ac6c7-4bf8-4d56-858b-f8f68c5434c3">



### Feature Selection

We applied the Boruta feature selection technique to identify the most significant features associated with Alzheimer's disease. This helped us understand which variables play a crucial role in predicting the diagnosis status.

<img width="600" alt="Screenshot 2024-06-09 at 12 53 09 am" src="https://github.com/MuhammadAhsanBughio/Alzheimer-Disease-Diagnosis/assets/139073097/fc990f0c-70a7-4fb6-9101-2a48069345fd">

### Logistic Regression Results

A logistic regression model was developed to predict the diagnosis status of Alzheimer's disease based on the significant predictor variables identified. The model's performance was evaluated using the following metrics:

- **Accuracy:** The percentage of correctly classified instances out of the total instances. The logistic regression model achieved an accuracy of 92.6%.
- **Precision:** The proportion of true positive predictions out of all positive predictions made by the model. The precision of the model was 90%.
- **Recall:** The proportion of true positive predictions out of all actual positive instances in the dataset. The recall rate was 80%.
- **F1 Score:** The harmonic mean of precision and recall, providing a balance between the two metrics. The F1 score of the model was 88.5%.

These performance metrics indicate that the logistic regression model successfully classified individuals into the Demented or Nondemented categories with reasonable accuracy. The significant predictor variables, including gender, age, MMSE scores, and brain volume, played crucial roles in predicting the diagnosis status of Alzheimer's disease.


## Results

Visualisations such as boxplots and histograms provided insights into the distribution and potential outliers of each variable. Clustering analysis revealed distinct groups, with one representing individuals at higher risk of dementia and the other exhibiting better cognitive function. Feature selection highlighted the importance of Clinical Dementia Rating (CDR) and MMSE in predicting Alzheimer's disease.

## Discussion

The analysis provided valuable insights into the dataset related to Alzheimer's disease. Several variables, including age, gender, education, socioeconomic status, cognitive function (MMSE), and brain volume, were found to be associated with the diagnosis of dementia. The logistic regression model demonstrated the significant impact of gender, age, MMSE scores, and brain volume on predicting the diagnosis of Alzheimer's disease. Early detection and intervention based on these predictors could potentially improve patient outcomes and quality of life.

## Conclusion

In conclusion, this analysis contributes to our understanding of the potential risk factors and characteristics associated with Alzheimer's disease. By leveraging various statistical techniques and machine learning algorithms, we gained insights into the dataset and identified key predictors of Alzheimer's disease.
