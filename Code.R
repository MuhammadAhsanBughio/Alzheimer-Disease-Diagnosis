MyData<-read.csv("project data.csv")

#Importing Libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)
library(GGally)
library(corrplot)
library(factoextra)
library(Boruta)
library(caret)
library(glmnet)
library(knitr)
# Converting M.F into numerical variable
# Counting the number of NA values in the "M.F" column
na_count <- sum(is.na(MyData$M.F))
# No NA vales so converting M.F into Numerical Variable
MyData$M.F <- ifelse(MyData$M.F == "M", 1, 0)
# Removing rows with Group = "Converted"
MyData <- subset(MyData, Group != "Converted")
# Removing rows with missing values
MyData <- na.omit(MyData)
# looking at the data structure
str(MyData)

## DESCRIPTIVE STATISTICS 
# Summary table
summary_table <- summary(MyData[, c("Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF")])
formatted_table<- kable(summary_table,align = "c")
formatted_table
# Count of Group
# Creating a table for the counts of Demented and Nondemented people
group_counts <- table(Group = MyData$Group)
print(group_counts)
# Creating a table for the counts of males and females
gender_counts <- table(Gender = ifelse(MyData$M.F == 0, "Female", "Male"))
print(gender_counts)
# Creating a table of counts for females and males in the Demented and Non-Demented groups
counts_table_group <- table(Gender = ifelse(MyData$M.F == 0, "Female", "Male"),
                            Group = MyData$Group)
# Calculating the percentages
percentage_table_group <- prop.table(counts_table_group, margin = 2) * 100
# Combining the counts and percentages into a single table
result_table <- cbind(counts_table_group, percentage_table_group)
# Printing the result table
print(result_table)

#BOX PLOTS
# Generate boxplots for each numerical variable by Group
numerical_vars <- c("Age", "EDUC", "SES","MMSE", "CDR", "eTIV", "nWBV", "ASF")
par(mfrow = c(2, 4))  # Set the layout for subplots
for (var in numerical_vars) {
  boxplot(MyData[[var]] ~ MyData$Group,
          main = paste("Boxplot of", var, "by Group"),
          xlab = "Group",
          ylab = var,
          col = c("lightblue", "lightgreen"),
          data = MyData)
}
# Reset the plotting layout
par(mfrow = c(1, 1))

## HISTOGRAM
# Generating histograms for each continiuos variable
par(mfrow = c(1, 6))  # Set the layout for subplots
numeric_vars <- c("Age", "SES","MMSE", "eTIV", "nWBV", "ASF")
for (var in numeric_vars) {
  hist(MyData[[var]],
       main = paste("Histogram of", var),
       xlab = var,
       col = "lightblue",
       breaks = "FD",
       data = MyData)
}
# Resetting the plotting layout
par(mfrow = c(1, 1))

## CORRELATION ANALYSIS
MyData_Correlation<-MyData[c("Group","Age","EDUC","SES","MMSE","CDR","eTIV","nWBV","ASF")]
ggpairs(MyData_Correlation)
# ASF and ETIV have a strong negative correlation, eliminating ASF column to reduce noise in the data
MyData <- MyData[, -which(names(MyData) == "ASF")]

## IMPLEMENTING CLUSTERING ALGORITHMS
# Pre Processing 
# Normalizing the numeric variables
numeric_vars <- c("Age", "EDUC", "SES","MMSE", "CDR", "eTIV", "nWBV")
MyData_normalized <- as.data.frame(scale(MyData[, numeric_vars]))
# View the normalized dataset
head(MyData_normalized)
# Encoding Categorical Attribute "Group"
MyData$Group <- ifelse(MyData$Group == "Demented", 1, 0)
# Combining the encoded Group column with normalized data and M.f
Combined_data <- cbind(Group=MyData$Group,M.F=MyData$M.F,MyData_normalized)
# View the combined data
head(Combined_data)

## K-MEANS CLUSTERING

# Determining the optimal number of clusters 
fviz_nbclust(Combined_data,kmeans,method="wss")+ geom_vline(xintercept=5,linetype=2)
fviz_nbclust(Combined_data,kmeans,method="silhouette")+ geom_vline(xintercept=5,linetype=2)
# choosing k=2 since the maximum silhouette score is at k=2
set.seed(123) 
kmeans2<-kmeans(Combined_data,centers=2,nstart=20) 
kmeans2 
str(kmeans2)
# To visualise the results the fviz_cluster function can be used: 
fviz_cluster(kmeans2,data=Combined_data)
# Scatter plot with Age and MMSE, colored by cluster
clustering_vector <- kmeans2$cluster
# Calculate the proportion of each diagnosis category within each cluster
prop_table <- table(clustering_vector, Combined_data$Group)
prop_table <- prop.table(prop_table, margin = 1)
prop_table

## Hierarchical Clustering

# Calculating the Distance Matrix
D<-dist(Combined_data,method="euclidean")
# Applying hierarchical clustering for complete linkage method
fit.complete<-hclust(D,method="complete") 
# for complete
plot(fit.complete)
# printing the dendrogram 
groups.fit.complete<-cutree(fit.complete,k=4)
# cutting tree into k=4 clusters 
# drawing dendrogram with red borders around the 4 clusters 
rect.hclust(fit.complete,k=4,border="red")
# Checking how many observations are in each cluster 
table(groups.fit.complete)
# Calculating the mean of each variable by clusters
# this will give the means for the last clustering call which is:fit.complete
aggregate(Combined_data,by=list(cluster=groups.fit.complete),mean)
# Calculate the frequency of each Group category within each cluster
diagnosis_freq <- table(groups.fit.complete, MyData$Group)
# Calculate the proportion of each diagnosis category within each cluster
diagnosis_prop <- prop.table(diagnosis_freq, margin = 1)
# Print the frequency and proportion tables
print(diagnosis_freq)
print(diagnosis_prop)
par(mfrow=c(1,1)) 
# LOGISTIC REGRESSION MODEL
## Putting ASF back to the data
MyData2<-read.csv("project data.csv")
# No NA vales so converting M.F into Numerical Values
MyData2$M.F <- ifelse(MyData2$M.F == "M", 1, 0)
# Removing rows with Group = "Converted"
MyData2 <- subset(MyData2, Group != "Converted")
# Removing rows with missing values
MyData2 <- na.omit(MyData2)
MyData_All<-cbind(MyData,ASF=MyData2$ASF)
str(MyData_All)
# Checking the significance of all the variables for the Logistic Model
glm.model<- glm(Group ~ M.F + Age + MMSE + nWBV + ASF + eTIV + SES + EDUC, data = MyData_All, family = binomial,control = list(maxit = 1000))
summary(glm.model)
# M.F, Age, MMSE, nWBV, EDUC are significant for the prediction model
# Setting a seed for reproducibility
set.seed(123)
# Determining the number of rows in the dataset
n <- nrow(MyData_All)
# Set the proportion of data for training
train_pro <- 0.7
# Calculating the number of rows for training
training_size <- round(train_pro * n)
# Randomly selecting row indices for training
training_indices <- sample(1:n, training_size, replace = FALSE)
# Spliting the data into training and testing subsets
training_data <- MyData_All[training_size, ]
testing_data <- MyData_All[-training_indices, ]
# Fitting the logistic regression model using the training data
glm.fit<- glm(Group ~ M.F + Age + MMSE + nWBV, data = MyData_All, family = binomial,control = list(maxit = 1000))
# View the summary of the model
summary(glm.fit)
# Make predictions on the testing data
predictions <- predict(glm.fit, newdata = testing_data, type = "response")
# Converting predicted probabilities to binary predictions
bin_predictions <- ifelse(predictions > 0.5, 1, 0)
# Evaluating the model's performance on the testing data
confusion_mat <- table(testing_data$Group, binary_predictions)
Acc <- sum(diag(confusion_mat)) / sum(confusion_mat)
Precision <- confusion_mat[2, 2] / sum(confusion_mat[, 2])
Recall <- confusion_mat[2, 2] / sum(confusion_mat[2, ])
F1_Score <- 2 * precision * Recall / (precision + recall)
# Printing the evaluation metrics
cat("Accuracy:", Acc, "\n")
cat("Recall:", Recall, "\n")
cat("F1 Score:", F1_Score, "\n")
cat("Precision:", Precision, "\n")

# FEATURE SELECTION
boruta1<-Boruta(Group~.,data=MyData_All,doTrace=1)
decision<-boruta1$finalDecision 
signif<-decision[boruta1$finalDecision%in%c("Confirmed")] 
print(signif) 
plot(boruta1,xlab="",main="Variable Importance",las=2,cex.axis=0.7)


