install.packages(c("tidyverse", "ggplot2", "broom", "corrplot", 
                   "pROC", "caret"))

library(tidyverse)
library(ggplot2)
library(broom)
library(corrplot)
library(pROC)
library(caret)

##1. Data cleaning and preprocessing
# 1.1 Import data
bc <- read_csv("/Users/quanshijiezuihaodesisi/Library/CloudStorage/OneDrive-kean.edu/MATH 3700/Project/data.csv")

# 1.2 View the basic structure
glimpse(bc)
summary(bc)

# 1.3 Delete unnecessary columns (id, Unnamed column, etc.)
bc <- bc %>%
  select(-id, -matches("Unnamed"))

# 1.4 Preprocessing data ("diagnosis" is transformed into a factor, and "B" is set as the reference)
bc <- bc %>%
  mutate(
    diagnosis = factor(diagnosis, levels = c("B", "M"))
  )

# 1.5 Check for deficiencies
colSums(is.na(bc))
 
#1.6 Delete the final missing invalid columns (ensure that the data is completely cleaned)
bc <- bc %>%
  select(-`...33`)
colSums(is.na(bc))

##2. Exploratory Data Analysis (EDA) and Visualization
#2.1 Distribution of Diagnosis (Benign vs. Malignant)
diag_dist <- bc %>%
  count(diagnosis) %>%
  mutate(prop = n / sum(n))

diag_dist

# 2.2 Distribution of Key Features by Diagnosis (Benign vs. Malignant)
# Class Imbalance
#radius_mean
ggplot(bc, aes(x = diagnosis, y = radius_mean, fill = diagnosis)) +
  geom_violin(trim = FALSE, alpha = 0.4) +
  geom_boxplot(width = 0.2, outlier.size = 0.5) +
  labs(
    title = "Distribution of mean radius by diagnosis",
    x = "Diagnosis",
    y = "Mean radius"
  ) +
  theme_minimal()

#area_mean
ggplot(bc, aes(diagnosis, area_mean, fill = diagnosis)) +
  geom_violin(trim = FALSE, alpha = 0.4) +
  geom_boxplot(width = 0.2, outlier.size = 0.5) +
  labs(title = "Distribution of mean area by diagnosis",
       x = "Diagnosis", y = "Mean area") +
  theme_minimal()

# Irregular Shape
#concavity_mean
ggplot(bc, aes(diagnosis, concavity_mean, fill = diagnosis)) +
  geom_violin(trim = FALSE, alpha = 0.4) +
  geom_boxplot(width = 0.2, outlier.size = 0.5) +
  labs(title = "Concavity (mean) by diagnosis",
       x = "Diagnosis", y = "Mean concavity") +
  theme_minimal()

#concave points_mean
ggplot(bc, aes(x = diagnosis, y = `concave points_mean`, fill = diagnosis)) +
  geom_violin(trim = FALSE, alpha = 0.4) +
  geom_boxplot(width = 0.2, outlier.size = 0.5) +
  labs(
    title = "Concave points (mean) by diagnosis",
    x = "Diagnosis",
    y = "Mean concave points"
  ) +
  theme_minimal()

# Texture Roughness
#texture_mean
ggplot(bc, aes(x = diagnosis, y = texture_mean, fill = diagnosis)) +
  geom_violin(trim = FALSE, alpha = 0.4) +
  geom_boxplot(width = 0.2, outlier.size = 0.5) +
  labs(
    title = "Distribution of mean texture by diagnosis",
    x = "Diagnosis",
    y = "Mean texture"
  ) +
  theme_minimal()

# 2.3 Correlation Matrix of Variables (Heatmap) for numeric variables only
num_vars <- bc %>%
  select(where(is.numeric))

corr_mat <- cor(num_vars)

corrplot(corr_mat,
         method = "color",
         type = "upper",
         tl.cex = 0.6,
         number.cex = 0.6)

## 3. Comparison of Five Key Features with Significant Differences (Benign vs. Malignant), Validating Section 2.2
t.test(radius_mean ~ diagnosis, data = bc)
t.test(area_mean ~ diagnosis, data = bc)
t.test(concavity_mean ~ diagnosis, data = bc)
t.test(`concave points_mean` ~ diagnosis, data = bc)
t.test(texture_mean ~ diagnosis, data = bc)

## 4. Data Preparation and Train/Test Split (Mean Features as the Primary Model)
# 4.1 Construction of the Modeling Dataset: Diagnosis and Mean Features
model_data <- bc %>%
  select(
    diagnosis,
    ends_with("_mean")
  )

# 4.2 Train/Test Split (70% Training set, 30% Testing set)
set.seed(123) 
train_index <- createDataPartition(model_data$diagnosis, p = 0.7, list = FALSE)

train <- model_data[train_index, ]
test  <- model_data[-train_index, ]

# Check Diagnosis Proportions in Training and Test Sets
prop.table(table(train$diagnosis))
prop.table(table(test$diagnosis))

## 5. Multivariate Logistic Regression (Mean Features as the Primary Model)
# 5.1 Multivariate Logistic Model
fit_logit <- glm(diagnosis ~ ., data = train, family = binomial)

summary(fit_logit)

# 5.2 Convert Coefficients to Odds Ratios (OR) and 95% Confidence Intervals
logit_results <- tidy(fit_logit, exponentiate = TRUE, conf.int = TRUE)

logit_results

## 5.3 Model Refinement Due to Multicollinearity: Refined Logistic Regression Model
bc <- bc %>%
mutate(concave_pts_100 = `concave points_mean` * 100)

train <- train %>%
  mutate(concave_pts_100 = `concave points_mean` * 100)

test <- test %>%
  mutate(concave_pts_100 = `concave points_mean` * 100)

fit_logit_refined <- glm(
  diagnosis ~ radius_mean + texture_mean + concave_pts_100,
  data = train,
  family = binomial
)

## 6. Test Set Prediction and Confusion Matrix
# 6.1 Predict Malignancy Probability on the Test Set (Using the Refined Logistic Regression Model)

test$pred_prob <- predict(fit_logit_refined,
                          newdata = test,
                          type = "response")

# 6.2 Convert Predicted Probabilities to Class Labels Using a 0.5 Threshold
test$pred_class <- ifelse(test$pred_prob >= 0.5, "M", "B") |>
  factor(levels = c("B", "M"))

# 6.3 Confusion Matrix and Classification Metrics
cm_refined <- confusionMatrix(test$pred_class,
                              test$diagnosis,
                              positive = "M")
cm_refined

logit_refined <- tidy(fit_logit_refined, exponentiate = TRUE, conf.int = TRUE)
logit_refined

## 7. ROC Curves and AUC Evaluation of the Classification Models
# 7.1 Construct ROC Objects
roc_obj <- roc(
  test$diagnosis,
  test$pred_prob,
  levels = c("B", "M"),   
  direction = "<"        
)

# 7.2 Calculate AUC Values
auc_val <- auc(roc_obj)
auc_val

# 7.3 Plot ROC Curves and Display AUC
plot(roc_obj,
     print.auc = TRUE,
     legacy.axes = TRUE,
     main = "ROC Curve for Optimized Logistic Model")

## 8. LASSO Logistic Regression for Robustness Analysis (Variable Selection)
library(dplyr)
library(caret)
library(glmnet)

# 8.1 Construct a New Dataset Using All Numeric Features and Diagnosis
model_data_all <- bc %>%
  select(diagnosis, where(is.numeric))   

# 8.2 Re-split the Data into Training and Test Sets (70% / 30%)
set.seed(123)
train_index2 <- createDataPartition(model_data_all$diagnosis,
                                    p = 0.7,
                                    list = FALSE)

train2 <- model_data_all[train_index2, ]
test2  <- model_data_all[-train_index2, ]

# 8.3 Prepare Matrix-form Predictors (X) and Response Variable (y)
x_train <- model.matrix(diagnosis ~ ., data = train2)[, -1]
y_train <- ifelse(train2$diagnosis == "M", 1, 0)  # M=1, B=0

# 8.4 Select the Optimal Lambda via 10-fold Cross-Validation (LASSO: alpha = 1)
set.seed(123)
cv_lasso <- cv.glmnet(
  x_train,
  y_train,
  family = "binomial",
  alpha  = 1,     # 1 = LASSO (L1 penalty)
  nfolds = 10
)


cv_lasso$lambda.min      
cv_lasso$lambda.1se      

# 8.5 Extract Non-zero Coefficients at lambda.min (Variables Selected by LASSO)
coef_min <- coef(cv_lasso, s = "lambda.min")
coef_min

nz_min_index <- which(coef_min != 0)

nz_min_vars  <- rownames(coef_min)[nz_min_index]
nz_min_vars

coef_1se <- coef(cv_lasso, s = "lambda.1se")
nz_1se_index <- which(coef_1se != 0)
nz_1se_vars  <- rownames(coef_1se)[nz_1se_index]
nz_1se_vars

## 9. Random Forest Model (Comparable to the Refined Logistic Regression Model)


# 9.1 Reconstruct the Modeling Dataset Using Only Predictor Variables (Excluding Predicted Probability Columns)
rf_data <- bc %>%
  select(diagnosis,
         radius_mean, texture_mean,
         concave_pts_100)

set.seed(123)
rf_index <- createDataPartition(rf_data$diagnosis, p = 0.7, list = FALSE)
rf_train <- rf_data[rf_index, ]
rf_test  <- rf_data[-rf_index, ]

prop.table(table(rf_train$diagnosis))
prop.table(table(rf_test$diagnosis))

# 9.2 Define Training Control: 5-fold Cross-Validation with ROC-based Model Selection
ctrl_rf <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,          
  summaryFunction = twoClassSummary
)
library(randomForest)
# 9.3 Train the Random Forest Model
set.seed(123)
rf_model <- train(
  diagnosis ~ .,
  data = rf_train,
  method = "rf",
  metric = "ROC",
  trControl = ctrl_rf
)

rf_model   

# 9.4 Make Predictions on the Test Set
rf_prob <- predict(rf_model, newdata = rf_test, type = "prob")[, "M"]
rf_class <- predict(rf_model, newdata = rf_test)

# 9.5 Confusion Matrix (Malignant as the Positive Class)
cm_rf <- confusionMatrix(rf_class,
                         rf_test$diagnosis,
                         positive = "M")
cm_rf

# 9.6 ROC Curves and AUC
roc_rf <- roc(
  rf_test$diagnosis,
  rf_prob,
  levels = c("B", "M"),
  direction = "<"
)

auc_rf <- auc(roc_rf)
auc_rf

plot(roc_rf,
     print.auc = TRUE,
     legacy.axes = TRUE,
     main = "ROC Curve – Random Forest")
