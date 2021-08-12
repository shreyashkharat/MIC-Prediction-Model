pop_data <- read.csv("/home/shreyashkharat/Datasets/insurance.csv", header = TRUE)
pop_data_fixed <- read.csv("/home/shreyashkharat/Datasets/insurance.csv", header = TRUE)
summary(pop_data)
pairs(~charges + age + bmi + children, data = pop_data_fixed)

# We need to create dummy variable for categorical variables
require("dummies")
pop_data <- dummy.data.frame(pop_data)
pop_data <- pop_data[, -2]
pop_data <- pop_data[, -5]
pop_data <- pop_data[, -9]

# Correlation
cor(pop_data)
round(cor(pop_data), 2)

#Model buliding
model1 <- lm(charges~., data = pop_data)
summary(model1)
# Clearly p-value of regionsoutheast and sexmale is > 0.15.
model1.1 <- lm(charges~age + bmi + children + smokeryes + regionnortheast + regionnorthwest, data = pop_data)
summary(model1.1)
pop_data <- pop_data[, -8]
pop_data <- pop_data[, -2]

#Test train split
set_a <- sort(sample(nrow(pop_data), nrow(pop_data)*.8))
training_set <- pop_data[set_a,]
test_set <- pop_data[-set_a,]

model_trained <- lm(charges~., data = training_set)
summary(model_trained)

model_test <- lm(charges~., data = test_set)
summary(model_test)

# Let's try Subset selection
require("leaps")
model_best_subset = regsubsets(charges~., data = pop_data)
summary(model_best_subset)$adjr2
which.max(summary(model_best_subset)$adjr2)
coef(model_best_subset, 6)
View(model_best_subset, 6)
# In model_best_subset we get adjusted R2 as 0.7497, which is greater than that of simple linear model.

# Let's try Ridge Regression
require("glmnet")
x = model.matrix(charges~., data = pop_data)[, -7]
y = pop_data$charges
grid = 10^seq(10, -2, length = 100)
model_ridge = glmnet(x, y, alpha = 0, lambda = grid)
summary(model_ridge)
# For best lambda
cv_fit = cv.glmnet(x, y, alpha = 0, lambda = grid)
plot(cv_fit)
opt_lambda <- cv_fit$lambda.min
# R2 calculation for opt_lambda
tss = sum((y - mean(y))^2)
y_predict = predict(model_ridge, s = opt_lambda, newx = x)
rss = sum((y_predict - y)^2)
rsq = 1 - rss/tss
# R2 for Ridge regression model is 0.7504
# Let's find R2 on test data when model is trained on training set.
x_training = model.matrix(charges~., data = training_set)[, -7]
y_training = training_set$charges
grid_train = 10^seq(10, -2, length = 100)
model_ridge_trained = glmnet(x_training, y_training, alpha = 0, lambda = grid_train)
# For best lambda
cv_fit_trained = cv.glmnet(x_training, y_training, alpha = 0, lambda = grid_train)
plot(cv_fit_trained)
opt_lambda_trained <- cv_fit_trained$lambda.min
# R2 calculation for opt_lambda_trained
x_test = model.matrix(charges~., data = test_set)[, -7]
y_test = test_set$charges
tss_test = sum((y_test - mean(y_test))^2)
y_predict_test = predict(model_ridge_trained, s = opt_lambda_trained, newx = x_test)
rss_test = sum((y_predict_test - y_test)^2)
rsq_test = 1 - rss_test/tss_test
# By Ridge regression on test data we get 0.7883 as R2.

# Let's try Lasso
x = model.matrix(charges~., data = pop_data)[, -7]
y = pop_data$charges
grid = 10^seq(10, -2, length = 100)
model_lasso = glmnet(x, y, alpha = 1, lambda = grid)
summary(model_lasso)
# For best lambda
cv_fit_l = cv.glmnet(x, y, alpha = 1, lambda = grid)
plot(cv_fit_l)
opt_lambda_l <- cv_fit_l$lambda.min
# R2 calculation for opt_lambda
tss_l = sum((y - mean(y))^2)
y_predict_l = predict(model_lasso, s = opt_lambda_l, newx = x)
rss_l = sum((y_predict_l - y)^2)
rsq_l = 1 - rss_l/tss_l
# R2 for lasso regression model is 0.7504.
# Let's find R2 on test data when model is trained on training set.
x_training = model.matrix(charges~., data = training_set)[, -7]
y_training = training_set$charges
grid_train = 10^seq(10, -2, length = 100)
model_lasso_trained = glmnet(x_training, y_training, alpha = 1, lambda = grid_train)
# For best lambda
cv_fit_trained_l = cv.glmnet(x_training, y_training, alpha = 1, lambda = grid_train)
plot(cv_fit_trained_l)
opt_lambda_trained_l <- cv_fit_trained_l$lambda.min
# R2 calculation for opt_lambda_trained
x_test_l = model.matrix(charges~., data = test_set)[, -7]
y_test_l = test_set$charges
tss_test_l = sum((y_test_l - mean(y_test_l))^2)
y_predict_test_l = predict(model_lasso_trained, s = opt_lambda_trained_l, newx = x_test_l)
rss_test_l = sum((y_predict_test_l - y_test_l)^2)
rsq_test_l = 1 - rss_test_l/tss_test_l
# By lasso we get R2 as 0.7882.

#Following are all R2 values by each method on test data.
# Simple Linear Model - 0.8015
# Best Subset selection - 0.7497 (Obviously low)
# Ridge Regression - 0.7883
# Lasso - 0.7882
# Above results were achieved for the attached Test Train split
# Results may vary due to different testing data, but R2 > 0.74 for any test-train split in pop_data.