# Required Packages: caTools, rpart, rpart.plot, randomForest, gbm, xgboost.
pop_data <- read.csv("/home/shreyashkharat/Datasets/insurance.csv", header = TRUE)
pop_data_fixed <- read.csv("/home/shreyashkharat/Datasets/insurance.csv", header = TRUE)
summary(pop_data)

# Train Test Split
set.seed(0)
require("caTools")
split = sample.split(pop_data, SplitRatio = 0.8)
train_set <- subset(pop_data, split == TRUE)
test_set <- subset(pop_data, split == FALSE)

# Regression Tree Building
require("rpart")
require("rpart.plot")
# Tree
reg_tree <- rpart(charges~., data = train_set, control = rpart.control(maxdepth = 5))
# Ploting the regression tree
rpart.plot(reg_tree, box.palette = "RdYlGn", digits = -3)
# Prediction
reg_predict <- predict(reg_tree, test_set)
# R^2 calculation
tss <- sum((test_set$charges - mean(test_set$charges))^2)
rss_reg <- sum((reg_predict - test_set$charges)^2)
rsq_reg <- 1 - rss_reg/tss
# The Simple Regression tree gives an accuracy of 0.8512.
# Let's try to increase the accuracy by pruning and ensemble techniques.

# Pruning the full tree
full_tree <- rpart(charges~., data = train_set, control = rpart.control(cp = 0))
rpart.plot(full_tree, box.palette = "RdYlGn", digits = -3)
# As the full_tree has to many branches let's see if pruning increases the accuracy.
# To do so, we need to find the cp corresponding to lowest rel. error
min_cp <- min_cp <- reg_tree$cptable[which.min(reg_tree$cptable[, "xerror"]), "CP"]
# Building the pruned_tree
pruned_tree <- prune(reg_tree, cp = min_cp)
rpart.plot(pruned_tree, box.palette = "RdYlGn", digits = -3)
# Let's assess the accuracy of full_tree and pruned_tree
full_predict <- predict(full_tree, test_set)
pruned_predict <- predict(pruned_tree, test_set)
rss_full <- sum((full_predict - test_set$charges)^2)
rss_pruned <- sum ((pruned_predict - test_set$charges)^2)
rsq_full <- 1 - rss_full/tss
rsq_pruned <- 1 - rss_pruned/tss
# The Pruned Regression tree gives an accuracy of 0.8512, which is same as Simple Regression Tree.
# The Full Regression tree gives an accuracy of 0.8691, which is a greater than that of Simple or Pruned.

# Ensemble Technique: 1. BAGGING
require("randomForest")
set.seed(0)
bagging = randomForest(formula = charges~., data = train_set, mtry = 6) # mtry is the no.of independent variables used for buliding model
bagging_predict <- predict(bagging, test_set)
# R2 calculation for bagging
rss_bagging = sum((bagging_predict - test_set$charges)^2)
rsq_bagging = 1 - rss_bagging/tss
# Using bagging model gives an accuracy of 0.8662, which is nearly equal to that of Full Regression Tree.

# Ensemble Technique: 2. RANDOM FOREST
require("randomForest")
set.seed(0)
random_forest = randomForest(formula = charges~., data = train_set, ntree = 1000) # ntree is the max no.of trees in model
random_forest_predict <- predict(random_forest, test_set)
# R2 calculation
rss_random <- sum((random_forest_predict - test_set$charges)^2)
rsq_random <- 1 - rss_random/tss
# Using Random Forest Model gives an accuracy of 0.8676, which is also nearly equal to that of Full Regression Tree.

# Ensemble Technique: 3. GBM or Gradient Boosting Model
require("gbm")
set.seed(0)
# Getting data set ready for gbm, ie making categorical variable in factor form
train_set$sex <- as.factor(train_set$sex)
train_set$smoker <- as.factor(train_set$smoker)
train_set$region <- as.factor(train_set$region)
test_set$sex <- as.factor(test_set$sex)
test_set$smoker <- as.factor(test_set$smoker)
test_set$region <- as.factor(test_set$region)
# Model building
model_gbm <- gbm(charges~., data = train_set, distribution = "gaussian", n.trees = 5000, interaction.depth = 6, shrinkage = 0.001, verbose = FALSE)
# For Regression models we use Gaussian distribution and for classification models we use Bernoulli distribution.
gbm_predict <- predict(model_gbm, test_set)
# R2 calculation
rss_gbm <- sum((gbm_predict - test_set$charges)^2)
rsq_gbm <- 1 - rss_gbm/tss
# Using Gradient Boosting Model gives an accuracy of 0.8824, which is largest so far.

# Ensemble Technique: 4. XGBoost or eXtreme Gradient Boosting Model
require("xgboost")
# For XGBoost the independent variables need to be in a DMatrix
train_X <- model.matrix(charges~. -1, data = train_set)
train_Y <- train_set$charges
test_X <- model.matrix(charges~. -1, data = test_set)
test_Y <- test_set$charges
DMatrix_train <- xgb.DMatrix(data = train_X, label = train_Y)
DMatrix_test <- xgb.DMatrix(data = test_X, label = test_Y)
# Model Building
model_xgboost <- xgboost(data = DMatrix_train, nrounds = 4000, objective = "reg:linear", eta = 0.001)
# Prediction
xgboost_predict <- predict(model_xgboost, DMatrix_test)
# R2 calculation
rss_xgb <- sum((xgboost_predict - test_set$charges)^2)
rsq_xgb <- 1 - rss_xgb/tss
# Using eXtreme Gradient Boosting Model gives an accuracy of 0.8655.
# Hence, GBM gave the highest accuracy of 0.8824.
