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

# Retraining the model

# Considering quadratic relation in age and charges.
pop_data$age2 <- pop_data$age^2
# Charges are affected when the BMI is high and the individual is a smoker.
# Transform bmi into categorical variable.
pop_data$bmi30 <- ifelse(pop_data$bmi > 30, 1, 0)

# Updating formula in linear model
model_updated <- lm(charges ~ age + bmi + children + bmi30 + bmi30*smokeryes + regionnorthwest + regionnortheast + age2, data = pop_data)
summary(model_updated)

# The updated model has an accuracy of 86% while the previous model had accuracy 76%.
