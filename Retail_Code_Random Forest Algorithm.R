
# Store should be opened in a particular area depending on sales, population, area, etc.

# Set working directory
setwd("C:/Users/staru/OneDrive/Desktop/Retail Project")

# Importing training and test dataset
store_train <- read.csv("store_train.csv", stringsAsFactors = FALSE)
store_test <- read.csv("store_test.csv", stringsAsFactors = FALSE)

# Library Needed For This Model
library(dplyr) # For Data Preparation
library(cvTools) # For cross validation
library(randomForest) # For building random forest

# Structure of the both training and test dataset
glimpse(store_train)
glimpse(store_test)

# Column Names of Training and Test Data
names(store_train)
names(store_test)

# Make difference which column name is not available on test data
setdiff(names(store_train), names(store_test))

# Check Missing Value in Imported Training and Test Data
sum(is.na(store_train))
sum(is.na(store_test))

# Summary of Training and Test Data
summary(store_train)
summary(store_test)

# Response variable is 'Store'
# Creating a empty column for response variable in store_test
store_test$store = NA

# Creating new column named 'data' in both store_train & store_test
# Which specifies if the data is from train or test set
store_train$data = 'train'
store_test$data = 'test'

# Combine both data sets into one for data cleaning purpose 
store_all <- rbind(store_train, store_test)

glimpse(store_all)

# Checking missing value of store_all data set
sum(is.na(store_all))
sapply(store_all, function(x) sum(is.na(x)))

# Data Preparation -------------------------------------------------------------------------------------------------------------

# Create Dummy Function Formula
CreateDummies = function(data, var, freq_cutoff = 0){
  t = table(data[,var])
  t = t[t > freq_cutoff]
  t = sort(t)
  categories = names(t)[-1] # Excluding first name as we need one less dummy variable
  
  for(cat in categories){
    name=paste(var,cat,sep="_") # Column name is written as var_cat, e.g., State_FL
    name=gsub(" ","",name)
    name=gsub("-","_",name)
    name=gsub("\\?","Q",name)
    name=gsub("<","LT_",name)
    name=gsub("\\+","",name)
    
    data[,name] = as.numeric(data[,var] == cat)
    # Column gets values 1 & 0 for respective category
  }
  
  data[,var]= NULL # Removing 'var' column from data
  return(data) # Return Data
}

glimpse(store_all)

# Store Variable
table(store_all$store)
unique(store_all$store)
prop.table(table(store_all$store))

store_all$store <- as.numeric(store_all$store == 1)
store_all$store <- as.factor(store_all$store)

glimpse(store_all)

# Let's check number of unique values in each column
sort(sapply(store_all, function(x) length(unique(x))))

# Name of all character columns
names(store_all)[sapply(store_all, function(x) is.character(x))]

# Name of all numerical columns
names(store_all)[sapply(store_all, function(x) is.numeric(x))]

# Drop Variables -  we cannot use these variables in modeling process
table(store_all$storecode) # 2572 unique values
table(store_all$Areaname) # 2572 unique values
table(store_all$countytownname) # 3176 unique values
table(store_all$countyname) # 1962 unique values
# Id variable we cannot use for this modeling process
# State_alpha is similar as State variable. we also drop this variable

store_all <- store_all %>% 
             select(-Id, -storecode, -Areaname, -countytownname, -countyname, -state_alpha)

glimpse(store_all)

# Create Dummy for categorical variables
char_logical = sapply(store_all, is.character)
cat_cols = names(store_all)[char_logical]
cat_cols

cat_cols = cat_cols[!(cat_cols %in% c('data','store'))]
cat_cols

for(col in cat_cols){
  store_all = CreateDummies(store_all, col, 50)
}

glimpse(store_all)

# Missing Values Treatment ----------------------------------------------------------------------------------------

# NA values in all the columns of store_all data set
sum(is.na(store_all))
sort(sapply(store_all, function(x) sum(is.na(x))))

# We can go ahead and separate training and test data BUT first we check NA values
store_all = store_all[!((is.na(store_all$store)) & store_all$data == 'train'), ]

# Imputing all missing values by mean function
for(col in names(store_all)) {
  
  if(sum(is.na(store_all[,col])) > 0 & !(col %in% c("data","store"))) {
    
    store_all[is.na(store_all[,col]),col] = mean(store_all[store_all$data == 'train',col], na.rm = T)
  }
}

# Store have 1431 missing values
sum(is.na(store_all)) # 1431 - These are missing value of store from Test Data

# Separate Train and Test data
store_train = store_all %>% filter(data == 'train') %>% select(-data)
store_test = store_all %>% filter(data == 'test') %>% select(-data,-store)

# Export Training and Test data set for future use
write.csv(store_train, "store_train_clean.csv", row.names = F)
write.csv(store_test, "store_test_clean.csv", row.names = F)

# -------------------------------------------------------------------------------------------------------------------------------------

# Model Building on Entire Training data ----------------------------------------------------------------------------------------------
# For Classification Random Forest we'll need to convert Response variable to factor type

glimpse(store_train)
store_train$store = as.factor(store_train$store)
table(store_train$store)

# Classification Random Forest with Parameter Tuning -----------------------------------------------------------------------------------

param = list(mtry = c(5,10,13),
             ntree = c(50,100,200,500,700,1000),
             maxnodes = c(5,10,15,20,30,50),
             nodesize = c(1,2,5,10))

# Function for getting all possible combinations : expand.grid()
all_comb = expand.grid(param) # Grid Search for all combination
#6*6*6*4 = 864 combinations of parameters,
# And for 10-fold CV, it would build 864*10 trees to find the best 
# performing parameters.

# Create function for ROC Curve and AUC Score
mycost_auc = function(y,yhat){
  roccurve = pROC::roc(y,yhat)
  score = pROC::auc(roccurve)
  return(score)
}

# Function for selecting random subset of Param
subset_paras = function(full_list_para, n = 10){
  
  all_comb = expand.grid(full_list_para)
  
  set.seed(1)
  
  s = sample(1:nrow(all_comb),n)
  
  subset_para = all_comb[s,]
  
  return(subset_para)
  
}

# Randomize Grid Search
num_trials = 55
my_params = subset_paras(param, num_trials)
# Note: A good value for num_trials is around 50-60

# CVTuning For Classification
myauc = 0

# Lets Start CVTuning For Classification
# This code will take couple
for(i in 1:num_trials){
  print(paste0('starting iteration :',i))
  # Uncomment the line above to keep track of progress
  params = my_params[i,]
  
  k = cvTuning(randomForest, store~., 
               data = store_train,
               tuning = params,
               folds = cvFolds(nrow(store_train), K = 10, type = "random"), # K = 5 or 10 gives good result
               cost = mycost_auc, 
               seed = 2,
               predictArgs = list(type = "prob")
  )
  print(k)
  print(class(k))
  score.this = k$cv[,2]
  print(paste0('CV Score: ', score.this))
  # Default cost is RMSE
  
  if(score.this > myauc){
    print(params)
    # Uncomment the line above to keep track of progress
    myauc = score.this
    print(myauc)
    # Uncomment the line above to keep track of progress
    best_params = params
  }
  
  print('DONE')
  # Uncomment the line above to keep track of progress
}

myauc
best_params

# mtry = Number of variables randomly sampled as candidates at each split. 
# ntree = Number of trees to grow
# maxnodes = Maximum number of terminal nodes trees in the forest can have
# nodesize = Minimum size of terminal nodes

st.rf.final = randomForest(store~.,
                           mtry = best_params$mtry,
                           ntree = best_params$ntree,
                           maxnodes = best_params$maxnodes,
                           nodesize = best_params$nodesize,
                           data = store_train
)

st.rf.final

# Performance of Score model On Training & Test data --------------------------------------------------------

# Probability Scores For Training and Test Data ----------------------------------------------------------------------------------------------

# Now if we needed to submit probability scores for the test data we can do at this point

# For Test Data Set (Probability Score)
test.prob.score <- predict(st.rf.final, newdata = store_test, type = 'prob')[,2]
test.prob.score[1:10] # Probability Score For Entire Test Data for class 1

# Export Prediction for submission
write.csv(test.prob.score, "probability_score_RFC.csv", row.names = F)

# For Training Data Set (Probability Score)
train_prob.score = predict(st.rf.final, newdata = store_train, type = 'prob')[,2]
train_prob.score[1:20] # Probability Score For Entire Training Data for class 1
store_train$store[1:20] # Real/Actual values of Entire Training Data

# AUC(Area Under Curve) (Higher the better)
library(pROC)

# For Training Data (Area under the curve: 0.8706)
auc(roc(store_train$store, train_prob.score)) # roc = Receiver operating characteristic curve
            
            
