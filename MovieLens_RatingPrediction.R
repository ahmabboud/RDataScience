# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
#search for matches to argument pattern within each element of a character vector 
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))
#sub and gsub perform replacement of the first and all matches respectively.

#str_split_fixed(string, pattern, n) Vectorised over string and pattern.
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
#if n is greater than the number of pieces, the result will be padded with empty strings.

colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")


# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]


# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")
# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# using rm() to removes unnecessary objects from the current workspace (R memory)
rm(dl, ratings, movies, test_index, temp, movielens, removed)
# note that rm(list=ls()) will remove all variable from the R Memory


# edx Dataset dimension
print(paste("edx rows=",nrow(edx)))
print(paste("edx cols=",ncol(edx)))
print(paste("edx names=",names(edx)))

# number of zero ratings
print(paste("number of zero ratings"))
print(sum(edx$rating==0))

# Number of unique Movies
print(paste("Number of unique Movies"))
print(length(unique(edx$movieId)))

# Number of Uniqe Users
print(paste("Number of Users"))
print(length(unique(edx$userId)))

# Max number of rating 
print(paste("Max number of ratings"))
edx %>% group_by(movieId,title) %>% filter(n() > 2000) %>% summarize(count=n()) %>% arrange(desc(count))%>% head(10)

#Most Givin Ratings
print(paste("Most Givin Ratings"))
edx %>% group_by(rating) %>% summarize(count=n()) %>% arrange(desc(count))%>% head(10)

# Partition dataset to test and training 
set.seed(2020,sample.kind = "Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1,
                                  p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

# The train set and the trest set considered without NA's 
print(sum(is.na(edx$rating)))
# Now Lets filter only records that are in the train set
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")


# Define validation RMSE function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#Data visiualization
qqnorm(SampleSet$rating);qqline(SampleSet$rating, col = 2)

# Use Just the Average as a prediction Model 
mu<- mean(train_set$rating)
naive_rmse <- RMSE(test_set$rating, mu)
rmse_results <- data_frame(method = "Just the average", RMSE = naive_rmse)
# Show RMSE results
rmse_results %>% knitr::kable()

# Add Movie effect (bi) to Model the prediction
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# Predict rating on the test set
predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i
# Calculating RMSE for this Model
model_1_rmse <- RMSE(predicted_ratings, test_set$rating)
# Adding results to the table of results
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect Model",
                                     RMSE = model_1_rmse ))
# Showing the Results table
rmse_results %>% knitr::kable()

# Add User effect bu to the Model

user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Predicting based on the test set
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred
# calculatin RMSE
model_2_rmse <- RMSE(predicted_ratings, test_set$rating)
# Adding Results to the table of results
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effects Model",  
                                     RMSE = model_2_rmse ))
# Showing the Results table
rmse_results %>% knitr::kable()


# Get Movie Titles
movie_titles <- edx %>% 
  select(movieId, title) %>%
  distinct()
# Here we can see that the top movies has low number of ratings 
train_set %>% dplyr::count(movieId) %>% 
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

# First lets chose an arbitrary lambda =4
lambda <- 4

movie_reg_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 
#Note that, bi is calculated as the solution to minimize eq1

user_reg_avgs <- train_set %>% 
  left_join(movie_reg_avgs, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

# plotting reularized vs original rating
tibble(original = movie_avgs$b_i, 
       regularlized = movie_reg_avgs$b_i, 
       n = movie_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.3)

# Showing a table of the 10 top movies after regularization.
train_set %>%
  dplyr::count(movieId) %>% 
  left_join(movie_reg_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

# Predict based on the new Model
predicted_ratings <- 
  test_set %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  left_join(user_reg_avgs, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred
# Calculating RMSE
model_3_rmse <- RMSE(predicted_ratings, test_set$rating)
# Adding Results to the Table
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized bi + bu Model",  
                                     RMSE = model_3_rmse ))
# Show table of results
rmse_results %>% knitr::kable()

lambdas <- seq(3, 6, 0.1)
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  
  movie_reg_avgs <- train_set %>% 
    group_by(movieId) %>% 
    summarize(b_i = sum(rating - mu)/(n()+l), n_i = n()) 
  #Note that, bi is calculated as the solution to minimize eq1
  
  user_reg_avgs <- train_set %>% 
    left_join(movie_reg_avgs, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  # Predict based on the new Model
  predicted_ratings <- 
    test_set %>% 
    left_join(movie_reg_avgs, by = "movieId") %>%
    left_join(user_reg_avgs, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})

# plot lambda
qplot(lambdas, rmses)  

lmbdBest <- lambdas[which.min(rmses)]
print(paste("Best Lambda=",lambda))


#Add the final Model to the rmse result to compare Models  
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized bi + bu, Best λ",  
                                     RMSE = min(rmses)))
# Show table of results
rmse_results %>% knitr::kable()


movie_reg_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lmbdBest), n_i = n()) 
#Note that, bi is calculated as the solution to minimize eq1

user_reg_avgs <- train_set %>% 
  left_join(movie_reg_avgs, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lmbdBest))

# Predict based on the new Model
predicted_ratings <- 
  test_set %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  left_join(user_reg_avgs, by = "userId") %>%
  mutate(pred_raw = mu + b_i + b_u) %>%
  mutate(pred=case_when(pred_raw < 0 ~ 0, pred_raw >5 ~5, TRUE ~ pred_raw )) %>%
  .$pred
# calculatin RMSE
model_4_rmse <- RMSE(predicted_ratings, test_set$rating)
#Add the final Model to the rmse result to compare Models  
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="After Filtering Outlayer",  
                                     RMSE =model_4_rmse ))
# Show table of results
rmse_results %>% knitr::kable()


## Adding Weighted mutual effect of movie and user  to the model

Cs <-seq (-1,1,0.1)
Rmse_C_All <- sapply( seq(-1,1,0.1), function(c){
  # Predict based on the new Model
  predicted_ratings <- 
    test_set %>% 
    left_join(movie_reg_avgs, by = "movieId") %>%
    left_join(user_reg_avgs, by = "userId") %>%
    mutate(pred_raw = mu + b_i + b_u + (c)*b_u*b_i) %>%
    mutate(pred=case_when(pred_raw < 0 ~ 0, pred_raw > 5 ~5, TRUE ~ pred_raw )) %>%
    .$pred
  # calculatin RMSE
  RMSE(predicted_ratings, test_set$rating)
})

# plot mutual Weight
qplot(Cs, Rmse_C_All)  

CBest <- Cs[which.min(Rmse_C_All)]
print(paste("Best C=",CBest ))

# Predict based on the new Model
predicted_ratings <- 
  test_set %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  left_join(user_reg_avgs, by = "userId") %>%
  mutate(pred_raw = mu + b_i + b_u + (CBest)*b_u*b_i) %>%
  mutate(pred=case_when(pred_raw < 0 ~ 0, pred_raw > 5 ~5, TRUE ~ pred_raw )) %>%
  .$pred
# calculatin RMSE
model_5_rmse <- RMSE(predicted_ratings, test_set$rating)
#Add the final Model to the rmse result to compare Models  
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="After Optimizing mutual weight",  
                                     RMSE =model_5_rmse ))
# Show table of results
rmse_results %>% knitr::kable()



#now we are ready to apply the final model on the validation set.
mu <- mean(edx$rating)

movie_reg_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lmbdBest), n_i = n()) 
#Note that, bi is calculated as the solution to minimize eq1

user_reg_avgs <- edx %>% 
  left_join(movie_reg_avgs, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lmbdBest))

# Predict based on the Final model using Validation Datase
predicted_ratings <- 
  validation %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  left_join(user_reg_avgs, by = "userId") %>%
  mutate(pred_raw = mu + b_i + b_u + (CBest)*b_u*b_i) %>%
  mutate(pred=case_when(pred_raw < 0 ~ 0, pred_raw > 5 ~5, TRUE ~ pred_raw )) %>%
  .$pred
# calculatin RMSE
model_6_rmse <- RMSE(predicted_ratings, validation$rating)
#Add the final Model to the rmse result to compare Models  
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Final Model on Validation",  
                                     RMSE =model_6_rmse ))
# Show table of results
rmse_results %>% knitr::kable()

ggplot(rmse_results, aes(method,RMSE,fill=method )) + geom_bar(stat="identity") + theme(legend.position = "none") +   coord_flip(ylim= c(0.85,0.88))
