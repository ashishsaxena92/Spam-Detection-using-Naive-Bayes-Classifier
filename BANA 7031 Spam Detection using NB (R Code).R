
#############################################################################################################
#############################################################################################################
#### Course Code & Name: BANA 7031 Probability Models                                              ##########
#### Assignment: Final Project                                                                     ##########
#### Submitted by: Ashish Saxena                                                                   ##########
#############################################################################################################
#############################################################################################################


# Listing required libraries
pkgs <- c(
  "tm",  # text mining functions
  "wordcloud",  # for word-cloud
  "SnowballC",  # for stemming
  "e1071",  # for stemming
  "caret",  # for confusion-matrix
  "ggplot2",  # for visualizations
  "dplyr",  # for transformation of data
  "caTools",  # for ROC curve
  "rms"  # for AUC value
)

# Installing required libraries
for (pkg in pkgs) {
  if (!(pkg %in% installed.packages()[, "Package"])) {
    install.packages(pkg, dependencies = T)
  }
}

# Loading the installed libraries into R environment
for (pkg in pkgs) {
    require(pkg, character.only = T)
}


# Setting up working directory - Please modify this based on location of dataset on your local machine
setwd('C:/Users/info/OneDrive/Desktop/University of Cincinnati/Spring Semester/BANA 7031 Probability Models/Project/')

# Reading input dataset
input = read.csv('spam.csv', stringsAsFactors = F)

# Creating combined text message from component fields
input$text = paste0(input$v2,input$X,input$X.1,input$X.2)

# Filtering for and renaming required fields from the dataset
spam = input[,c("v1","text")]
colnames(spam) = c("Flag", "Text")

# Converting "Flag" to factor (categorical) variable for classification response
spam$Flag = as.factor(spam$Flag)

# Analyzing structure of dataset obtained
str(spam)

# Analyzing distribution of response classes in the dataset
table(spam$Flag)/nrow(spam)
cnt_spam = spam%>%
  group_by(Flag)%>%
  summarise(cnt = length(Text))

ggplot(data = cnt_spam, aes(x = Flag, y = cnt)) +
  geom_bar(stat="identity",color='steelblue',fill='steelblue')

# Create test-train sample with 70-30 split
set.seed(7031)
index= sample(nrow(spam), 0.7*nrow(spam))
train = spam[index,]
train_labels = train$Flag
table(train$Flag)/nrow(train)
test = spam[-index,]
table(test$Flag)/nrow(test)
test_labels = test$Flag

# Defining data processing function for pre-processing of data
clean_data = function(spam){
  
  # Creating corpus of the dataset
  sms_corp = Corpus(VectorSource(spam$Text))
  
  # Converting all text to lower-case
  corpus_clean = tm_map(sms_corp, tolower)
  
  # Removing numbers from the dataset
  corpus_clean = tm_map(corpus_clean, removeNumbers)
  
  # Removing stop words for better training of the model
  corpus_clean = tm_map(corpus_clean, removeWords, stopwords())
  
  # Removing leading/trailing whitespace from data
  corpus_clean = tm_map(corpus_clean, stripWhitespace)
  
  # Removing punctuations from data for better training
  replacePunctuation <- function(x) {gsub("[[:punct:]]+", " ", x)}
  corpus_clean = tm_map(corpus_clean, replacePunctuation)
  
  # Stemming data for standardization and hence better training
  corpus_clean = tm_map(corpus_clean, wordStem)
  return(corpus_clean)
}

# Plotting word-cloud of legitimate (ham) data
wordcloud(clean_data(spam[spam$Flag=="ham",]), min.freq = 50, random.order = FALSE)

# Plotting word-cloud of spam data
wordcloud(clean_data(spam[spam$Flag=="spam",]), min.freq = 20, random.order = FALSE)


# Processing train & test datasets for document-term matrix
corp_train = clean_data(train)
corp_test = clean_data(test)

# Creating respective Document-Term Matrix (DTM) from processed datasets
train_dtm = DocumentTermMatrix(corp_train)
test_dtm = DocumentTermMatrix(corp_test)

# Defining frequency thresholds for DTM to reduce number of features and hence processing time
train_freq = findFreqTerms(train_dtm, 5)
test_freq = findFreqTerms(test_dtm, 5)

# Filtering for corresponding terms obtained against specified thresholds
train_freq_dtm = train_dtm[,train_freq]
test_freq_dtm = test_dtm[,test_freq]

# Function to convert features in sparse matrix from numerical to categorical 
convert_counts <- function(x) {x <- ifelse(x > 0, "Yes", "No")}

# Applying user-defined function to DTM for each column
cnt_train <- apply(train_freq_dtm, MARGIN = 2,
                   convert_counts)

cnt_test <- apply(test_freq_dtm, MARGIN = 2,
                   convert_counts)

# Creating final model based on train data
nb_binary = naiveBayes(cnt_train, train_labels)

# Predicting results from model and test data
results = predict(nb_binary, cnt_test, type = "class")
results2 = predict(nb_binary, cnt_test, type = "raw")

# Plotting the confusion matrix for result analysis
confusionMatrix(results,test_labels)

# Plotting the AUC
rcorr.cens(results2[,1],ifelse(test_labels=="ham",1,0))[1]
colAUC(results2[,1], ifelse(test_labels=="ham",1,0), plotROC=TRUE)
