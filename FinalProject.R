library(tidyverse)
library(readr)
library(tidytext)
library(wordcloud2)
library(forcats)
library(dplyr)
library(readtext) 
library(ggplot2)
library(SnowballC)
library(topicmodels)
library(stm)
library(ldatuning)
library(knitr)
library(LDAvis)
library(stringr)
library(servr)
library(scales)
library(reshape2)
library(NLP)


################ Prepare ################
#Purpose: The purpose of this project is to explore the trends related to Artificial Intelligence in the New York Times newspaper from 1986 to 2016.The media plays a powerful role to control the public perceptions.Understanding of the media’s attitude toward AI can help to understand people’s perspectives likewise.  

#Data Sources: As data source, this project used the Fast & Horvitz(2016) AI news annotated dataset. This dataset includes 

#Target Audience: This analysis may help AI educators and researchers who want to develop educational learning activities related to text mining classification task. They can use this analysis as an example to reveal possible learning opportunities. 

# Research Questions
#RQ1:How the AI was discussed in the public newspaper between 1986 and 2016?

#RQ2:What are the most salient ideas related to AI?

#RQ3:What kind of topics covered in the pessimistic and optimistic articles?

################ Wrangle ################

robot_ai_all_public <- read_csv("data/robot-ai-all-public.csv")

View(robot_ai_all_public)

ai_public <- robot_ai_all_public %>%
  select("Article ID", "Article Date", "Paragraph number","NYT section",Paragraph, Title, "AI Mood")

#Rename column names

ai_public_new <- ai_public %>%
  rename(
    ID = "Article ID",
    Date = "Article Date",
    Paragraph_number= "Paragraph number",
    Section= "NYT section",
    AI_mood= "AI Mood"
  )

glimpse(ai_public_new)

ai_duplicated<- distinct(ai_public_new,ID,Date,Paragraph_number,Section,Paragraph, Title)
glimpse(ai_duplicated)


#/// This part is not necessary anymore.
#Identify and removing duplicated rows 

#Creating a data frame
#ai_public_new_df<- data.frame(ai_public_new)

# Remove duplicates from data frame:
#ai_public_new_df[!duplicated(ai_public_new_df),]

#Delete duplicate rows
#ai_public_new_df.un <- ai_public_new_df[!duplicated(ai_public_new_df), ]

#head(ai_public_new_df.un)

# We still have duplicated rows
#ai_public_new_df.un <- unique(ai_public_new_df.un)

# Remove the duplicated rows based on ID
#ai_public_new_df2.un <- ai_public_new_df %>%
  #distinct(ID, .keep_all = TRUE)

#head(ai_public_new_df2.un)

ai_public_removed <- ai_duplicated
head(ai_public_removed)

#Term Frequency-Inverse Document Frequency
ai_words <- ai_public_removed %>%
  unnest_tokens(word, Paragraph) %>%
  count(Title, word, sort = TRUE)

head(ai_words)

total_words <- ai_words %>%
  group_by(Title) %>%
  summarise(total = sum(n))

total_words

ai_totals <- left_join(ai_words, total_words)
head(ai_totals)

ai_tf_idf <- ai_totals %>%
  bind_tf_idf(word, Title, n)

ai_tf_idf

view(ai_tf_idf)

#tidying text 

head(stop_words)
view(stop_words)

ai_public_tidy <- ai_public_removed %>%
  na.omit() %>%
  unnest_tokens(output = word, input = Paragraph) %>%
  anti_join(stop_words, by = "word")

ai_public_tidy

ai_counts <- ai_public_tidy %>%
  filter(word != "tri") %>%
  filter(word != "im") %>%
  filter(word != "ms") %>%
  filter(word !="ii") %>%
  count(word, sort = TRUE) 

ai_clean <- anti_join(ai_counts, stop_words)
head(ai_clean)

# let's quickly see the wordcloud

wordcloud2(ai_clean,
           color = ifelse(ai_clean[, 2] > 1000, 'black', 'gray'))

#see in a basic bar chart
ai_clean %>%
  filter(n > 100) %>% # keep rows with word counts greater than 100
  mutate(word = reorder(word, n)) %>% #reorder the word variable by n and replace with new variable called word
  ggplot(aes(n, word)) + # create a plot with n on x axis and word on y axis
  geom_col() # make it a bar plot

#Finding Frequencies

ai_frequencies <- ai_public_tidy %>%
  count(Section,Title, word, sort = TRUE) %>%
  mutate(proportion = n / sum(n))

ai_frequencies %>%
  slice_max(proportion, n = 10) %>%
  ungroup() %>%
  group_by(Section) %>%
  ggplot(aes(proportion, fct_reorder(word, proportion), fill = Title)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~Title, ncol = 2, scales = "free") +
  labs(y = NULL, x = NULL)

ai_frequencies_bysection <- ai_public_tidy %>%
  count(Section, word, sort = TRUE) %>%
  group_by(Section) %>%
  mutate(proportion = n / sum(n))

ai_frequencies_bysection

ai_frequencies_bysection %>%
  slice_max(proportion, n = 3) %>%
  ungroup() %>%
  group_by(Section) %>%
  ggplot(aes(proportion, fct_reorder(word, proportion), fill = Section)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~Section, ncol = 3, scales = "free") +
  labs(y = NULL, x = NULL)

#creating multiple graphs

ai_title_counts <- ai_public_tidy %>%
  count(Title, word)

total_words <- ai_title_counts %>%
  group_by(Title) %>%
  summarize(total = sum(n))

ai_words <- left_join(ai_title_counts, total_words)

ai_tf_idf <- ai_words %>%
  bind_tf_idf(word, Title, n)

ai_tf_idf %>%
  filter(tf_idf >= "2.0") %>%
  group_by(Title) %>%
  slice_max(tf_idf, n = 5) %>%
  ungroup() %>%
  mutate(Title=as.factor(Title),
         word=reorder_within(word, tf_idf, Title)) %>%
  ggplot(aes(word, tf_idf, fill = Title)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~Title, ncol = 2, scales = "free") +
  coord_flip() +
  scale_x_reordered() +
  labs(title = "Words Unique to Each Article", x = "tf-idf value", y = NULL)

### Topic Modeling Part ###

#Creating a Document Term Matrix

ai_dtm <- ai_public_tidy %>%
  count(Section, word) %>%
  cast_dtm(Section, word, n)

temp <- textProcessor(ai_public_removed$Paragraph, 
                      metadata = ai_public_removed,  
                      lowercase=TRUE, 
                      removestopwords=TRUE, 
                      removenumbers=TRUE,  
                      removepunctuation=TRUE, 
                      wordLengths=c(3,Inf),
                      stem=TRUE,
                      onlycharacter= FALSE, 
                      striphtml=TRUE, 
                      customstopwords=NULL
                      )
meta <- temp$meta
vocab <- temp$vocab
docs <- temp$documents

stemmed_ai <- ai_public_removed %>%
  unnest_tokens(output = word, input = Paragraph) %>%
  anti_join(stop_words, by = "word") %>%
  mutate(stem = wordStem(word))

stemmed_ai

stemmed_ai <- ai_public_removed %>%
  unnest_tokens(output = word, input = Paragraph) %>%
  anti_join(stop_words, by = "word") %>%
  mutate(stem = wordStem(word)) %>%
  count(ID, stem) %>%
  cast_dtm(ID,stem,n)

stemmed_ai

stemmed_dtm_ai <- ai_public_removed  %>%
  unnest_tokens(output = word, input = Paragraph) %>%
  anti_join(stop_words, by = "word") %>%
  mutate(stem = wordStem(word)) %>%
  count(word, stem, sort = TRUE) %>%
  cast_dtm(word, stem, n)

stemmed_dtm_ai

##Fitting a Topic Modeling with LDA
ai_lda<- LDA(ai_dtm, 
              k = 10, 
              control = list(seed = 588)
)

ai_lda

docs <- temp$documents
meta <- temp$meta
vocab <- temp$vocab

#Fitting a Structural Topic Model

ai_stm <- stm(documents=docs,
                          data=meta,
                          vocab=vocab,
                          K=10,
                          max.em.its=25,
                          verbose = FALSE)

plot.STM(ai_stm, n = 5)

k_metrics_ai <- FindTopicsNumber(
  ai_dtm,
  topics = seq(10, 75, by = 5),
  metrics = "Griffiths2004",
  method = "Gibbs",
  control = list(),
  mc.cores = NA,
  return_models = FALSE,
  verbose = FALSE,
  libpath = NULL
)

FindTopicsNumber_plot(k_metrics_ai)


findingk <- searchK(docs, 
                    vocab, 
                    K = c(5:15),
                    data = meta, 
                    verbose=FALSE)

plot(findingk)

#The LDAvis Explorer
toLDAvis(mod = ai_stm, docs = docs)

#RQ3:What kind of topics covered in the pessimistic and optimistic articles?

bing <- get_sentiments("bing")

bing

sentiment_bing <- inner_join(ai_public_tidy, bing, by = "word")

sentiment_bing

summary_bing <- sentiment_bing %>% 
  group_by(Section) %>% 
  count(sentiment, sort = TRUE) %>% 
  spread(sentiment, n) %>%
  mutate(sentiment = positive - negative) %>%
  mutate(lexicon = "bing") %>%
  relocate(lexicon)

summary_bing

summary_bing_word <- sentiment_bing %>% 
  group_by(word) %>% 
  count(sentiment, sort = TRUE) %>% 
  spread(sentiment, n) %>%
  mutate(sentiment = positive - negative) %>%
  mutate(lexicon = "bing") %>%
  relocate(lexicon)

summary_bing_word

bing_counts <- sentiment_bing %>%
  count(sentiment, sort = TRUE)

bing_counts %>%
  mutate(sentiment = reorder(sentiment,n)) %>%
  ggplot(aes(n, sentiment)) +
  geom_col() +
  labs(x = "Bing Sentiment", y = NULL) +
  theme_minimal()
