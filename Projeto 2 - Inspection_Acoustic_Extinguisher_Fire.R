'O teste hidrostático extintor é um procedimento estabelecido pelas normas da ABNT NBR 12962/2016, que determinam que todos os extintores devem ser testados a cada cinco
anos, com a finalidade de identificar eventuais vazamentos, além de também verificar a resistência do material do extintor.

Com isso, o teste hidrostático extintor pode ser realizado em baixa e alta pressão, de acordo com estas normas em questão. O procedimento é realizado por profissionais técnicos
da área e com a utilização de aparelhos específicos e apropriados para o teste, visto que eles devem fornecer resultados com exatidão.

Seria possível usar Machine Learning para prever o funcionamento de um extintor de incêndio com base em simulações feitas em computador e assim incluir uma camada adicional
de segurança nas operações de uma empresa? Esse é o objetivo do Projeto com Feedback 2.

Usando dados reais disponíveis publicamente, seu trabalho é desenvolver um modelo de Machine Learning capaz de prever a eficiência de extintores de incêndio.'

setwd('C:/Users/Matheus Keitaro/OneDrive - Tecsoil Automação e Sistemas S.A/Área de Trabalho/DSA/R e Azure Machine learning/Projetos-1-2/Projeto 2')
getwd()

library(dplyr)
library(tidyr)
library(caret)
library(rpart)
library(readxl)
library(corrplot)
library(caTools)
library(ggplot2)
library(cowplot)
library(e1071)
library(gbm)

df <- read_excel('Acoustic_Extinguisher_Fire_Dataset.xlsx')

#Verificando o dataset
View(df)
str(df)
summary(df)
dim(df)

#verificando valores NA
colSums(is.na(df))

#Verificando se a variavel target esta balanceada
count(df, STATUS, sort=TRUE)

#Convertendo a variavel FUEL para factor
df$FUEL <- as.factor(df$FUEL)

#selecionando variaveis numericas
df_num <- df %>% select(where(is.numeric))
df_cat <- df %>% select(where(is.factor))

#Verificando a correlação entre as variaveis
corr <- cor(df_num)
corrplot(corr, method = 'color')

#Convertendo a variavel STATUS para factor
df$STATUS <- as.factor(df$STATUS)

#Splitando os dados
set.seed(1234)
sample <- sample.split(df$STATUS, SplitRatio = 0.7)
df_train <- subset(df, sample == TRUE)
df_test <- subset(df, sample == FALSE)

str(df_train)

#Criando os modelos
modelo1 <- glm(STATUS~., data = df_train, family = "binomial")
modelo2<- rpart(STATUS ~., data=df_train)
modelo3 <- svm(STATUS~., data = df_train, kernel = 'sigmoid')
modelo4 <- naiveBayes(STATUS~., data = df_train)
modelo5 <- gbm(STATUS~., data = df_train, distribution = "multinomial", cv.folds = 10, shrinkage = .01, n.minobsinnode = 10, n.trees = 500)

?gbm
#Predição 1
summary(modelo1)
pred <- predict(modelo1, df_test, type = "response")
pred<- ifelse(pred >0.5, 1, 0)

anova(modelo1, test = "Chisq")

result <- confusionMatrix(as.factor(pred), df_test$STATUS)

#Predição 2
pred2 <- predict(modelo2, df_test)
pred2 <- data.frame(pred2)
pred2$pred <- ifelse(pred2$X0 > pred2$X1, 0, 1)
result2 <- confusionMatrix(as.factor(pred2$pred), df_test$STATUS)


#Predição 3
pred3 <- predict(modelo3, df_test)
result3 <- confusionMatrix(as.factor(pred3), df_test$STATUS)

#Predição 4
pred4 <- predict(modelo4, df_test)
result4 <- confusionMatrix(as.factor(pred4), df_test$STATUS) 

#Predição 5
summary(modelo5)
pred5 = predict.gbm(object = modelo5, newdata = df_test, n.trees = 500, type = "response")
pred5 <- data.frame(pred5)
pred5$pred <- ifelse(pred5$X0.500 > pred5$X1.500, 0, 1)
result5 <- confusionMatrix(as.factor(pred5$pred), df_test$STATUS)


data.frame(logistic = round(result$overall, 3),
           rpart = round(result2$overall,3),
           svm = round(result3$overall,3),
           NaiveBayes = round(result4$overall,3),
           gradBoost = round(result5$overall, 3))

data.frame(logistic = round(result$byClass, 3),
           rpart = round(result2$byClass,3),
           svm = round(result3$byClass,3),
           NaiveBayes = round(result4$byClass,3),
           gradBoost = round(result5$byClass, 3))

result$table
result5$table


'O modelo gradientboost, foi o que apresentou maior acuracia, e um numero maior no F1 score. Porém o modelo de regressão logistica
apresentou uma eficacia muito próxima do GB. Os dois modelos entregaram um valor de acuracia e F1 score superior a 90%.

Abaixo foi realizada a cross-validation para o metodo de regressão logistica.'

library(plyr)
# False positive rate
fpr <- NULL

# False negative rate
fnr <- NULL

# Number of iterations
k <- 500

# Initialize progress bar
pbar <- create_progress_bar('text')
pbar$init(k)

# Accuracy
acc <- NULL

set.seed(123)

for(i in 1:k){
  # Train-test splitting
  # 95% of samples -> fitting
  # 5% of samples -> testing
  smp_size <- floor(0.95 * nrow(df))
  index <- sample(seq_len(nrow(df)),size=smp_size)
  train <- df[index, ]
  test <- df[-index, ]
  
  # Fitting
  model <- glm(STATUS~.,family=binomial,data=train)
  
  # Predict results
  results_prob <- predict(model,subset(test,select=c(1:6)),type='response')
  
  # If prob > 0.5 then 1, else 0
  results <- ifelse(results_prob > 0.5,1,0)
  
  # Actual answers
  answers <- test$STATUS
  
  # Accuracy calculation
  misClasificError <- mean(answers != results)
  
  # Collecting results
  acc[i] <- 1-misClasificError
  
  # Confusion matrix
  dim(results)
  dim(answers)
  cm <- confusionMatrix(data=as.factor(results), reference=answers)
  fpr[i] <- cm$table[2]/(nrow(df)-smp_size)
  fnr[i] <- cm$table[3]/(nrow(df)-smp_size)
  
  
  pbar$step()
  print(i)
}

# Average accuracy of the model
mean(acc)

par(mfcol=c(1,2))

# Histogram of accuracy
hist(acc,xlab='Accuracy',ylab='Freq',
     col='cyan',border='blue',density=30)

# Boxplot of accuracy
boxplot(acc,col='cyan',border='blue',horizontal=T,xlab='Accuracy',
        main='Accuracy CV')

# Confusion matrix and plots of fpr and fnr
mean(fpr)
mean(fnr)
hist(fpr,xlab='% of fnr',ylab='Freq',main='FPR',
     col='cyan',border='blue',density=30)
hist(fnr,xlab='% of fnr',ylab='Freq',main='FNR',
     col='cyan',border='blue',density=30)

'A cross-validation indica uma acuracia media de 90%, incluindo que 75% dos testes rodados, atingiu uma acuracia entre 87% a 91%.'