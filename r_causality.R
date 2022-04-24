library(readr)
library(jtools) # for summ()
library(interactions)
library(ggplot2)

data <- read_csv("sample/data_435.csv")
dataint <- read_csv("sample/data_518.csv")

par(mfrow=c(2,2))
hist(data$A)
hist(dataint$A)
hist(data$B)
hist(dataint$B)

hist(data$C)
hist(dataint$C)
hist(data$D)
hist(dataint$D)


#fiti <- lm(A ~ B + C, data = data)
#interaction.plot(data$A, data$B, data$X1)
