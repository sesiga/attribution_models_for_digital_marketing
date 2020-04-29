rm(list=ls())
library(ChannelAttribution)
data(PathData)
Data$path[1]
help("write.csv")
write.csv(Data,file='C:/Users/sesig/Documents/master data science/tfm/r_dataset/r_data.csv',sep=',',dec='.',quote=FALSE,row.names=FALSE)

df = data.frame('x'=c(5000,10000,20000,30000,50000,100000),
                'y'=c(6,11.57,23,41.7,76,274))
plot(df$x,df$y,type='l')
plot(df$x,log(df$y),type='l')
mod = lm(y~log(x),data=df)
summary(mod)
df_pred = data.frame('x'=16500000)
pred = predict(mod,newdata=df_pred)
pred
exp(pred)
