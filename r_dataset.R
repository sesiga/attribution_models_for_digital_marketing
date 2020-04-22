rm(list=ls())
library(ChannelAttribution)
data(PathData)
Data$path[1]
help("write.csv")
write.csv(Data,file='C:/Users/sesig/Documents/master data science/tfm/r_dataset/r_data.csv',sep=',',dec='.',quote=FALSE,row.names=FALSE)
