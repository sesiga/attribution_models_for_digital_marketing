#### Criteo data set ########
library(data.table)  #permite trabajar con grandes data.frames de forma más rápida, tiene funciones como fread para leer los ficheros
                     #el objeto de base es data.table en vez de data.frame, y la sintaxis es un poco distinta de la habitual (dplyr por ejemplo)
                     #la sintaxis de dplyr se puede usar en un objeto data.table, pero es mucho más lenta  
                     #link útil sobre la sintaxis de data.table: https://walterwzhang.github.io/notes/2016/12/29/dplyr-and-datatable-in-R
library(fst) #para guardar y cargar datos en formato fst (más rápido)
library(ggplot2)

#### Convertir el fichero a .fst para que sea más fácil cargarlo en próximas ocasiones - SÓLO SE HACE UNA VEZ
dt <- fread("C:/Users/sesig/Documents/master data science/tfm/criteo_attribution_dataset/criteo_ordered_dataset.tsv", sep="\t", header=TRUE, showProgress=TRUE)
write.fst(dt, "criteo_dataset.fst")

#### Lectura de los datos y ordenar por usuario

dt <- read.fst("criteo_dataset.fst", as.data.table=TRUE)
names(dt)

dt <- setorder(dt, uid, timestamp)

Conv_por_usuario <- dt[, .(Num_Conv= sum(conversion)), by = uid]  #agrupamos por usuario y calculamos el número de conversiones
table(Conv_por_usuario[, .(Num_Conv)])  #Quizá los de 2 o más conversiones se pueden separar en distintos usuarios
# 0       1       2       3       4       5       6       7       8       9      10      11      12      13 
# 5814396  190891   58770   27207   14954    9119    6088    4242    3011    2356    1762    1476    1130     913 
# 14      15      16 Conversion      18      19      20      21      22      23      24      25      26      27 
# 742     663     553     483     391     344     297     229     223     182     180     156     122     119 

barplot(table(Conv_por_usuario[, .(Num_Conv)]) )

dt = dt[, "Num_Conv":=sum(conversion), by=uid]
Conv2 = dt[Num_Conv==2]  #los usuarios con 2 conversiones (algunas son dos conversiones de verdad, distinta conver
#sion_timestamp, otras tienen la misma conversion_timestamp)

head(Conv2, 30)


#### Distribución de las campañas, cuántas veces aparece cada una. Se aprecia un posible punto de corte sobre 50000.
# Una posibilidad sería quedarse con las campañas que aparezcan al menos 50000 veces
Campaings <- setorder( dt[, .(Num_Camp = .N), by = campaign], - Num_Camp)
ggplot(Campaings, aes(x=1:nrow(Campaings), y = Num_Camp)) + geom_bar(stat="identity")+ 
       xlab("Campañas") + ylab("Frecuencia Absoluta (núm. de impactos)")

#### Distribución del número de impactos por cliente. Se podría cortar y quedarse con los usuarios con 10 o menos impactos
Impactos <- setorder( dt[, .(Num_Imp = .N), by = uid], - Num_Imp)   #agrupamos por cliente y calculamos el número de impactos (.N significa núm. de filas)
                                                                    #En setorder el primer argumento es la tabla, y el segundo la columna por la que ordenar, con un
                                                                    # - delante para indicar orden descendiente
# TARDA DEMASIADO
# ggplot(Impactos, aes(x=1:nrow(Impactos), y = Num_Imp)) + geom_bar(stat="identity")+ scale_y_log10() +
#   xlab("Usuarios") + ylab("Frecuencia Absoluta (núm. de impactos)")  

boxplot(Impactos[, .(Num_Imp)])
table(Impactos[, .(Num_Imp)])
# 1       2       3       4       5       6       7       8       9      10      11      12      13      14      15 
# 3175158 1190099  591322  342342  217097  146627  103041   75947   56243   43501   34029   26847   21574   17563   14531 
# 16      17      18      19      20      21      22      23      24      25      26      27      28      29      30 
# 11850   10004    8345    7133    6027    5199    4372    3832    3286    2810    2311    2140    1961    1684    1455 
# 31      32      33      34      35      36      37      38      39      40      41      42      43      44      45 
# 1260    1151    1029     901     801     701     635     606     539     493     431     398     360     342     285 

barplot(table(Impactos[, .(Num_Imp)]))

#### Conversiones y no-conversiones

Conversion <- dt[, .(Conv = max(conversion)), by = uid]  #lista de los usuarios que convierten (1) o no (0)
table(Conversion[, .(Conv)])

dt_conv <- dt [Num_Conv>=1 ]  #data table con los usuarios que convierten
dt_noconv <- dt [Num_Conv==0 ]  #data table con los usuarios que NO convierten

