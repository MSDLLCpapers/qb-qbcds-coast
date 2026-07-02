# load the gene expression matrix
chen <- get(load('GSE87544_1443737Cells.Expresssion.Matrix.log_tpm+1_.renamed.RData'))
head(chen)
dim(chen)

# load the labels of the cells
labels <- read.csv('GSE87544_1443737Cells.SVM.cluster.identity.renamed.csv')
head(labels)
str(labels)

# check if the orders of the cells in two data matrices are consistent
sum(!labels[,1]==colnames(chen))

# extract OPC and MO cells
Y<-labels[,3]
index <- Y=='MO'|Y=='OPC'
X<-chen[,index]
Y<-Y[index]

# keep the genes that can be detected in more than 30% of the cells (OPC and MO cells)
perc <- rowMeans(X > 0)
keep <- (perc > 0.3)
X1 <- X[keep, ]
cat('Global 30% filter:', nrow(X1), 'genes,', ncol(X1), 'cells\n')

# save the names of the genes, the new expression matrix, and the new labels
gene_names<-rownames(X1)
write.table(gene_names,file='gene_names.txt',quote=F,row.names=F,col.names=F)

rownames(X1)<-colnames(X1)<-NULL
write.table(X1,file='chen_X.txt',quote=F,row.names=F,col.names=F)

Ynew<-matrix(0,ncol(X1),2)
Ynew[Y=='MO',1]<-1
Ynew[Y=='OPC',2]<-1
Ynew<-as.data.frame(Ynew)
write.table(Ynew,file='chen_Y.txt',quote=F,row.names=F,col.names=F)

# per-group filter: keep genes detected in >10% of cells in at least one group
perc_mo <- rowMeans(X[, Y=='MO'] > 0)
perc_opc <- rowMeans(X[, Y=='OPC'] > 0)
keep2 <- (perc_mo > 0.1) | (perc_opc > 0.1)
X2 <- X[keep2, ]
cat('Per-group 10% filter:', nrow(X2), 'genes,', ncol(X2), 'cells\n')

gene_names2<-rownames(X2)
write.table(gene_names2,file='gene_names2.txt',quote=F,row.names=F,col.names=F)

rownames(X2)<-colnames(X2)<-NULL
write.table(X2,file='chen_X2.txt',quote=F,row.names=F,col.names=F)

Ynew2<-matrix(0,ncol(X2),2)
Ynew2[Y=='MO',1]<-1
Ynew2[Y=='OPC',2]<-1
Ynew2<-as.data.frame(Ynew2)
write.table(Ynew2,file='chen_Y2.txt',quote=F,row.names=F,col.names=F)