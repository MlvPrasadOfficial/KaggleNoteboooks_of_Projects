# You can write R code here and then click "Run" to run it on our platform

library(readr)
library(Rtsne)
# The competition datafiles are in the directory ../input
# Read competition data files:
train <- read_csv("../input/train.csv")
test <- read_csv("../input/test.csv")
train$label <- as.factor(train$label)

# shrinking the size for the time limit
numTrain <- 10000
set.seed(1)
rows <- sample(1:nrow(train), numTrain)
train <- train[rows,]
# using tsne
set.seed(1) # for reproducibility
tsne <- Rtsne(train[,-1], dims = 2, perplexity=30, verbose=TRUE, max_iter = 500)
# visualizing
colors = rainbow(length(unique(train$label)))
names(colors) = unique(train$label)
plot(tsne$Y, t='n', main="tsne")
text(tsne$Y, labels=train$label, col=colors[train$label])

# compare with pca
pca = princomp(train[,-1])$scores[,1:2]
plot(pca, t='n', main="pca")
text(pca, labels=train$label,col=colors[train$label])

# Generate output files with write_csv(), plot() or ggplot()
# Any files you write to the current directory get shown as outputs
