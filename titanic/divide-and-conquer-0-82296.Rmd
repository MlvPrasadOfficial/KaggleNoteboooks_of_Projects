---
title: 'Divide and Conquer [0.82296]'
date: "`r Sys.Date()`"
output:
  html_document:
    number_sections: true
    toc: true
    theme: cosmo
    highlight: tango
    code_folding : hide
---

# Introduction and Preliminaries

This *RMarkdown* might be helpful for those that tried to look at this
data set at various angles but are having trouble getting a **public score
better than 0.8**. It grew out of studying many approaches and scripts. The kernel is not intended as a "getting started" one as there are
many other kernels that achieve that purpose. I'd encourage starting and understanding those first. I wrote
it originally as an R script; I thank [*Heads or Tails*](https://www.kaggle.com/headsortails) for encouraging me to
give it a shot converting it to RMarkdown. 

Many kernels report cross validation is typically higher than the public score 
with this data set. The imbalance might be caused of legitimate statistical differences
between the train and test because they are rather small. In this [**other kernel**](https://www.kaggle.com/pliptor/how-am-i-doing-with-my-score) this topic is further discussed.
My belief, however, is that the discrepancy is germane to a compounding set of procedures that overfits many kernels. This kernel will produce a model that predicts
with a **cross validation accuracy of 0.826** and the actual **public score is 0.82296**. 

Beginners may benefit as well as this kernel demonstrates a method on how to fill missing values and
apply **cross validation** to predict scores. It also tries to answer the frequent question of why a score of 100%
is almost provably impossible by looking at a narrow class of female passengers with very similar profile in the **Fare** section of this kernel. Yet, the survival fate of these women is mixed.

The Titanic data set is fun with many features for a playground set but I came to realize that using all
the features is more an invitation for over fitting than helpful for building a good model.
Therefore this kernel is about keeping only potentially relevant features
and discarding as much as possible to avoid over-fitting. Cabin and Age features have many missing values
so we decided to mostly purge them rather than attempting to fill missing values for a better than 80%
accuracy. In fact, a May 2018 [**kernel**](https://www.kaggle.com/cdeotte/titanic-using-name-only-0-81818) by [**Chris Deotte**](https://www.kaggle.com/cdeotte) produces
a public score above 80% using just the Name feature and a simple rule!

One of the most important charts appears to be Sex vs Pclass vs Survival so
we first focus on it. 

This script breaks the problem in sub-stages (**divide**) where a *score* feature
so called **log likelihood ratio** is slowly extracted out of the raw features.
In the end, we make the final model and prediction (**conquer**) using just the
log likelihood as an aggregated feature. One may argue this is a form of a **meta-algorithm**,
along the lines of **boosting**. [This link has more information on **meta-algorithms**](https://stats.stackexchange.com/questions/18891/bagging-boosting-and-stacking-in-machine-learning)

It is still a work in progress. There appears to be places for improvement
and we are working on them.

**Acknowledgments**

The following kernels heavily influenced techniques, ideas, and presentation of this kernel.

[Titanic: Getting Started With R - Full Guide to 0.81340](https://www.kaggle.com/c/titanic/discussion/6821)
by [Trevor Stephens](https://www.kaggle.com/trevorstephens).

[Exploring Survival on the Titanic](https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic)
by [Megan Risdal](https://www.kaggle.com/mrisdal).

[Tidy TitaRnic](https://www.kaggle.com/headsortails/tidy-titarnic)
by [Heads or Tails](https://www.kaggle.com/headsortails).

Many others.

## Load libraries

```{r Load Libraries, echo=TRUE, message=FALSE, warning=FALSE}
library(plyr);      # load plyr prior to dplyr to avoid warnings
library(caret);     # A nice machine learning wrapper library 
library(dplyr);     # data manipulation
library(gridExtra); # multiple plots in one page
library(rpart.plot);# nice plots for rpart trees 
```

## Load data

```{r, message=FALSE, warning=FALSE}
inpath  <- "../data/";
outpath <- "../data/";
kaggle <- F;
if(!dir.exists(inpath)) {
        kaggle  <- T;
	inpath  <- "../input/"; # changing path to Kaggle's environment
	outpath <- "";
}
train <- read.csv(paste0(inpath,"train.csv"));
test  <- read.csv(paste0(inpath,"test.csv"));
```
## Combine data

We combine the test and train sets for joint pre-processing of features. It is mainly used to the imputation of missing data.
Combining train and test sets for this step ensures better statistics is obtained for the features before predicting missing ones. 
This topic is also exemplified in [**this kernel**](https://www.kaggle.com/pliptor/how-am-i-doing-with-my-score). By not performing a joint imputation
your score is less likely to achieve the best score.
We also compute `tr_idx` for train indices. Test indices may be expressed by R's complementary index `-tr_idx`. 

```{r, message = FALSE}
test$Survived <- NA;
comb <- rbind(train, test);
tr_idx <- seq(nrow(train)) # train indices. test indices may be expressed by -tr_idx 
```

## Let us fix a couple of errors in the data set

Thousands of kagglers worked on this data set. Some folks really dug into the details. The following fixes SibSp/Parch values for two passengers (Id=280 and Id=1284) according to [**this kernel**](https://www.kaggle.com/c/titanic/discussion/39787) because a 16 year old can't have a 13 year old son! He goes further and confirm it with historical data. This is mostly an illustration. While I was hoping for an extra point, adding/removing this step doesn't change the final predictions.

```{r, message = FALSE}
comb$SibSp[comb$PassengerId==280] = 0
comb$Parch[comb$PassengerId==280] = 2
comb$SibSp[comb$PassengerId==1284] = 1
comb$Parch[comb$PassengerId==1284] = 1
```

# Completing missing data

The first step will be completing missing data. There are four features with missing data.

1. Fare values: `r sum(is.na(comb$Fare))` passenger.
2. Embarked values: `r sum(comb$Embarked=="")` passengers.
3. Cabin values: `r round(sum(comb$Cabin=='')/nrow(comb)*100)`% missing.
4. Age values: `r round(sum(is.na(comb$Age))/nrow(comb)*100)`% missing.

## Completing Fare 

There are different ways and assisting tools for completing missing data. We will use
*rpart* because it is simple and it tolerates other missing features during training.

```{r, echo=TRUE, message=FALSE, warning=FALSE}
trControl <- trainControl(method="repeatedcv", number=7, repeats=5); 
faremiss <- which(is.na(comb$Fare)); # missing fares (only PassengerId = 1044 is missing)
model_f   <- train( Fare ~ Pclass + Sex + Embarked + SibSp + Parch,
                  data = comb %>% dplyr::filter(!is.na(Fare)),
                   trControl = trControl, method="rpart", na.action = na.pass, tuneLength = 5);
comb$Fare[faremiss] = predict(model_f, comb[faremiss,]); 
comb$FareFac <- factor(comb$Fare); 
```

rpart trees can be plotted nicely with the rpart.plot package

```{r, echo=TRUE, message=FALSE, warning=FALSE}
rpart.plot(model_f$finalModel)
```

Smallish trees like this help interpretation of features. We can see that
naturally passengers in **Pclass** 1 have paid more **Fare**. It is known
that the **Fare** values are the aggregate for all members in a group such as a
family. This is the reason for higher **Fare** values when **Parch** 
and **SibSp** are larger. We can also see more obscure information such as males
in Pclass 1 without spouses and children paid less **Fare** than
females. However, notice also the female **Fare** value (97) is about twice the
value for males (53). It is probably an indication many females without spouses
or children in **Pclass** 1 were not traveling alone.

Let's now check how reliable the estimates are.

```{r, echo=TRUE, message=FALSE, warning=FALSE}
print(model_f$results)
```

The RMSE is large so it will only serve as a ball park. However, we
will not worry much about it because only one passenger had a missing fare
value.

## Completing Embarked

Embarked is a feature that is not used in the remaining of the kernel. It is
one of those features we decided to drop to prevent over fitting.  I'd comment
that this might look odd because I know that having [Sex + Pclass + Embarked is
better than Sex +
Pclass](https://www.kaggle.com/pliptor/what-s-the-expected-maximum-score). It
is a good topic for investigation. I guess it interacts badly with *groups*.

## Completing Cabin

There are too many (70%) missing values for this feature. We will not attempt to
complete missing values in order to prevent adding noise. 

Many kernels simply engineer an indicator for the presence of this feature.
It may help your kernel.

## Completing Age

About `r round(sum(is.na(comb$Age))/nrow(comb)*100)`% of the values are missing. 
The next plot shows that the Age feature has missing values primarily in Pclass 3.

```{r, message = FALSE}
ggplot(comb, aes(Pclass,fill=!is.na(Age))) + geom_bar(position="dodge") + labs(title="Passenger Has Age",fill="Has Age")
```

We decide at this point to dismiss Age information for Pclass 3.  Having to
complete a large percentage of missing values may add more noise to the
prediction system for a goal of better than 80% accuracy.

### How Age impacts Pclass 1 and 2

From the previous observations, we plot the density for survival per age for the remaining Pclass 1 and 2.

```{r, message=FALSE, warning=FALSE}
ggplot(comb[tr_idx,] %>% dplyr::filter(Pclass!=3), aes(Age)) + 
  geom_density(alpha=0.5, aes(fill=factor(Survived))) + labs(title="Survival density per Age for Pclass 1 and 2");
```

The previous graph supports the case that children under around 14 for Pclass 1 and 2
have high likelihood of survival and other age bands are likely to have little
impact for predictions. We create the feature $Minor$ that indicates
children below 14 in Pclass 1 and Pclass 2. 

```{r, message = FALSE}
child <- 14;
comb$Minor <- ifelse(comb$Age<child&comb$Pclass!=3, 1, 0);
comb$Minor <- ifelse(is.na(comb$Minor), 0, comb$Minor);
```

# Extract group indicators and finding groups

There is little doubt that knowing how people formed groups (such as a family) 
or belonged to certain groups (such as children) is key in this data set. Several
good kernels demonstrate that. We attempt to identify various types of groups in addition to
the typical *family* and *title*. One of the main differences in this 
kernel is that we don't engineer at *title* feature from the *name* feature as it is done in most kernels. We believe it is either
not very useful or a potential source for over fitting.

## Compute frequencies

First we will compute the size of groups for Ticket and FareFac.
Here's a fancy way of how we may compute frequencies of features by group in R. These will
be used to determine group sizes.

```{r, message = FALSE}
comb$TFreq <- ave(seq(nrow(comb)), comb$Ticket,  FUN=length);
comb$FFreq <- ave(seq(nrow(comb)), comb$FareFac, FUN=length);
```

## Family

Here a Surname feature is engineered from the Name feature. It will be used later as one of the
group indicators.

```{r, echo=TRUE, message=FALSE, warning=FALSE}
comb$Surname <- sapply(as.character(comb$Name), FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]});
```

## Fares

An interesting characteristic of the fare prices in this data set is that they are very granular.
They are so finely granulated some obscure potential groups would be left unnoticed without using it.
For example, only two passengers paid the exact amount of 6.75 for their fare, embarked at the same port, had ticked numbers very 
close, etc. Perhaps identifying these tiny groups gives us an edge for an extra point or two.

```{r, message = FALSE}
print(comb %>% dplyr::filter(Fare=="6.75")) # A young couple?
```

Is this a young couple? Note that if it is, it would had been undetected by typical procedures to identify "families". This example is also
a remarkable demonstration of why achieving a score of 100% is very unlikely. We have a young female in Pclass=3 with no relatives that didn't make it to the survived list. Let's list others with a similar profile: 

```{r, message = FALSE}
comb %>% dplyr::filter(Sex=="female",Pclass==3,SibSp==0,Parch==0,Age>15, Age<20)
```

It becomes clear that they all have a similar profile, with no family ties, and survival rate is around 63%. We have seven of them to make a prediction in the test set and unless someone comes up with a better idea, the optimum bet is to bet all of those with a similar profile survived. However, most likely around 37% of them perished and a prediction error is probably inevitable making a score of 100% an improbable achievement using only statistics and machine learning. Conversely, if you do
see a kernel that does extremely well, you may want to understand how they may be handling those "tough" types of profiles.

## Finding groups

We now assign group identifications (GID) to each passenger. The assignment follows the following rules:

1. The maximum group size is 11.
2. First we look for families by Surname and break potentially identical family names by appending a family size.
3. Single families by the above rule are labeled 'Single'.
4. Look at the 'Single' group and assign a GID to those that share a Ticket value.
5. Look at the 'Single' group and assign a GID to those that share a Fare value.

```{r echo=TRUE, message=FALSE, warning=FALSE}
maxgrp <- 12
# Family groups larger than 1
comb$GID <- paste0(comb$Surname,as.character(comb$SibSp + comb$Parch + 1))
comb$GID[comb$SibSp + comb$Parch == 0] <- 'Single'

# Ticket group
group <-(comb$GID=='Single') & (comb$TFreq>1) & (comb$TFreq<maxgrp)
comb$GID[group] <- comb$Ticket[group]

# Fare group
group <- (comb$GID=='Single') & (comb$FFreq>1) & (comb$FFreq<maxgrp)
comb$GID[group] <- comb$FareFac[group]

comb$GID <- factor(comb$GID);
```

# Engineer SLogL feature

Secret sauce #2: Engineer a log likelihood ratio survival feature (SLogL). 
The idea is to consolidate all features into a single number indicative of Survival.
Log likelihood ratio is a transformation from a binary random variable such as Survival to
a point in the real line. SLogL gets bigger when survival is likely and gets smaller (negative)
when survival is less likely. SLogL is zero when survival is fifty-fifty. The toss of a fair coin has
a log likelihood ratio of zero. 

Say you have this binary random variable (Survived) and there are multiple "features" that affect it. There is an underlying assumption that the features must be independent (one reason why I make an effort not to bring too many features unless needed because I know that it would violate the theory otherwise. Some of them are highly correlated. Sex and (Mr, Mrs) for those that process **Title** are examples. Otherwise you'll start double counting (overfitting) unless you take other preventive measures.

Say then you have three independent features A, B, and C that influence Survived. Feature A says that probability of survival is PA (and by exclusion death is 1-PA). Then the log likelihood contribution to SLogL by A is computed by SLogA = log(PA/(1-PA)). From the assumption of feature independence and the definition of log likelihood, SLogL becomes SLogA + SLogB + SlogC. In other words, we add the log likelihood ratio contributions of each of the independent features. In a real world, the features may be correlated. In this dataset, if A is Sex and B is Title you'll have twice the proper contribution to SLogL with respect to Sex.

More information about log likelihood ratio can be found [here](http://onlinelibrary.wiley.com/doi/10.1002/0471739219.app1/pdf).

Previously I wrote the method is related to [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression) (mainly because
of the logit function). However, Chris Deotte pointed out in the comments section that this process is more akin to naive Bayes including detailed formulas. Thanks Chris. 

```{r, message = FALSE}
# define function to compute log likelihood of a/(1-a)
logl <- function(a) {
	a <- max(a,0.1); # avoids log(0)
	a <- min(a,0.9); # avoids division by 0
	return (log(a/(1-a)));
}
```

## Pclass and Sex

The next relatively simple plot lets us draw many interesting conclusions.

```{r, echo=TRUE, message=FALSE, warning=FALSE}
p <- list();
item <- 1;
ylim <- 300;
for(class in c(1:3)){
	for(sex in c("male", "female")) {
		p[[item]] <- ggplot(comb[tr_idx,] %>% dplyr::filter(Pclass==class, Sex==sex), aes(x=Survived)) + 
		  geom_bar(aes(fill=Survived)) + scale_y_continuous(limits=c(0,ylim)) + 
		  theme(legend.position="none") + labs(title=paste('Pclass=', as.character(class), sex));
		item <- item + 1;
	}
}
do.call(grid.arrange, p)
```

1. The fate of passengers in Pclass 2 is almost certain: females almost all survive, males almost all perish.
2. Pclass 1 males are not as lucky as in Pclass 2 but females almost all survive.
3. The luck for females in Pclass 3 is mostly a flip of a coin, close to 50%. **I believe this is
the one of the strongest indicators for the limits of achivable score in this dataset.**
4. Males in Pclass 1 has survival rate that is higher than others. **I believe this brings
difficulties similarly as in the case above because the probability of survival gets closer to 50%**

From items 3 and 4, I believe that concentrating efforts on them might produce some extra points. This kernel
doesn't make direct effort towards that but it is an idea to have in mind.

Now we compute the log likelihood ratio of survival for each of the 6 areas in the grid. We use
`dplyr`'s grouping by multiple columns using `.dots`. 

```{r, message = FALSE}
SLogL_SexPclass <- comb[tr_idx,]  %>% 
  dplyr::group_by(.dots=c('Sex','Pclass')) %>% dplyr::summarize(SLogL=logl(mean(Survived)))
print(SLogL_SexPclass)
```

SLogL is proportional to survival. We can see numerically that females in Pclass 1 and 2 have similar
survival likelihood. Females in Pclass 3 are right at the edge of survival as it was also interpreted in the previous plot. 

The next code chunk left-merges the above table to `comb`.
```{r, message = FALSE}
comb  <- merge(x = comb, y= SLogL_SexPclass, by = c('Sex','Pclass'), all.x = TRUE) %>% 
         arrange(PassengerId)
```

## How Survived relates to Ticket and Fare frequencies?

The following plot confirms there is information about survival by looking at the frequency of duplicate tickets and fares

```{r, echo=TRUE, message=FALSE, warning=FALSE}
ggplot(comb[tr_idx,], aes(x=FFreq, y=TFreq, color=factor(Survived)))+
  geom_density_2d() + labs(title="Ticket Frequency and Fare Frequency Density");
```

```{r, echo=TRUE, message=FALSE, warning=FALSE}
pf <- ggplot(comb[tr_idx,] %>% dplyr::filter(Sex=='female'), aes(x=FFreq, y=TFreq, color=Survived)) +
  geom_density_2d() + labs(title="TFreq and FFreq Density (female)");
pm <- ggplot(comb[tr_idx,] %>% dplyr::filter(Sex=="male"), aes(x=FFreq, y=TFreq, color=Survived)) +
  geom_density_2d() + labs(title="TFreq and FFreq Density (male)");
grid.arrange(pf, pm, ncol=2);
```

## TFreq vs Pclass vs SLogL graph

Observe in this chart the SLogL has a few concentrations of undecided outcomes. They are mainly around
an SLogL of zero as expected.
The next steps will diffuse these areas in preparation for the last **conquer** step.

```{r, echo=TRUE, message=FALSE, warning=FALSE}
ggplot(comb[tr_idx,], aes(x=Pclass, y=SLogL)) + geom_jitter(aes(color=Survived)) + 
  facet_grid(  . ~ TFreq,  labeller=label_both) + labs(title="SLogL vs Pclass vs TFreq")
```

# Adjusting SLogL according to groups

```{r, echo=TRUE, message=FALSE, warning=FALSE}
ggplot(comb[tr_idx,], aes(x=FFreq, y=TFreq, color=Survived, alpha=0.5)) +
  geom_count(position=position_dodge(width=5)) + labs(title="Ticket and Fare Frequencies");
```


## Next we reward or penalize groups of people

Group sizes have been argued as one of the driving factors for survival in many of the earlier kernels. 
Large groups may had a harder time to organize themselves moving towards the life boats. 


```{r, echo=TRUE, message=FALSE, warning=FALSE}
ticket_stats <- comb %>% group_by(Ticket) %>% summarize(l = length(Survived), na = sum(is.na(Survived)), c = sum(Survived, na.rm=T));
```

### By increasing the log likelihood score for groups larger than one that have survivors

Note we apply this bias only to groups that contain individuals we need to predict.
Applying to all seems to add noise. It is something that's worth more investigation (version 54
adds the note below).

Note: Daliun2 was the first to ask for mode details about the next block and Martin Krasser
asked for more details at version 53. We can apply the same bias to the entire set
by removing `ticket_stats$na[i] > 0 &` in the next block. Thanks to Martin, I decided to 
revisit this section in the eyes of cross validation (which I didn't have in earlier versions of this kernel).
The interesting observation is that by applying the bias to all, the cross validation goes up
but the standard deviation of the cross validation also goes up, which confirms the
model gets noisier. With respect to public score, it appears the maintaining the standard deviation
low is more important. 

```{r, echo=TRUE, message=FALSE, warning=FALSE}
for ( i in 1:nrow(ticket_stats)) {
	plist <- which(comb$Ticket==ticket_stats$Ticket[i]);
	if(ticket_stats$na[i] > 0 & ticket_stats$l[i] > 1 & ticket_stats$c[i] > 0) {
			comb$SLogL[plist] <- comb$SLogL[plist] + 3;
	}
}
```

### The sconst variable was used to penalize singles prior to version 30

However, after testing a range of small constants (positive or negative other than 0), we concluded that this is 
being used by the final optimizer to differentiate singles from not singles 
rather than a being interpreted directly as penalty or reward.

Motivated Akshay's question in the comments section, I decided to dig a little deeper here.
I checked the differences in the output when **sconst** was set to zero. We can find that when we do so,
27 females in the Single category in Pclass 3 switch state from survived to perished. No other changes are observed in the output.
Earlier we commented that by just looking at Sex and Pclass chart, females in Pclass 3 had a fifty-fifty chance. This
extra piece of information breaks that balance and creates a big (favorable to a better cross validation and public score) swing
in the output. 

More comments and code to follow on this. Stay tuned.

```{r, echo=TRUE, message=FALSE, warning=FALSE}
sconst <- -2.1;
comb$SLogL[comb$GID=="Single"] <- comb$SLogL[comb$GID=="Single"] - sconst;
```

### By penalizing large group sizes (See TFreq vs Pclass vs SLogL graph)

```{r, echo=TRUE, message=FALSE, warning=FALSE}
comb$SLogL[comb$TFreq ==  7] <- comb$SLogL[comb$TFreq == 7]  - 3;
comb$SLogL[comb$TFreq ==  8] <- comb$SLogL[comb$TFreq == 8]  - 1;
comb$SLogL[comb$TFreq == 11] <- comb$SLogL[comb$TFreq == 11] - 3;
```

### By promoting high likelihood of survival for Minors in Pclass 1 and Pclass 2.

```{r, echo=TRUE, message=FALSE, warning=FALSE}
comb$SLogL[comb$Minor==1] <- 8;
```

## Plot SLogL after adjustments

Note how the previous steps diffused previously bundled areas. Now we may choose an optimizer algorithm to perform the final classification in the **conquer** stage of this process.

```{r, echo=TRUE, message=FALSE, warning=FALSE}
ggplot(comb[tr_idx,], aes(x=Pclass, y=SLogL)) + geom_jitter(aes(color=Survived)) + 
  facet_grid(  . ~ TFreq,  labeller=label_both) + labs(title="SLogL vs Pclass vs TFreq")
```

# Conquer

In this stage we simply perform the final prediction using just the target *Survived* and feature *SLogL* alone.  KNN is used
in the kernel but replacing it by 'rf' lets the optimization be done by the random forest algorithm. Random forest and other 
algorithms such as treebag all produce the same output. It indirectly demonstrates the SLogL feature after adjustments
seems to have a smooth distribution in space and many optimization algorithms have no trouble finding a solution for it.

## Transform Survived to a categorical target for binary classification

Survived will be treated as a categorical target.

```{r, message = FALSE}
comb$Survived <- factor(comb$Survived);
```

## Cross validation

The next chunk models, predicts and prints the cross validation. Cross
validation is an essential step in competitions to help up us not only
calibrate the parameters of our model but estimate the prediction accuracy with
unseen data.

```{r}
set.seed(2017);
trControl <- trainControl(method="repeatedcv", number=7, repeats = 5); 
fms <- formula("Survived ~ SLogL"); 
model_m <- train(fms, data = comb[tr_idx,],
	 metric="Accuracy", trControl = trControl, method = "knn"); 
comb$Pred <- predict(model_m, comb);
print(model_m$results)
```

Note how the cross validation Accuracy score is within very close range of what this kernel obtains as a public score (0.82296), which demonstrates
how over fitting is being avoided while achieving a high accuracy.

## Write submission file

```{r}
df_final <- data.frame(PassengerId = comb$PassengerId[-tr_idx], Survived=comb$Pred[-tr_idx]);
write.csv(df_final, paste0(outpath,"pred_divconq.csv"), row.names =F, quote=F)
```

# Conclusions 

A good public score can be achieved using the approach of engineering a single scalar (SLogL) and then running a final optimization on top of it. The used 
method can be seen as **boosting** or a form of **meta-algorithm**.
Cross validation of the model demonstrates that the achieved public score of **0.82296** is very close to predictions. 



 
