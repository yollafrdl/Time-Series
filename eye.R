data("crqa", package = "crqa")
typeof(RDts1)
nrow(RDts1)
head(RDts1)
tail(RDts1)
barplot(table(RDts1), ylab="Count", col="darkblue")
plot(as.ts(RDts1), col="darkblue", ylab="Position")

library(mxnet)

seq_1 <- RDts1[1:100,]
seq_2 <- RDts1[101:200,]
seq_3 <- RDts1[201:300,]
data <- cbind(seq_1,seq_2,seq_3)

lab_1 <- RDts1[2:101,]
lab_2 <- RDts1[102:201,]
lab_3 <- RDts1[202:301,]
lab <- cbind(lab_1,lab_2,lab_3)

data_set = aperm((array(c(data), dim = c(100,3))))
labels_set = aperm(array(c(lab), dim = c(100,3)))
dim(data_set)
dim(labels_set)

trainDat <- list(data = data_set, label = labels_set)
summary(trainDat)

batch_size = 1
seq.len = 3
num.hidden = 25
num.embed = 2
num_layer = 1
num_round = 2
update.period = 1
learning.rate = 0.3
num_labels = 7
input_size = 7
moment = 0.9

require(mxnet)
mx.set.seed(0)

model <- mx.lstm(trainDat, eval.data = NULL, ctx = mx.cpu(),
                 num.round = num_round,
                 update.period = update.period,
                 num.lstm.layer = num_layer,
                 seq.len = seq.len,
                 num.hidden = num.hidden,
                 num.embed = num.embed,
                 num.label = num_labels,
                 batch.size = batch_size,
                 input.size = input_size,
                 initializer = mx.init.uniform(0.1),
                 learning.rate = learning.rate,
                 momentum = moment)

mx.set.seed(2018)
pred <- mx.lstm.inference(num.lstm.layer = num_layer,
                          input.size = input_size,
                          num.hidden = num.hidden,
                          num.embed = num.embed,
                          num.label = num_labels,
                          ctx = mx.cpu(),
                          arg.params = model$arg.params)

model_probs <- mx.lstm.forward(pres, RDts1[301,1])
model_probs$prob

RDts1[302,1]

probs <- 1:700
eye_class <- 1:100
dim(probs) <- c(num_labels,100)
for (i in (1:100)) {
  temp = as.numeric(RDts1[i + 302, 1])
  mx.set.seed(2018)
  model_prob <- mx.lstm.forward(pred, temp, FALSE)
  prob = as.array(model_prob$prob)
  eye_class[i] <- which.max(prob)
}

eye_class <- as.data.frame(eye_class)
eye_class[,1][eye_class[,1] == 7] <- 10
final_pred <- as.ts(eye_class[,1])
final_pred <- as.numeric(unlist(final_pred))
final_pred <- ts(matrix(final_pred),
                 start = c(1),
                 end = c(100),
                 frequency = 1)

result <- cbind(RDts1[303:402,1], final_pred)
plot(result)
