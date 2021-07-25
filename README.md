# Objective:
To build a model to predict if a passenger survived the sinking of the Titanic or not by using provided datasets.

# Running the Code:
To run the code keep train.csv and test.csv in the same folder as code.py . Run code.py.
This will print the prediction on the test set as the output and also generates result.csv containing the results.

# Methodology:

# Data Preprocessing:
We preprocess the dataset using Pandas. First we imported the data and tried to get some insights about it. We noticed that there were some missing values in the dataset. To check the number of missing values, we used

len(raw_data.loc[raw_data['Age'].isnull()])

>> 177

len(raw_data.loc[raw_data['Cabin'].isnull()])

>> 687

len(raw_data.loc[raw_data['Embarked'].isnull()])

>> 2

There were no missing values in any other column. 

Around 77% of the Cabin values are missing, which cannot be fixed. So we simply drop that column. Since it is difficult to convert Name and Ticket to numeric values. So we dropped those two columns too.   

To fix the missing values in age column we fill the missing values with the mean of all the given values.

Next, we have Embarked column which has three possible values S, C and Q. We represent this information using one-hot encoding. So we drop original Embarked column and add three new columns to the dataframe named isS, isC and isQ, which take the value True if the Embarked values is S, C or Q respectively. Here the missing values will not cause any problems because in such a case all the three columns will have the value False. 

The next problem is the range of values. All the values lie between 0 and 1, except for Age and Fare. So we make the values of Age and Fare range from 0 to 1 using

Age := Age - min(Age) / max(Age) - min(Age)

Fare := Fare - min(Fare) / max(Fare) - min(Fare)

So finally we have 890 training examples and 9 features, which have either numeric or binary values. The binary value can be easily converted to numeric values 0 and 1. 

# The Model:
We used Numpy to build the model. The given problem is a classification problem with a binary output. So, we use Logistic Regression to build our model.
In Logistic Regression the prediction P(x1,x2,..,xn) is given by

P(x1,x2,..,xn) = σ(t) = 1 / ( 1 + e^(-t)) 

where,                      

t = w1.x1 + w2.x2 + … + wn.xn + b

where w1, w2, w3, ...., wn and b are parameters whose values will be learned during training. 

We first vectorize the input to be able to use numpy. We create a 9 x 890 matrix X_train which contains a different training example in each column and the values of the 9 different features in 9 different rows. We also create a 1 x 890 matrix Y_train , which contains a label 0 or 1 in each different column for each training example. 

We have the parameter matrix w = [w1 w2 w3 …. w9] and a parameter b. We initialize all the parameters to 0. 

We have, T = w.X_train + [ b b b … b ](890 times b),    

which can be written as

T = w.X_train + b     

as numpy will broadcast b to necessary dimensions. 

Where T is 1 x 890 matrix with column containing the value of t for each different training example. 

Then we calculate Prediction = σ(T), where Prediction is the 1 x 890 containing the containing the prediction for each training example in each different column.

# The Cost Function:
We use the Binary Cross Entropy to calculate the loss or error in our Prediction. The Binary Cross Entropy is given by

L(pi,yi) = - yi.log(pi) - (1-yi).log(1-pi)

L(p,y) = (1/m)∑_(i = 1)^m- yi.log(pi) - (1-yi).log(1-pi)

Where m = 890 and yi and pi are labels and prediction for training example i. 

# Optimizing the Loss:
We use Gradient Descent to minimize our loss function. We have, 

w1 := w1 - lr.∂L(p,y)/∂w1

w2 := w2 - lr.∂L(p,y)/∂w2

....

w9 := w9 - lr.∂L(p,y)/∂w9

b := b - lr.∂L(p,y)/∂b

∂L(pi,yi)/∂ti = (∂L(pi,yi)/∂pi) . (∂pi/∂ti)

= (∂L(pi,yi)/∂pi) . σ’(ti) 

= (– yi/pi + (1 – yi)/(1 – pi)) . (pi( 1 - pi))

= pi - yi

In vectorized form we have, dT = Prediction - Y

where dT is matrix containing the the value ∂L(p,y)/∂t for each training example in each different column. 

∂L(pi,yi)/∂wj = (∂L(pi,yi)/∂ti) . (∂ti/∂wj)

∴ ∂L(pi,yi)/∂wj = (∂L(pi,yi)/∂ti) . wj

∴ ∂L(p,y)/∂wj = (1/m)∑_(i = 1)^m(∂L(pi,yi)/∂ti) . wj
	
∂L(pi,yi)/∂b = (∂L(pi,yi)/∂ti) . (∂ti/∂b)
  
∴ ∂L(pi,yi)/∂b = (∂L(pi,yi)/∂ti) . 1

∴ ∂L(p,y)/∂b = (1/m)∑_(i = 1)^m(∂L(pi,yi)/∂ti)

In Vectorized form, 

dw = dT.dot(X_train.transpose())/m

db = np.sum(dT,axis = 1,keepdims = True)/m

dw = [∂L(p,y)/∂w1 ∂L(p,y)/∂w2 ... ∂L(p,y)/∂w9]

db = [∂L(p,y)/∂b]

To update the parameters, we do 

w := w - lr.dw 

b := b - lr.db

We repeat the same process till the loss is minimized 

![image](https://user-images.githubusercontent.com/18099362/126904357-07a39932-709a-473c-a613-e67ec275b1a6.jpeg)

 
Loss becomes almost constant after 1000 iterations, thus we stop after 1000 iterations.

To choose the value of learning rate lr, we try different values of learning rate, and plot accuracy vs learning rate and pick the learning rate with maximum accuracy.

![image](https://user-images.githubusercontent.com/18099362/126904363-148219ec-6dad-4e5e-b230-123984c901fe.jpeg)

We find lr = 0.9 gives maximum accuracy. 

# Predicting the Final Output:
Our model outputs a number between 0 and 1. We say that the passenger survived if output is greater than 0.5 otherwise we say that he died. Thus in the Prediction matrix we replace every entry greater than 0.5 by 1 and every other entry by 0

Prediction = np.where(Prediction>0.5,1,0)

Then we calculate the accuracy by checking the number of correct predictions.

accuracy = (np.sum(np.where(Y_train ==Prediction,1,0))/Prediction.shape[1]) * 100

print(accuracy)

>> 80.3591470258137

To get the output for the test set, we just replace X_train by X_test and use our model to calculate the Prediction. We also replace nan values in Prediction by 0.5 . 

Prediction = np.where(np.isnan(Prediction),0.5,Prediction)


# Flowchart:
![Screen Shot 2021-07-25 at 8 53 57 PM](https://user-images.githubusercontent.com/18099362/126904396-7d3194b1-33a4-4db9-9bd1-499a8d9fb71e.png)
 

# Results:
We get an accuracy of 80.359 % on the training set. 

The results for the test set are contained in the result.csv file.
