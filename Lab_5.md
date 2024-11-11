
Getting Started with Data Mining Techniques
===========================================

In 2003, Linden, Smith, and York of Amazon.com published a paper
entitled Item-to-Item Collaborative Filtering, which explained how
product recommendations at Amazon work. Since then, this class of
algorithmg has gone on to dominate the industry standard for
recommendations. Every website or app with a sizeable user base, be it
Netflix, Amazon, or Facebook, makes use of some form of collaborative
filters to suggest items (which may be movies, products, or friends):


![](./images/a10e3bca-ab33-450c-a0f4-ddf773ad5652.jpg)



As described in the first lab, collaborative filters try to leverage
the power of the community to give reliable, relevant, and sometime,
even surprising recommendations. If Alice and Bob largely like the same
movies (say The Lion King, Aladdin, and Toy Story) and Alice also likes
Finding Nemo, it is extremely likely that Bob, who hasn\'t watched
Finding Nemo, will like it too.

We will be building powerful collaborative filters in the next lab.
However, before we do that, it is important that we have a good grasp of
the underlying techniques, principles, and algorithms that go into
building collaborative filters.

Therefore, in this lab, we will cover the following topics:

-   **Similarity measures**: Given two items, how do we mathematically
    quantify how different or similar they are to each other? Similarity
    measures help us in answering this question.\
    We have already made use of a similarity measure (the cosine score)
    while building our content recommendation engine. In this lab,
    we will be looking at a few other popular similarity scores.
-   **Dimensionality reduction**: When building collaborative filters,
    we are usually dealing with millions of users rating millions of
    items. In such cases, our user and item vectors are going to be of a
    dimension in the order of millions. To improve performance, speed up
    calculations, and avoid the curse of dimensionality, it is often a
    good idea to reduce the number of dimensions considerably, while
    retaining most of the information. This section of the lab will
    describe techniques that do just that.
-   **Supervised learning**: Supervised learning is a class of machine
    learning algorithm that makes use of label data to infer a mapping
    function that can then be used to predict the label (or class) of
    unlabeled data. We will be looking at some of the most popular
    supervised learning algorithms, such as support vector machines,
    logistic regression, decision trees, and ensembling.
-   **Clustering**: Clustering is a type of unsupervised learning where
    the algorithm tries to divide all the data points into a certain
    number of clusters. Therefore, without the use of a label dataset,
    the clustering algorithm is able to assign classes to all the
    unlabel points. In this section, we will be looking at k-means
    clustering, a simple but powerful algorithm popularly used in
    collaborative filters.
-   **Evaluation methods and metrics**: We will take a look at a few
    evaluation metrics that are used to gauge the performance of these
    algorithms. The metrics include accuracy, precision, and recall.

The topics covered in this lab merit an entire textbook. Since this
is a hands-on recommendation engine tutorial, we will not be delving too
deeply into the functioning of most of the algorithms. Nor will we code
them up from scratch. What we will do is gain an understanding of how
and when they work, their advantages and disadvantages, and their
easy-to-use implementations using the scikit-learn library.



Problem statement
=================

Collaborative filtering algorithms try to solve the prediction problem
(as described in the Lab 1,
*Getting Started with Recommender Systems*). In other words, we are
given a matrix of i users and j items. The value in the ith row and the
jth column (denoted by rij) denotes the rating given by user i to item
j:


![](./images/d6fd11cf-ee20-4af0-bcd8-bd9a3c3acc5b.png)




Matrix of i users and j items


<div>

Our job is to complete this matrix. In other words, we need to predict
all the cells in the matrix that we have no data for. For example, in
the preceding diagram, we are asked to predict whether user E will like
the music player item. To accomplish this task, some ratings are
available (such as User A liking the music player and video games)
whereas others are not (for instance, we do not know whether Users C and
D like video games).

</div>



Similarity measures
===================

From the rating matrix in the previous section, we see that every user
can be represented as a j-dimensional vector where the kth dimension
denotes the rating given by that user to the kth item. For instance, let
1 denote a like, -1 denote a dislike, and 0 denote no rating. Therefore,
user B can be represented as (0, 1, -1, -1). Similarly, every item can
also be represented as an i-dimensional vector where the kth dimension
denotes the rating given to that item by the kth user. The video games
item is therefore represented as (1, -1, 0, 0, -1).

We have already computed a similarity score for like-dimensional vectors
when we built our content-based recommendation engine. In this section,
we will take a look at the other similarity measures and also revisit
the cosine similarity score in the context of the other scores.



Euclidean distance
==================

The Euclidean distance can be defined as the length of the line segment
joining the two data points plotted on an *n*-dimensional Cartesian
plane. For example, consider two points plotted in a 2D plane:


![](./images/1c808a35-3c9d-4bbe-a6ae-e858a3961159.png)




Euclidean distance


The distance, d, between the two points gives us the Euclidean distance
and its formula in the 2D space is given in the preceding graph.

More generally, consider two *n*-dimensional points (or vectors):

-   **v1**: (q1, q2,\...., qn)
-   **v2**: (r1, r2,\....., rn)

Then, the Euclidean score is mathematically defined as:


![](./images/5327766a-fae3-4b13-8664-dfec476932a1.png)


Euclidean scores can take any value between 0 and infinity. The lower
the Euclidean score (or distance), the more similar the two vectors are
to each other. Let\'s now define a simple function using NumPy, which
allows us to compute the Euclidean distance between two *n*-dimensional
vectors using the aforementioned formula:


```
#Function to compute Euclidean Distance. 
def euclidean(v1, v2):
    
    #Convert 1-D Python lists to numpy vectors
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    #Compute vector which is the element wise square of the difference
    diff = np.power(np.array(v1)- np.array(v2), 2)
    
    #Perform summation of the elements of the above vector
    sigma_val = np.sum(diff)
    
    #Compute square root and return final Euclidean score
    euclid_score = np.sqrt(sigma_val)
    
    return euclid_score
```


Next, let\'s define three users who have rated five different movies:


```
#Define 3 users with ratings for 5 movies
u1 = [5,1,2,4,5]
u2 = [1,5,4,2,1]
u3 = [5,2,2,4,4]
```


From the ratings, we can see that users 1 and 2 have extremely different
tastes, whereas the tastes of users 1 and 3 are largely similar. Let\'s
see whether the Euclidean distance metric is able to capture this:


```
euclidean(u1, u2)

OUTPUT:
7.4833147735478827
```


The Euclidean distance between users 1 and 2 comes out to be
approximately 7.48:


```
euclidean(u1, u3)

OUTPUT:
1.4142135623730951
```


Users 1 and 3 have a much smaller Euclidean score between them than
users 1 and 2. Therefore, in this case, the Euclidean distance was able
to satisfactorily capture the relationships between our users.



Pearson correlation
===================

Consider two users, Alice and Bob, who have rated the same five movies.
Alice is extremely stingy with her ratings and never gives more than a 4
to any movie. On the other hand, Bob is more liberal and never gives
anything below a 2 when rating movies. Let\'s define the matrices
representing Alice and Bob and compute their Euclidean distance:


```
alice = [1,1,3,2,4]
bob = [2,2,4,3,5]

euclidean(alice, bob)

OUTPUT:
2.2360679774997898
```


We get a Euclidean distance of about 2.23. However, on closer
inspection, we see that Bob always gives a rating that is one higher
than Alice. Therefore, we can say that Alice and Bob\'s ratings are
extremely correlated. In other words, if we know Alice\'s rating for a
movie, we can compute Bob\'s rating for the same movie with high
accuracy (in this case, by just adding 1).

Consider another user, Eve, who has the polar opposite tastes to Alice:


```
eve = [5,5,3,4,2]

euclidean(eve, alice)

OUTPUT:
6.324555320336759
```


We get a very high score of 6.32, which indicates that the two people
are very dissimilar. If we used Euclidean distances, we would not be
able to do much beyond this. However, on inspection, we see that the sum
of Alice\'s and Eve\'s ratings for a movie always add up to 6.
Therefore, although very different people, one\'s rating can be used to
accurately predict the corresponding rating of the other. Mathematically
speaking, we say Alice\'s and Eve\'s ratings are strongly negatively
correlated.

Euclidean distances place emphasis on magnitude, and in the process, are
not able to gauge the degree of similarity or dissimilarity well. This
is where the Pearson correlation comes into the picture. The Pearson
correlation is a score between -1 and 1, where -1 indicates total
negative correlation (as in the case with Alice and Eve) and 1 indicates
total positive correlation (as in the case with Alice and Bob), whereas
0 indicates that the two entities are in no way correlated with each
other (or are independent of each other).

Mathematically, the Pearson correlation is defined as follows:


![](./images/d8d9ac65-0fe2-452f-a394-ecf52c6be691.png)


Here, ![](./images/cf6be388-6b92-4140-9b16-06c11f0d25d7.png)
denotes the mean of all the elements in vector *i*.

The SciPy package gives us access to a function that computes the
Pearson Similarity Scores:


```
from scipy.stats import pearsonr

pearsonr(alice, bob)

OUTPUT:
(1.0, 0.0)
pearsonr(alice, eve)

OUTPUT:
(-1.0, 0.0)
```


The first element of our list output is the Pearson score. We see that
Alice and Bob have the highest possible similarity score, whereas Alice
and Eve have the lowest possible score.\
Can you guess the similarity score for Bob and Eve?



Cosine similarity
=================

In the previous lab, we mathematically defined the cosine similarity
score and used it extensively while building our content-based
recommenders:


![](./images/6d9a8b2c-2455-46e6-af8d-24e85bb2a810.png)


Mathematically, the Cosine similarity is defined as follows:


![](./images/c2d6d4ef-6faa-45c4-a75d-db6d6faa9690.png)


The cosine similarity score computes the cosine of the angle between two
vectors in an *n*-dimensional space. When the cosine score is 1 (or
angle is 0), the vectors are exactly similar. On the other hand, a
cosine score of -1 (or angle 180 degrees) denotes that the two vectors
are exactly dissimilar to each other.

Now, consider two vectors, x and y, both with zero mean. We see that
when this is the case, the Pearson correlation score is exactly the same
as the cosine similarity Score. In other words, for centered vectors
with zero mean, the Pearson correlation is the cosine similarity score.

Different similarity scores are appropriate in different scenarios. For
cases where the magnitude is important, the Euclidean distance is an
appropriate metric to use. However, as we saw in the case described in
the Pearson correlation subsection, magnitude is not as important to us
as correlation. Therefore, we will be using the Pearson and the cosine
similarity scores when building our filters.



Clustering
==========

One of the main ideas behind collaborative filtering is that if user A
has the same opinion of a product as user B, then A is also more likely
to have the same opinion as B on another product than that of a randomly
chosen user.

Clustering is one of the most popular techniques used in
collaborative-filtering algorithms. It is a type of unsupervised
learning that groups data points into different classes in such a way
that data points belonging to a particular class are more similar to
each other than data points belonging to different classes:


![](./images/f30db156-43ab-479f-aa83-cdb67d2f265f.png)



For example, imagine that all our users were plotted on a
two-dimensional Cartesian plane, as shown in the preceding graph. The
job of a clustering algorithm is to assign classes to every point on
this plane. Just like the similarity measures, there is no one
clustering algorithm to rule them all. Each algorithm has its specific
use case and is suitable only in certain problems. In this section, we
will be looking only at the k-means clustering algorithm, which will
perform a satisfactory job is assigning classes to the collection of
preceding points. We will also see a case where k-means will not prove
to be suitable.



k-means clustering
==================

The k-means algorithm is one of the simplest yet most popular machine
learning algorithms. It takes in the data points and the number of
clusters (k) as input.

Next, it randomly plots k different points on the plane (called
centroids). After the k centroids are randomly plotted, the following
two steps are repeatedly performed until there is no further change in
the set of k centroids:

-   Assignment of points to the centroids: Every data point is assigned
    to the centroid that is the closest to it. The collection of data
    points assigned to a particular centroid is called a cluster.
    Therefore, the assignment of points to k centroids results in the
    formation of k clusters.
-   Reassignment of centroids: In the next step, the centroid of every
    cluster is recomputed to be the center of the cluster (or the
    average of all the points in the cluster). All the data points are
    then reassigned to the new centroids:


![](./images/df507754-0da0-43a7-8918-9ec46d18fe33.png)


The preceding screenshot shows a visualization of the steps involved in
a k-means clustering algorithm, with the number of assigned clusters as
two.


We will not be implementing the k-means algorithm from scratch. Instead,
we will use its implementation provided by scikit-learn. As a first
step, let\'s access the data points as plotted in the beginning of this
section:


```
#Import the function that enables us to plot clusters
from sklearn.datasets.samples_generator import make_blobs

#Get points such that they form 3 visually separable clusters
X, y = make_blobs(n_samples=300, centers=3,
                       cluster_std=0.50, random_state=0)


#Plot the points on a scatterplot
plt.scatter(X[:, 0], X[:, 1], s=50)
```


One of the most important steps while using the k-means algorithm is
determining the number of clusters. In this case, it can be clearly seen
from the plot (and the code) that we\'ve plotted the points in such a
way that they form three clearly separable clusters. Let\'s now apply
the k-means algorithm via scikit-learn and assess its performance:


```
#Import the K-Means Class
from sklearn.cluster import KMeans

#Initializr the K-Means object. Set number of clusters to 3, 
#centroid initilalization as 'random' and maximum iterations to 10
kmeans = KMeans(n_clusters=3, init='random', max_iter=10)

#Compute the K-Means clustering 
kmeans.fit(X)

#Predict the classes for every point
y_pred = kmeans.predict(X)

#Plot the data points again but with different colors for different classes
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50)

#Get the list of the final centroids
centroids = kmeans.cluster_centers_

#Plot the centroids onto the same scatterplot.
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=100, marker='X')
```


We see that the algorithm proves to be extremely successful in
identifying the three clusters. The three final centroids are also
marked with an X on the plot:


![](./images/60b43706-0325-4a32-a270-ec3193c351f3.png)




Choosing k
==========

As stated in the previous subsection, choosing a good value of k is
vital to the success of the k-means clustering algorithm. The number of
clusters can be anywhere between 1 and the total number of data points
(where each point is assigned to its own cluster).

Data in the real world is seldom of the type explored previously, where
the points formed well defined, visually separable clusters on a
two-dimensional plane. There are several methods available to determine
a good value of K. In this section, we will explore the Elbow method of
determining k.

The Elbow method computes the sum of squares for each value of k and
chooses the elbow point of the sum-of-squares v/s K plot as the best
value for k. The elbow point is defined as the value of k at which the
sum-of-squares value for every subsequent k starts decreasing much more
slowly.

The sum of squares value is defined as the sum of the distances of each
data point to the centroid of the cluster to which it was assigned.
Mathematically, it is expressed as follows:


![](./images/209c345a-a658-4081-9f33-8d91c766aa37.png)


Here, Ck is the kth cluster and uk is the corresponding centroid of Ck.

Fortunately for us, scikit-learn\'s implementation of k-means
automatically computes the value of sum-of-squares when it is computing
the clusters. Let\'s now visualize the Elbow plot for our data and
determine the best value of K:


```
#List that will hold the sum of square values for different cluster sizes
ss = []

#We will compute SS for cluster sizes between 1 and 8.
for i in range(1,9):
    
    #Initialize the KMeans object and call the fit method to compute clusters 
    kmeans = KMeans(n_clusters=i, random_state=0, max_iter=10, init='random').fit(X)
    
    #Append the value of SS for a particular iteration into the ss list
    ss.append(kmeans.inertia_)

#Plot the Elbow Plot of SS v/s K
sns.pointplot(x=[j for j in range(1,9)], y=ss)
```



![](./images/6b410c80-031b-44d6-9e5a-8bb5f1450210.png)



From the plot, it is clear that the Elbow is at K=3. From what we
visualized earlier, we know that this is indeed the optimum number of
clusters for this data.



Other clustering algorithms
===========================

The k-means algorithm, although very powerful, is not ideal for every
use case.

To illustrate, let\'s construct a plot with two half moons. Like the
preceding blobs, scikit-learn gives us a convenient function to plot
half-moon clusters:


```
#Import the half moon function from scikit-learn
from sklearn.datasets import make_moons

#Get access to points using the make_moons function
X_m, y_m = make_moons(200, noise=.05, random_state=0)

#Plot the two half moon clusters
plt.scatter(X_m[:, 0], X_m[:, 1], s=50)
```



![](./images/c23431ea-c235-4672-8fbc-12a742546d04.png)



Will the k-means algorithm be able to figure out the two half moons
correctly? Let\'s find out:


```
#Initialize K-Means Object with K=2 (for two half moons) and fit it to our data
kmm = KMeans(n_clusters=2, init='random', max_iter=10)
kmm.fit(X_m)

#Predict the classes for the data points
y_m_pred = kmm.predict(X_m)

#Plot the colored clusters as identified by K-Means
plt.scatter(X_m[:, 0], X_m[:, 1], c=y_m_pred, s=50)
```


Let\'s now visualize what k-means thinks the two clusters that exist for
this set of data points are:


![](./images/f399828f-41b5-48ce-b1c3-e37c170b798a.png)



We see that the k-means algorithm doesn\'t do a very good job of
identifying the correct clusters. For clusters such as these half moons,
another algorithm, called spectral clustering, with nearest-neighbor,
affinity performs much better.

We will not go into the workings of spectral clustering. Instead, we
will use its scikit-learn implementation and assess its performance
directly:


```
#Import Spectral Clustering from scikit-learn
from sklearn.cluster import SpectralClustering

#Define the Spectral Clustering Model
model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors')

#Fit and predict the labels
y_m_sc = model.fit_predict(X_m)

#Plot the colored clusters as identified by Spectral Clustering
plt.scatter(X_m[:, 0], X_m[:, 1], c=y_m_sc, s=50)
```



![](./images/43fa2b76-f6b8-46fb-8c2d-4535e14d548e.png)



We see that spectral clustering does a very good job of identifying the
half-moon clusters.

We have seen that different clustering algorithms are appropriate in
different cases. The same applies to cases of collaborative filters. For
instance, the surprise package, which we will visit in the next lab,
has an implementation of a collaborative filter that makes use of yet
another clustering algorithm, called co-clustering. We will wrap up our
discussion of clustering and move on to another important data mining
technique: dimensionality reduction.



Dimensionality reduction
========================

Most machine learning algorithms tend to perform poorly as the number of
dimensions in the data increases. This phenomenon is often known as the
curse of dimensionality. Therefore, it is a good idea to reduce the
number of features available in the data, while retaining the maximum
amount of information possible. There are two ways to achieve this:

-   **Feature selection**: This method involves identifying the features
    that have the least predictive power and dropping them altogether.
    Therefore, feature selection involves identifying a subset of
    features that is most important for that particular use case. An
    important distinction of feature selection is that it maintains the
    original meaning of every retained feature. For example, let\'s say
    we have a housing dataset with price, area, and number of rooms as
    features. Now, if we were to drop the area feature, the remaining
    price and number of rooms features will still mean what they did
    originally.
-   **Feature extraction**: Feature extraction takes in *m*-dimensional
    data and transforms it into an *n*-dimensional output space (usually
    where *m* \>\> *n*), while retaining most of the information.
    However, in doing so, it creates new features that have no inherent
    meaning. For example, if we took the same housing dataset and used
    feature extraction to output it into a 2D space, the new features
    won\'t mean price, area, or number of rooms. They will be devoid of
    any meaning.

In this section, we will take a look at an important feature-extraction
method: **Principal component analysis** (or **PCA**).



Principal component analysis
============================

**Principal component analysis** is an unsupervised feature extraction
algorithm that takes in *m*-dimensional input to create a set of *n*
(*m* \>\> *n*) linearly uncorrelated variables (called principal
components) in such a way that the *n* dimensions lose as little
variance (or information) as possible due to the loss of the (*m*-*n*)
dimensions.

The linear transformation in PCA is done in such a way that the first
principal component holds the maximum variance (or information). It does
so by considering those variables that are highly correlated to each
other. Every principal component has more variance than every succeeding
component and is orthogonal to the preceding component.

Consider a three-dimensional space where two features are highly
correlated to each other and relatively uncorrelated to the third:


![](./images/9f8c85c1-cbc2-44e8-8b2a-9c4a34fcd4e0.png)


Let\'s say that we want to convert this into a two-dimensional space. To
do this, PCA tries to identify the first principal component, which will
hold the maximum possible variance. It does so by defining a new
dimension using the two highly correlated variables. Now, it tries to
define the next dimension in such a way that it holds the maximum
variance, is orthogonal to the first principal component constructed,
and also is uncorrelated to it. The two new dimensions (or principal
components), PC 1 and PC 2, are shown in the preceding figure.

Understanding the PCA algorithm requires linear algebraic concepts that
are beyond the scope of this course. Instead, we will use the black box
implementation of PCA that [scikit-learn] gives us and consider a
use case with the well-known Iris dataset.

The first step is to load the Iris dataset from the UCI machine learning
repository into a pandas DataFrame:


```
# Load the Iris dataset into Pandas DataFrame
iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", 
                 names=['sepal_length','sepal_width','petal_length','petal_width','class'])

#Display the head of the dataframe
iris.head()
```



![](./images/6a8745dd-6316-4c05-9afb-81f42946e2d7.png)


The PCA algorithm is extremely sensitive to scale. Therefore, we are
going to scale all the features in such a way that they have a mean of 0
and a variance of 1:


```
#Import Standard Scaler from scikit-learn
from sklearn.preprocessing import StandardScaler

#Separate the features and the class
X = iris.drop('class', axis=1)
y = iris['class']

# Scale the features of X
X = pd.DataFrame(StandardScaler().fit_transform(X), 
                 columns = ['sepal_length','sepal_width','petal_length','petal_width'])

X.head()
```



![](./images/0365714a-25dc-45f9-a203-f1eddb655995.png)


We\'re now in a good place to apply the PCA algorithm. Let\'s transform
our data into the two-dimensional space:


```
#Import PCA
from sklearn.decomposition import PCA

#Intialize a PCA object to transform into the 2D Space.
pca = PCA(n_components=2)

#Apply PCA
pca_iris = pca.fit_transform(X)
pca_iris = pd.DataFrame(data = pca_iris, columns = ['PC1', 'PC2'])

pca_iris.head()
```



![](./images/d5b1f7a0-f605-4645-a932-062be42e66e6.png)


The [scikit-Learn]\'s PCA implementation also gives us information
about the ratio of variance contained by each principal component:


```
pca.explained_variance_ratio

OUTPUT:
array([ 0.72770452, 0.23030523])

```


We see that the first principal component holds about 72.8% of the
information, whereas the second principal component holds about 23.3%.
In total, 95.8% of the information is retained, whereas 4.2% of the
information is lost in removing two dimensions.

Finally, let\'s visualize our data points by class in the new 2D plane:


```
#Concatenate the class variable
pca_iris = pd.concat([pca_iris, y], axis = 1)

#Display the scatterplot
sns.lmplot(x='PC1', y='PC2', data=pca_iris, hue='class', fit_reg=False)
```



![](./images/d48fdcaf-da25-44dc-826b-40695edab808.png)





Other dimensionality reduction techniques
=========================================



Linear-discriminant analysis
============================

Like PCA, linear-discriminant analysis is a linear transformation method
that aims to transform *m*-dimensional data into an *n*-dimensional
output space.

However, unlike PCA, which tries to retain the maximum information, LDA
aims to identify a set of *n* features that result in the maximum
separation (or discrimination) of classes. Since LDA requires labeled
data in order to determine its components, it is a type of supervised
learning algorithm.

Let\'s now apply the LDA algorithm to the Iris dataset:


```
#Import LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#Define the LDA Object to have two components
lda = LinearDiscriminantAnalysis(n_components = 2)

#Apply LDA
lda_iris = lda.fit_transform(X, y)
lda_iris = pd.DataFrame(data = lda_iris, columns = ['C1', 'C2'])

#Concatenate the class variable
lda_iris = pd.concat([lda_iris, y], axis = 1)

#Display the scatterplot
sns.lmplot(x='C1', y='C2', data=lda_iris, hue='class', fit_reg=False)

```



![](./images/902577ac-de3d-4433-89f0-5f0c0ee1f823.png)



We see that the classes are much more separable than in PCA.



Singular value decomposition
============================

Singular value decomposition, or SVD, is a type of matrix analysis
technique that allows us to represent a high-dimensional matrix in a
lower dimension. SVD achieves this by identifying and removing the less
important parts of the matrix and producing an approximation in the
desired number of dimensions.

The SVD approach to collaborative filtering was first proposed by Simon
Funk and proved to be extremely popular and effective during the Netflix
prize competition. Unfortunately, understanding SVD requires a grasp of
linear algebraic topics that are beyond the scope of this course. However,
we will use a black box implementation of the SVD collaborative filter
as provided by the [surprise] package in the next lab.



Supervised learning
===================

Supervised learning is a class of machine learning algorithm that takes
in a series of vectors and their corresponding output (a continuous
value or a class) as input, and produces an inferred function that can
be used to map new examples.

An important precondition for using supervised learning is the
availability of labeled data. In other words, it is necessary that we
have access to input for which we already know the correct output.

Supervised learning can be classified into two types: classification and
regression. A classification problem has a discrete set of values as the
target variable (for instance, a like and a dislike), whereas a
regression problem has a continuous value as its target (for instance,
an average rating between one and five).

Consider the rating matrix defined earlier. It is possible to treat
(*m-1*) columns as the input and the m^th^ column as the target
variable. In this way, it should be possible to predict an unavailable
value in the m^th^ column by passing in the corresponding (m-1)
dimensional vector.

Supervised learning is one of the most mature subfields of machine
learning and, as a result, there are plenty of potent algorithms
available for performing accurate predictions. In this section, we will
look at some of the most popular algorithms used successfully in a
variety of applications (including collaborative filters).



k-nearest neighbors
===================

**k-nearest neighbors** (**k-NN**) is perhaps the simplest machine
learning algorithm. In the case of classification, it assigns a class to
a particular data point by a majority vote of its *k* nearest neighbors.
In other words, the data point is assigned the class that is the most
common among its k-nearest neighbors. In the case of regression, it
computes the average value for the target variable based on its
k-nearest neighbors.

Unlike most machine learning algorithms, k-NN is non-parametric and lazy
in nature. The former means that k-NN does not make any underlying
assumptions about the distribution of the data. In other words, the
model structure is determined by the data. The latter means that k-NN
undergoes virtually no training. It only computes the k-nearest
neighbors of a particular point in the prediction phase. This also means
that the k-NN model needs to have access to the training data at all
times and cannot discard it during prediction like its sister
algorithms.



Classification
==============


![](./images/d37d07d3-2ce0-4165-ba28-5157422bb1f5.png)



k-NN classification is best explained with the help of an example.
Consider a dataset that has binary classes (represented as the blue
squares and the red triangles). k-NN now plots this into *n*-dimensional
space (in this case, two dimensions).

Let\'s say we want to predict the class of the green circle. Before the
k-NN algorithm can make predictions, it needs to know the number of
nearest neighbors that have to be taken into consideration (the value of
*k*). *k* is usually odd (to avoid ties in the case of binary
classification).

Consider the case where *k=3*.

k-NN computes the distance metric (usually the Euclidean distance) from
the green circle to every other point in the training dataset and
selects the three data points that are closest to it. In this case,
these are the points contained in the solid inner circle.\
The next step is to determine the majority class among the three points.
There are two red triangles and one blue square. Therefore, the green
circle is assigned the class of red triangle.

Now, consider the case where *k=5*.

In this case, the nearest neighbors are all the points contained within
the dotted outer circle. This time around, we have two red triangles and
three blue squares. Therefore, the green circle is assigned the class of
blue square.

From the preceding case, it is clear that the value of *k* is extremely
significant in determining the final class assigned to a data point. It
is often a good practice to test different values of *k* and assess its
performance with your cross-validation and test datasets.



Regression
==========

k-NN regression works in almost the same way. Instead of classes, we
compute the property values of the k-NN.

Imagine that we have a housing dataset and we\'re trying to predict the
price of a house. The price of a particular house will therefore be
determined by the average of the prices of the houses of its *k* nearest
neighbors. As with classification, the final target value may differ
depending on the value of *k*.


For the rest of the algorithms in this section, we will go through only
the classification process. However, just like k-NN, most algorithms
require only very slight modifications to be suitable for use in a
regression problem.




Support vector machines
=======================

The support vector machine is one of the most popular classification
algorithms used in the industry. It takes in an *n*-dimensional dataset
as input and constructs an (*n-1*) dimensional hyperplane in such a way
that there is maximum separation of classes.

Consider the visualization of a binary dataset in the following
screenshot:


![](./images/4cb564ba-4cc1-4bb3-aa45-4849836ccaf5.png)


The preceding graph shows three possible hyperplanes (the straight
lines) that separate the two classes. However, the solid line is the one
with the maximum margin. In other words, it is the hyperplane that
maximally separates the two classes. Also, it divides the entire plane
into two regions. Any point below the hyperplane will be classified as a
red square, and any point above will be classified as a blue circle.

The SVM model is only dependent on [support vectors]*;* these are
the points that determine the maximum margin possible between the two
classes. In the preceding graph, the filled squares and circles are the
support vectors. The rest of the points do not have an effect on the
workings of the SVM:


![](./images/122d0c7b-0c42-4c81-8483-c8a36025727e.png)


SVMs are also capable of separating classes that are not linearly
separable (such as in the preceding figure). It does so with special
tools, called radial kernel functions, that plot the points in a higher
dimension and attempt to construct a maximum margin hyperplane there.



Decision trees
==============

Decision trees are extremely fast and simple tree-based algorithms that
branch out on features that result in the largest information gain*.*
Decision trees, although not very accurate, are extremely interpretable.

We will not delve into the inner workings of the decision tree, but we
will see it in action via a visualization:


![](./images/dc4b45e3-bf2a-43ab-84ad-b2fda443aa18.png)


Let\'s say we want to classify the Iris dataset using a decision tree. A
decision tree performing the classification is shown in the preceding
diagram. We start at the top and go deeper into the tree until we reach
a leaf node.

For example, if the petal width of a flower is less than 0.8 cm, we
reach a leaf node and it gets classified as setosa*.* If not, it goes
into the other branch and the process continues until a leaf node is
reached.

Decision trees have an element of randomness in their workings and come
up with different conditions in different iterations. As stated before,
they are also not very accurate in their predictions. However, their
randomness and fast execution make them extremely popular in ensemble
models, which will be explained in the next section.



Ensembling
==========

The main idea behind ensembling is that the predictive power of multiple
algorithms is much greater than a single algorithm. Decision trees are
the most common base algorithm used when building ensembling models.



Bagging and random forests
==========================

Bagging is short for bootstrap aggregating. Like most other ensemble
methods, it averages over a large number of base classification models
and averages their results to deliver its final prediction.

These are the steps involved in building a bagging model:

1.  A certain percentage of the data points are sampled (say 10%). The
    Sampling is done with replacement. In other words, a particular data
    point can appear in multiple iterations.
2.  A baseline classification model (typically a decision tree) is
    trained on this sampled data.
3.  This process is repeated until *n* number of models are trained. The
    final prediction delivered by the bagging model is the average of
    all the predictions of all the base models.

An improvement on the bagging model is the random forest model. In
addition to sampling data points, the random forest ensemble method also
forces each baseline model to randomly select a subset of the features
(usually a number equal to the square root of the total number of
features):


![](./images/ef09911f-dd49-461f-9d27-baa0ad54e2a6.png)


Selecting a subset of samples, as well as features, to build the
baseline decision trees greatly enhances the randomness of each
individual tree. This, in turn, increases the robustness of the random
forest and allows it to perform extremely well with noisy data.

Additionally, building baseline models from a subset of features and
analyzing their contribution to the final prediction also allows the
random forest to determine the importance of each feature. It is
therefore possible to perform feature-selection using random forests
(recall that feature-selection is a type of dimensionality reduction).



Boosting
========

The bagging and the random forest models train baseline models that are
completely independent of each other. Therefore, they do not learn from
the mistakes that each learner has made. This is where boosting comes
into play.

Like random forests, boosting models build a baseline model using a
subset of samples and features. However, while building the next
learners, the boosting model tries to rectify the mistakes that the
previous learners made. Different boosting algorithms do this in
different ways.

For example, the original boosting algorithm simply added 50% of the
misclassified samples to the second learner, and all the samples that
the first two learners disagree upon to build the third and final
learner. This ensemble of three learners was then used to make
predictions.

Boosting algorithms are extremely robust and routinely provide high
performance. This makes them extremely popular in data science
competitions and, as far as we are concerned, in building powerful
collaborative filters.

The [scikit-learn] gives us access to implementations of all the
algorithms described in this section. The usage of every algorithm is
almost the same. As an illustration, let\'s apply gradient boosting to
classify the Iris dataset:


```
#Divide the dataset into the feature dataframe and the target class series.
X, y = iris.drop('class', axis=1), iris['class']

#Split the data into training and test datasets. 
#We will train on 75% of the data and assess our performance on 25% of the data

#Import the splitting function
from sklearn.model_selection import train_test_split

#Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#Import the Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

#Apply Gradient Boosting to the training data
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)

#Compute the accuracy on the test set
gbc.score(X_test, y_test)

OUTPUT:
0.97368421052631582
```


We see that the classifier achieves a [97.3]% accuracy on
the unseen test data. Like random forests, gradient boosting machines
are able to gauge the predictive power of each feature. Let\'s plot the
feature importances of the Iris dataset:


```
#Display a bar plot of feature importances
sns.barplot(x= ['sepal_length','sepal_width','petal_length','petal_width'],       y=gbc.feature_importances_)

```



![](./images/66f0e4eb-ffbc-4c0d-9549-39053234113c.png)





Evaluation metrics
==================

In this section, we will take a look at a few metrics that will allow us
to mathematically quantify the performance of our classifiers,
regressors, and filters.



Accuracy
========

Accuracy is the most widely used metric to gauge the performance of a
classification model. It is the ratio of the number of correct
predictions to the total number of predictions made by the model:


![](./images/c4e870eb-dca2-45be-900d-2439cd5b75f4.png)




Root mean square error
======================

The **Root Mean Square Error** (or **RMSE**) is a metric widely used to
gauge the performance of regressors. Mathematically, it is represented
as follows:


![](./images/619cb11f-b47a-428e-92d9-05bf08b0a841.png)


Here,
![](./images/7ca4e738-dbfe-4c82-9f9c-d6ac7ff621a0.png)
is the i^th^ real target value and
![](./images/7ae9ea8f-9150-4c9a-8028-df08658eff7e.png)
is the i^th^ predicted target value.



Binary classification metrics
=============================

Sometimes, accuracy does not give us a good estimate of the performance
of a model.

For instance, consider a binary class dataset where 99% of the data
belongs to one class and only 1% of the data belongs to the other class.
Now, if a classifier were to always predict the majority class for every
data point, it would have 99% accuracy. But that wouldn\'t mean that the
classifier is performing well.

For such cases, we make use of other metrics. To understand them, we
first need to define a few terms:

-   **True positive** (**TP**): True positive refers to all cases where
    the actual and the predicted classes are both positive
-   **True negative** (**TN**): True negative refers to all cases where
    the actual and the predicted classes are both negative
-   **False positive** (**FP**): These are all the cases where the
    actual class is negative but the predicted class is positive
-   **False negative** (**FN**): These are all the cases where the
    actual class is positive but the predicted class is negative

To illustrate, consider a test that tries to determine whether a person
has cancer. If the test predicts that a person does have cancer when in
fact they don\'t, it is a false positive. On the other hand, if the test
fails to detect cancer in a person actually suffering from it, it is a
false negative.



Precision
=========

The precision is the ratio of the number of positive cases that were
correct to all the cases that were identified as positive.
Mathematically, it looks like this:


![](./images/b5aa14fe-1440-4bad-bc9a-f602f717df1a.png)




Recall
======

The recall is the ratio of the number of positive cases that were
identified to the all positive cases present in the dataset:


![](./images/44d25d38-9af6-489d-8231-4f4aaa40ae3a.png)




F1 score
========

The F1 score is a metric that conveys the balance between precision and
recall. It is the harmonic mean of the precision and recall. An F1 score
of 1 implies perfect precision and recall, whereas a score of 0 implies
precision and recall are not possible:


![](./images/dcd94ad1-96f6-4e27-84c9-d6f42e1efee2.png)




Summary
=======

In this lab, we have covered a lot of topics that will help us to
build powerful collaborative filters. We took a look at clustering, a
form of unsupervised learning algorithm that could help us to segregate
users into well defined clusters. Next, we went through a few
dimensionality reduction techniques to overcome the curse of
dimensionality and improve the performance of our learning algorithms.

The subsequent section dealt with supervised learning algorithms, and
finally we ended the lab with a brief overview of various evaluation
metrics.

The topics covered in this lab merit an entire course and we did not
analyze the techniques in the depth usually required of machine learning
engineers. However, what we have learned in this lab should be
sufficient to help us build and understand collaborative filters, which
is one of the main objectives of this course. In case you\'re interested,
a more detailed treatment of the topics presented in this lab is
available in an excellent course entitled *Python Machine Learning* by
Sebastian Thrun.
