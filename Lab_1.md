
Getting Started with Recommender Systems
========================================

Almost everything we buy or consume today is influenced by some form of
recommendation; whether that\'s from friends, family, external reviews,
and, more recently, from the sources selling you the product. When you
log on to Netflix or Amazon Prime, for example, you will see a list of
movies and television shows the service thinks you will like based on
your past watching (and rating) history. Facebook suggests people it
thinks you may know and would probably like to add. It also curates a
News Feed for you based on the posts you\'ve liked, the people you\'ve
be-friended, and the pages you\'ve followed. Amazon recommends items to
you as you browse for a particular product. It shows you similar
products from a competing source and suggests auxiliary items
*frequently bought together* with the product.

So, it goes without saying that providing a good recommendation is at
the core of successful business for these companies. It is in Netflix\'s
best interests to engage you with content that you love so that you
continue to subscribe to its service; the more relevant the items Amazon
shows you, the greater your chances -- and volume -- of purchases will
be, which directly translates to greater profits. Equally, establishing
*friendship* is key to Facebook\'s power and influence as an almost
omnipotent social network, which it then uses to churn money out of
advertising.

In this introductory lab, we will acquaint ourselves with the world
of recommender systems, covering the following topics:

-   What is a recommender system? What can it do and not do?
-   The different types of recommender systems



Technical requirements
======================

You will be required to have Python installed on a system. Finally, to
use the Git repository of this course, the user needs to install Git.

What is a recommender system?
=============================

Recommender systems are pretty self-explanatory; as the name suggests,
they are systems or techniques that recommend or suggest a particular
product, service, or entity. However, these systems can be classified
into the following two categories, based on their approach to providing
recommendations.



The prediction problem
======================

In this version of the problem, we are given a matrix of *m* users and
*n* items. Each row of the matrix represents a user and each column
represents an item. The value of the cell in the i^th^ row and the j^th^
column denotes the rating given by user *i* to item *j*. This value is
usually denoted as r~ij~.

For instance, consider the matrix in the following screenshot:


![](./images/61ad9d1e-fc57-48f6-ad4c-fefab39324c4.png)


This matrix has seven users rating six items. Therefore, m = 7 and n =
6. User 1 has given the item 1 a rating of 4. Therefore, r~11~ = 4.

Let us now consider a more concrete example. Imagine you are Netflix and
you have a repository of 20,000 movies and 5,000 users. You have a
system in place that records every rating that each user gives to a
particular movie. In other words, you have the rating matrix (of shape
5,000 × 20,000) with you.

However, all your users will have seen only a fraction of the movies you
have available on your site; therefore, the matrix you have is sparse.
In other words, most of the entries in your matrix are empty, as most
users have not rated most of your movies.

The prediction problem, therefore, aims to predict these missing values
using all the information it has at its disposal (the ratings recorded,
data on movies, data on users, and so on). If it is able to predict the
missing values accurately, it will be able to give great
recommendations. For example, if user *i* has not used item *j*, but our
system predicts a very high rating (denoted by
![](./images/57a418b0-5fe0-41be-a202-275bc4162f83.png)~ij~),
it is highly likely that *i* will love *j* should they discover it
through the system.



The ranking problem
===================

Ranking is the more intuitive formulation of the recommendation problem.
Given a set of *n* items, the ranking problem tries to discern the top
*k* items to recommend to a particular user, utilizing all of the
information at its disposal.


![](./images/43494356-5ecc-4ae1-9e89-72ca91a248bc.png)


Imagine you are Airbnb, much like the preceding example. Your user has
input the specific things they are looking for in their host and the
space (such as their location, and budget). You want to display the top
10 results that satisfy those aforementioned conditions. This would be
an example of the ranking problem.

It is easy to see that the prediction problem often boils down to the
ranking problem. If we are able to predict missing values, we can
extract the top values and display them as our results.

In this course, we will look at both formulations and build systems that
effectively solve them.



Types of recommender systems
============================

In recommender systems, as with almost every other machine learning
problem, the techniques and models you use (and the success you enjoy)
are heavily dependent on the quantity and quality of the data you
possess. In this section, we will gain an overview of three of the most
popular types of recommender systems in decreasing order of data they
require to inorder function efficiently.



Collaborative filtering
=======================

Collaborative filtering leverages the power of community to provide
recommendations. Collaborative filters are one of the most popular
recommender models used in the industry and have found huge success for
companies such as Amazon. Collaborative filtering can be broadly
classified into two types.



User-based filtering
====================

The main idea behind user-based filtering is that if we are able to find
users that have bought and liked similar items in the past, they are
more likely to buy similar items in the future too. Therefore, these
models recommend items to a user that similar users have also liked.
Amazon\'s *Customers who bought this item also bought* is an example of
this filter, as shown in the following screenshot:


![](./images/67526260-a36e-4c9b-9356-8f00000a5085.png)


Imagine that Alice and Bob mostly like and dislike the same video games.
Now, imagine that a new video game has been launched on the market.
Let\'s say Alice bought the game and loved it. Since we have discerned
that their tastes in video games are extremely similar, it\'s likely
that Bob will like the game too; hence, the system recommends the new
video game to Bob.



Item-based filtering
====================

If a group of people have rated two items similarly, then the two items
must be similar. Therefore, if a person likes one particular item,
they\'re likely to be interested in the other item too. This is the
principle on which item-based filtering works. Again, Amazon makes good
use of this model by recommending products to you based on your browsing
and purchase history, as shown in the following screenshot:


![](./images/97e8384b-0e6c-4515-a5dc-900511facc4d.png)


Item-based filters, therefore, recommend items based on the past ratings
of users. For example, imagine that Alice, Bob, and Eve have all given
*War and Peace* and *The Picture of Dorian Gray* a rating of
excellent*.* Now, when someone buys *The Brothers Karamazov,* the system
will recommend *War and Peace* as it has identified that, in most cases,
if someone likes one of those books, they will like the other, too.



Shortcomings
============

One of the biggest prerequisites of a collaborative filtering system is
the availability of data of past activity. Amazon is able to leverage
collaborative filters so well because it has access to data concerning
millions of purchases from millions of users.

Therefore, collaborative filters suffer from what we call the **cold
start problem***.* Imagine you have started an e-commerce website -- to
build a good collaborative filtering system, you need data on a large
number of purchases from a large number of users. However, you don\'t
have either, and it\'s therefore difficult to build such a system from
the start.



Content-based systems
=====================

Unlike collaborative filters, content-based systems do not require data
relating to past activity. Instead, they provide recommendations based
on a user profile and metadata it has on particular items.

Netflix is an excellent example of the aforementioned system. The first
time you sign in to Netflix, it doesn\'t know what your likes and
dislikes are, so it is not in a position to find users similar to you
and recommend the movies and shows they have liked.


![](./images/272885cb-c998-496d-a642-c9172b282e11.png)


As shown in the previous screenshot, what Netflix does instead is ask
you to rate a few movies that you *have* watched before. Based on this
information and the metadata it already has on movies, it creates a
watchlist for you. For instance, if you enjoyed the *Harry Potter* and
*Narnia* movies, the content-based system can identify that you like
movies based on fantasy novels and will recommend a movie such as *Lord
of the Rings* to you.

However, since content-based systems don\'t leverage the power of the
community, they often come up with results that are not as impressive or
relevant as the ones offered by collaborative filters. In other words,
content-based systems usually provide recommendations that are
*obvious.* There is little novelty in a *Lord of the Rings*
recommendation if *Harry Potter* is your favorite movie.



Knowledge-based recommenders
============================


![](./images/840de401-e39f-49f8-a2b7-7fc6ddaac6b2.png)


Knowledge-based recommenders are used for items that are very rarely
bought. It is simply impossible to recommend such items based on past
purchasing activity or by building a user profile. Take real estate, for
instance. Real estate is usually a once-in-a-lifetime purchase for a
family. It is not possible to have a history of real estate purchases
for existing users to leverage into a collaborative filter, nor is it
always feasible to ask a user their real estate purchase history.

In such cases, you build a system that asks for certain specifics and
preferences and then provides recommendations that satisfy those
aforementioned conditions. In the real estate example, for instance, you
could ask the user about their requirements for a house, such as its
locality, their budget, the number of rooms, and the number of storeys,
and so on. Based on this information, you can then recommend properties
that will satisfy all of the above conditions.

Knowledge-based recommenders also suffer from the problem of low
novelty, however. Users know full-well what to expect from the results
and are seldom taken by surprise.



Hybrid recommenders
===================

As the name suggests, hybrid recommenders are robust systems that
combine various types of recommender models, including the ones we\'ve
already explained. As we\'ve seen in previous sections, each model has
its own set of advantages and disadvantages. Hybrid systems try to
nullify the disadvantage of one model against an advantage of another.

Let\'s consider the Netflix example again. When you sign in for the
first time, Netflix overcomes the cold start problem of collaborative
filters by using a content-based recommender, and, as you gradually
start watching and rating movies, it brings its collaborative filtering
mechanism into play. This is far more successful, so most practical
recommender systems are hybrid in nature.

In this course, we will build a recommender system of each type and will
examine all of the advantages and shortcomings described in the previous
sections.



Summary
=======

In this lab, we gained an overview of the world of recommender
systems. We saw two approaches to solving the recommendation problem;
namely, prediction and ranking. Finally, we examined the various types
of recommender systems and discussed their advantages and disadvantages.

In the next lab, we will learn to process data with pandas, the data
analysis library of choice in Python. This, in turn, will aid us in
building the various recommender systems we\'ve introduced.