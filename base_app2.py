"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/TfidfVectorizer.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Climate Change Belief Analysis")
	st.subheader("Climate change tweet classification")
	st.image('resources/imgs/cover photo.jpeg') 

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Problem statement","Information", "EDA", "Model description","Prediction"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("Since the beginning of industrialisation, however, the composition of the atmosphere has changed as a result of greenhouse gas emissions. This global warming caused by human beings intensifies the natural greenhouse effect and is leading to detectable changes to the climate. ")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page
	if selection == "Problem statement":
		st.info("Problem statement of climate change")
		# You can read a markdown file from supporting resources folder
		st.markdown("We are given a task to predict based on peoples opinions on climate change whether or not they believe if it is real or not. This would add value to companies market research efforts in estimating how their product or services may be received.We are required to use one of the machine learning models Classification to classify whether or not a prson believes in climate change based on the data collected on tweeter.")
	if selection == "EDA":
		st.info("Exploratory Data Analysis")
		st.subheader("distribution of data")
		st.image('./Bar chart.png') 
		st.markdown("by exploring the dataset we can tell that the dataset cointains few data for negative and neutral labels and higher distribution for positive and highly positive")
		st.subheader("Most common words")
		st.image('./Word cloud.png') 
		st.markdown("We can see that some people do believe and some don't belive in climate change. Government support is seen as an approach to address the challenge. Fighting climate change is among the headlines. Climate change and global warming are the most frequent ones. Donald Trump doesn't believe in climate change.")
		st.subheader("distribution of words")
		st.image('./Word count.png') 
		st.markdown("We can see that the word climate appears most in positive hashtags")
	if selection == "Model description":
		st.info("Description")
		st.subheader("LogisticRegression")
		# You can read a markdown file from supporting resources folder
		st.markdown("Logistic Regression uses the probability of a data point to belonging to a certain class to classify each datapoint to it's best estimated class. It assumes that the outcome is a binary or dichotomous variable e.g(yes vs no, positive vs negative, 1 vs 0), there is a linear relationship between the logit of the outcome and each predictor variables, there are no influential values e.g(outliers) in the continuious predictors.")
		st.subheader("naive Bayes")
		st.markdown("Naive Bayes is a classification algorithm that uses the principle of Bayes theorem to make classifications and assumes independent variables to be statistically independent from each other.")
		st.subheader("Linear Support Vector")
		st.markdown("Support Vector Machine(SVM) seeks the best decision boundary which separates two classes with the highest generalization ability. SVM wants the smallest distance between data points and the decision boundary to be as large as possible. This alogrithm also comes with parameters that can be used to obtain optimal results.")
	# Building out the predication page
	if selection == "Prediction":
		st.subheader("Prediction")
		st.info("LogisticRegression()")
		st.info("MultinomialNB()")
		st.info("LinearSVC()")

		
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","The three base models used in our analysis were logistic regression, naive Bayes and linear support vector. Logistic regression performed best")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("models/Random_Forest_Classifier.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
