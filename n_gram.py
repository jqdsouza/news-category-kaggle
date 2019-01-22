import pandas as pd 
import nltk, os, re, string, time

from pandas import ExcelWriter
from nltk.util import ngrams 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import Word
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')

"""
Author: Justin D'Souza 

Dependencies: python -m textblob.download_corpora

Estimated time for script implementation:  30 mins
Actual time for script implementation:     25 mins (8:00PM - 8:25PM, Jan. 20 2018)
Program execution time : 				  ~2.5 mins 
"""

def get_unigram_distribution(df):
	"""
	Gets unigram frequency distribution.

	Args:
		df: input dataframe
	"""

	unigram_df = df.text.str.split(expand=True).stack().value_counts()
	unigram_df = unigram_df.reset_index()
	unigram_df.columns = ['unigram','frequency']

	return unigram_df

def get_ngram_distribution(df, n):
	"""
	Gets frequency distribution of n-grams.

	Args:
		df: input dataframe
		n: 	integer denoting n-gram type
	"""

	counts = Counter()

	for text in df['tokenized_text']:
		counts.update(nltk.ngrams(text, n))

	freq_df = pd.DataFrame.from_dict(counts, orient = 'index')
	freq_df = freq_df.reset_index()
	freq_df.columns = ['{0}-gram'.format(n),'frequency']
	freq_df = freq_df.sort_values('frequency', ascending=False)

	return freq_df

def replace_bad_chars(string):
    return re.sub(r'[^a-zA-Z0-9\s]', ' ', string)

def preprocess_data(df):
	"""
	Applies simple NLP preprocessing techniques on input dataframe.

	Args:
		df: input dataframe
	"""

	## aggregate headline and short_descriptions cols
	df['text'] = df.headline + " " + df.short_description

	## get rid of non-alphanumeric chars
	df['text'] = df['text'].apply(replace_bad_chars)

	## convert to lowercase
	df['text'] = df['text'].str.lower() 

	#3remove stop words
	stop_words = set(stopwords.words('english'))
	df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))

	## lemmatize text
	df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

	## tokenize text
	df['tokenized_text'] = df['text'].apply(word_tokenize) 

	## get length of text and get rid of bad text based on length threshold
	## threshold = 2 standard deviations below mean length
	df['text_length'] = df['tokenized_text'].apply(len)
	threshold = df['text_length'].mean() - 2*(df['text_length'].std())
	df = df[df.text_length >= threshold]

	return df

def read_data():
	df = pd.read_json('News_Category_Dataset_v2.json', lines=True)
	
	return df

if __name__ == "__main__":
	start = time.time()
	df = read_data()
	preproc_df = preprocess_data(df)

	## output unigrams
	unigram_df = get_unigram_distribution(preproc_df)
	# unigram_df.to_csv("unigram.csv", index=False)

	## ouput bigrams
	bigram_df = get_ngram_distribution(preproc_df, 2)
	# bigram_df.to_csv("bigram.csv", index=False)

	writer = pd.ExcelWriter('n-grams.xlsx', engine='xlsxwriter')

	## Write each df to a different worksheet
	unigram_df.to_excel(writer, sheet_name='unigram', index=False)
	bigram_df.to_excel(writer, sheet_name='bigram', index=False)

	writer.save()

	end = time.time()

	# print("Time to run program: ", end - start)


