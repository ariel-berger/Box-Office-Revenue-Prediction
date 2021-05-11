import pandas as pd
import re
from sklearn.preprocessing import MultiLabelBinarizer, KBinsDiscretizer
from sklearn.impute import SimpleImputer
import ast
import numpy as np


class preprocessor:
	"""
	preprocessor Class
	2 public methods, transform and fit_transform
	"""

	def __init__(self, number_of_bins=8, number_of_companies=20, number_of_countries=10,
				 missing_budget_imputing=140000):
		self.missing_budget_imputing = missing_budget_imputing
		self.number_of_bins = number_of_bins
		self.number_of_companies = number_of_companies
		self.number_of_countries = number_of_countries
		self.kbins_list = []
		self.mlb = None
		self.genre_mlb = None
		self.top_companies = None
		self.top_countries = None

	def __encode_genres(self, df):
		"""
		takes the genres feature and fit transform to one (or more) hot encoding.
		:param df:dataframe
		:return: dataframe
		"""
		self.genre_mlb = MultiLabelBinarizer()
		df = df.join(pd.DataFrame(self.genre_mlb.fit_transform(df.pop("genres")), columns=self.genre_mlb.classes_,
								  index=df.index),
					 lsuffix='l')
		return df

	def __encode_genre_transform(self, df):
		"""
		takes the genres feature and transform to one (or more) hot encoding
		from pre-fitted mlb
		:param df:dataframe
		:return: dataframe
		"""
		return df.join(
			pd.DataFrame(self.genre_mlb.transform(df.pop("genres")), columns=self.genre_mlb.classes_, index=df.index),
			lsuffix='l')

	@staticmethod
	def __col_to_len_bin(df, col_name, number_of_bins):
		"""
		change to discrete number_of_bins bins continues values to improve generalization
		:param df: dataframe
		:param col_name: string
		:param number_of_bins: integer
		:return: KBinsDiscretizer object and dataframe
		"""
		df[col_name] = df[col_name].apply(len)
		kbin = KBinsDiscretizer(n_bins=number_of_bins, encode='ordinal')
		df[col_name] = kbin.fit_transform(df[col_name].values.reshape(-1, 1))
		return kbin, df

	@staticmethod
	def __top_prod_companies(df, number_of_companies):
		"""
		find the number_of_companies most frequent companies and return as list
		:param df: dataframe
		:param number_of_companies: integer
		:return: list of strings
		"""
		mlb = MultiLabelBinarizer()
		temp = pd.DataFrame(mlb.fit_transform(df['production_companies']), columns=mlb.classes_, index=df.index)
		return temp.sum().sort_values(ascending=False)[:number_of_companies].index.values

	@staticmethod
	def __top_prod_countries(df, number_of_countries):
		"""
		find the number_of_countries most frequent countries and return as list
		:param df: dataframe
		:param number_of_countries: integer
		:return: list of strings
		"""
		mlb = MultiLabelBinarizer()
		temp = pd.DataFrame(mlb.fit_transform(df['production_countries']), columns=mlb.classes_, index=df.index)
		return temp.sum().sort_values(ascending=False)[:number_of_countries].index.values

	def __bin_columns(self, df):
		"""
		fit and transform
		:param df: dataframe
		:return: dataframe
		"""
		cols_to_bin = ['cast', 'crew', 'spoken_languages', 'Keywords']
		for col_name in cols_to_bin:
			kbin, df = self.__col_to_len_bin(df, col_name, self.number_of_bins)
			self.kbins_list.append(kbin)
		return df

	def __bin_columns_transform(self, df):
		"""
		transform
		:param df: dataframe
		:return: dataframe
		"""
		cols_to_bin = ['cast', 'crew', 'spoken_languages', 'Keywords']
		for col_name, kbin in zip(cols_to_bin, self.kbins_list):
			df[col_name] = df[col_name].apply(len)
			df[col_name] = kbin.transform(df[col_name].values.reshape(-1, 1))
		return df

	@staticmethod
	def __parse_dates(df):
		"""
		parse the date to dateparts - year, month, week-day and ohe the month and the week-days.
		:param df: dataframe
		:return: dataframe
		"""
		df['release_date'] = pd.to_datetime(df['release_date'])
		df['release_date'] = df['release_date'].fillna(df['release_date'].median())
		df['year'] = df['release_date'].dt.year
		df['month'] = df['release_date'].dt.month
		df['day'] = df['release_date'].dt.weekday
		df = pd.get_dummies(df, columns=['month', 'day'])
		return df

	@staticmethod
	def __parse_json(df):
		"""
		parse text columns to dict and remain only the important value (usually the name)
		:param df: dataframe
		:return: dataframe
		"""
		col_names = ['genres', 'production_companies', 'production_countries', 'cast', 'crew', 'spoken_languages',
					 'Keywords']
		value_names = ['name', 'name', 'iso_3166_1', 'name', 'name', 'name', 'name']
		for col_name, value_name in zip(col_names, value_names):
			# df[col_name] = df[col_name].fillna("{}")
			df[col_name] = df[col_name].apply(literal_eval_error_handling)
			df[col_name] = df[col_name].apply(lambda x: [i[value_name] for i in x])
		return df

	@staticmethod
	def __fillnan(df):
		"""
		fill nan to text columns.
		:param df: dataframe
		:return: dataframe
		"""
		col_names = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']
		for col_name in col_names:
			df[col_name] = df[col_name].fillna(df[col_name].median())
		return df

	def __top_countries_and_companies(self, df):
		"""
		find the top countries and companies.
		add binary column that checks if the movie company is on of the top companies
		and one (or more) hot encoded the top countries
		:param df: dataframe
		:return: dataframe
		"""
		self.top_countries = set(self.__top_prod_countries(df, self.number_of_countries))
		self.top_companies = set(self.__top_prod_companies(df, self.number_of_companies))
		df['is_top_prod'] = df['production_companies'].apply(
			lambda x: 1 if len(set(x).intersection(self.top_companies)) >= 1 else 0)
		for country in self.top_countries:
			df[country] = df['production_countries'].apply(lambda x: 1 if country in x else 0)
		return df

	def __top_countries_and_companies_transform(self, df):
		"""
		from the top countries and companies:
		add binary column that checks if the movie company is on of the top companies
		and one (or more) hot encoded the top countries
		:param df: dataframe
		:return: dataframe
		"""
		df['is_top_prod'] = df['production_companies'].apply(
			lambda x: 1 if len(set(x).intersection(self.top_companies)) >= 1 else 0)
		for country in self.top_countries:
			df[country] = df['production_countries'].apply(lambda x: 1 if country in x else 0)
		return df

	def fit_transform(self, df):
		"""
		fit and transform the df from raw data to data that ready to enter ML models
		:param df: dataframe
		:return: dataframe
		"""
		df = self.__parse_json(df)
		df = self.__fillnan(df)
		df = self.__parse_dates(df)
		df['budget'] = df['budget'].apply(lambda x: self.missing_budget_imputing if int(x) == 0 else x)
		df['has_collections'] = df['belongs_to_collection'].isna().astype(int)
		df['homepage'] = df['homepage'].isna().astype(int)
		df['is_en'] = df['original_language'].apply(lambda x: 1 if x == 'en' else 0)
		df = self.__encode_genres(df)
		df = self.__top_countries_and_companies(df)
		df = self.__bin_columns(df)
		df.drop(
			['release_date', 'original_language', 'production_countries', 'production_companies', 'id', 'backdrop_path',
			 'imdb_id', 'poster_path', 'video', 'belongs_to_collection', 'status', 'runtime',
			 'original_title', 'overview', 'tagline', 'title'], axis=1, inplace=True)
		return df

	def transform(self, df):
		"""
		transform the df from raw data to data that ready to enter ML models
		:param df: dataframe
		:return: dataframe
		"""
		df = self.__parse_json(df)
		df = self.__fillnan(df)
		df = self.__parse_dates(df)
		df['budget'] = df['budget'].apply(lambda x: self.missing_budget_imputing if int(x) == 0 else x)
		df['has_collections'] = df['belongs_to_collection'].isna().astype(int)
		df['homepage'] = df['homepage'].isna().astype(int)
		df['is_en'] = df['original_language'].apply(lambda x: 1 if x == 'en' else 0)
		df = self.__encode_genre_transform(df)
		df = self.__top_countries_and_companies_transform(df)
		df = self.__bin_columns_transform(df)
		df.drop(
			['release_date', 'original_language', 'production_countries', 'production_companies', 'id', 'backdrop_path',
			 'imdb_id', 'poster_path', 'video', 'belongs_to_collection', 'status', 'runtime',
			 'original_title', 'overview', 'tagline', 'title'], axis=1, inplace=True)
		return df


def literal_eval_error_handling(value):
	try:
		return ast.literal_eval(value)
	except:
		return ast.literal_eval("{}")
