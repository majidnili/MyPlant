from bs4 import BeautifulSoup
import requests
import urllib2
from string import ascii_uppercase
from bs4 import NavigableString
import json

from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import TruncatedSVD
import numpy as np
import scipy as sc
import string
from scipy.sparse import hstack, vstack
from collections import defaultdict

import matplotlib.pyplot as plt


def get_html():
    '''#scraping data:'''
    html_str = "http://www.missouribotanicalgarden.org/PlantFinder/ \
                PlantFinderListResults.aspx?letter="
    missouri = "http://www.missouribotanicalgarden.org"

    plants = set()
    for letter in ascii_uppercase:
        html = html_str + letter
        r = requests.get(html)
        soup = BeautifulSoup(r.content, 'html.parser')
        links = soup.find_all('a', target='_self')
        for link in links:
            if link['href'][-1] == letter:
                plants.add(missouri + link['href'])
    plants = sorted(list(plants))
    return plants


def make_soup(plants):
    '''text'''
    plant_dict = dict()
    for plant in plants:
        r = requests.get(plant)
        soup = BeautifulSoup(r.content, 'html.parser')
        name = soup.find('span', id="dnn_srTitle_lblTitle").text
        plant_dict[name] = soup
    return plant_dict


def make_temp_dict(plant_dict):
    '''This function extract inforamtion from web pages made by make_soup function
    and make a dictionary for each plant.'''
    missouri = "http://www.missouribotanicalgarden.org"
    temp_dict = dict()
    for key, value in plant_dict.iteritems():
        key = key.lower()
        try:
            key = key.strip()
        except:
            pass
    # extracting data from right column (table):
        table = dict()
        for x in value.select('.column-right > div'):
            for y in x.contents:
                if type(y) == NavigableString:
                    try:
                        txt = y.strip()
                    except:
                        txt = y
                    categ = txt.split(':')
                    try:
                        table[categ[0]] = categ[1]
                    except:
                        pass

                else:
                    pass
    # extracting text part:
        text = list()
        for x in value.select('.row > p'):
            for y in x.contents:
                if type(y) == NavigableString:
                    try:
                        text.append(y.strip() + ' ')
                    except:
                        text.append(y)
                else:
                    pass
        table['body'] = ''.join(text)
        temp_dict[key] = table.copy()

    # extracting url and image-url (if available):
        link = value.select('form')

        temp_dict[key]['url'] = missouri + link[0]["action"]

        try:
            link = value.select('.main-pic > a')
            imgUrl = missouri + link[0]['href']
            r = requests.get(imgUrl)
            soup2 = BeautifulSoup(r.content, 'html.parser')
            link2 = soup2.select('img')
            temp_dict[key]['img-url'] = link2[0]['src']
        except:
            pass

    return temp_dict


def category_to_num(temp_dict):
    '''# convert some categorical features of plant profile \
    (right column table) to numerical values:'''
    # check if the plant blooms (ignore details of blooming/flower)
    set_flower = set()
    bloom_dict = dict()
# list various terms for blooming plants:
    for key in temp_dict:
        try:
            set_flower.add(temp_dict[key]["Flower"])
        except:
            pass
    for i in set_flower:
        if i == ' Insignificant' or i == ' Fragrant, Insignificant' or\
           i == ' Insignificant, Good Dried' \
           or i == ' Showy, Insignificant':
            bloom_dict[i] = -1.
        else:
            bloom_dict[i] = 1.

# Maintenance level:
    temp_list = [' Low', ' Low-Medium', ' Medium', ' Medium-High', ' High']
    maintn_dict = dict()
    for i, k in enumerate(temp_list):
        maintn_dict[k] = float(i+1)

# light requirement:
    temp_list = [' Full Shade', ' Part shade to full shade', ' Part shade',
                 ' Full sun to part shade', ' Full sun']
    light_dict = dict()
    for i, k in enumerate(temp_list):
        light_dict[k] = float(i+1)

# watering condition:
    temp_list = [' Dry', ' Dry to medium', ' Medium', ' Medium to wet', ' Wet']
    water_dict = dict()
    for i, k in enumerate(temp_list):
        water_dict[k] = float(i+1)

    dict_list = bloom_dict, maintn_dict, light_dict, water_dict
    return dict_list


def make_table(dict_list, temp_dict, keys):
    '''make a matrix of some of the features of the table part \
       of plant profile built by category_to_num function:'''
    plant_table = defaultdict(list)
    name_list = ["Flower", "Maintenance", "Sun", "Water"]
    name_list2 = ["Height", "Spread"]

    for key in keys:
        for name, feature in zip(name_list, dict_list):
            try:
                level = temp_dict[key][name]
                plant_table[key].append(feature[level])
            except:
                plant_table[key].append(-1.)

    # try new feature: sun_level/water_level
    # higher number is for desert-like plants
        if plant_table[key][2] != -1. and plant_table[key][3] != -1.:
            plant_table[key].append(plant_table[key][2]/plant_table[key][3])
        else:
            plant_table[key].append(-1.)

    # make columns from size of the plants:
        for feature in name_list2:
            try:
                range_ = temp_dict[key][feature]
                low = float(range_.split()[0])
                high = float(range_.split()[2])
                plant_table[key].append(low)
                plant_table[key].append(high)
            except:
                plant_table[key].append(-1.)
                plant_table[key].append(-1.)

        try:
            range_ = temp_dict[key]['Zone']
            low = float(range_.split()[0])
            high = float(range_.split()[-1])
            plant_table[key].append(low)
            plant_table[key].append(high)
        except:
            plant_table[key].append(-1.)
            plant_table[key].append(-1.)

        # check if plant is houseplant
        words = ['houseplant', 'indoor']
        houseplant = dict()
        exclude = set(string.punctuation)
        houseplant[key] = 0.
        for word in temp_dict[key]["body"].split():
            word_ = ''.join(ch for ch in list(word) if ch not in exclude)
            if word_ in words:
                houseplant[key] = 1.
                break

        plant_table[key].append(houseplant[key])
    return plant_table


def tokenize(doc):
    '''ducument'''
    wordnet = WordNetLemmatizer()
    return [wordnet.lemmatize(word) for word in word_tokenize(doc.lower())]


def tokenizer_(descriptions, tokenize, ngram=(1, 4), maxf=None):
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=ngram,
                            strip_accents='unicode', max_features=maxf,
                            tokenizer=tokenize)
    vectorized = tfidf.fit_transform(descriptions)
    return vectorized, tfidf


def cosine_dist(plant, vectorized, njobz=1):
    sims = pairwise_distances(vectorized[plant, :], vectorized,
                              metric='cosine', n_jobs=njobz)
    return sims


def plot_figure(eig):
    '''plot percentage of variance explained by each of the
       selected components to decide how many components to keep'''
    plt.figure(figsize=(10, 6))
    plt.plot(eig, linewidth=5)
    plt.legend(prop={'size': 10})
    plt.ylabel("magnitude", fontsize=20)
    plt.xlabel("feature", fontsize=20)
    plt.tick_params(axis='both', labelsize=20)


if __name__ == "__main__":
    plants = get_html()
    plant_dict = make_soup(plants)
    temp_dict = make_temp_dict(plant_dict)

    # save plants' names in list so they can be referred to in the list order
    keys = [key for key in temp_dict]

    dict_list = category_to_num(temp_dict)
    plant_table = make_table(dict_list, temp_dict, keys)

    # make matrix from table part from make_table functions:
    table = np.zeros(len(plant_table) * len(plant_table[keys[0]])).\
        reshape(len(plant_table), len(plant_table[keys[0]]))
    for i, key in enumerate(keys):
        table[i, :] = plant_table[key]

    # compile a descriptions list from plant profile for nlp:
    descriptions = []
    for key in keys:
        text = [v for k, v in temp_dict[key].iteritems()
                if k != "url" and k != "img-url"]
        descriptions.append(''.join(text))

    vectorized_all, tfidf_all = tokenizer_(descriptions, tokenize)

    svd = TruncatedSVD(n_components=1000, algorithm="arpack")
    svd.fit(vectorized_all)
    reduced = svd.fit_transform(vectorized_all)
    eig = svd.explained_variance_ratio_

    plot_figure(eig)

    tfidf_redu = reduced[:, :150]
    # normalize the table matrix
    nrm = np.linalg.norm(table, axis=1)
    norm_table = table.copy()
    for i in xrange(table.shape[0]):
        norm_table[i, :] = table[i, :]/nrm[i]
    # normalize the svd matrix
    nrm = np.linalg.norm(tfidf_redu, axis=1)
    norm_tfidf_redu = tfidf_redu.copy()
    for i in xrange(tfidf_redu.shape[0]):
        norm_tfidf_redu[i, :] = tfidf_redu[i, :]/nrm[i]
    # concatinate the normalized svd and table matrices:
    final_mat = np.concatenate((norm_table, norm_tfidf_redu), axis=1)
    # calculate pairwise distances for selected plants
    plant = "anthurium andraeanum"
    sim = cosine_dist(keys.index(plant), final_mat)
    similar = sim_all[0].argsort()[:40]
    sim_plants = [(j, keys[j], sim_all[0][j]) for j in similar]
