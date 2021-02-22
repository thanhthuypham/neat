from flask import Flask, render_template
from flask import request
from flask import Response

import numpy as np
import pandas as pd
import requests 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib.figure import Figure
import io
import base64

try:
  import googleclouddebugger
  googleclouddebugger.enable(
    breakpoint_enable_canary=True
  )
except ImportError:
  pass


app = Flask(__name__)

###Write function to query data by search terms:
def query_book_data_batch(url):
    """
    Download book per batch based on the url
    
    Arg: 
    url(string): the url to get book data
    
    Output: 
    Data Frame of bookd data per batch
    """
    
    #Query the data
    r = requests.get(url)
    json_data = r.json()
    items_list = json_data['items']
    
    #Construct the table
    title = []
    authors = []
    description = []
    date = []
    rating = []

    for i in range (0, len(items_list)): 
        title.append(dict(items_list[i]['volumeInfo'])['title'])
        for label, field in [(authors, 'authors'), (description, 'description'), (date, 'publishedDate'), (rating, 'averageRating')]:
            if field in dict(items_list[i]['volumeInfo']).keys():
                label.append(dict(items_list[i]['volumeInfo'])[field])  
            else:
                label.append('NA')

    #Zip data in to data frame
    df_book = pd.DataFrame(zip(title, authors, description, date, rating), columns = ['Title', 'Authors', 'Description', 'Date', 'Rating'])
    
    return df_book


###Write function to query tops 300 volumes:
def query_book_data(search_term, startIndex=0, maxResults=30):
    """
    Download book content on Google Books based on the search team, start index and number of results per query:
    
    Args:
    search_terms(strings): of search term
    startIndex(int): default = 0, the position to start query an interger from 1 to max length of the book list
    maxResults: default = 30, number of books to get in one request
    
    Returns: 
    Data Frame with 5 colums: Title, Authors, Description, Year, Avg Rating 
    """
    df_output = pd.DataFrame(columns=['Title', 'Authors', 'Description', 'Date', 'Rating'])
    for sindex in np.arange(0, 300, 30): 
        url = "https://www.googleapis.com/books/v1/volumes?q="+str(search_term)+"&startIndex="+str(sindex)+"&maxResults="+str(maxResults)+"&projection=lite&fields=items(volumeInfo)"
        df_output = pd.concat([df_output, query_book_data_batch(url)], axis=0)
    return df_output

###Write function to plot top terms in one domain
def domain_knowledge_explorer(search_term):
    """
    Plot top popular phrases that appear in the Google Book library based on search tearm
    
    Args: 
    search_term(str): search term
    
    Returns:
    Chart that shows top popular words
    """
    #Extract the description
    ref_lib = query_book_data(str(search_term))
    ref_lib_1 = ref_lib[['Description']].dropna(subset=['Description']).reset_index()
    doc = ref_lib_1['Description']
    
     #Count words frequency: n-word
    Countvec = CountVectorizer(max_features=50, ngram_range=(2,3), lowercase = True, stop_words = 'english')
    words_matrix = Countvec.fit_transform(doc)
    words_array = words_matrix.toarray()
    #Create dataframe of most popular n_word phrases:
    words_df = pd.DataFrame(words_array, columns = Countvec.get_feature_names())
    
    #Count words frequency: 1-word
    Countvec_mono = CountVectorizer(max_features=10, ngram_range=(1,1), lowercase = True, stop_words = 'english')
    words_matrix_mono = Countvec_mono.fit_transform(doc)
    words_array_mono = words_matrix_mono.toarray()
    #Create dataframe of most popular 1-word phrases:
    words_df_mono = pd.DataFrame(words_array_mono, columns = Countvec_mono.get_feature_names())
    
    #Create dataframe of most popular phrases:
    all_words_df = pd.concat([words_df_mono, words_df], axis = 1)
    top_words_predrop = pd.DataFrame(all_words_df.sum(axis=0).sort_values(ascending = False), columns = ['count'])
    top_words = top_words_predrop.drop(top_words_predrop.index[2:7], axis=0)
    
  
    #Plot
    fig = Figure(figsize=(12,6))
    ax = fig.subplots()
    sns.set_style('whitegrid')
    max_count=top_words['count'].max()
    ax.plot(top_words[:50]['count']/50, marker='o', c='black')
    ax.text(18, (max_count + 2)/50, "Count of most popular word/ phrase: "+str(max_count), c = 'black', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    
    i=0
    for word in top_words[:50].index:
        ax.text(i, (top_words.loc[word]['count']+5)/50, word, rotation=45, c = (0.1, i/len(top_words), 0.5), fontsize=10)
        i = i+1

    ax.set_xlabel('Most popular words/ phrases', fontsize=12)
    ax.set_ylabel('Scaled frequency', fontsize=12)
    ax.set_title('Most popular words/ phrases in '+str(search_term)+' domain', fontsize=14)
    
    # Save the chart to a temporary buffer.
    buf = io.BytesIO()
    fig.savefig(buf, format="png")

    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return data


### Display the results on web app
data = 'String'
@app.route("/")
def index():
    search_term = request.args.get("search_term", "")
    if search_term: 
        global data
        data = domain_knowledge_explorer(search_term)
    return render_template('index.html', search_term=search_term, data=data)

###Run the app
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)