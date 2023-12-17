import numpy as np
import plotly_express as px
from sklearn.feature_extraction.text import CountVectorizer

def bar_chart(x, y, title):
    import plotly_express as px
    """
    Plot bar chart.
    Args:
        x (pandas series): Categorical variable
        y (pandas series): Numerical (discrete) variable 
        title (str): Title of the visualization.
    """
    fig = px.bar(
        x = x,
        y = y,
        text = x,
        color_discrete_sequence=["#691633"],
        template="simple_white",
        width=400,        
        height=400
    )
        
    fig.update_layout(
        font=dict(family="Times New Roman",size=12),
        yaxis={"categoryorder":"max ascending"},
        title={
        "text": title,
        "y":0.98,
        "x":0.5,
        "xanchor": "center",
        "yanchor": "top"}
        )
        
    fig.update_yaxes(title_text = "", visible = True)
    fig.update_xaxes(title_text = "", visible = False)
        
    return fig.show()

def histogram(df, x, n_bins, title): 
    """
    Plot Histogram.
    Args:
        df (pandas dataframe): Input DataFrame 
        x (str): Categorical variable. The attribute assignment should be a string.
        n_bins (int): Bin size 
        title (str): Title of the visualization
    """
    fig = px.histogram(
        data_frame=df,
        x = x,
        marginal= "box",
        color_discrete_sequence=["#691633"],
        nbins=n_bins,
        template="simple_white",
        width=1000,        
        height=600
    )
    
    fig.update_layout(
        font_family="Times New Roman", 
        title={
            'text': title,
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'}
    )
    
    return fig.show()

def get_top_n_gram(corpus, n_words = 30, n_min = 2, n_max = 3):
    """
    Plot Top n_gram words.
    Args:
        corpus (pandas series): List of tokens
        n_words (int): Number of words
        n_min (int): Minimum number of n_gram size
        n_max (int): Maximum number of n_gram size
    """

    count_vectorizer = CountVectorizer(ngram_range=(n_min, n_max), lowercase = False ).fit(corpus)
    bow_emd = count_vectorizer.transform(corpus)
    
    words = bow_emd.sum(axis=0).round(2)
    words_freq = [(word, words[0, idx]) for word, idx in count_vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n_words]


def confusion_matrix_plot(ground_truth, prediction):
    """
    Plot Confusion Matrix.
    Args:
        ground_truth (pandas series): Ground truth values
        prediction (pandas series): Predicted values 
    """

    from sklearn.metrics import confusion_matrix 

    cm = confusion_matrix(ground_truth, prediction)

    fig = px.imshow(
        cm,
        x = ground_truth.unique(),
        y = ground_truth.unique(),
        color_continuous_scale='Viridis',
        height = 700,
        width = 1000
    )

    fig.update_layout(
        font_family="Times New Roman", 
        title={
        "text": "Confusion Matrix",
        "y":0.97,
        "x":0.5,
        "xanchor": "center",
        "yanchor": "top"}
    )

    fig.update_yaxes(title_text = "Prediction", visible = True)
    fig.update_xaxes(title_text = "Ground Truth", visible = True, side = "bottom")

    fig.show()

