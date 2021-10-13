###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

#import matplotlib.pyplot as pl
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score, accuracy_score
import scipy


def distribution(data, transformed = False):
    """
    Visualization code for displaying skewed distributions of features
    """
    
    # Create figure
    fig = plt.figure(figsize = (11,5));

    # Skewed feature plotting
    for i, feature in enumerate(['capital-gain','capital-loss']):
        ax = fig.add_subplot(1, 2, i+1)
        ax.hist(data[feature], bins = 25, color = '#00A0A0')
        ax.set_title("'%s' Feature Distribution"%(feature), fontsize = 14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of Records")
        ax.set_ylim((0, 2000))
        ax.set_yticks([0, 500, 1000, 1500, 2000])
        ax.set_yticklabels([0, 500, 1000, 1500, ">2000"])

    # Plot aesthetics
    if transformed:
        fig.suptitle("Log-transformed Distributions of Continuous Census Data Features", \
            fontsize = 16, y = 1.03)
    else:
        fig.suptitle("Skewed Distributions of Continuous Census Data Features", \
            fontsize = 16, y = 1.03)

    fig.tight_layout()
    fig.show()


def evaluate(results, accuracy, f1):
    """
    Visualization code to display results of various learners.
    
    inputs:
      - learners: a list of supervised learners
      - stats: a list of dictionaries of the statistic results from 'train_predict()'
      - accuracy: The score for the naive predictor
      - f1: The score for the naive predictor
    """
  
    # Create figure
    fig, ax = plt.subplots(2, 3, figsize = (15,10))
    
    # Constants
    bar_width = 0.3
    colors = ['#A00000','#00A0A0','#00A000']
    
    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
            for i in np.arange(3):
                
                # Creative plot code
                ax[j//3, j%3].bar(i+k*bar_width, results[learner][i][metric], width = bar_width, color = colors[k])
                ax[j//3, j%3].set_xticks([0.45, 1.45, 2.45])
                ax[j//3, j%3].set_xticklabels(["1%", "10%", "100%"])
                ax[j//3, j%3].set_xlabel("Training Set Size")
                ax[j//3, j%3].set_xlim((-0.1, 3.0))
    
    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")
    
    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")
    
    # Add horizontal lines for naive predictors
    ax[0, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[0, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    
    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color = colors[i], label = learner))
    plt.legend(handles = patches, bbox_to_anchor = (-.80, 2.53), \
               loc = 'upper center', borderaxespad = 0., ncol = 3, fontsize = 'x-large')
    
    # Aesthetics
    plt.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize = 10, y = 1.10)
    plt.tight_layout()
    plt.show()
    

def feature_plot(importances, X_train, y_train):
    
    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:5]]
    values = importances[indices][:5]

    # Creat the plot
    fig = plt.figure(figsize = (9,5))
    plt.title("Normalized Weights for First Five Most Predictive Features", fontsize = 16)
    plt.bar(np.arange(5), values, width = 0.6, align="center", color = '#00A000', \
          label = "Feature Weight")
    plt.bar(np.arange(5) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#00A0A0', \
          label = "Cumulative Feature Weight")
    plt.xticks(np.arange(5), columns)
    plt.xlim((-0.5, 4.5))
    plt.ylabel("Weight", fontsize = 12)
    plt.xlabel("Feature", fontsize = 12)
    
    plt.legend(loc = 'upper center')
    plt.tight_layout()
    plt.show()  


def gen_corr_matrix(df, v_cols, h_cols): 
    '''
    Function to generate correlation matrix between selected columns that can be subsequently used to generate a heatmap. 
    
    INPUT: 
        df - dataframe
        v_cols - list of numerical features to be ploted on the vertical axis of the heat map
        h_cols - list of numerical features to be ploted on the horizontal axis of the heat map
    
    OUTPUT:
        Returns a dataframe containing the pearson's correlation coefficient for each feature pair in v_cols and h
    
    '''
    # Function to generate heatmap of correlation matrix between selected columns. 
    # Credit: julianstanley (https://stackoverflow.com/questions/45487145/pandas-correlation-between-list-of-columns-x-whole-dataframe)
    
    
    #Create a new dictionary
    plotDict = {}
    # Loop across each of the two lists that contain the items you want to compare
    for gene1 in v_cols:
        for gene2 in h_cols:
            # Do a pearsonR comparison between the two items you want to compare
            tempDict = {(gene1, gene2): scipy.stats.pearsonr(df[gene1],df[gene2])}
            # Update the dictionary each time you do a comparison
            plotDict.update(tempDict)
    # Unstack the dictionary into a DataFrame
    dfOutput = pd.Series(plotDict).unstack()
    # Optional: Take just the pearsonR value out of the output tuple
    dfOutputPearson = dfOutput.apply(lambda x: x.apply(lambda x:x[0]))
    return dfOutputPearson
    


def heatmap_full(df_corr, map_title, annot, x, y):
    '''
    DESCRIPTION: Function to generate a Full heatmap plot
    
    INPUT: 
        df_corr (pandas dataframe) - a dataframe containing the correlation values
        map_title (string) - Title of the heatmap plot. Appears on the top of the heatmap plot
        annot (boolean) - Boolean to indicate if the heatmap should be annotated or not.
        x (int): width of heatmap in inches
        y (int): height of heatmap in inches
    
    Return:
        Plots a Full heatmap of the features in df_corr
    '''
    fig, ax = plt.subplots(figsize=(x, y))
    # mask
    #mask = np.triu(np.ones_like(df_corr, dtype=np.bool))
    # adjust mask and df

    #mask = mask[1:, :-1]
    corr = df_corr.copy()

    # color map

    #cmap = sb.diverging_palette(0, 230, 90, 60, as_cmap=True)
    cmap = 'coolwarm'
    
    # plot heatmap
    sb.heatmap(corr, annot=annot, fmt=".2f", 
               linewidths=5, cmap=cmap, vmin=-1, vmax=1, 
               cbar_kws={"shrink": .8}, square=True)
    
    
    #Code to limit what is labelled on the heatmap to above a certain threshold
    for t in ax.texts:
        if np.abs(float(t.get_text()))>=0.05:
            t.set_text(t.get_text()) #if the value is greater than 0.4 then I set the text 
        else:
            t.set_text("") # if not it sets an empty text

        
    ax.set(facecolor = 'white')
    
    # ticks
    yticks = [i.upper() for i in corr.index]
    xticks = [i.upper() for i in corr.columns]
    plt.yticks(plt.yticks()[0], labels=yticks, rotation=0)
    plt.xticks(plt.xticks()[0], labels=xticks)

    # title
    title = map_title
    plt.title(title, loc='left', fontsize=20)
    
    fig.tight_layout()
    plt.show()


def heatmap_lower_triangle_only(df_corr, map_title, annot, x, y):
    '''
    DESCRIPTION: Function to generate a heatmap plot showing only the lower triangle of the heatmap
    
    INPUT: 
        df_corr (pandas dataframe) - a dataframe containing the correlation values
        map_title (string) - Title of the heatmap plot. Appears on the top of the heatmap plot
        annot (boolean) - Boolean to indicate if the heatmap should be annotated or not.
        x (int): width of heatmap in inches
        y (int): height of heatmap in inches
    
    Return:
        Plots a heatmap of the features in df_corr showing only the lower triangle of the heatmap
    '''
        
    fig, ax = plt.subplots(figsize=(x, y))
    # mask
    mask = np.triu(np.ones_like(df_corr, dtype=np.bool))
    # adjust mask and df

    mask = mask[1:, :-1]
    corr = df_corr.iloc[1:,:-1].copy()

    #mask = mask[0:, :]
    #corr = df_corr.iloc[0:,:].copy()

    
    # color map

    #cmap = sb.diverging_palette(0, 230, 90, 60, as_cmap=True)
    cmap = 'coolwarm'

    # plot heatmap
    sb.heatmap(corr, mask=mask, annot=annot, fmt=".2f", 
               linewidths=5, cmap=cmap, vmin=-1, vmax=1, 
               cbar_kws={"shrink": .8}, square=True)
    
    
    #Code to limit what is labelled on the heatmap to above a certain threshold
    for t in ax.texts:
        if np.abs(float(t.get_text()))>=0.05:
            t.set_text(t.get_text()) #if the value is greater than 0.4 then I set the text 
        else:
            t.set_text("") # if not it sets an empty text
    ax.set(facecolor = 'white')
    
    # ticks
    yticks = [i.upper() for i in corr.index]
    xticks = [i.upper() for i in corr.columns]
    plt.yticks(plt.yticks()[0], labels=yticks, rotation=0)
    plt.xticks(plt.xticks()[0], labels=xticks)

    # title
    title = map_title
    plt.title(title, loc='left', fontsize=20)
    fig.tight_layout()
    plt.show()