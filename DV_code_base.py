import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np

########################################
def main(what_plot = 1 , df = None ):
    print('you chose :' , what_plot)
    print(df.head())

    #Histo-gram
    if what_plot == 1:
        #normal plot
        plt.figure()
        sns.histplot(df.cont_skew , bins = 'auto'  , kde = True)#bins --> 'fd' , 'scott' 
        
        plt.title('normal')
        
        plt.xlabel('val')
        plt.ylabel('count')
        
        plt.show()

        #adding the mean and median and changing the style to classic '
        plt.figure()
        sns.histplot(df.cont_skew , bins = 'scott'  , kde = True)

        plt.title('normal :')
        
        plt.xlabel('val')
        plt.ylabel('count')
        
        plt.axvline(df.cont_skew.mean() , color = 'red' , label = 'mean')
        plt.axvline(df.cont_skew.median() , color = 'green' , label = 'median')
        plt.legend()

        plt.style.use('classic')
        # saving the plot 
        plt.savefig('histogram.png' , dpi = 300) #dpi --> resolution of the image
        plt.show()

    #Density Plot   

    if what_plot == 2:
        #normal
        plt.figure()
        sns.kdeplot(df.cont_skew , fill = True , bw_method='' , color='purple' ) #bw_method --> 'scott' , 'silverman' , or for automatic : 
        
        plt.title('Density Plot')
        
        plt.xlabel('val')
        plt.ylabel('Density')
        
        plt.style.use('classic')

        plt.show()
        # 

    #ECDF plot 
    if what_plot == 3 : 
        plt.figure()
        sns.ecdfplot(df.cont_skew , color = 'orange')
        
        plt.title('ECDF Plot')
        
        plt.xlabel('val')
        plt.ylabel('ECDF')
        
        plt.axvline(df.cont_skew.mean() , color = 'red' , label = 'mean' , linestyle = '--')
        plt.axvline(df.cont_skew.median() , color = 'green' , label = 'median')
        plt.legend()

        plt.style.use('classic')

        plt.show()
    #Q-Q plot
    if what_plot == 4 : 
        from scipy import stats
        plt.figure()
        stats.probplot(df.cont_skew , dist = 'norm' , plot = plt) # dist --> 'norm' ,
       # 'expon' , 'lognorm' , 'weibull_min' , 'weibull_max' , 'gamma' , 'beta' , 'cauchy' , 'laplace' , 'gumbel_r' , 'gumbel_l'
        
        plt.title('Q-Q Plot')
        
        plt.xlabel('Theoretical Quantiles')
        plt.ylabel('Sample Quantiles')
        
        plt.style.use('classic')

        plt.show()

    #box plot 
    if what_plot == 5 :
        plt.figure()
        sns.boxplot(x = df.cont_out , color = 'cyan' , showfliers = True , ) #showfliers --> outliers
        
        plt.title('Box Plot')
        
        plt.xlabel('val')
        plt.ylabel('')

        plt.axvline(df.cont_out.mean() , color = 'red' , label = 'mean' , linestyle = '--')
       
        plt.legend()

        plt.style.use('classic')

        plt.show()
        ############
        plt.style.use('classic')
        plt.figure(figsize=(8, 3))

        sns.boxplot(
            x=df.cont_out,
            color='cyan',
            showfliers=True,
            width=0.4
            ,notch=True
            ,boxprops=dict(alpha=0.7)
            ,flierprops=dict(marker='o', color='red', alpha=0.5)
            ,medianprops=dict(color='blue', linewidth=2)
            ,whiskerprops=dict(color='magenta', linewidth=1.5)
            ,capprops=dict(color='magenta', linewidth=1.5)
            ,showmeans=True, meanprops=dict(marker='D', color='orange', alpha=0.7)
        )

        # Mean line
        mean_val = df.cont_out.mean()
        plt.axvline(mean_val, color='red', linestyle='--', label='Mean')

        # Median line (important!)
        median_val = df.cont_out.median()
        plt.axvline(median_val, color='blue', linestyle='-', label='Median')

        plt.title('better style ')
        plt.xlabel('Value')
        
        plt.legend()
        plt.tight_layout()
        plt.show()
    #violin plot
    if what_plot == 6 :
        plt.figure()
        sns.violinplot(x = df.cont_skew , color = 'lightblue'  ,bw_method= 'silverman' ) #bw_method --> 'scott' , 'silverman'
        
        plt.title('Violin Plot')
        
        plt.xlabel('val')
        plt.ylabel('')

        plt.show()
        
        sns.set_theme(style="whitegrid")

        #plt.style.use('classic')
        plt.figure(figsize=(8, 4))

        sns.violinplot(
            x=df.cont_skew,
            color='lightgray',
            bw_method='silverman', # bw_method --> 'scott' , 'silverman'
            inner='quartile'  # shows median + quartiles
            

            ,cut=0  # cut=0 to limit the tails to the data range
            , bw_adjust=0.5  # adjust bandwidth for smoother or more detailed plot
        )

        # Add mean line
        mean_val = df.cont_skew.mean()
        plt.axvline(mean_val, color='red', linestyle='--', label='Mean')

        plt.title('Violin Plot of cont_skew')
        plt.xlabel('Value')

        plt.legend()
        sns.despine()
        plt.show()
    #beeswarm plotS
    if what_plot == 7 :
        sns.set_theme(style="whitegrid")
        plt.figure()
        sns.swarmplot(x = df.cont_skew , color = 'magenta')
        
        plt.title('Beeswarm Plot')
        
        plt.xlabel('val')
        plt.ylabel('')

        plt.axvline(df.cont_skew.mean(), color='red', linestyle='-.', label='Mean' , alpha = 0.7)

     
        sns.despine()
        plt.tight_layout()
        plt.show()
    #Ridge plot
    if what_plot == 8 :
        plt.style.use('fast')# fast is the best 
        # now this is the list with all the styles : 
        #['bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 
        # 'grayscale', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark-palette', 
        # 'seaborn-dark', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 
        # 'seaborn-notebook', 
        # 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 
        # 'seaborn-ticks', 'seaborn-whitegrid', 'seaborn-white', 'Solarize_Light2',
        #  '_classic_test_patch']

        plt.figure()
        sns.kdeplot(df.cont_skew , fill = True , bw_method='scott' , color='purple' ) #bw_method --> 'scott' , 'silverman'
        
        plt.title('Ridge Plot')
        
        plt.xlabel('val')
        plt.ylabel('Density')
        
       
        sns.despine()

        plt.show()
    #Line plot
    if what_plot == 9 :
        plt.style.use('fast')
        plt.figure()
        sns.lineplot(x = range(len(df.cont_norm)), y = df.cont_skew , color = 'black')
        
        plt.title('Line Plot')
        
        plt.xlabel('Index')
        plt.ylabel('val')
        
        sns.despine()

        plt.show()
        #########
        plt.style.use('default')   # better than 'fast' for readability
        plt.figure(figsize=(9, 4))

        x = range(len(df.cont_norm))
        y = df.cont_skew
        # Add mean reference line
        mean_val = y.mean()
        plt.axhline(mean_val, color='red', linestyle='--', label='Mean')

        # Optional: highlight trend (rolling average)
        rolling = y.rolling(window=250).mean()
        plt.plot(x, rolling, color='blue', linewidth=2, label='Rolling Mean')

        plt.title('Trend of cont_skew over Index')
        plt.xlabel('Index')
        plt.ylabel('Value')

        plt.grid(alpha=0.3)
        sns.despine()

        plt.legend()
        plt.tight_layout()
        plt.show()
###################### part 3 
    #bar plot and its ver (cat data)
    #normal
    if what_plot == 10:
        plt.style.use('fast')
        plt.figure()

        sns.countplot(x=df.segment, palette='Set2' , order = df.segment.value_counts().index)  

        
        plt.title('TTTT')
        plt.xlabel('segment')
        plt.ylabel('Count')

        sns.despine()
        plt.show()

        ############## horziontal 
        plt.style.use('fast')
        plt.figure()

        sns.countplot(y=df.segment, palette='Set2' , order = df.segment.value_counts().index)  

        
        plt.title('TTTT')
        plt.xlabel('segment')
        plt.ylabel('Count')

        sns.despine()
        plt.show()
        
        return
    #stacked <BAD> <IGNORE>
    if what_plot == 11:
        plt.style.use('fast')
        plt.figure()
        sns.countplot(x=df.segment , y = df.education )

        plt.title('TTTT')
        plt.xlabel('segment')
        plt.ylabel('Count')

        sns.despine()

        plt.show()

        return
    #side by side (dodged)
    if what_plot == 12:
        plt.style.use('fast')
        plt.figure()
        
        sns.countplot(x=df.segment,
                       hue = df.education ,width=0.8 , 
                       palette='Set1' , order = df.segment.value_counts().index)  
        
        
        plt.title('TTTT')
        plt.xlabel('segment')
        plt.ylabel('Count')

        sns.despine()
        plt.show()
        return
    #pareto
    if what_plot == 130000:
        
        return
    #Lollipop
    if what_plot == 13:
        plt.style.use('fast')
        plt.figure()

        counts = df['segment'].value_counts().sort_values(ascending=False)

        #plt.vlines(counts.index, 0, counts.values, color='steelblue', linewidth=2)   # fix: use vlines not stem OR hlines as to get it horizontal 
        #plt.plot(counts.index, counts.values, 'o', color='steelblue', markersize=8)  # fix: dots on top and when H just switch these places 

        plt.hlines(df['segment'].value_counts().sort_values(ascending=False).index  , 
                   0,  df['segment'].value_counts().sort_values(ascending=False).values
                   , color='steelblue', linewidth=2)   # fix: use vlines not stem OR hlines as to get it horizontal 
        plt.plot( df['segment'].value_counts().sort_values(ascending=False).values 
                 ,df['segment'].value_counts().sort_values(ascending=False).index, 
                 'o', color='steelblue', markersize=8)  # fix: dots on top

        plt.title('Lollipop Plot - Segment')
        plt.xlabel('Segment')
        plt.ylabel('Count')
        sns.despine()
        
        plt.show()
    #

    if what_plot == 101: # this to add the count and %  ontop of the plot : 
            sns.set_theme(style="whitegrid")
            plt.figure(figsize=(8, 4))

            ax = sns.countplot(
                x=df.segment,
                palette='Set2',
                order=df.segment.value_counts().index
            )

            # Add values on top of each bar
            total_count = len(df.segment)
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(
                    f'{int(height)}',
                    (p.get_x() + p.get_width() / 2, height),
                    ha='center',
                    va='bottom'
                )

   
            plt.title('Count of Segments')
            plt.xlabel('Segment')
            plt.ylabel('Count')

            plt.grid(axis='y', alpha=0.3) # this so the grid is less visible 
            
            sns.despine()
            plt.tight_layout()
            plt.show()
#################
            sns.set_theme(style="whitegrid")
            plt.figure(figsize=(8, 4))

            ax = sns.countplot(
                x=df.segment,
                palette='Set2',
                order=df.segment.value_counts().index
            )

            # Add values on top of each bar
            total_count = len(df.segment)
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(
                    f'{float(height/total_count) * 100} %',
                    (p.get_x() + p.get_width() / 2, height),
                    ha='center',
                    va='bottom'
                )

   
            plt.title('Count of Segments')
            plt.xlabel('Segment')
            plt.ylabel('Count')

            plt.grid(axis='y', alpha=0.3) # this so the grid is less visible 
            
            sns.despine()
            plt.tight_layout()
            plt.show()
###############################
##############################
#BIVARIATE PLOTS
#num x num 
    #scatter plot
    #basic
    if what_plot == 14:
        plt.style.use('fast')
        plt.figure()
        plt.scatter(
            x = df.cont_norm,
            y = df.spend,
            
            alpha=0.5,
            edgecolors='none'
        )
        plt.show()
        ######
        #here with a line 
        plt.style.use('fast')
        plt.figure()
        plt.scatter(
            x = df.cont_norm,
            y = df.spend,
            
            alpha=0.5,
            edgecolors='none'
        )
        sns.regplot(x=df.cont_norm, y=df.spend, scatter=False, color='red')  # Add a regression line , basicly another plot 
        plt.show()
        # a polynomial line
        plt.style.use('fast')
        plt.figure()
        plt.scatter(
            x = df.cont_norm,
            y = df.spend,
            
            alpha=0.5,
            edgecolors='none'
        )
        sns.regplot(x=df.cont_norm, y=df.spend, scatter=False, color='red' , order = 2)  # remember ML and how we upped the degree of the poly 
        # so now it is in par 'order' 
        plt.show()
        ###############
        # now a diffrent style is the hexbin plot which is like a 2D histogram and it is good for large datasets 
        plt.style.use('fast')
        plt.figure()
        plt.hexbin(x=df.cont_norm, y=df.spend, gridsize=30, cmap='Blues', edgecolors='none')  # cmap --> 'Blues' , 'Greens' , 'Reds' , 'Purples' , 'Oranges'
        plt.colorbar(label='Count in bin')
        plt.title('Hexbin Plot')
        plt.xlabel('cont_norm')
        plt.ylabel('spend')
        plt.show()
        #2d 
        plt.style.use('fast')
        plt.figure()
        sns.kdeplot(x=df.cont_norm, y=df.spend, fill=True, cmap='Blues', bw_method='scott')  # cmap --> 'Blues' , 'Greens' , 'Reds' , 'Purples' , 'Oranges' , bw_method --> 'scott' , 'silverman'
        plt.title('2D Density Plot')
        plt.xlabel('cont_norm')
        plt.ylabel('spend')
        plt.show()



        pass
    
    #heatmap
    if what_plot == 15:
        plt.style.use('fast')
        plt.figure()
        #first make the corr matrix 
        corr_matrix = df[['cont_out', 'spend' , 'cont_norm' , 'score']].corr()

        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5 ) # cmap --> 'coolwarm' , 'viridis' , 'plasma' , 'inferno' , 'magma'
        plt.xticks(rotation=45) # to rotate the x labels
        plt.yticks(rotation=0) # to keep the y labels horizontal

        plt.title('Correlation Heatmap') 
        plt.show()
        pass















    ################################
    ####not much used things  section 
    #####################
    # use less of it 
    #now to pie chart
    if what_plot == 16:
        plt.style.use('fast')
        plt.figure()
        segment_counts = df['segment'].value_counts()
        plt.pie(segment_counts, labels=segment_counts.index, autopct='%1.1f%%', startangle=90, # to be clockwise
                colors=sns.color_palette('Set2', len(segment_counts)) # color
                 ,counterclock=False) # orderedd
         # and we want it to be clockwise so we can add the start angle and make it 90   & ordered 
        # colors --> 'Set1' , 'Set2' , 'Set3' , 'Pastel1' , 'Pastel2' , 'Dark2' , 'Accent'
        plt.title('Segment Distribution')
        
        plt.show()
        pass
    ###########################################
    ######################
    ### MISC knowledge 
    # percentage labels on bar plot
    import matplotlib.ticker as mtick
    if what_plot == 17:
        plt.figure(figsize=(8,5))

        # get counts
        counts = df['segment'].value_counts() #NOTE
        total = counts.sum() #NOTE

        # plot
        ax = sns.countplot(
            x='segment',
            data=df,
            order=counts.index,
            palette='Set2'
        )

        # add percentage labels
        for container in ax.containers:             #NOTE
            labels = [f'{(v/total):.1%}' for v in container.datavalues] #NOTE 
            ax.bar_label(container, labels=labels, padding=3) #NOTE 

        plt.title('Segment Distribution (%)')
        plt.xlabel('Segment')
        plt.ylabel('Count')

        # clean numbers (optional)
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}')) #NOTE

        plt.show()
        pass
    

def useful_thing():
    #palettes / colorsBrewer
    #a more better way to stander the color when like bar plot
    
        #palette --> 'Set1' , 'Set2' , 'Set3' , 'Pastel1' , 'Pastel2' , 'Dark2' , 'Accent'
    ##############################
    # bins are gotten via an equation 
    #as well as band width


    pass
plot = 15
if __name__ == "__main__":
    
   
    np.random.seed(42)
    n = 500

    # Base variables
    cont_norm = np.random.normal(0, 1, n)
    cont_skew = np.random.lognormal(0, 0.6, n)

    x = np.concatenate([np.random.normal(5, 1, n - 5), np.random.normal(12, 0.5, 5)])
    np.random.shuffle(x)
    cont_out = x

    visits = np.random.poisson(3, n)

    segment = np.random.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])

    education_levels = ["High school", "Bachelor", "Master", "PhD"]
    education = pd.Categorical(
        np.random.choice(education_levels, size=n, p=[0.3, 0.4, 0.2, 0.1]),
        categories=education_levels,
        ordered=True
    )

    df = pd.DataFrame({
        "id":        np.arange(1, n + 1),
        "cont_norm": cont_norm,
        "cont_skew": cont_skew,
        "cont_out":  cont_out,
        "visits":    visits,
        "segment":   segment,
        "education": education
    })

    # Derived / mutated variables
    df["spend"]      = 120 + 18 * df["cont_norm"] + 7 * df["visits"] + np.random.normal(0, 15, n)

    df["score"]      = 65 + 8 * df["cont_norm"] - 4 * np.log1p(df["cont_skew"]) + np.random.normal(0, 6, n)

    df["engagement"] = 40 + 3 * df["visits"] + 2 * df["cont_norm"] + np.random.normal(0, 5, n)

    # scale() in R standardizes to mean=0, std=1 — equivalent to scipy zscore or manual
    cont_out_scaled  = (df["cont_out"] - df["cont_out"].mean()) / df["cont_out"].std()
    visits_scaled    = (df["visits"]   - df["visits"].mean())   / df["visits"].std()
    df["risk_index"] = 0.5 * cont_out_scaled + 0.4 * visits_scaled + np.random.normal(0, 0.6, n)
    main(plot , df )
