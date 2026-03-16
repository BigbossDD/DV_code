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
        sns.histplot(df.cont_skew , bins = 'fd'  , kde = True)#bins --> 'fd' , 'scott' 
        
        plt.title('normal')
        
        plt.xlabel('val')
        plt.ylabel('count')
        
        plt.show()

        #adding the mean and median and changing the style to classic '
        plt.figure()
        sns.histplot(df.cont_skew , bins = 'scott'  , kde = True)

        plt.title('normal')
        
        plt.xlabel('val')
        plt.ylabel('count')
        
        plt.axvline(df.cont_skew.mean() , color = 'red' , label = 'mean')
        plt.axvline(df.cont_skew.median() , color = 'green' , label = 'median')
        plt.legend()

        plt.style.use('classic')
        
        plt.show()

    #Density Plot   

    if what_plot == 2:
        #normal
        plt.figure()
        sns.kdeplot(df.cont_skew , fill = True , bw_method='scott' , color='purple' ) #bw_method --> 'scott' , 'silverman'
        
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
        stats.probplot(df.cont_skew , dist = 'norm' , plot = plt)
        
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
    #violin plot
    if what_plot == 6 :
        plt.figure()
        sns.violinplot(x = df.cont_skew , color = 'red'  ,bw_method= 'silverman' ) #bw_method --> 'scott' , 'silverman'
        
        plt.title('Violin Plot')
        
        plt.xlabel('val')
        plt.ylabel('')

        plt.style.use('classic')

        plt.show()
    #beeswarm plot
    if what_plot == 7 :
        plt.figure()
        sns.swarmplot(x = df.cont_skew , color = 'magenta')
        
        plt.title('Beeswarm Plot')
        
        plt.xlabel('val')
        plt.ylabel('')

        plt.style.use('classic')

        plt.show()
    #Ridge plot
    if what_plot == 8 :
        plt.figure()
        sns.kdeplot(df.cont_skew , fill = True , bw_method='scott' , color='purple' ) #bw_method --> 'scott' , 'silverman'
        
        plt.title('Ridge Plot')
        
        plt.xlabel('val')
        plt.ylabel('Density')
        
        plt.style.use('classic')

        plt.show()
    #Line plot
    if what_plot == 9 :
        plt.figure()
        sns.lineplot(x = range(len(df.cont_skew)), y = df.cont_skew , color = 'blue')
        
        plt.title('Line Plot')
        
        plt.xlabel('Index')
        plt.ylabel('val')
        
        plt.style.use('classic')

        plt.show()
    return


if __name__ == "__main__":
    plot = 8
    print('starting !!!')
    np.random.seed(42)

    n = 500

    # cont_norm
    cont_norm = np.random.normal(0, 1, n)

    # cont_skew (log-normal distribution)
    cont_skew = np.random.lognormal(0, 0.6, n)

    # cont_out (normal data with 5 outliers)
    x = np.concatenate([
        np.random.normal(5, 1, n - 5),
        np.random.normal(12, 0.5, 5)
    ])
    np.random.shuffle(x)
    cont_out = x

    # visits (Poisson)
    visits = np.random.poisson(3, n)

    # segment (categorical with probabilities)
    segment = np.random.choice(
        ["A", "B", "C"],
        size=n,
        p=[0.5, 0.3, 0.2]
    )

    # education (ordered categorical)
    education_levels = ["High school", "Bachelor", "Master", "PhD"]
    education = np.random.choice(
        education_levels,
        size=n,
        p=[0.3, 0.4, 0.2, 0.1]
    )

    education = pd.Categorical(
        education,
        categories=education_levels,
        ordered=True
    )

    # Create DataFrame
    df = pd.DataFrame({
        "cont_norm": cont_norm,
        "cont_skew": cont_skew,
        "cont_out": cont_out,
        "visits": visits,
        "segment": segment,
        "education": education
    })
    main(plot , df )
