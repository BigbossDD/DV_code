import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
def Q2(data):
    #1- Produce a visualisation to show which of the nine areas has the best and the 
    #area with the worst chance of scoring a goal.
    plt.style.use('fast')
    plt.figure()

    sns.countplot(x = data.Position , hue=data.Goal , palette= 'Set2' , order = data.Position.value_counts().index)
    
    plt.title('the best and the area with the worst chance of scoring a goal')
    
    plt.xlabel('Position')
    plt.xticks(rotation=45)

    plt.ylabel('precentage of goals')

    plt.legend(title='Goal', loc='upper right')
    #and now adding the percentage of goals on top of the bars  <MIA , find a way to do it>
    
    sns.despine()
    plt.show()


    #2- Produce a visualisation to show which of the three main areas (Left, Middle, 
    #Right) has the best chance of scoring a goal and which has the worst chance of 
    #scoring.
    if data.Position.str.contains('L').any():
        data['Main_area'] = data.Position.apply(lambda x: 'Left' if 'L' in x else ('Middle' if 'C' in x else 'Right'))
    plt.style.use('fast')
    plt.figure()

    sns.countplot(x = data.Main_area , hue =data.Goal , palette= 'Set2',order=data.Main_area.value_counts().index   )

    #<maybe we will needed to add the % on top so , find a way to do it>
    sns.despine()
    plt.show()
    #3- Produce a visualisation showing the relationship between the foot used to 
    #convert the penalty and the nine areas.
    
    import matplotlib.ticker as mtick

    # compute percentages
    df = (
        data.groupby(['Position', 'Foot'])
        .size()
        .groupby(level=0,group_keys=False)
        .apply(lambda x: x / x.sum())
        .reset_index(name='pct')
    )

    # pivot for stacking
    pivot = df.pivot(index='Position', columns='Foot', values='pct')

    # plot stacked bars
    pivot.plot(kind='bar', stacked=True)

    # format
    plt.ylabel('Percentage')
    plt.xlabel('Position')
    plt.title('Foot distribution by Position')

    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))

    plt.legend(title='Foot')
    plt.show()
    #4- In no more than 150 words, give briefly a description of your methodology and 
    #conclusion for the three visualisations above. What is your recommendation for 
    #the team's manager?
   
   
    #Bonus: Use statistical test(s) to provide an exploratory test to confirm the 
    #association's results or lack-off between the position/main area and the scoring 
    #(2 points).

    return 
def Q3(data):
    pass


def main():
    data_q2 = pd.read_csv("/home/bigboss/PSUT/DV_code/MID_train_part2/data/Question 2.csv")
    #print(data_q2.head())
    '''
        ID  Game Goal Position   Foot
    0   1     1  Yes       LC  Right
    1   2     2  Yes       HR  Right
    2   3     3  Yes       MR  Right
    3   4     4  Yes       HL  Right
    4   5     5  Yes       MC   Left

HL: High Left
ML: Middle Left
LL: Low Left
HC: High Centre
MC: Middle Centre
LC: Low Centre
HR: High Right
MR: Middle Right
LR: Low Right

    '''
    Q2(data_q2)


    ###################



    return 



if __name__ == "__main__":  
    main()