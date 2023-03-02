import csv
import pandas as pd
import random
import math
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import log_loss, f1_score, accuracy_score

def getIndividualStat(statName,team,df,avg=False):
    """Calculate the summed total of a given stat for a given team.
    
    Parameters:
        statName(String) - the name of the stat to be found.
        team(String) - the abbrieviation of the desired team.
        df(DataFrame) - the available game data.
        gameWindow(Int) - the number of recent games to use.
        avg(Bool) - whether or not to calculate the average.
    
    Returns:
        totalFor(Float) - the sum or average of the stat for the team.
        totalAgainst(Float) - the sum or average of the stat againsst the team.
    """


    #create the dfs for the away and home games
    dfAway = df[(df["Away_Team"] == team)]
    dfHome = df[(df["Home_Team"] == team)]

    #create the strings
    awayString = "Away_" + statName
    homeString = "Home_" + statName

    #get the data for away games
    awayStatFor = dfAway[awayString]
    awayStatAgainst = dfAway[homeString]

    #get the data for home games
    homeStatFor = dfHome[homeString]
    homeStatAgainst = dfHome[awayString]

    #determine if we are calculating the sum or average
    match avg:
        case False:
            #get totals
            totalFor = (awayStatFor.sum() + homeStatFor.sum())
            totalAgainst = (awayStatAgainst.sum() + homeStatAgainst.sum())
            return totalFor, totalAgainst
        case True:
            #get averages
            #account for division by zero errors
            if (awayStatFor.shape[0] + homeStatFor.shape[0]) == 0:
                totalFor = 0
            else:
                totalFor = (awayStatFor.sum() + homeStatFor.sum())/(awayStatFor.shape[0] + homeStatFor.shape[0])

            if (awayStatAgainst.shape[0] + homeStatAgainst.shape[0]) == 0:
                totalAgainst = 0
            else:
                totalAgainst = (awayStatAgainst.sum() + homeStatAgainst.sum())/(awayStatAgainst.shape[0] + homeStatAgainst.shape[0])

            return totalFor, totalAgainst

def getWinsLoses(team,df):
    """Calculate the number of wins and loses for a team.
    
    Parameters:
        team(String) - the abbrieviation of the desired team.
        df(DataFrame) - the available game data.
        gameWindow(Int) - the number of recent games to use.
    
    Returns:
        wins(Int) - the number of games the team won.
        loses(Int) - the number of games the team lost.
    """

    #create the dfs for the
    dfAway = df[(df["Away_Team"] == team)]
    dfHome = df[(df["Home_Team"] == team)]

    #track wins and loses
    wins = 0
    loses = 0

    #count away wins
    for index, game in dfAway.iterrows():
        if game['Away_Score'] > game['Home_Score']:
            wins += 1
        else:
            loses += 1

    #count home wins
    for index, game in dfHome.iterrows():
        if game['Home_Score'] > game['Away_Score']:
            wins += 1
        else:
            loses += 1

    return [wins,loses]

def getStreak(team,df):
    """Calculate the number of wins or loses that have occured in a row for a team.
    
    Parameters:
        team(String) - the abbrieviation of the desired team.
        df(DataFrame) - the available game data.
        gameWindow(Int) - the number of recent games to use.
    
    Returns:
        streak(Int) - the number of games won or lost in a row.
    """

    #keep track of the streak
    streak = 0

    #iterate backwards through the games
    for index, game in df.iloc[::-1].iterrows():
        if game['Away_Team'] == team:
            if game['Away_Score'] > game['Home_Score']:
                if streak < 0:
                    break
                else:
                    streak += 1
            elif game['Away_Score'] < game['Home_Score']:
                if streak > 0:
                    break
                else:
                    streak -= 1
        elif game['Home_Team'] == team:
            if game['Away_Score'] < game['Home_Score']:
                if streak < 0:
                    break
                else:
                    streak += 1
            elif game['Away_Score'] > game['Home_Score']:
                if streak > 0:
                    break
                else:
                    streak -= 1
        else:
            print("Data Parsing has created an error in the streak variable")


    return streak

def calculateCORSI(shotAttemptsFor,shotAttemptsAgainst,average=False):
    """Calculate the CORSI, given shot attempts.
    
    Parameters:
        shotAttemptsFor(Int) - the number of shot attempts for.
        shotAttemptsAgainst(Int) - the number of shot attempts against.
        average(Bool) - if the calculate should be the average or the sum.
    
    Returns:
        CORSI - the average or sum for the CORSI
    """
    match average:
        case False:
            return shotAttemptsFor - shotAttemptsAgainst
        case True:
            #account for divided by zero errors
            if shotAttemptsAgainst == 0:
                return 0
            else:
                return shotAttemptsFor/(shotAttemptsAgainst + shotAttemptsFor)
            
def collectDataForTeam(team,df,gameDate,gameWindow):
    """Collect all the stats for a certain team before a game.
    
    Parameters:
        team(String) - the abbrieviation of the desired team.
        df(DataFrame) - the available game data.
        gameDate(DateTime) - the date the current game took place on
        gameWindow(Int) - the number of recent games to use.
    
    Returns:
        totals(List) - a list with all the statistics to be added to the main dataframe.
    """
    #only select required games
    df = df[(df["Away_Team"] == team) | (df["Home_Team"] == team)].tail(gameWindow)
    
    #list where the items to be placed in a row are stored
    totals = []

    
    #collect stats
    goals = getIndividualStat("Score",team,df,gameWindow)
    goalsAvg = getIndividualStat("Score",team,df,gameWindow,True)
    goals5v5 = getIndividualStat("Score5v5",team,df,gameWindow)
    goals5v5Avg = getIndividualStat("Score5v5",team,df,gameWindow,True)
    goalsClose5v5 = getIndividualStat("ScoreClose5v5",team,df,gameWindow)
    goalsClose5v5Avg = getIndividualStat("ScoreClose5v5",team,df,gameWindow,True)
    shots = getIndividualStat("Shots",team,df,gameWindow)
    shotsAvg = getIndividualStat("Shots",team,df,gameWindow,True)
    shotAttempts = getIndividualStat("Shot_Attempts",team,df,gameWindow)
    shotAttempts5v5 = getIndividualStat("Shot_Attempts5v5",team,df,gameWindow)
    shotAttemptsClose5v5 = getIndividualStat("Shot_AttemptsClose5v5",team,df,gameWindow)
    faceOffs = getIndividualStat("FO",team,df,gameWindow)
    hits = getIndividualStat("Hits",team,df,gameWindow)
    hitsAvg = getIndividualStat("Hits",team,df,gameWindow,True)
    pims = getIndividualStat("PIM",team,df,gameWindow)
    pimsAvg = getIndividualStat("PIM",team,df,gameWindow,True)
    blocks = getIndividualStat("Blocks",team,df,gameWindow)
    blocksAvg = getIndividualStat("Blocks",team,df,gameWindow,True)
    giveAways = getIndividualStat("Give",team,df,gameWindow)
    giveAwaysAvg = getIndividualStat("Give",team,df,gameWindow,True)
    takeAways = getIndividualStat("Take",team,df,gameWindow)
    takeAwaysAvg = getIndividualStat("Take",team,df,gameWindow,True)
    PPO = getIndividualStat("PPO",team,df,gameWindow)
    PPG = getIndividualStat("PPG",team,df,gameWindow)

    #get wins and loses
    winsLoses = getWinsLoses(team,df,gameWindow)

    #calculate shooting percentage
    if shots[0] == 0:
        shootingPer = 0
    else:
        shootingPer = goals[0]/shots[0]
    
    #calculate save percentage
    if shots[1] == 0:
        savePer = 0
    else:
        savePer = 1 - (goals[1]/shots[1])

    #PowerPlay Percentage
    if PPO[0] == 0:
        PPPer = 0
    else:
        PPPer = PPG[0]/PPO[0]

    #PenaltyKill Percentage
    if PPO[1] == 0:
        PKPer = 0
    else:
        PKPer = 1 - (PPG[1]/PPO[1])

    #calculate the PDO
    PDO = shootingPer + savePer
    
    #calculate CORSI
    shotAttemptsFor = shotAttempts[0]
    shotAttemptsAgainst = shotAttempts[1]
    CORSISum = calculateCORSI(shotAttemptsFor,shotAttemptsAgainst)
    CORSIAvg = calculateCORSI(shotAttemptsFor,shotAttemptsAgainst,True)

    #calculate CORSI for 5v5
    shotAttemptsFor5v5 = shotAttempts5v5[0]
    shotAttemptsAgainst5v5 = shotAttempts5v5[1]
    CORSI5v5Sum = calculateCORSI(shotAttemptsFor5v5,shotAttemptsAgainst5v5)
    CORSI5v5Avg = calculateCORSI(shotAttemptsFor5v5,shotAttemptsAgainst5v5,True)

    #calculate CORSI for 5v5 close situations
    shotAttemptsFor5v5Close = shotAttemptsClose5v5[0]
    shotAttemptsAgainst5v5Close = shotAttemptsClose5v5[1]
    CORSI5v5CloseSum = calculateCORSI(shotAttemptsFor5v5Close,shotAttemptsAgainst5v5Close)
    CORSI5v5CloseAvg = calculateCORSI(shotAttemptsFor5v5Close,shotAttemptsAgainst5v5Close,True)

    #calculate faceoff percentage
    if ((faceOffs[0] + faceOffs[1]) == 0):
        FOPer = 0
    else:
        FOPer = faceOffs[0]/(faceOffs[0] + faceOffs[1])
    
    #the totals
    totals = totals + [winsLoses[0],winsLoses[1],goals[0],goals[1],goalsAvg[0],goalsAvg[1],goals5v5[0],goals5v5[1],
            goals5v5Avg[0],goals5v5Avg[1],goalsClose5v5[0],goalsClose5v5[1],goalsClose5v5Avg[0],goalsClose5v5Avg[1],
            shots[0],shots[1],shotsAvg[0],shotsAvg[1],CORSISum,CORSIAvg,CORSI5v5Sum,CORSI5v5Avg,
            CORSI5v5CloseSum,CORSI5v5CloseAvg,FOPer,hits[0],hits[1],hitsAvg[0],hitsAvg[1],pims[0],pims[1],pimsAvg[0],pimsAvg[1],
            blocks[0],blocks[1],blocksAvg[0],blocksAvg[1],giveAways[0],giveAways[1],giveAwaysAvg[0],giveAwaysAvg[1],
            takeAways[0],takeAways[1],takeAwaysAvg[0],takeAwaysAvg[1],PPPer,PKPer,shootingPer,savePer,PDO]
 
    return totals
    
def createFrame(df,dfOut,gameWindow):
    """Create the dataframe of games.
    
    Parameters:
        df(DataFrame) - the available game data.
        dfOut(DataFrame) - the empty dataframe that each game will be added to.
        gameWindow(Int) - the number of recent games to use.
    
    Returns:
        dfOut(DataFrame) - the filled dataframe that contains all games.
    """
    #iterate through all games
    for i in df.Game_Id.unique():
        #home and away teams as well as the season
        away = df[df['Game_Id'] == i].Away_Team.unique()[0]
        home = df[df['Game_Id'] == i].Home_Team.unique()[0]
        season = df[df['Game_Id'] == i].season.unique()[0]
        date = df[df['Game_Id'] == i].Date.unique()[0]
        regOrOT = df[df['Game_Id'] == i].RegOrOT.unique()[0]

        #determine if the home or away team won
        if df[df['Game_Id'] == i]['Home_Score'].values[0] > df[df['Game_Id'] == i]['Away_Score'].values[0]:
            outcome = 1
        elif df[df['Game_Id'] == i]['Home_Score'].values[0] < df[df['Game_Id'] == i]['Away_Score'].values[0]:
            outcome = 0
        else:
            continue

        #make sure the right data is being used
        gameData = df[(df["Date"] < date) & (df["season"] == season)]
        
        #get away data for away teams
        awayTeamData = collectDataForTeam(away,gameData,date,gameWindow)

        #get home data
        homeTeamData = collectDataForTeam(home,gameData,date,gameWindow)

        #begin the row with the game, away team and home team ids.
        lst = [i,regOrOT,away,home,season]

        #represent features as home_value - away_value
        for j in range(len(awayTeamData)):
            lst.append(homeTeamData[j] - awayTeamData[j])
        
        #add the outcome
        lst.append(outcome)
        
        #add the row to the dataframe
        dfOut.loc[len(dfOut)] = lst

    return dfOut

def removeEarlyGames(df,games=20):
    """Remove the early games from each season.
    
    Parameters:
        df(DataFrame) - the available game data.
        games(int) - the number of early games to remove.
    
    Returns:
        df(DataFrame) - the updated DataFrame.
    """
    #get all team names
    teams = df.Home_Team.unique()

    #get all seasons
    seasons = df.season.unique()

    #where the games that will be removed are stored
    gamesToRemove = set()

    for i in seasons:
        #get the games from a given season
        season = df[df["season"] == i].sort_values("Game_Id")
        for j in teams:
            #get the teams first x games
            earlyGames = season[(season["Away_Team"] == j) | (season["Home_Team"] == j)].head(games)
            gameList = earlyGames.Game_Id.unique()
            gamesToRemove.update(gameList)
    
    #remove games from the dataframe
    df = df[~df.Game_Id.isin(list(gamesToRemove))]

    return df

def movingWindow(classifier,df,numOfWindows=19):
    """Use a moving window to evaluate the model.
    
    Parameters:
        classifier(Sklearn) - the classifier used for the data.
        df(DataFrame) - the available game data.
        numOfWindows(int) - the number of windows used for evaluation.
    
    Returns:
        accResults(list) - a list that contains all the results from the accuracy evaluation.
        f1Results(list) - a list that contains all the results from the f1 evaluation.
        logLoss(list) - a list that contains all the results from the log loss evaluation.
    """
    #reassign df
    newDF = df

    #store the number of rows and desired training/testing sizes
    rows = newDF.shape[0] #newDF
    trainingSize = 2460 #two seasons
    testSize = 1230 #one season

    #shifting interval
    interval = math.floor((rows - trainingSize - testSize)/numOfWindows)

    #where the train and test indices are stored
    train = []
    test = []
    i = 0

    #create the list of training and testing indices
    while i < (rows - trainingSize - testSize):
        train.append([i,i+trainingSize])
        test.append([i+trainingSize+1,i+trainingSize+testSize+1])
        i += interval

    #where results are stored
    accResults = []
    f1Results = []
    logLoss = []

    #separate X and y
    newXDF = newDF.loc[:,newDF.columns != 'Outcome']
    newYDF = newDF['Outcome'].astype('int32')

    for i in range(len(train)):
        #use RFECV and get the best features
        tempYTrain = newYDF[train[i][0]:train[i][1]+1]
        tempXTrain = newXDF[train[i][0]:train[i][1]+1]

        #create testing data
        tempXTest = newXDF[test[i][0]:test[i][1]+1]
        tempYTest = newYDF[test[i][0]:test[i][1]+1]

        #Score
        classifier.fit(tempXTrain,tempYTrain)
        preds = classifier.predict(tempXTest)
        proba = classifier.predict_proba(tempXTest)
        accResults.append(accuracy_score(tempYTest,preds))
        f1Results.append(f1_score(tempYTest,preds))
        logLoss.append(log_loss(tempYTest,proba))

    return accResults, f1Results, logLoss


def main(create,model):
    """The main function that controls all parts of the model creation.
    
    Parameters:
        create(bool) - if the data needs to be created. 
        model(String) - a string representing which ML model is to be used.
    """
    #if making the data is required
    if create:
        #all available data
        data = pd.read_csv('NHLData.csv')

        #make date column into datetime object
        data['Date'] = pd.to_datetime(data['Date'],format='%Y-%m-%d')

        #columns
        baseCols = ['Game_Id',
                    'RegOrOT',
                    'Away_Team',
                    'Home_Team',
                    'season',
                    "Wins",
                    "Loses",
                    "Goals",
                    "GoalsAgainst",
                    "GoalsAvg",
                    "GoalsAgainstAvg",
                    "Goals5v5",
                    "GoalsAgainst5v5",
                    "Goals5v5Avg",
                    "GoalsAgainst5v5Avg",
                    "GoalsClose5v5",
                    "GoalsAgainstClose5v5",
                    "GoalsClose5v5Avg",
                    "GoalsAgainstClose5v5Avg",
                    "Shots",
                    "ShotsAgainst",
                    "ShotsAvg",
                    "ShotsAgainstAvg",
                    "CORSI",
                    "CORSIAvg",
                    "CORSI5v5",
                    "CORSI5v5Avg",
                    "CORSIClose5v5",
                    "CORSIClose5v5Avg",
                    "FO",
                    "Hits",
                    "HitsAgainst",
                    "HitsAvg",
                    "HitsAgainstAvg",
                    "PIMS",
                    "PIMSAgainst",
                    "PIMSAvg",
                    "PIMSAgainstAvg",
                    "Blocks",
                    "BlocksAgainst",
                    "BlocksAvg",
                    "BlocksAgainstAvg",
                    "Give",
                    "GiveAgainst",
                    "GiveAvg",
                    "GiveAgainstAvg",
                    "Take",
                    "TakeAgainst",
                    "TakeAvg",
                    "TakeAgainstAvg",
                    "XGFor",
                    "XGAgainst",
                    "XGForAvg",
                    "XGAgainstAvg",
                    "XGFor5v5",
                    "XGAgainst5v5",
                    "XGFor5v5Avg",
                    "XGAgainst5v5Avg",
                    "XGFor5v5Close",
                    "XGAgainst5v5Close",
                    "XGFor5v5CloseAvg",
                    "XGAgainst5v5CloseAvg",
                    "PP%",
                    "PK%",
                    "sh%",
                    "sv%",
                    "PDO%",
                    "Outcome"]

        #train and test df
        trainDF = pd.DataFrame(columns=baseCols)

        #fill the dataframe
        newTrainDF = createFrame(data.copy(),trainDF.copy(),82)
        newDF = removeEarlyGames(newTrainDF)
        newDF = newDF[newDF['RegOrOT'] != 'OT']
        newDF = newDF.drop(['Game_Id','RegOrOT','Away_Team','Home_Team','season'], axis=1)

        #create csv
        newDF.to_csv("NHLOutput.csv",index=False)

    #logistic regression
    if model == "LR":
        classifier = LogisticRegression(max_iter=10000) 
    #SVM
    elif model == "SVM":
        classifier = SVC(probability=True)
    #random forest
    elif model == "RF":
        classifier = RandomForestClassifier(n_estimators=500,random_state=random.seed(1415))
    #xg boost
    elif model == "XG":
        classifier = XGBClassifier(n_estimators=500,random_state=random.seed(1415))
    #extra trees
    elif model == "EX":
        classifier = ExtraTreesClassifier(n_estimators=500,random_state=random.seed(1415))
    #decision trees
    elif model == "DT":
        classifier = DecisionTreeClassifier(random_state=random.seed(1415))

    #where results are stored
    scores = []
    f1 = []
    logs = []
    featureImportance = []

    #read in the game data
    newTrainDF = pd.read_csv("NHLOutput.csv")

    #score the model
    score, f1Score, logLoss, featureImp = movingWindow(classifier,newTrainDF)
    scores.append(score)
    f1.append(f1Score)
    logs.append(logLoss)
    featureImportance.append(featureImp)
    
    #write the moving window results to a csv
    with open('MWResults' + model + '.csv','w', encoding='UTF8', newline='') as f:

        writer = csv.writer(f)

        row = ["Accuracy","","F1 Score","","Log Loss"]
        writer.writerow(row)

        for i in range(len(scores[0])):
            row = []
            row.append(scores[i])
            row.append("")
            row.append(f1[i])
            row.append("")
            row.append(logs[i])
    
            writer.writerow(row)

    f.close

    return scores

main(True,"LR")
 