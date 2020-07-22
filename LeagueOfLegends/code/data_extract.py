##---Data Extract Class----#

# 1. Pull champion data from Riot Games & competitive esports data from Oracle's Elixir website
# 2. Create raw data input

#import packages
import pandas as pd
import requests
import json
import pickle
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
import pdb

import warnings
warnings.filterwarnings('ignore')

class DataExtract:

    __slots__ = ['x_train', 'x_test']

    def __int__(self, matches_df):
        self.matches_df = matches_df

    #----Extract data from dictionary for each champion-patch-----#
    @staticmethod
    def extract_champions_data(champion_dict, champion_name, patch_number):

        #Initialize porter object
        porter = PorterStemmer()

        #Split into base stats & spells & classes
        stats = champion_dict['stats']
        passive = champion_dict['passive']
        spells = champion_dict['spells']
        classes = champion_dict['tags']

        #Some champions are only in one class. Make sure its still in a list []
        if type(classes) == 'str':
            classes = [classes]

        #Keywords for important game mechanics
        hard_cc_list = ['airborne', 'charm', 'flee', 'taunt', 'sleep', 'stun', 'supression', 'suspension', 
                        'stasis', 'pull', 'knock']
        soft_cc_list = ['blind', 'cripple', 'disarm', 'ground', 'knockdown', 'nearsight', 'root', 'silence', 'slow']
        gap_closer_list = ['dash', 'blink', 'leap', 'launch', 'movementspeed', 'teleport']
        vision_list = ['vision', 'sight']
        protection_list = ['shield', 'heal']

        #initialize variables
        hard_cc_value = 0
        soft_cc_value = 0
        spells_average_range_value = 0
        gap_closer_value = 0
        protection_value = 0

        #Passive
        passive = passive['description']
        passive = passive.lower()
        passive = passive.replace('movement speed', 'movementspeed') #specific logic for move speed
        passive = [porter.stem(x) for x in word_tokenize(passive)]
        hard_cc = len(list(set(passive) & set(hard_cc_list))) #unique key words
        soft_cc = len(list(set(passive) & set(soft_cc_list))) #unique key words
        protection = len(list(set(passive) & set(protection_list)))
        gap_closer = len(list(set(passive) & set(gap_closer_list)))

        #Update talley with passive
        hard_cc_value = hard_cc_value + hard_cc
        soft_cc_value = soft_cc_value + soft_cc
        protection_value = protection_value + protection
        gap_closer_value = gap_closer_value + gap_closer

        #Four spells
        for i in range(4):

            #----Extract text information from tooltip
            tooltip = spells[i]['tooltip'].lower()
            tooltip = tooltip.replace('movement speed', 'movementspeed') #specific logic for move speed
            tooltip = [porter.stem(x) for x in word_tokenize(tooltip)]
            hard_cc = len(list(set(tooltip) & set(hard_cc_list))) #unique key words
            soft_cc = len(list(set(tooltip) & set(soft_cc_list))) #unique key words
            protection = len(list(set(tooltip) & set(protection_list)))
            gap_closer = len(list(set(tooltip) & set(gap_closer_list)))
            spells_range = np.mean(spells[i]['range'])

            #Update talley
            hard_cc_value = hard_cc_value + hard_cc
            soft_cc_value = soft_cc_value + soft_cc
            protection_value = protection_value + protection
            gap_closer_value = gap_closer_value + gap_closer
            spells_average_range_value = np.mean([spells_average_range_value, spells_range])

             #-----Setup Dataframe---------#
             dict_temp = {'champion' : champion_name,
                            'patch' : patch,
                            'hard_cc_value' : hard_cc_value, 
                            'soft_cc_value' : soft_cc_value,
                            'spells_average_range_value' : spells_average_range_value,
                            'gap_closer_value' : gap_closer_value,
                            'protection_value' : protection_value,
                            'classes': [classes]} #make this a list

             dict_temp = {**stats, **dict_temp}
             output_df = pd.DataFrame(dict_temp, index = [0])
             output_df = output_df.set_index(['champion', 'patch'])

        return(output_df)

    #--- Pull data from Riot Games website and use extract_champions_data function-----#
    def data_pull(self, champions_list, patches, url):

        champion_output = pd.DataFrame()
        loops = len(patches)
        counter = 0

        for patch in patches:
            for champion in champions_list:

                url = url.format(patch, champion)

                try:
                    data = requests.get(url).json()
                    data = data['data'][champion]
                    champion = extract_champions_data(data, champion, patch)
                    champion_output = pd.concat([champion_output, champion])
                    #print('Added:' + champion)
                    #print(champion)
                except:
                    continue
                    #print('Request failed')
                    #do nothing if champion not in this patch, e.g. new releases

            counter = counter + 1
            print('Completed patch: ' + patch + ' | ' + str(counter) + '/' + str(loops))

        champion_output = champion_output.reset_index()

        #One-hot encode champion classes. This function is pretty sweet
        mlb = MultiLabelBinarizer()
        champion_output = champion_output.join(pd.DataFrame(mlb.fit_transform(champion_output.pop('classes')),
                                                        columns=mlb.classes_,
                                                        index=champion_output.index))

        self.champion_output = champion_output


    def create_player_profiles(self, window):

        #We will use a moving average based on the window.
        #Treat this as time dependent, 
        player_df = self.matches_df
        player_df = player_df[['date', 'gameid','player', 'position', 'side', 'champion', 'patch','kills','deaths',
                                'assists', 'damagetochampions', 'wardsplaced', 'wardskilled', 'totalgold', 
                                'total_cs','monsterkills', 'elementaldrakes']]

        player_df = player_df.sort_values(['player', 'date'], ascending = True)
        player_df = player_df.reset_index(drop=True) #need to drop index in new pandas version for groupby mean 

        #Values fields
        value_fields = ['player','kills', 'deaths', 'assists', 'damagetochampions','wardsplaced', 'wardskilled',
                        'totalgold', 'total_cs','monsterkills', 'elementaldrakes']

        player_df2 = player_df[value_fields]
        player_df2= player_df2.groupby(['player']).rolling(window).mean().shift(-window+1).reset_index().fillna(method = 'ffill')
        player_df2 = player_df2.drop('level_1', axis = 1)

        #Recombine
        player_df = player_df[['date', 'gameid', 'position', 'side', 'champion', 'patch']]
        player_df = pd.concat([player_df, player_df2], axis = 1)
        player_df['patch'] = player_df['patch'].apply(lambda x : reformat_patch_number(str(x)))
        self.player_profiles = player_df


    def create_team_profiles(self, window):
        #We will use a moving average based on the window.
        #Treat this as time dependent, 
        team_df = self.matches_df[matches_df['position'] == 'team']
        team_df = team_df[['date', 'gameid', 'side','team','gamelength','elementaldrakes']]
        team_df = team_df.sort_values(['team', 'date'], ascending = True)
        team_df = team_df.reset_index(drop=True) #need to drop index in new pandas version for groupby mean 

        #Values fields
        value_fields = ['team', 'gamelength', 'elementaldrakes']
        team_df2 = team_df[value_fields]
        team_df2 = team_df2.groupby(['team']).rolling(window).mean().shift(-window+1).reset_index().fillna(method = 'ffill')
        team_df2 = team_df2.drop('level_1', axis = 1)

        #Convert drakes to binary variable for soul point
        team_df2.loc[team_df2['elementaldrakes'] >= 3, 'soul_point'] = 1
        team_df2.loc[team_df2['elementaldrakes'] < 3, 'soul_point'] = 0
        team_df2.loc[team_df2['elementaldrakes'] < 3, 'soul_point'] = 0
        team_df2 = team_df2.drop(['elementaldrakes'], axis = 1)

        #Recombine
        team_df = team_df[['date', 'gameid', 'side']]
        team_df = pd.concat([team_df, team_df2], axis = 1)
        team_df = team_df.drop(['team'], axis = 1)
        self.team_profiles = team_df


    def head_to_head_players(self):

        #Split into blue and red
        player_profiles = self.player_profiles
        champion_output = self.champion_output
        player_profiles = player_profiles.set_index(['champion','patch'])
        champion_output = champion_output.set_index(['champion','patch'])

        player_profiles = pd.merge(player_profiles, champion_output, how = 'left', 
                                    left_index = True, right_index = True)
        player_profiles = player_profiles.reset_index()
        player_profiles = player_profiles.drop(['champion', 'patch', 'player'], axis = 1)

        blue = player_profiles[player_profiles['side'] == 'Blue']
        red = player_profiles[player_profiles['side'] == 'Red']
        blue = blue.drop('side', axis = 1)
        red = red.drop('side', axis = 1)
        blue = blue.set_index(['gameid', 'position', 'date'])
        red = red.set_index(['gameid', 'position', 'date'])

        #Rename red columns
        original_columns = red.columns
        red_columns = ['red_' + x for x in original_columns]
        red.columns = red_columns

        #Merge
        blue = pd.merge(blue, red, how = 'left', left_index = True, right_index = True)

        #Create delta columns
        delta_columns = []
        for column in original_columns:
            column_name = 'delta_' + column
            blue[column_name] = blue[column] - blue['red_' + column]
            delta_columns = delta_columns + [column_name]

        blue = blue[delta_columns]
        blue = blue.reset_index()
        blue = blue.drop(['position', 'date'], axis = 1)
        blue = blue.groupby(['gameid'], as_index = False).sum()
        self.h_t_h_players = blue  

    def head_to_head_teams(self):

        team_profiles = self.team_profiles
        blue_team_df = team_profiles[team_profiles['side'] == 'Blue']
        red_team_df = team_profiles[team_profiles['side'] == 'Red']
        blue_team_df = blue_team_df.drop('side', axis = 1)
        red_team_df = red_team_df.drop('side', axis = 1)
        blue_team_df = blue_team_df.set_index(['gameid', 'date'])
        red_team_df = red_team_df.set_index(['gameid', 'date'])

        #Rename red columns
        original_columns = red_team_df.columns
        red_columns = ['red_' + x for x in original_columns]
        red_team_df.columns = red_columns

        #Merge
        blue_team_df = pd.merge(blue_team_df, red_team_df, how = 'left', left_index = True, right_index = True)
        blue_team_df['average_gamelength'] = (blue_team_df['gamelength'] + blue_team_df['red_gamelength'])/2
        blue_team_df = blue_team_df.reset_index()
        blue_team_df = blue_team_df.drop(['date', 'gamelength', 'red_gamelength'], axis = 1)
         
        self.h_t_h_teams = blue_team_df


    def create_output(self):

        h_t_h_players = self.h_t_h_players
        h_t_h_teams = self.h_t_h_teams
        match

        h_t_h_players = h_t_h_players.set_index(['gameid'])
        h_t_h_teams = h_t_h_teams.set_index(['gameid'])
        h_t_h_complete = pd.merge(h_t_h_players, h_t_h_teams, how = 'left',
                                    left_index = True, right_index = True)
 
        target = match_data[['gameid', 'side', 'result']]
        target = target[(target['side'] == 'Blue')].drop('side', axis = 1).drop_duplicates()
        target = target.set_index('gameid')
        final = pd.merge(h_t_h_complete, target, how = 'left', left_index = True, right_index = True)

        x = final[final.columns[~final.columns.isin(['result'])]]
        y = final['result']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.full_output = final


    @staticmethod
    def save_pickle(variable, path):

        output_file = open(path,'wb')
        pickle.dump(x_train, output_file)
        output.close()

    @staticmethod
    def read_pickle(path):

        input_file = open(path,'rb')
        variable = pickle.load(input_file)
        input_file.close()
        return(variable)


DATA_DIR = '../data/'
MODEL_DIR = '../models/'
PATCH_END = 10.1
TRAIN_FILE = 
TEST_FILE =
WINDOW = 5

#If we run this directly through terminal or shell
if __name__ == "__main__":

    champions_list =
    patches_list = 

    data_extract_pipeline = DataExtract()

    data_extract_pipeline.data_pull(self, champions_list, patches, url)
    data_extract_pipeline.create_player_profiles(window = WINDOW)
    data_extract_pipeline.create_team_profiles(window = WINDOW)
    data_extract_pipeline.head_to_head_players()
    data_extract_pipeline.head_to_head_teams()
    data_extract_pipeline.create_output()

    #Save output









