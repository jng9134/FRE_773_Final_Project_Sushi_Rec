import os
assert 'COMET_API_KEY' in os.environ and os.environ['COMET_API_KEY']
assert 'MY_PROJECT_NAME' in os.environ and os.environ['MY_PROJECT_NAME']
print("Running experiment for project: {}".format(os.environ['MY_PROJECT_NAME']))
from comet_ml import Experiment

from metaflow import FlowSpec, step, Parameter, IncludeFile, current
from datetime import datetime

# make sure we are running locally for this
assert os.environ.get('METAFLOW_DEFAULT_DATASTORE', 'local') == 'local'
assert os.environ.get('METAFLOW_DEFAULT_ENVIRONMENT', 'local') == 'local'

class sushi_rec_flow(FlowSpec):
    
    CUSTOMER_FEATURES_FILE = IncludeFile(
        'customer_features',
        help = 'csv file with customer (user) features',
        is_text = False,
        default = 'customer_features.csv')
    
    SUSHI_FEATURES_FILE = IncludeFile(
        'sushi_features',
        help = 'csv file with sushi (item) featues',
        is_text = False,
        default = 'sushi_features.csv')

    SUSHI_RATINGS_FILE = IncludeFile(
        'sushi_ratings',
        help = 'csv file with user item interactions',
        is_text = False,
        default = 'sushi_ratings_data.csv')

    TEST_SPLIT = Parameter(
        name = 'test_split',
        help = 'Determining the split of the dataset for testing',
        default = 0.25
    )

    NUM_USERS = Parameter(
        name = 'num_users',
        help = 'number of users in dataset',
        default = 100
    )

    NUM_ITEMS = Parameter(
        name = 'num_items',
        help = 'number of items in dataset',
        default = 5000
    )

    SAMPLE_USER = Parameter(
        name = 'sample_user',
        help = 'any user in the data set for testing a recommendation',
        default = 800
    )

    @step
    def start(self):
        """
        Start up and print out some info to make sure everything is ok metaflow-side

        """
        print("Starting up at {}".format(datetime.utcnow()))
        # debug printing - this is from https://docs.metaflow.org/metaflow/tagging
        # to show how information about the current run can be accessed programmatically
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        print("username: %s" % current.username)
        self.next(self.load_data)
    
    """
    Because of how lightfm works, I could not separte the train, test, and evaluate steps and do the slicing
    So that is why everything is duplicated many times
    In metaflow I run differnt models on DIFFERENT SLICES OF DATA in parallel
    """

    @step
    def load_data(self): 
        """
        Read the data in from the static file
        """
        import pandas as pd

        self.customer_features_df = pd.read_csv('customer_features.csv')
        self.sushi_features_df = pd.read_csv('sushi_features.csv')
        self.sushi_ratings_df = pd.read_csv('sushi_ratings_data.csv')

        self.sushi_ratings_no_user = self.sushi_ratings_df.drop(columns=['user_id'])

        #initially in the dataset there is only 100 items and 5000 users 
        #making this list to use later for lightfm model
        self.sushi_id = list(range(0,self.NUM_USERS))
        self.user_id = list(range(0,self.NUM_ITEMS))

        #split data
        gender = list(self.customer_features_df.loc[:,'gender'])
        ages = list(self.customer_features_df.loc[:,'age'])
        east_or_west = list(self.customer_features_df.loc[:,'east_or_west'])
        most_frequently_sold = list(self.sushi_features_df.loc[:,'most_frequently'])

        males = []
        females = []
        iterator = 0
        for i in range(len(gender)):
            if(gender[iterator] == 0):
                males.append(iterator)
            else:
                females.append(iterator)
            iterator+=1

        self.sushi_ratings_no_user_males_only = self.sushi_ratings_no_user.drop(index= females)
        self.sushi_ratings_no_user_females_only = self.sushi_ratings_no_user.drop(index= males)

        under_30 = []
        over_30 = []

        iterator = 0
        for i in range(len(ages)):
            if(ages[iterator] >= 2):
                over_30.append(iterator)
            else:
                under_30.append(iterator)
            iterator+=1

        self.sushi_ratings_no_user_under_30 = self.sushi_ratings_no_user.drop(index= over_30)
        self.sushi_ratings_no_user_over_30 = self.sushi_ratings_no_user.drop(index= under_30)

        east = []
        west = []

        iterator = 0
        for i in range(len(east_or_west)):
            if(east_or_west[iterator] == 0):
                east.append(iterator)
            else:
                west.append(iterator)
            iterator+=1

        self.sushi_ratings_no_user_east = self.sushi_ratings_no_user.drop(index= west)
        self.sushi_ratings_no_user_west = self.sushi_ratings_no_user.drop(index= east)


        more_common = []
        less_common = []

        iterator = 0
        for i in range(len(most_frequently_sold)):
            if(most_frequently_sold[iterator] < .65):
                less_common.append(iterator)
            else:
                more_common.append(iterator)
            iterator += 1

        self.sushi_ratings_no_user_common = self.sushi_ratings_no_user.drop(self.sushi_ratings_no_user.columns[less_common], axis = 1)
        self.sushi_ratings_no_user_less_common = self.sushi_ratings_no_user.drop(self.sushi_ratings_no_user.columns[more_common], axis = 1)

        # go to the next step
        self.next(self.prepare_data_for_model, self.prepare_male, self.prepare_female, 
        self.prepare_under_30, self.prepare_over_30, self.prepare_east, self.prepare_west, 
        self.prepare_common, self.prepare_uncommon)

    @step
    def prepare_data_for_model(self):
        from lightfm.data import Dataset
        from lightfm import LightFM
        from lightfm import cross_validation
        from lightfm.evaluation import precision_at_k
        from lightfm.evaluation import auc_score
        from lightfm.evaluation import recall_at_k
        import numpy as np
        # find all interactions where an interaction is defined as someone who positively rated a sushi (greater than 2)
        # returns the interaction matrix where if a user rated a sushi >= 2 then it will be have a 'True' in that cell, otherwise it will have 'False'
        # can test different models - all ratings with weights, all ratings higher than 2, all ratings higher than 3, etc
        self.positive_rankings= self.sushi_ratings_no_user >= 3

        # interaction_list is a list of tuples that represent whether a user liked an item - find all the 'True' values from df
        # for example [(1,6),(1,8),(2,2)...]
        # represents user 1 liked item 6 and 8, while user 2 liked item 2...
        self.interaction_list = [(self.positive_rankings[col][self.positive_rankings[col].eq(True)].index[i], self.positive_rankings.columns.get_loc(col)) for col in self.positive_rankings.columns for i in range(len(self.positive_rankings[col][self.positive_rankings[col].eq(True)].index))]
        self.weight_list = []


        # find get the scores of each True item in matrix 
        # the scores will be used as weights for the lightfm model
        for x in self.interaction_list:
            weight = self.sushi_ratings_no_user.iloc[x] / 4
            self.weight_list.append((weight,))

        self.interaction_weight_list = []
        for (x, y) in zip(self.interaction_list, self.weight_list):
            self.interaction_weight_list.append(x + y)

        #creating a lightfm dataset using the list of user_id (0-4999) and list of sushi_id(0-99)
        self.dataset = Dataset()
        self.dataset.fit(
            set(self.user_id), 
            set(self.sushi_id)
            )
   
        self.interactions, self.weights = self.dataset.build_interactions(
            self.interaction_weight_list)
       
        self.train_split, self.test_split = cross_validation.random_train_test_split(self.weights, test_percentage = 0.25, random_state = 2022 )
        

        # Train/fit model
        self.model = LightFM(
            no_components=150,
            learning_rate=0.05,
            loss='warp-kos',
            random_state=2023)

        self.model.fit(
            self.train_split,
            epochs=10, verbose=True)

        self.test_set_likes = {key: [] for key in self.test_split.nonzero()[0]}
        for test_row, test_col in zip(*self.test_split.nonzero()):
            self.test_set_likes[test_row].append(test_col)

    
        temp = self.sushi_ratings_df.copy()
        temp = temp.replace(-1,np.NaN)
        temp = temp.describe().drop(columns='user_id')
        mean_rate = temp.transpose().reset_index().rename(columns={'index':'sushi'})
        mean_rate = mean_rate.sort_values(by=['mean'],ascending=False)
        self.top_ten = mean_rate['sushi'][:10].reset_index()
        self.top_ten = self.top_ten['index'].tolist()

        self.train_auc = auc_score(self.model, self.test_split).mean()
        self.precision = precision_at_k(self.model, self.test_split, self.train_split, k=10).mean()
        self.hit_rate = recall_at_k(self.model, self.test_split, self.train_split, k = 10).mean()

        print(self.train_auc)
        print(self.precision)
        print(self.hit_rate)

        #preparing Popularity Baseline
        self.test_set_likes = {key: [] for key in self.test_split.nonzero()[0]}
        for test_row, test_col in zip(*self.test_split.nonzero()):
            self.test_set_likes[test_row].append(test_col)

    
        temp = self.sushi_ratings_df.copy()
        temp = temp.replace(-1,np.NaN)
        temp = temp.describe().drop(columns='user_id')
        mean_rate = temp.transpose().reset_index().rename(columns={'index':'sushi'})
        mean_rate = mean_rate.sort_values(by=['mean'],ascending=False)
        self.top_ten = mean_rate['sushi'][:10].reset_index()
        self.top_ten = self.top_ten['index'].tolist()

        #calculating baseline popularity (top 10 highest avg rated sushis)
        Total_hit_ = 0
        Total_test_case_ = 0
        for user_x in self.test_set_likes.keys():
            hit = 0
            for a in self.test_set_likes[user_x]:
                Total_test_case_ += 1
                if a in self.top_ten:
                    hit += 1
                    Total_hit_ += 1

        self.popular_hit_rate_ = Total_hit_/ Total_test_case_
        print("The test case total {} The hitted total {} The hit rate of the system is {}".format(Total_test_case_, Total_hit_,self.popular_hit_rate_))

        
        

        self.next(self.join)
    
    @step
    def prepare_male(self):
        #prepares data male data only
        from lightfm.data import Dataset
        from lightfm import LightFM
        from lightfm import cross_validation
        from lightfm.evaluation import precision_at_k
        from lightfm.evaluation import auc_score
        from lightfm.evaluation import recall_at_k
        import numpy as np
        # find all interactions where an interaction is defined as someone who positively rated a sushi (greater than 2)
        # returns the interaction matrix where if a user rated a sushi >= 2 then it will be have a 'True' in that cell, otherwise it will have 'False'
        # can test different models - all ratings with weights, all ratings higher than 2, all ratings higher than 3, etc
        self.positive_rankings= self.sushi_ratings_no_user_males_only >= 3

        # interaction_list is a list of tuples that represent whether a user liked an item - find all the 'True' values from df
        # for example [(1,6),(1,8),(2,2)...]
        # represents user 1 liked item 6 and 8, while user 2 liked item 2...
        self.interaction_list = [(self.positive_rankings[col][self.positive_rankings[col].eq(True)].index[i], self.positive_rankings.columns.get_loc(col)) for col in self.positive_rankings.columns for i in range(len(self.positive_rankings[col][self.positive_rankings[col].eq(True)].index))]
        self.weight_list = []


        # find get the scores of each True item in matrix 
        # the scores will be used as weights for the lightfm model
        for x in self.interaction_list:
            weight = self.sushi_ratings_no_user.iloc[x] / 4
            self.weight_list.append((weight,))

        self.interaction_weight_list = []
        for (x, y) in zip(self.interaction_list, self.weight_list):
            self.interaction_weight_list.append(x + y)

        #creating a lightfm dataset using the list of user_id (0-4999) and list of sushi_id(0-99)
        self.dataset = Dataset()
        self.dataset.fit(
            set(self.user_id), 
            set(self.sushi_id)
            )
   
        self.interactions, self.weights = self.dataset.build_interactions(
            self.interaction_weight_list)
       
        self.train_split, self.test_split = cross_validation.random_train_test_split(self.weights, test_percentage = 0.25, random_state = 2022 )
        

        # Train/fit model
        self.model = LightFM(
            no_components=150,
            learning_rate=0.05,
            loss='warp-kos',
            random_state=2023)

        self.model.fit(
            self.train_split,
            epochs=10, verbose=True)

        self.test_set_likes = {key: [] for key in self.test_split.nonzero()[0]}
        for test_row, test_col in zip(*self.test_split.nonzero()):
            self.test_set_likes[test_row].append(test_col)

    
        temp = self.sushi_ratings_df.copy()
        temp = temp.replace(-1,np.NaN)
        temp = temp.describe().drop(columns='user_id')
        mean_rate = temp.transpose().reset_index().rename(columns={'index':'sushi'})
        mean_rate = mean_rate.sort_values(by=['mean'],ascending=False)
        self.top_ten = mean_rate['sushi'][:10].reset_index()
        self.top_ten = self.top_ten['index'].tolist()

        self.train_auc = auc_score(self.model, self.test_split).mean()
        self.precision = precision_at_k(self.model, self.test_split, self.train_split, k=10).mean()
        self.hit_rate = recall_at_k(self.model, self.test_split, self.train_split, k = 10).mean()

        print(self.train_auc)
        print(self.precision)
        print(self.hit_rate)
        
        

        self.next(self.join)
    
    @step
    def prepare_female(self):
        from lightfm.data import Dataset
        from lightfm import LightFM
        from lightfm import cross_validation
        from lightfm.evaluation import precision_at_k
        from lightfm.evaluation import auc_score
        from lightfm.evaluation import recall_at_k
        import numpy as np
        # find all interactions where an interaction is defined as someone who positively rated a sushi (greater than 2)
        # returns the interaction matrix where if a user rated a sushi >= 2 then it will be have a 'True' in that cell, otherwise it will have 'False'
        # can test different models - all ratings with weights, all ratings higher than 2, all ratings higher than 3, etc
        self.positive_rankings= self.sushi_ratings_no_user_females_only >= 3

        # interaction_list is a list of tuples that represent whether a user liked an item - find all the 'True' values from df
        # for example [(1,6),(1,8),(2,2)...]
        # represents user 1 liked item 6 and 8, while user 2 liked item 2...
        self.interaction_list = [(self.positive_rankings[col][self.positive_rankings[col].eq(True)].index[i], self.positive_rankings.columns.get_loc(col)) for col in self.positive_rankings.columns for i in range(len(self.positive_rankings[col][self.positive_rankings[col].eq(True)].index))]
        self.weight_list = []


        # find get the scores of each True item in matrix 
        # the scores will be used as weights for the lightfm model
        for x in self.interaction_list:
            weight = self.sushi_ratings_no_user.iloc[x] / 4
            self.weight_list.append((weight,))

        self.interaction_weight_list = []
        for (x, y) in zip(self.interaction_list, self.weight_list):
            self.interaction_weight_list.append(x + y)

        #creating a lightfm dataset using the list of user_id (0-4999) and list of sushi_id(0-99)
        self.dataset = Dataset()
        self.dataset.fit(
            set(self.user_id), 
            set(self.sushi_id)
            )
   
        self.interactions, self.weights = self.dataset.build_interactions(
            self.interaction_weight_list)
       
        self.train_split, self.test_split = cross_validation.random_train_test_split(self.weights, test_percentage = 0.25, random_state = 2022 )
        

        # Train/fit model
        self.model = LightFM(
            no_components=150,
            learning_rate=0.05,
            loss='warp-kos',
            random_state=2023)

        self.model.fit(
            self.train_split,
            epochs=10, verbose=True)

        self.test_set_likes = {key: [] for key in self.test_split.nonzero()[0]}
        for test_row, test_col in zip(*self.test_split.nonzero()):
            self.test_set_likes[test_row].append(test_col)

    
        temp = self.sushi_ratings_df.copy()
        temp = temp.replace(-1,np.NaN)
        temp = temp.describe().drop(columns='user_id')
        mean_rate = temp.transpose().reset_index().rename(columns={'index':'sushi'})
        mean_rate = mean_rate.sort_values(by=['mean'],ascending=False)
        self.top_ten = mean_rate['sushi'][:10].reset_index()
        self.top_ten = self.top_ten['index'].tolist()

        self.train_auc = auc_score(self.model, self.test_split).mean()
        self.precision = precision_at_k(self.model, self.test_split, self.train_split, k=10).mean()
        self.hit_rate = recall_at_k(self.model, self.test_split, self.train_split, k = 10).mean()

        print(self.train_auc)
        print(self.precision)
        print(self.hit_rate)
        
        self.next(self.join)

    @step
    def prepare_under_30(self):
        from lightfm.data import Dataset
        from lightfm import LightFM
        from lightfm import cross_validation
        from lightfm.evaluation import precision_at_k
        from lightfm.evaluation import auc_score
        from lightfm.evaluation import recall_at_k
        import numpy as np
        # find all interactions where an interaction is defined as someone who positively rated a sushi (greater than 2)
        # returns the interaction matrix where if a user rated a sushi >= 2 then it will be have a 'True' in that cell, otherwise it will have 'False'
        # can test different models - all ratings with weights, all ratings higher than 2, all ratings higher than 3, etc
        self.positive_rankings= self.sushi_ratings_no_user_under_30 >= 3

        # interaction_list is a list of tuples that represent whether a user liked an item - find all the 'True' values from df
        # for example [(1,6),(1,8),(2,2)...]
        # represents user 1 liked item 6 and 8, while user 2 liked item 2...
        self.interaction_list = [(self.positive_rankings[col][self.positive_rankings[col].eq(True)].index[i], self.positive_rankings.columns.get_loc(col)) for col in self.positive_rankings.columns for i in range(len(self.positive_rankings[col][self.positive_rankings[col].eq(True)].index))]
        self.weight_list = []


        # find get the scores of each True item in matrix 
        # the scores will be used as weights for the lightfm model
        for x in self.interaction_list:
            weight = self.sushi_ratings_no_user.iloc[x] / 4
            self.weight_list.append((weight,))

        self.interaction_weight_list = []
        for (x, y) in zip(self.interaction_list, self.weight_list):
            self.interaction_weight_list.append(x + y)

        #creating a lightfm dataset using the list of user_id (0-4999) and list of sushi_id(0-99)
        self.dataset = Dataset()
        self.dataset.fit(
            set(self.user_id), 
            set(self.sushi_id)
            )
   
        self.interactions, self.weights = self.dataset.build_interactions(
            self.interaction_weight_list)
       
        self.train_split, self.test_split = cross_validation.random_train_test_split(self.weights, test_percentage = 0.25, random_state = 2022 )
        

        # Train/fit model
        self.model = LightFM(
            no_components=150,
            learning_rate=0.05,
            loss='warp-kos',
            random_state=2023)

        self.model.fit(
            self.train_split,
            epochs=10, verbose=True)

        self.test_set_likes = {key: [] for key in self.test_split.nonzero()[0]}
        for test_row, test_col in zip(*self.test_split.nonzero()):
            self.test_set_likes[test_row].append(test_col)

    
        temp = self.sushi_ratings_df.copy()
        temp = temp.replace(-1,np.NaN)
        temp = temp.describe().drop(columns='user_id')
        mean_rate = temp.transpose().reset_index().rename(columns={'index':'sushi'})
        mean_rate = mean_rate.sort_values(by=['mean'],ascending=False)
        self.top_ten = mean_rate['sushi'][:10].reset_index()
        self.top_ten = self.top_ten['index'].tolist()

        self.train_auc = auc_score(self.model, self.test_split).mean()
        self.precision = precision_at_k(self.model, self.test_split, self.train_split, k=10).mean()
        self.hit_rate = recall_at_k(self.model, self.test_split, self.train_split, k = 10).mean()

        print(self.train_auc)
        print(self.precision)
        print(self.hit_rate)

        self.next(self.join)

    @step
    def prepare_over_30(self):
        from lightfm.data import Dataset
        from lightfm import LightFM
        from lightfm import cross_validation
        from lightfm.evaluation import precision_at_k
        from lightfm.evaluation import auc_score
        from lightfm.evaluation import recall_at_k
        import numpy as np
        # find all interactions where an interaction is defined as someone who positively rated a sushi (greater than 2)
        # returns the interaction matrix where if a user rated a sushi >= 2 then it will be have a 'True' in that cell, otherwise it will have 'False'
        # can test different models - all ratings with weights, all ratings higher than 2, all ratings higher than 3, etc
        self.positive_rankings= self.sushi_ratings_no_user_over_30 >= 3

        # interaction_list is a list of tuples that represent whether a user liked an item - find all the 'True' values from df
        # for example [(1,6),(1,8),(2,2)...]
        # represents user 1 liked item 6 and 8, while user 2 liked item 2...
        self.interaction_list = [(self.positive_rankings[col][self.positive_rankings[col].eq(True)].index[i], self.positive_rankings.columns.get_loc(col)) for col in self.positive_rankings.columns for i in range(len(self.positive_rankings[col][self.positive_rankings[col].eq(True)].index))]
        self.weight_list = []


        # find get the scores of each True item in matrix 
        # the scores will be used as weights for the lightfm model
        for x in self.interaction_list:
            weight = self.sushi_ratings_no_user.iloc[x] / 4
            self.weight_list.append((weight,))

        self.interaction_weight_list = []
        for (x, y) in zip(self.interaction_list, self.weight_list):
            self.interaction_weight_list.append(x + y)

        #creating a lightfm dataset using the list of user_id (0-4999) and list of sushi_id(0-99)
        self.dataset = Dataset()
        self.dataset.fit(
            set(self.user_id), 
            set(self.sushi_id)
            )
   
        self.interactions, self.weights = self.dataset.build_interactions(
            self.interaction_weight_list)
       
        self.train_split, self.test_split = cross_validation.random_train_test_split(self.weights, test_percentage = 0.25, random_state = 2022 )
        

        # Train/fit model
        self.model = LightFM(
            no_components=150,
            learning_rate=0.05,
            loss='warp-kos',
            random_state=2023)

        self.model.fit(
            self.train_split,
            epochs=10, verbose=True)

        self.test_set_likes = {key: [] for key in self.test_split.nonzero()[0]}
        for test_row, test_col in zip(*self.test_split.nonzero()):
            self.test_set_likes[test_row].append(test_col)

    
        temp = self.sushi_ratings_df.copy()
        temp = temp.replace(-1,np.NaN)
        temp = temp.describe().drop(columns='user_id')
        mean_rate = temp.transpose().reset_index().rename(columns={'index':'sushi'})
        mean_rate = mean_rate.sort_values(by=['mean'],ascending=False)
        self.top_ten = mean_rate['sushi'][:10].reset_index()
        self.top_ten = self.top_ten['index'].tolist()

        self.train_auc = auc_score(self.model, self.test_split).mean()
        self.precision = precision_at_k(self.model, self.test_split, self.train_split, k=10).mean()
        self.hit_rate = recall_at_k(self.model, self.test_split, self.train_split, k = 10).mean()

        print(self.train_auc)
        print(self.precision)
        print(self.hit_rate)

        self.next(self.join)
    
    @step 
    def prepare_east(self):
        from lightfm.data import Dataset
        from lightfm import LightFM
        from lightfm import cross_validation
        from lightfm.evaluation import precision_at_k
        from lightfm.evaluation import auc_score
        from lightfm.evaluation import recall_at_k
        import numpy as np
        # find all interactions where an interaction is defined as someone who positively rated a sushi (greater than 2)
        # returns the interaction matrix where if a user rated a sushi >= 2 then it will be have a 'True' in that cell, otherwise it will have 'False'
        # can test different models - all ratings with weights, all ratings higher than 2, all ratings higher than 3, etc
        self.positive_rankings= self.sushi_ratings_no_user_east >= 3

        # interaction_list is a list of tuples that represent whether a user liked an item - find all the 'True' values from df
        # for example [(1,6),(1,8),(2,2)...]
        # represents user 1 liked item 6 and 8, while user 2 liked item 2...
        self.interaction_list = [(self.positive_rankings[col][self.positive_rankings[col].eq(True)].index[i], self.positive_rankings.columns.get_loc(col)) for col in self.positive_rankings.columns for i in range(len(self.positive_rankings[col][self.positive_rankings[col].eq(True)].index))]
        self.weight_list = []


        # find get the scores of each True item in matrix 
        # the scores will be used as weights for the lightfm model
        for x in self.interaction_list:
            weight = self.sushi_ratings_no_user.iloc[x] / 4
            self.weight_list.append((weight,))

        self.interaction_weight_list = []
        for (x, y) in zip(self.interaction_list, self.weight_list):
            self.interaction_weight_list.append(x + y)

        #creating a lightfm dataset using the list of user_id (0-4999) and list of sushi_id(0-99)
        self.dataset = Dataset()
        self.dataset.fit(
            set(self.user_id), 
            set(self.sushi_id)
            )
   
        self.interactions, self.weights = self.dataset.build_interactions(
            self.interaction_weight_list)
       
        self.train_split, self.test_split = cross_validation.random_train_test_split(self.weights, test_percentage = 0.25, random_state = 2022 )
        

        # Train/fit model
        self.model = LightFM(
            no_components=150,
            learning_rate=0.05,
            loss='warp-kos',
            random_state=2023)

        self.model.fit(
            self.train_split,
            epochs=10, verbose=True)

        self.test_set_likes = {key: [] for key in self.test_split.nonzero()[0]}
        for test_row, test_col in zip(*self.test_split.nonzero()):
            self.test_set_likes[test_row].append(test_col)

    
        temp = self.sushi_ratings_df.copy()
        temp = temp.replace(-1,np.NaN)
        temp = temp.describe().drop(columns='user_id')
        mean_rate = temp.transpose().reset_index().rename(columns={'index':'sushi'})
        mean_rate = mean_rate.sort_values(by=['mean'],ascending=False)
        self.top_ten = mean_rate['sushi'][:10].reset_index()
        self.top_ten = self.top_ten['index'].tolist()

        self.train_auc = auc_score(self.model, self.test_split).mean()
        self.precision = precision_at_k(self.model, self.test_split, self.train_split, k=10).mean()
        self.hit_rate = recall_at_k(self.model, self.test_split, self.train_split, k = 10).mean()

        print(self.train_auc)
        print(self.precision)
        print(self.hit_rate)

        self.next(self.join)
    
    @step
    def prepare_west(self):
        from lightfm.data import Dataset
        from lightfm import LightFM
        from lightfm import cross_validation
        from lightfm.evaluation import precision_at_k
        from lightfm.evaluation import auc_score
        from lightfm.evaluation import recall_at_k
        import numpy as np
        # find all interactions where an interaction is defined as someone who positively rated a sushi (greater than 2)
        # returns the interaction matrix where if a user rated a sushi >= 2 then it will be have a 'True' in that cell, otherwise it will have 'False'
        # can test different models - all ratings with weights, all ratings higher than 2, all ratings higher than 3, etc
        self.positive_rankings= self.sushi_ratings_no_user_west >= 3

        # interaction_list is a list of tuples that represent whether a user liked an item - find all the 'True' values from df
        # for example [(1,6),(1,8),(2,2)...]
        # represents user 1 liked item 6 and 8, while user 2 liked item 2...
        self.interaction_list = [(self.positive_rankings[col][self.positive_rankings[col].eq(True)].index[i], self.positive_rankings.columns.get_loc(col)) for col in self.positive_rankings.columns for i in range(len(self.positive_rankings[col][self.positive_rankings[col].eq(True)].index))]
        self.weight_list = []


        # find get the scores of each True item in matrix 
        # the scores will be used as weights for the lightfm model
        for x in self.interaction_list:
            weight = self.sushi_ratings_no_user.iloc[x] / 4
            self.weight_list.append((weight,))

        self.interaction_weight_list = []
        for (x, y) in zip(self.interaction_list, self.weight_list):
            self.interaction_weight_list.append(x + y)

        #creating a lightfm dataset using the list of user_id (0-4999) and list of sushi_id(0-99)
        self.dataset = Dataset()
        self.dataset.fit(
            set(self.user_id), 
            set(self.sushi_id)
            )
   
        self.interactions, self.weights = self.dataset.build_interactions(
            self.interaction_weight_list)
       
        self.train_split, self.test_split = cross_validation.random_train_test_split(self.weights, test_percentage = 0.25, random_state = 2022 )
        

        # Train/fit model
        self.model = LightFM(
            no_components=150,
            learning_rate=0.05,
            loss='warp-kos',
            random_state=2023)

        self.model.fit(
            self.train_split,
            epochs=10, verbose=True)

        self.test_set_likes = {key: [] for key in self.test_split.nonzero()[0]}
        for test_row, test_col in zip(*self.test_split.nonzero()):
            self.test_set_likes[test_row].append(test_col)

    
        temp = self.sushi_ratings_df.copy()
        temp = temp.replace(-1,np.NaN)
        temp = temp.describe().drop(columns='user_id')
        mean_rate = temp.transpose().reset_index().rename(columns={'index':'sushi'})
        mean_rate = mean_rate.sort_values(by=['mean'],ascending=False)
        self.top_ten = mean_rate['sushi'][:10].reset_index()
        self.top_ten = self.top_ten['index'].tolist()

        self.train_auc = auc_score(self.model, self.test_split).mean()
        self.precision = precision_at_k(self.model, self.test_split, self.train_split, k=10).mean()
        self.hit_rate = recall_at_k(self.model, self.test_split, self.train_split, k = 10).mean()

        print(self.train_auc)
        print(self.precision)
        print(self.hit_rate)

        self.next(self.join)

    @step
    def prepare_common(self):
        from lightfm.data import Dataset
        from lightfm import LightFM
        from lightfm import cross_validation
        from lightfm.evaluation import precision_at_k
        from lightfm.evaluation import auc_score
        from lightfm.evaluation import recall_at_k
        import numpy as np
        # find all interactions where an interaction is defined as someone who positively rated a sushi (greater than 2)
        # returns the interaction matrix where if a user rated a sushi >= 2 then it will be have a 'True' in that cell, otherwise it will have 'False'
        # can test different models - all ratings with weights, all ratings higher than 2, all ratings higher than 3, etc
        self.positive_rankings= self.sushi_ratings_no_user_common >= 3

        # interaction_list is a list of tuples that represent whether a user liked an item - find all the 'True' values from df
        # for example [(1,6),(1,8),(2,2)...]
        # represents user 1 liked item 6 and 8, while user 2 liked item 2...
        self.interaction_list = [(self.positive_rankings[col][self.positive_rankings[col].eq(True)].index[i], self.positive_rankings.columns.get_loc(col)) for col in self.positive_rankings.columns for i in range(len(self.positive_rankings[col][self.positive_rankings[col].eq(True)].index))]
        self.weight_list = []


        # find get the scores of each True item in matrix 
        # the scores will be used as weights for the lightfm model
        for x in self.interaction_list:
            weight = self.sushi_ratings_no_user.iloc[x] / 4
            self.weight_list.append((weight,))

        self.interaction_weight_list = []
        for (x, y) in zip(self.interaction_list, self.weight_list):
            self.interaction_weight_list.append(x + y)

        #creating a lightfm dataset using the list of user_id (0-4999) and list of sushi_id(0-99)
        self.dataset = Dataset()
        self.dataset.fit(
            set(self.user_id), 
            set(self.sushi_id)
            )
   
        self.interactions, self.weights = self.dataset.build_interactions(
            self.interaction_weight_list)
       
        self.train_split, self.test_split = cross_validation.random_train_test_split(self.weights, test_percentage = 0.25, random_state = 2022 )
        

        # Train/fit model
        self.model = LightFM(
            no_components=150,
            learning_rate=0.05,
            loss='warp-kos',
            random_state=2023)

        self.model.fit(
            self.train_split,
            epochs=10, verbose=True)

        self.test_set_likes = {key: [] for key in self.test_split.nonzero()[0]}
        for test_row, test_col in zip(*self.test_split.nonzero()):
            self.test_set_likes[test_row].append(test_col)

    
        temp = self.sushi_ratings_df.copy()
        temp = temp.replace(-1,np.NaN)
        temp = temp.describe().drop(columns='user_id')
        mean_rate = temp.transpose().reset_index().rename(columns={'index':'sushi'})
        mean_rate = mean_rate.sort_values(by=['mean'],ascending=False)
        self.top_ten = mean_rate['sushi'][:10].reset_index()
        self.top_ten = self.top_ten['index'].tolist()

        self.train_auc = auc_score(self.model, self.test_split).mean()
        self.precision = precision_at_k(self.model, self.test_split, self.train_split, k=10).mean()
        self.hit_rate = recall_at_k(self.model, self.test_split, self.train_split, k = 10).mean()

        print(self.train_auc)
        print(self.precision)
        print(self.hit_rate)

        self.next(self.join)
    
    @step
    def prepare_uncommon(self):
        from lightfm.data import Dataset
        from lightfm import LightFM
        from lightfm import cross_validation
        from lightfm.evaluation import precision_at_k
        from lightfm.evaluation import auc_score
        from lightfm.evaluation import recall_at_k
        import numpy as np
        # find all interactions where an interaction is defined as someone who positively rated a sushi (greater than 2)
        # returns the interaction matrix where if a user rated a sushi >= 2 then it will be have a 'True' in that cell, otherwise it will have 'False'
        # can test different models - all ratings with weights, all ratings higher than 2, all ratings higher than 3, etc
        self.positive_rankings= self.sushi_ratings_no_user_less_common >= 3

        # interaction_list is a list of tuples that represent whether a user liked an item - find all the 'True' values from df
        # for example [(1,6),(1,8),(2,2)...]
        # represents user 1 liked item 6 and 8, while user 2 liked item 2...
        self.interaction_list = [(self.positive_rankings[col][self.positive_rankings[col].eq(True)].index[i], self.positive_rankings.columns.get_loc(col)) for col in self.positive_rankings.columns for i in range(len(self.positive_rankings[col][self.positive_rankings[col].eq(True)].index))]
        self.weight_list = []


        # find get the scores of each True item in matrix 
        # the scores will be used as weights for the lightfm model
        for x in self.interaction_list:
            weight = self.sushi_ratings_no_user.iloc[x] / 4
            self.weight_list.append((weight,))

        self.interaction_weight_list = []
        for (x, y) in zip(self.interaction_list, self.weight_list):
            self.interaction_weight_list.append(x + y)

        #creating a lightfm dataset using the list of user_id (0-4999) and list of sushi_id(0-99)
        self.dataset = Dataset()
        self.dataset.fit(
            set(self.user_id), 
            set(self.sushi_id)
            )
   
        self.interactions, self.weights = self.dataset.build_interactions(
            self.interaction_weight_list)
       
        self.train_split, self.test_split = cross_validation.random_train_test_split(self.weights, test_percentage = 1, random_state = 2022 )
        

        # Train/fit model
        self.model = LightFM(
            no_components=150,
            learning_rate=0.05,
            loss='warp-kos',
            random_state=2023)

        self.model.fit(
            self.train_split,
            epochs=10, verbose=True)

        self.test_set_likes = {key: [] for key in self.test_split.nonzero()[0]}
        for test_row, test_col in zip(*self.test_split.nonzero()):
            self.test_set_likes[test_row].append(test_col)

    
        temp = self.sushi_ratings_df.copy()
        temp = temp.replace(-1,np.NaN)
        temp = temp.describe().drop(columns='user_id')
        mean_rate = temp.transpose().reset_index().rename(columns={'index':'sushi'})
        mean_rate = mean_rate.sort_values(by=['mean'],ascending=False)
        self.top_ten = mean_rate['sushi'][:10].reset_index()
        self.top_ten = self.top_ten['index'].tolist()

        self.train_auc = auc_score(self.model, self.test_split).mean()
        self.precision = precision_at_k(self.model, self.test_split, self.train_split, k=10).mean()
        self.hit_rate = recall_at_k(self.model, self.test_split, self.train_split, k = 10).mean()

        print(self.train_auc)
        print(self.precision)
        print(self.hit_rate)

        self.next(self.join)
    
    @step
    def join(self, inputs):
        self.test_set_likes = inputs.prepare_data_for_model.test_set_likes
        self.model = inputs.prepare_data_for_model.model
        self.interactions = inputs.prepare_data_for_model.interactions
        self.positive_rankings = inputs.prepare_data_for_model.positive_rankings
        self.top_ten = inputs.prepare_data_for_model.top_ten

        self.popular_hit_rate_ = inputs.prepare_data_for_model.popular_hit_rate_
        self.reg_hit_rate = inputs.prepare_data_for_model.hit_rate
        self.male_hit_rate = inputs.prepare_male.hit_rate
        self.female_hit_rate = inputs.prepare_female.hit_rate
        self.under_30_hit_rate = inputs.prepare_under_30.hit_rate
        self.over_30_hit_rate = inputs.prepare_over_30.hit_rate
        self.east_hit_rate = inputs.prepare_east.hit_rate
        self.west_hit_rate = inputs.prepare_west.hit_rate
        self.common_hit_rate = inputs.prepare_common.hit_rate
        self.uncommon_hit_rate = inputs.prepare_uncommon.hit_rate

        self.merge_artifacts(inputs, include=['sushi_id'])

        print("regular model hit rate: %d" %inputs.prepare_data_for_model.hit_rate)
        print("male split hit rate: %d" %inputs.prepare_male.hit_rate)
        print("female split hit rate: %d" %inputs.prepare_female.hit_rate)
        self.next(self.give_predictions)

    @step
    def give_predictions(self):
        import numpy as np
        user_x = self.SAMPLE_USER
        self.n_users, self.n_items = self.interactions.shape

        prediction_scores = list(self.model.predict(user_x, np.arange(self.n_items)))
        predictions = np.flip(list(np.argsort(prediction_scores)))

        # create a dictionary where we match each item number to sushi name 
        # so when we make a reccomendation instead of returning 5 we can give back item 5's sushi name
        col_names = (list(self.positive_rankings.columns))
        res = dict(zip(self.sushi_id, col_names))

        #print the recommended items from model
        counter = 0
        print("Recommendations for user {:d}:".format(user_x))
        for x in range(10):
            value = predictions[counter]
            sushi = res[value]
            print("     " + sushi)
            counter+=1
        counter = 0

        #print User's likes from the test set to compare to Recomendation 
        print("Test set for user {:d}:".format(user_x))


        if user_x in self.test_set_likes.keys():
            for a in self.test_set_likes[user_x]:
                sushi = a
                print(res[sushi])
        else:
            print("test set was empty - not enough likes for user to split 20%")

        self.next(self.end)

    @step
    def end(self):
        exp = Experiment(project_name=os.environ['MY_PROJECT_NAME'], auto_param_logging=False)

        params={"Model_type":"LightFM",
                "Loss_Function":"warp-kos",
                }
        
        metrics = {
                    "Model_Hit_Rate": self.reg_hit_rate,
                    "Male_Hit_Rate": self.male_hit_rate,
                    "Female_Hit_Rate": self.female_hit_rate,
                    "Under_30_Hit_Rate": self.under_30_hit_rate,
                    "Over_30_Hit_Rate": self.over_30_hit_rate,
                    "East_Japan_Hit_Rate": self.east_hit_rate,
                    "West_Japan_Hit_Rate": self.west_hit_rate,
                    "Uncommon_Sushi_Hit_Rate": self.uncommon_hit_rate,
                    "Common_Sushi_Hit_Rate": self.common_hit_rate,
                    "Popularity_Baseline_Hit_Rate": self.popular_hit_rate_}
        try:
            exp.log_parameters(params)
        except:
            print("log_parameters failed")  
        try:
            exp.log_metrics(metrics)
        except:
            print("log_metrics failed")

        print("Finished at {}!".format(datetime.utcnow()))
        

if __name__ == '__main__':
    sushi_rec_flow()