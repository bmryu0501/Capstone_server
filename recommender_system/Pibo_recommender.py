#import
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression
#import os
#import numpy as np
#import implicit
#import scipy.sparse as sparse
import pymysql
import pickle

'''
Recommend class related to achievement evaluation


## TODO ##
1. grid_search for hyperparameter tuning
2. implement sending success or fail if update is successful or not
3. implement sending recommended tasks
4. make prediction table in DB
   --> update DB when update is called
   --> load DB when recommend is called and use it to recommend (if needed, choose randomly from top N)
'''

class recommend_SVD:
    '''
    Recommend class using SVD
    Surprise library is used
    '''

    def __init__(self, update=False):
        '''
        Initialize class

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        # varibles
        self.__alpha = 0.4
        self.__beta = 0.6

        if update:
            # connect to mysql DB
            self.__connectDB()
            # set achievement evaluation data
            self.data_achievement = self.__setAchievement()
            # set engagement level data
            self.data_engagement = self.__setEngagement()

            self.__closeDB()
            # update model
            self.update_model_achievement()
            self.update_model_engagement()




    def __connectDB(self):
        '''
        Connect to mysql DB

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        # connect to mysql DB
        self.conn = pymysql.connect(
            user='capstone2',
            passwd='sirlab2020',
            db='Capstone_DB',
            charset='utf8'
        )

        # set cursor
        self.curs = self.conn.cursor(pymysql.cursors.DictCursor)


    def __closeDB(self):
        '''
        Close mysql DB

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        # close mysql DB
        self.conn.close()


    def recommend_achievement(self, user_id, num_task=1):
        '''
        Recommend by achievement evaluation

        Parameters
        ----------
        user_id : int
            User id
        num_task : int (default=1)
            Number of tasks to recommend

        Returns
        -------
        float
            Estimated achievement evaluation
        '''

        # load model
        model = pickle.load(open('model_achievement.pkl', 'rb'))
        
        # return depends on the number of tasks to recommend
        # recommend single task
        if num_task == 1:
            # predict
            pred = model.predict(user_id, self.__trainset_achievement.all_items())
            # return estimated achievement evaluation
            return 
        else:
            pass #TODO: return multiple tasks

    def recommend_engagement(self, user_id, num_task=1):
        '''
        Recommend by engagement level

        Parameters
        ----------
        user_id : int
            User id
        num_task : int (default=1)
            Number of tasks to recommend

        Returns
        -------
        float
            Estimated engagement level
        '''
        
        # load model
        model = pickle.load(open('model_engagement.pkl', 'rb'))

        # return depends on the number of tasks to recommend
        if num_task == 1:
            pass #TODO: return single task
        else: 
            pass #TODO: return multiple tasks

    def __setAchievement(self):
        '''
        Set achievement evaluation data

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        # load data from mysql DB
        sql = "SELECT * FROM achievement"
        self.curs.execute(sql)
        self.data_achievement = pd.DataFrame(self.curs.fetchall())

        # preprocess data
        self.data_achievement['NotAchieved'] = 100 - (self.data_achievement['Score_Parent'] * self.__alpha +
                                                      self.data_achievement['Score_Expert'] * self.__beta)
        # drop duplicated data
        self.data_achievement = self.data_achievement.drop_duplicates(['UID', 'TID'], keep='last')
        # set reader
        reader = Reader(rating_scale=(0, 100))
        # set data
        self.data_achievement = Dataset.load_from_df(self.data_achievement[['UID', 'TID', 'NotAchieved']], reader=reader)
        # set data
        data = Dataset.load_from_df(self.data_achievement, reader)

        # split data into train set and test set
        self.__trainset_achievement, self.__testset_achievement = train_test_split(data, test_size=.25)

    def __setEngagement(self):
        '''
        Set engagement level data

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        # load data from mysql DB
        sql = "SELECT * FROM engagement"
        self.curs.execute(sql)
        self.data_engagement = pd.DataFrame(self.curs.fetchall())
        
        # preprocess data
        # drop duplicated data
        self.data_engagement = self.data_engagement.drop_duplicates(['UID', 'TID'], keep='last')
        # set reader
        reader = Reader(rating_scale=(0, 100))
        # set data
        self.data = Dataset.load_from_df(self.data_engagement[['UID', 'TID', 'Engagement_Level']], reader=reader)

        # split data into train set and test set
        self.__trainset_engagement, self.__testset_engagement = train_test_split(self.data, test_size=.25)

    def set_alpha_beta(self, alpha, beta):
        '''
        Set alpha and beta

        Parameters
        ----------
        alpha : float
            Weight of parent score
        beta : float
            Weight of expert score

        Returns
        -------
        None
        '''

        self.__alpha = alpha
        self.__beta = beta

    def update_model_achievement(self):
        '''
        Update model for achievement evaluation

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        # update data
        self.connectDB()
        self.setAchievement()
        self.closeDB()

        # set model
        self.model_achievement = SVD(n_factors=10)
        # train model
        self.model_achievement.fit(self.__trainset_achievement)

        # save model
        pickle.dump(self.model_achievement, open('model_achievement.pkl', 'wb'))

    
    def update_model_engagement(self):
        '''
        Update model for engagement level

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        # update data
        self.connectDB()
        self.setEngagement()
        self.closeDB()

        # set model
        self.model_engagement = SVD(n_factors=10)
        # train model
        self.model_engagement.fit(self.__trainset_engagement)

        # save model
        pickle.dump(self.model_engagement, open('model_engagement.pkl', 'wb'))



""" Old version
class recommend_achievement:
    '''
    Recommend with explicit recommendation based on achievement evaluation.
    Surprise package is used.

    TODO
    전문가 평가 이전에는 부모 100%
    전문가 평가가 이루어진 날부터 날짜가 멀어질 수록 전문가의 계수값 하락
    '''

    def __init__(self, data, is_file_name=False, col_parent='Parent', col_expert='Expert', num_task=50):
        '''
        Make recommend_achievement class

        parameters:
        data: pandas data frame with achievement evaluation 
              or file name to use (csv format)
        [is_file_name: True if data is file name] = False
        [col_parent: column name of achievement evaluation by parents] = 'parent'
        [col_expert: column name of achievement eavluation by expert] = 'expert'
        [num_task: number of tasks] = 50
        '''
        if is_file_name:
            self.__data = pd.read_csv(data)
        else:
            self.__data = data
        
        self.__col_parent = col_parent
        self.__col_expert = col_expert
        self.__num_task = num_task

    
    def recommend(self, uid, num_recommend = 1, alpha=0.4, beta=0.6, test_size=0.25):
        '''
        This funciton recommend with explicit recommendation based on achievement evaluation

        parameters:
        uid: user id to recommend
        [num_recommend: number of recommendation] = 1
        [alpha: weight of parents' evaluation] = 0.4
        [beta: weight of expert's evaluation] = 0.6
        [test_size: size for test set] = 0.25
        
        return:
        if num_recommend is 1, return most not achieved task ID
        array with recommended task ID as num_recommend
        '''
        
        self.__alpha = alpha
        self.__beta = beta

        self.__data['NotAchieved'] = 100 - (self.__data[self.__col_parent] * self.__alpha +
                                            self.__data[self.__col_expert] * self.__beta)
        # drop duplicated data
        self.__data = self.__data.drop_duplicates(['UserID', 'TaskID'], keep='last')

        reader = Reader(rating_scale=(0, 100))

        # set data
        data = Dataset.load_from_df(self.__data[['UserID', 'TaskID', 'NotAchieved']], reader=reader)

        # split data to train, test
        train, test = train_test_split(data, test_size, random_state=42)

        # set model and train with train set
        self.__model = SVD(K=10)
        self.__model.fit(train)

        predictions = self.__model.test(test)
        self.rmse = accuracy.rmse(predictions, verbose=False)

        recommend_arr = []
        for i in range(0, self.__num_task):
            recommend_arr.append(self.__model.predict(uid, i))
        
        # sort tasks by NotAchieved value
        recommend_arr.sort(key=lambda x : -x[3])

        if num_recommend == 1:
            return recommend_arr[0][1]
        else:
            __ret = []
            for i in range(0, num_recommend):
                __ret.append(recommend_arr[i][1])
            return __ret[:num_recommend]
        

    def setAlphaBeta(self, alpha, beta):
        self.__alpha = alpha
        self.__beta = beta

    def update():
        pass

class recommend_engagement:
    '''
    This class recommend task based on engagement level with implicit feedback

    get data and user ID, then recommend task(s)
    '''

    def __init__(self, file_name, usecols=['UserID', 'TaskID', 'Engagement Level']):
        os.environ['MKL_NUM_THREADS'] = '1'

        # read csv and remove duplicated data
        self.__df = pd.read_csv(file_name, usecols=usecols)
        self.__df = self.__df.drop_duplicates([usecols[0], usecols[1]], keep='last', ignore_index=True)

        # make users and tasks array
        self.__users = list(self.__df[usecols[0]].unique())
        self.__tasks = list(self.__df[usecols[1]].unique())
        engagement = list(self.__df[usecols[2]])

        # make sparse matrix
        rows = self.__df[usecols[0]].astype('category').cat.codes
        cols = self.__df[usecols[1]].astype('category').cat.codes

        self.__user_task = sparse.csr_matrix((engagement, (rows, cols)))
        self.__task_user = self.__user_task.T.tocsr()

        # fit ALS model
        self.__model = implicit.als.AlternatingLeastSquares(num_threads=8)
        self.__model.fit(self.__task_user, show_progress=False)

    def recommend(self, uid, num_recommend=1, filter_already_played_task=True):
        '''
        This function recommend task(s)
        If user is a first-time user,
        the task which has highest engagement level will be recommended

        parameters:
        uid: user id to recommend
        [n: number of tasks to be recommended]

        return:
        if n == 1, return task ID
        else, return array of task IDs
        '''

        uid_index = self.__users.index(uid)
        self.__recommendations = self.__model.recommend(uid_index, self.__user_task, N=num_recommend, filter_already_liked_items=filter_already_played_task)

        '''
        TODO
        need recommendation for first-time users

        improvement
        limit data collection to models that had at least n tasks played
        https://www.ethanrosenthal.com/2016/10/19/implicit-mf-part-1/
        '''

        ret = []
        for i in self.__recommendations:
            ret.append(self.__tasks.index(i[0]))
        if num_recommend == 1:
            return ret[0]
        return ret[:num_recommend]"""


"""
class preference_to_engagement_level:
    '''
    This class estimate engagement level using linear regression
    '''
    def __init__(self, file_name, fit=True):
        self.__path = os.path.abspath(file_name)
        self.__df = pd.read_csv(file_name)
        if fit:
            self.fit()

    def fit(self):
        x = self.__df[['eye tracking', 'emotion', 'speech']]
        y = self.__df[['engagement level']]

        self.__lr = LinearRegression()
        self.__lr.fit(x, y)

    def insert_preference(self, uid, tid, factors):
        engagement_level = self.__lr.predict(np.array([factors]))[0][0]
        engagement_level = round(engagement_level)
        self.__df.loc[len(self.__df)] = [len(self.__df), uid, tid] + factors
        self.__df.to_csv(self.__path, index=False)
"""