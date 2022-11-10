#import
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
#from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os
import numpy as np
import implicit
import scipy.sparse as sparse

'''
Recommend class related to achievement evaluation


## TODO ##
accuracy 측정할 거 만들기
Matrix Factorization 시 factor 개수 조정해서 최적 개수 찾기
recommender 두 개 parameter 일관성 맞추기
recommend_preference에서 engagement level predict랑 실제 estimated 구분
recommend_preference에서 없는 user가 입력으로 들어왔을 때 예외처리
recommend_preference에서 train set으로 나눌지 안나눌지 고민하고 넣든말든
'''

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
            self.__achievement = pd.read_csv(data)
        else:
            self.__achievement = data
        
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

        self.__achievement['NotAchieved'] = 100 - (self.__achievement[self.__col_parent] * self.__alpha +
                                                   self.__achievement[self.__col_expert] * self.__beta)
        # drop duplicated data
        self.__achievement = self.__achievement.drop_duplicates(['UserID', 'TaskID'], keep='last')

        reader = Reader(rating_scale=(0, 100))

        # set data
        data = Dataset.load_from_df(self.__achievement[['UserID', 'TaskID', 'NotAchieved']], reader=reader)

        # split data to train, test
        train, test = train_test_split(data, test_size, random_state=42)

        # set model and train with train set
        self.__model = SVD()
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


"""
class recommend_preference:
    '''
    This class recommend task based on preference with implicit feedback

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
        return ret[:num_recommend]
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