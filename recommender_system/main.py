from Pibo_recommender import *

recom_achieve = recommend_achievement('../dataset/achievement.csv')
print(recom_achieve.recommend(0, num_recommend=10))
print(recom_achieve.rmse)


# recom_pref = recommend_preference('./dataset/preference_score.csv')
# print(recom_pref.recommend(30, num_recommend=10))