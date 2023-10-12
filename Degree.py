import pickle

from MyRecEdit.HyperGraphHelper import HyperGraphHelper
from MyRecEdit.MyRecEditTrain import MyRecEditTrain
import math
import os
import time
from datetime import datetime

from source.config.projectConfig import projectConfig
from source.scikit.service.DataProcessUtils import DataProcessUtils
from source.utils.ExcelHelper import ExcelHelper
from source.utils.pandas.pandasHelper import pandasHelper


class Degeree:
    @staticmethod
    def TestAlgorithm(project, dates, alpha=0.8, K=20, c=1, TrainPRDisIsComputed=False,
                      HyperGraphIsCreated=False, re=0.25, ct=0.25, ic=0.25, rc=0.25):

        recommendNum = 5
        obj = {}
        for date in dates:
            project, date, PR_degree = Degeree.algorithmBody(date, project,
                                                             recommendNum,
                                                             alpha=alpha, K=K,
                                                             c=c,
                                                             TrainPRDisIsComputed=TrainPRDisIsComputed,
                                                             HyperGraphIsCreated=HyperGraphIsCreated,
                                                             re=re, ct=ct,
                                                             ic=ic,
                                                             rc=rc)
            obj[date] = PR_degree
        return obj

    @staticmethod
    def preProcess(df_review, df_commit, df_issue_comment, df_review_comment, dates):
        df_review.dropna(how='any', inplace=True)
        df_commit.dropna(how='any', inplace=True)
        df_issue_comment.dropna(how='any', inplace=True)
        df_review_comment.dropna(how='any', inplace=True)
        df_review['label'] = df_review['pr_created_at'].apply(
            lambda x: (time.strptime(x, "%Y-%m-%d %H:%M:%S").tm_year == dates[2] and
                       time.strptime(x, "%Y-%m-%d %H:%M:%S").tm_mon == dates[3]))
        df_commit['label'] = df_commit['pr_created_at'].apply(
            lambda x: (time.strptime(x, "%Y-%m-%d %H:%M:%S").tm_year == dates[2] and
                       time.strptime(x, "%Y-%m-%d %H:%M:%S").tm_mon == dates[3]))
        df_issue_comment['label'] = df_issue_comment['pr_created_at'].apply(
            lambda x: (time.strptime(x, "%Y-%m-%d %H:%M:%S").tm_year == dates[2] and
                       time.strptime(x, "%Y-%m-%d %H:%M:%S").tm_mon == dates[3]))
        df_review_comment['label'] = df_review_comment['pr_created_at'].apply(
            lambda x: (time.strptime(x, "%Y-%m-%d %H:%M:%S").tm_year == dates[2] and
                       time.strptime(x, "%Y-%m-%d %H:%M:%S").tm_mon == dates[3]))
        df_review.reset_index(drop=True, inplace=True)
        df_commit.reset_index(drop=True, inplace=True)
        df_issue_comment.reset_index(drop=True, inplace=True)
        df_review_comment.reset_index(drop=True, inplace=True)

        convertDict, reviewConvertDict = DataProcessUtils.changeMyRecStringToNumber_v2(df_review, df_commit,
                                                                                       df_issue_comment,
                                                                                       df_review_comment)
        df_review['comment_at'] = df_review['review_created_at'].apply(lambda x: time.strptime(x, "%Y-%m-%d %H:%M:%S"))
        df_review['day'] = df_review['comment_at'].apply(
            lambda x: 10000 * x.tm_year + 100 * x.tm_mon + x.tm_mday)

        train_data = df_review.loc[df_review['label'] == False].copy(deep=True)
        train_data_commit = df_commit.loc[df_commit['label'] == False].copy(deep=True)
        train_data_issue_comment = df_issue_comment.loc[df_issue_comment['label'] == False].copy(deep=True)
        train_data_review_comment = df_review_comment.loc[df_review_comment['label'] == False].copy(deep=True)

        test_data = df_review.loc[df_review['label']].copy(deep=True)
        test_data = test_data.loc[test_data['create_before_pr']].copy(deep=True)
        test_data_commit = df_commit.loc[df_commit['label']].copy(deep=True)
        test_data_commit = test_data_commit.loc[test_data_commit['create_before_pr']].copy(deep=True)
        train_data.drop(columns=['label'], inplace=True)
        train_data_commit.drop(columns=['label'], inplace=True)
        train_data_issue_comment.drop(columns=['label'], inplace=True)
        train_data_review_comment.drop(columns=['label'], inplace=True)
        test_data.drop(columns=['label'], inplace=True)
        test_data_commit.drop(columns=['label'], inplace=True)
        train_data.reset_index(drop=True, inplace=True)
        train_data_commit.reset_index(drop=True, inplace=True)
        train_data_issue_comment.reset_index(drop=True, inplace=True)
        train_data_review_comment.reset_index(drop=True, inplace=True)
        train_data.fillna(value='', inplace=True)
        test_data.reset_index(drop=True, inplace=True)
        test_data_commit.reset_index(drop=True, inplace=True)
        test_data.fillna(value='', inplace=True)
        trainDict = dict(list(train_data.groupby('pr_number')))
        trainCommitDict = dict(list(train_data_commit.groupby('pr_number')))
        trainIssueCommentDict = dict(list(train_data_issue_comment.groupby('pr_number')))
        trainReviewCommentDict = dict(list(train_data_review_comment.groupby('pr_number')))
        testDict = dict(list(test_data.groupby('pr_number')))
        testCommitDict = dict(list(test_data_commit.groupby('pr_number')))
        train_data_number_group = train_data.drop_duplicates(['pr_number'])['pr_number']
        train_commit_data_number_group = train_data_commit.drop_duplicates(['pr_number'])['pr_number']
        train_issue_data_number_group = train_data_issue_comment.drop_duplicates(['pr_number'])['pr_number']
        train_review_data_number_group = train_data_review_comment.drop_duplicates(['pr_number'])['pr_number']

        test_data_number_group = test_data.drop_duplicates(['pr_number'])['pr_number']
        test_commit_data_number_group = test_data_commit.drop_duplicates(['pr_number'])['pr_number']

        test_data_y = {}
        test_data_y_workload = {}
        for pull_number in test_data_number_group:
            tempDf = testDict[pull_number]
            reviewers = []
            reviewers_info = []
            for row in tempDf.itertuples(index=False, name='Pandas'):
                r = getattr(row, 'review_user_login')
                day = getattr(row, 'day')
                if r not in reviewers:
                    reviewers.append(r)
                if (r, day) not in reviewers_info:
                    reviewers_info.append((r, day))
            test_data_y[pull_number] = reviewers
            test_data_y_workload[pull_number] = reviewers_info

        train_data_y = {}
        train_data_y_workload = {}
        train_data_committer = {}
        train_data_issue_commenter = {}
        train_data_review_commenter = {}
        for pull_number in train_data_number_group:
            tempDf = trainDict[pull_number]
            reviewers = []
            reviewers_info = []
            for row in tempDf.itertuples(index=False, name='Pandas'):
                r = getattr(row, 'review_user_login')
                day = getattr(row, 'day')
                if r not in reviewers:
                    reviewers.append(r)
                if (r, day) not in reviewers_info:
                    reviewers_info.append((r, day))
            train_data_y[pull_number] = reviewers
            train_data_y_workload[pull_number] = reviewers_info

        for pull_number in train_commit_data_number_group:
            committers = list(trainCommitDict[pull_number].drop_duplicates(['commit_user_login'])['commit_user_login'])
            train_data_committer[pull_number] = committers
        for pull_number in train_issue_data_number_group:
            issue_commenter = list(
                trainIssueCommentDict[pull_number].drop_duplicates(['comment_user_login'])['comment_user_login'])
            if len(issue_commenter) == 1 and issue_commenter[0] == '':
                continue
            train_data_issue_commenter[pull_number] = issue_commenter
        for pull_number in train_review_data_number_group:
            review_commenter = list(
                trainReviewCommentDict[pull_number].drop_duplicates(['review_comment_user_login'])[
                    'review_comment_user_login'])
            if len(review_commenter) == 1 and review_commenter[0] == '':
                continue
            train_data_review_commenter[pull_number] = review_commenter

        test_data_committer = {}
        for pull_number in test_commit_data_number_group:
            committers = list(testCommitDict[pull_number].drop_duplicates(['commit_user_login'])['commit_user_login'])
            test_data_committer[pull_number] = committers

        train_data.drop(columns=['comment_at', 'day'], inplace=True)
        test_data.drop(columns=['comment_at', 'day'], inplace=True)
        return train_data, train_data_commit, train_data_issue_comment, train_data_review_comment, train_data_y, \
               train_data_y_workload, train_data_committer, train_data_issue_commenter, train_data_review_commenter, \
               test_data, test_data_commit, test_data_y, test_data_y_workload, test_data_committer, convertDict, \
               reviewConvertDict, df_review, df_review_comment, df_commit, df_issue_comment

    @staticmethod
    def algorithmBody(date, project, recommendNum=5, alpha=0.98, K=20, c=1, TrainPRDisIsComputed=False,
                      HyperGraphIsCreated=False, re=0.25, ct=0.25, ic=0.25, rc=0.25):

        df_review = None
        df_commit = None
        df_issue_comment = None
        df_review_comment = None
        for i in range(date[0] * 12 + date[1], date[2] * 12 + date[3] + 1):
            y = int((i - i % 12) / 12)
            m = i % 12
            if m == 0:
                m = 12
                y = y - 1

            filename = None

            if i < date[2] * 12 + date[3]:
                filename_review = projectConfig.getMyRecEditDataPath() + os.sep + f'{project}/MyRecEdit_ALL_{project}_data_review_{y}_{m}_to_{y}_{m}.tsv'
                filename_commit = projectConfig.getMyRecEditDataPath() + os.sep + f'{project}/MyRecEdit_ALL_{project}_data_commit_{y}_{m}_to_{y}_{m}.tsv'
                filename_issue_comment = projectConfig.getMyRecEditDataPath() + os.sep + f'{project}/MyRecEdit_ALL_{project}_data_issue_comment_{y}_{m}_to_{y}_{m}.tsv '
                filename_review_comment = projectConfig.getMyRecEditDataPath() + os.sep + f'{project}/MyRecEdit_ALL_{project}_data_review_comment_{y}_{m}_to_{y}_{m}.tsv'
            else:
                filename_review = projectConfig.getMyRecEditDataPath() + os.sep + f'{project}/MyRecEdit_ALL_{project}_data_review_{y}_{m}_to_{y}_{m}.tsv'
                filename_commit = projectConfig.getMyRecEditDataPath() + os.sep + f'{project}/MyRecEdit_ALL_{project}_data_commit_{y}_{m}_to_{y}_{m}.tsv'
                filename_issue_comment = projectConfig.getMyRecEditDataPath() + os.sep + f'{project}/MyRecEdit_ALL_{project}_data_issue_comment_{y}_{m}_to_{y}_{m}.tsv'
                filename_review_comment = projectConfig.getMyRecEditDataPath() + os.sep + f'{project}/MyRecEdit_ALL_{project}_data_review_comment_{y}_{m}_to_{y}_{m}.tsv'

            if df_review is None:
                df_review = pandasHelper.readTSVFile(filename_review, pandasHelper.INT_READ_FILE_WITH_HEAD)
            else:
                temp = pandasHelper.readTSVFile(filename_review, pandasHelper.INT_READ_FILE_WITH_HEAD)
                df_review = df_review.append(temp)

            if df_commit is None:
                df_commit = pandasHelper.readTSVFile(filename_commit, pandasHelper.INT_READ_FILE_WITH_HEAD)
            else:
                temp = pandasHelper.readTSVFile(filename_commit, pandasHelper.INT_READ_FILE_WITH_HEAD)
                df_commit = df_commit.append(temp)

            if df_issue_comment is None:
                df_issue_comment = pandasHelper.readTSVFile(filename_issue_comment,
                                                            pandasHelper.INT_READ_FILE_WITH_HEAD)
            else:
                temp = pandasHelper.readTSVFile(filename_issue_comment, pandasHelper.INT_READ_FILE_WITH_HEAD)
                df_issue_comment = df_issue_comment.append(temp)

            if df_review_comment is None:
                df_review_comment = pandasHelper.readTSVFile(filename_review_comment,
                                                             pandasHelper.INT_READ_FILE_WITH_HEAD)
            else:
                temp = pandasHelper.readTSVFile(filename_review_comment, pandasHelper.INT_READ_FILE_WITH_HEAD)
                df_review_comment = df_review_comment.append(temp)

        df_review.reset_index(inplace=True, drop=True)
        df_commit.reset_index(inplace=True, drop=True)
        df_issue_comment.reset_index(inplace=True, drop=True)
        df_review_comment.reset_index(inplace=True, drop=True)
        df_commit = df_commit.loc[df_commit['is_review_commit'] == True].copy(deep=True)
        df_issue_comment = df_issue_comment.loc[df_issue_comment['is_review_issue_comment'] == True].copy(deep=True)
        df_review_comment = df_review_comment.loc[df_review_comment['is_review_review_comment'] == True].copy(deep=True)

        train_data, train_data_commit, train_data_issue_comment, train_data_review_comment, train_data_y, \
        train_data_y_workload, train_data_committer, train_data_issue_commenter, train_data_review_commenter, \
        test_data, test_data_commit, test_data_y, test_data_y_workload, test_data_committer, convertDict, \
        reviewConvertDict, df_review, df_review_comment, df_commit, df_issue_comment = \
            Degeree.preProcess(df_review, df_commit, df_issue_comment, df_review_comment, date)

        prList = list(set(test_data['pr_number']))
        prList.sort()

        project, date, PR_degree = Degeree.RecommendByMyRecEdit(train_data,
                                                                train_data_commit,
                                                                train_data_issue_comment,
                                                                train_data_review_comment,
                                                                train_data_y,
                                                                train_data_y_workload,
                                                                train_data_committer,
                                                                train_data_issue_commenter,
                                                                train_data_review_commenter,
                                                                test_data,
                                                                test_data_commit,
                                                                test_data_y,
                                                                test_data_y_workload,
                                                                test_data_committer,
                                                                date,
                                                                project, convertDict,
                                                                recommendNum=recommendNum,
                                                                K=K, alpha=alpha, c=c,
                                                                TrainPRDisIsComputed=TrainPRDisIsComputed,
                                                                HyperGraphIsCreated=HyperGraphIsCreated
                                                                , re=re, ct=ct, ic=ic, rc=rc)

        trainSize = (train_data.shape[0], test_data.shape[0])
        print(trainSize)

        return project, date, PR_degree

    @staticmethod
    def RecommendByMyRecEdit(train_data,
                             train_data_commit,
                             train_data_issue_comment,
                             train_data_review_comment,
                             train_data_y,
                             train_data_y_workload,
                             train_data_committer,
                             train_data_issue_commenter,
                             train_data_review_commenter,
                             test_data,
                             test_data_commit,
                             test_data_y,
                             test_data_y_workload,
                             test_data_committer,
                             date,
                             project, convertDict,
                             recommendNum=5,
                             K=20, alpha=0.8, c=1,
                             TrainPRDisIsComputed=False,
                             HyperGraphIsCreated=False
                             , re=0.25, ct=0.25, ic=0.25, rc=0.25):

        start = datetime.now()

        c_total_time = 0

        prCreatedTimeMap = {}
        for pr, temp_df in dict(list(train_data.groupby('pr_number'))).items():
            t1 = list(temp_df['pr_created_at'])[0]
            t1 = time.strptime(t1, "%Y-%m-%d %H:%M:%S")
            t1 = int(time.mktime(t1))
            prCreatedTimeMap[pr] = t1

        tempData = train_data[['pr_number', 'filename']].copy(deep=True)
        tempData.drop_duplicates(inplace=True)
        tempData.reset_index(inplace=True, drop=True)
        pathDict = dict(list(tempData.groupby('pr_number')))


        tempData = train_data[['pr_number', 'review_user_login']].copy(deep=True)
        tempData.drop_duplicates(inplace=True)
        reviewerFreqDict = {}
        for r, temp_df in dict(list(tempData.groupby('review_user_login'))).items():
            reviewerFreqDict[r] = temp_df.shape[0]

        ###
        if not os.path.exists(
                f'./trainPrDis/trainPrDis_{K}/{project}/{project}_{date[0]}_{date[1]}_{date[2]}_{date[3]}_trainPrDis.pkl'):
            TrainPRDisIsComputed = False
        if not TrainPRDisIsComputed:
            trainPrDis = MyRecEditTrain.getTrainDataPrDistance(train_data, K, pathDict, date, prCreatedTimeMap)
            if not os.path.exists(f'./trainPrDis/trainPrDis_{K}/{project}/'):
                os.makedirs(f'./trainPrDis/trainPrDis_{K}/{project}/')
            with open(
                    f'./trainPrDis/trainPrDis_{K}/{project}/{project}_{date[0]}_{date[1]}_{date[2]}_{date[3]}_trainPrDis.pkl',
                    'wb') as f:
                pickle.dump(trainPrDis, f)
        else:
            with open(
                    f'./trainPrDis/trainPrDis_{K}/{project}/{project}_{date[0]}_{date[1]}_{date[2]}_{date[3]}_trainPrDis.pkl',
                    'rb') as file:
                trainPrDis = pickle.loads(file.read())
        PR_degree = None
        return project, date, PR_degree

