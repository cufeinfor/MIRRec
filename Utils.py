import math
import os
import time

from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from source.config.projectConfig import projectConfig
from source.scikit.service.DataProcessUtils import DataProcessUtils
from source.utils.pandas.pandasHelper import pandasHelper


class Utils:
    DATES_ONE_YEAR = [(2018, 6, 2019, 6), (2018, 7, 2019, 7), (2018, 8, 2019, 8), (2018, 9, 2019, 9),
                      (2018, 10, 2019, 10), (2018, 11, 2019, 11), (2018, 12, 2019, 12), (2019, 1, 2020, 1),
                      (2019, 2, 2020, 2), (2019, 3, 2020, 3), (2019, 4, 2020, 4), (2019, 5, 2020, 5),
                      (2019, 6, 2020, 6), (2019, 7, 2020, 7), (2019, 8, 2020, 8), (2019, 9, 2020, 9),
                      (2019, 10, 2020, 10), (2019, 11, 2020, 11), (2019, 12, 2020, 12), (2020, 1, 2021, 1),
                      (2020, 2, 2021, 2), (2020, 3, 2021, 3), (2020, 4, 2021, 4), (2020, 5, 2021, 5),
                      (2020, 6, 2021, 6), (2020, 7, 2021, 7), (2020, 8, 2021, 8), (2020, 9, 2021, 9),
                      (2020, 10, 2021, 10), (2020, 11, 2021, 11)]
    DATES_ONE_MONTH = [(2019, 6, 2019, 6), (2019, 7, 2019, 7), (2019, 8, 2019, 8), (2019, 9, 2019, 9),
                       (2019, 10, 2019, 10), (2019, 11, 2019, 11), (2019, 12, 2019, 12), (2020, 1, 2020, 1),
                       (2020, 2, 2020, 2), (2020, 3, 2020, 3), (2020, 4, 2020, 4), (2020, 5, 2020, 5),
                       (2020, 6, 2020, 6), (2020, 7, 2020, 7), (2020, 8, 2020, 8), (2020, 9, 2020, 9),
                       (2020, 10, 2020, 10), (2020, 11, 2020, 11), (2020, 12, 2020, 12), (2021, 1, 2021, 1),
                       (2021, 2, 2021, 2), (2021, 3, 2021, 3), (2021, 4, 2021, 4), (2021, 5, 2021, 5),
                       (2021, 6, 2021, 6), (2021, 7, 2021, 7), (2021, 8, 2021, 8), (2021, 9, 2021, 9),
                       (2021, 10, 2021, 10), (2021, 11, 2021, 11)]
    DATES_ALL = [(2018, 6, 2019, 6), (2018, 6, 2019, 7), (2018, 6, 2019, 8), (2018, 6, 2019, 9), (2018, 6, 2019, 10),
                 (2018, 6, 2019, 11), (2018, 6, 2019, 12), (2018, 6, 2020, 1), (2018, 6, 2020, 2), (2018, 6, 2020, 3),
                 (2018, 6, 2020, 4), (2018, 6, 2020, 5), (2018, 6, 2020, 6), (2018, 6, 2020, 7), (2018, 6, 2020, 8),
                 (2018, 6, 2020, 9), (2018, 6, 2020, 10), (2018, 6, 2020, 11), (2018, 6, 2020, 12), (2018, 6, 2021, 1),
                 (2018, 6, 2021, 2), (2018, 6, 2021, 3), (2018, 6, 2021, 4), (2018, 6, 2021, 5), (2018, 6, 2021, 6),
                 (2018, 6, 2021, 7), (2018, 6, 2021, 8), (2018, 6, 2021, 9), (2018, 6, 2021, 10), (2018, 6, 2021, 11)]
    DATES_ALL_MONTH = [(2018, 6), (2018, 7), (2018, 8), (2018, 9), (2018, 10), (2018, 11),
                       (2018, 12), (2019, 1), (2019, 2), (2019, 3), (2019, 4), (2019, 5),
                       (2019, 6), (2019, 7), (2019, 8), (2019, 9), (2019, 10), (2019, 11),
                       (2019, 12), (2020, 1), (2020, 2), (2020, 3), (2020, 4), (2020, 5),
                       (2020, 6), (2020, 7), (2020, 8), (2020, 9), (2020, 10), (2020, 11),
                       (2020, 12), (2021, 1), (2021, 2), (2021, 3), (2021, 4), (2021, 5),
                       (2021, 6), (2021, 7), (2021, 8), (2021, 9), (2021, 10), (2021, 11)]
    DATES_TEST_MONTH = [(2019, 6), (2019, 7), (2019, 8), (2019, 9), (2019, 10), (2019, 11),
                        (2019, 12), (2020, 1), (2020, 2), (2020, 3), (2020, 4), (2020, 5),
                        (2020, 6), (2020, 7), (2020, 8), (2020, 9), (2020, 10), (2020, 11),
                        (2020, 12), (2021, 1), (2021, 2), (2021, 3), (2021, 4), (2021, 5),
                        (2021, 6), (2021, 7), (2021, 8), (2021, 9), (2021, 10), (2021, 11)]
    PROJECTS_ALL = ['opencv', 'jekyll', 'spring-boot', 'xbmc', 'django', 'cakephp', 'joomla-cms',
                    'symfony', 'rails', 'akka', 'scala', 'tensorflow', 'electron',
                    'bitcoin', 'react', 'gatsby', 'angular', 'node', 'scikit-learn', 'ansible', 'pandas']
    PROJECTS_TEN = ['bitcoin', 'electron', 'opencv', 'xbmc', 'react', 'angular', 'django',
                    'symfony', 'rails', 'scala']
    MAX_CHANGE_FILES = 100
    DATES = [(2018, 6), (2018, 7), (2018, 8), (2018, 9), (2018, 10), (2018, 11), (2018, 12),
             (2019, 1), (2019, 2), (2019, 3), (2019, 4), (2019, 5), (2019, 6),
             (2019, 7), (2019, 8), (2019, 9), (2019, 10), (2019, 11), (2019, 12),
             (2020, 1), (2020, 2), (2020, 3), (2020, 4), (2020, 5), (2020, 6),
             (2020, 7), (2020, 8), (2020, 9), (2020, 10), (2020, 11), (2020, 12),
             (2021, 1), (2021, 2), (2021, 3), (2021, 4), (2021, 5), (2021, 6),
             (2021, 7), (2021, 8), (2021, 9), (2021, 10), (2021, 11), (2021, 12)]
    MONTH = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    DAYS = ['31', '28', '31', '30', '31', '30', '31', '31', '30', '31', '30', '31']

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
    def getOriginData(project, date):
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
        return df_review, df_commit, df_issue_comment, df_review_comment

    @staticmethod
    def getOriginSelfData(project, date):
        df_review_self = None
        df_commit_self = None
        df_issue_comment_self = None
        df_review_comment_self = None
        projectFileNamePrex = project + '_self'
        for i in range(date[0] * 12 + date[1], date[2] * 12 + date[3] + 1):
            y = int((i - i % 12) / 12)
            m = i % 12
            if m == 0:
                m = 12
                y = y - 1

            filename = None

            if i < date[2] * 12 + date[3]:
                filename_review = projectConfig.getMyRecEditDataPath() + os.sep + f'{projectFileNamePrex}/MyRecEditReviewedBySelf_ALL_{project}_data_review_{y}_{m}_to_{y}_{m}.tsv'
                filename_commit = projectConfig.getMyRecEditDataPath() + os.sep + f'{projectFileNamePrex}/MyRecEditReviewedBySelf_ALL_{project}_data_commit_{y}_{m}_to_{y}_{m}.tsv'
                filename_issue_comment = projectConfig.getMyRecEditDataPath() + os.sep + f'{projectFileNamePrex}/MyRecEditReviewedBySelf_ALL_{project}_data_issue_comment_{y}_{m}_to_{y}_{m}.tsv '
                filename_review_comment = projectConfig.getMyRecEditDataPath() + os.sep + f'{projectFileNamePrex}/MyRecEditReviewedBySelf_ALL_{project}_data_review_comment_{y}_{m}_to_{y}_{m}.tsv'
            else:
                filename_review = projectConfig.getMyRecEditDataPath() + os.sep + f'{projectFileNamePrex}/MyRecEditReviewedBySelf_ALL_{project}_data_review_{y}_{m}_to_{y}_{m}.tsv'
                filename_commit = projectConfig.getMyRecEditDataPath() + os.sep + f'{projectFileNamePrex}/MyRecEditReviewedBySelf_ALL_{project}_data_commit_{y}_{m}_to_{y}_{m}.tsv'
                filename_issue_comment = projectConfig.getMyRecEditDataPath() + os.sep + f'{projectFileNamePrex}/MyRecEditReviewedBySelf_ALL_{project}_data_issue_comment_{y}_{m}_to_{y}_{m}.tsv'
                filename_review_comment = projectConfig.getMyRecEditDataPath() + os.sep + f'{projectFileNamePrex}/MyRecEditReviewedBySelf_ALL_{project}_data_review_comment_{y}_{m}_to_{y}_{m}.tsv'

            if df_review_self is None:
                df_review_self = pandasHelper.readTSVFile(filename_review, pandasHelper.INT_READ_FILE_WITH_HEAD)
            else:
                temp = pandasHelper.readTSVFile(filename_review, pandasHelper.INT_READ_FILE_WITH_HEAD)
                df_review_self = df_review_self.append(temp)

            if df_commit_self is None:
                df_commit_self = pandasHelper.readTSVFile(filename_commit, pandasHelper.INT_READ_FILE_WITH_HEAD)
            else:
                temp = pandasHelper.readTSVFile(filename_commit, pandasHelper.INT_READ_FILE_WITH_HEAD)
                df_commit_self = df_commit_self.append(temp)

            if df_issue_comment_self is None:
                df_issue_comment_self = pandasHelper.readTSVFile(filename_issue_comment,
                                                                 pandasHelper.INT_READ_FILE_WITH_HEAD)
            else:
                temp = pandasHelper.readTSVFile(filename_issue_comment, pandasHelper.INT_READ_FILE_WITH_HEAD)
                df_issue_comment_self = df_issue_comment_self.append(temp)

            if df_review_comment_self is None:
                df_review_comment_self = pandasHelper.readTSVFile(filename_review_comment,
                                                                  pandasHelper.INT_READ_FILE_WITH_HEAD)
            else:
                temp = pandasHelper.readTSVFile(filename_review_comment, pandasHelper.INT_READ_FILE_WITH_HEAD)
                df_review_comment_self = df_review_comment_self.append(temp)

        df_review_self.reset_index(inplace=True, drop=True)
        df_commit_self.reset_index(inplace=True, drop=True)
        df_issue_comment_self.reset_index(inplace=True, drop=True)
        df_review_comment_self.reset_index(inplace=True, drop=True)
        df_commit_self = df_commit_self.loc[df_commit_self['is_review_commit'] == True].copy(deep=True)
        df_issue_comment_self = df_issue_comment_self.loc[
            df_issue_comment_self['is_review_issue_comment'] == True].copy(deep=True)
        df_review_comment_self = df_review_comment_self.loc[
            df_review_comment_self['is_review_review_comment'] == True].copy(deep=True)
        return df_review_self, df_commit_self, df_issue_comment_self, df_review_comment_self

    @staticmethod
    def GetPRCommitFileData(project, date):
        df_file = None
        projectName = project + '_file'
        for i in range(date[0] * 12 + date[1], date[2] * 12 + date[3] + 1):
            y = int((i - i % 12) / 12)
            m = i % 12
            if m == 0:
                m = 12
                y = y - 1
            if i < date[2] * 12 + date[3]:
                filename_file = projectConfig.getMyRecEditDataPath() + os.sep + f'{projectName}/MyRecEdit_ALL_{project}_data_file_{y}_{m}_to_{y}_{m}.tsv'
            else:
                filename_file = projectConfig.getMyRecEditDataPath() + os.sep + f'{projectName}/MyRecEdit_ALL_{project}_data_file_{y}_{m}_to_{y}_{m}.tsv'

            if df_file is None:
                df_file = pandasHelper.readTSVFile(filename_file, pandasHelper.INT_READ_FILE_WITH_HEAD)
            else:
                temp = pandasHelper.readTSVFile(filename_file, pandasHelper.INT_READ_FILE_WITH_HEAD)
                df_file = df_file.append(temp)

        df_file.reset_index(inplace=True, drop=True)
        df_file['label'] = df_file['pr_created_at'].apply(
            lambda x: (time.strptime(x, "%Y-%m-%d %H:%M:%S").tm_year == date[2] and
                       time.strptime(x, "%Y-%m-%d %H:%M:%S").tm_mon == date[3]))
        train_data_file = df_file.loc[df_file['label'] == False].copy(deep=True)
        test_data_file = df_file.loc[df_file['label']].copy(deep=True)
        return train_data_file, test_data_file

    @staticmethod
    def GetCommitFileData(project):
        commit_file_path = f'../dataset/commitFileRelation/ALL_{project}_data_commit_file_relation.tsv'
        df_commit_file = pandasHelper.readTSVFile(commit_file_path, pandasHelper.INT_READ_FILE_WITH_HEAD)
        df_commit_file = df_commit_file[['file_commit_sha', 'file_filename', 'file_status', 'file_additions',
                                         'file_deletions', 'file_changes']].copy(
            deep=True)
        df_commit_file.columns = ['sha', 'file_name', 'file_status', 'file_additions', 'file_deletions', 'file_changes']
        df_commit_file.drop_duplicates(inplace=True)
        counts_series = df_commit_file['sha'].value_counts()
        dict_count = {'sha': counts_series.index, 'counts': counts_series.values}
        df_commit_file_counts = pd.DataFrame(dict_count)
        df_commit_file = pd.merge(left=df_commit_file, right=df_commit_file_counts, left_on='sha', right_on='sha')
        df_commit_file = df_commit_file.drop(
            df_commit_file[df_commit_file['counts'] >= Utils.MAX_CHANGE_FILES].index)

        return df_commit_file

    @staticmethod
    def GetCommitData(project):
        path = f'../dataset/prCommitData_edit/ALL_{project}_pr_commit_data.tsv'
        df_commit = pandasHelper.readTSVFile(path, pandasHelper.INT_READ_FILE_WITH_HEAD)
        df_commit = df_commit[['pull_number', 'sha', 'commit_author_login', 'commit_commit_author_date']].copy(
            deep=True)
        df_commit.columns = ['pull_number', 'sha', 'commit_user_login', 'commit_at']
        return df_commit


    @staticmethod
    def GetPRData(project):
        path = f'../dataset/pullrequestData/ALL_{project}_data_pullrequest.tsv'
        df = pandasHelper.readTSVFile(path, pandasHelper.INT_READ_FILE_WITH_HEAD)
        df = df[['number', 'user_login', 'created_at', 'closed_at', 'merged_at', 'merged']].copy(deep=True)
        df.columns = ['pr_number', 'author_user_login', 'pr_created_at', 'pr_closed_at', 'pr_merged_at', 'is_merged']
        return df

    @staticmethod
    def GetReviewData(project):
        path = f'../dataset/reviewData/ALL_{project}_data_review.tsv'
        df = pandasHelper.readTSVFile(path, pandasHelper.INT_READ_FILE_WITH_HEAD)
        df = df[['pull_number', 'user_login', 'submitted_at', 'id', 'commit_id']].copy(deep=True)
        df.columns = ['pr_number', 'review_user_login', 'reviewed_at', 'review_id', 'review_commit_id']
        return df

    @staticmethod
    def GetReviewComment(project):
        path = f'../dataset/reviewCommentData/ALL_{project}_data_review_comment.tsv'
        df = pandasHelper.readTSVFile(path, pandasHelper.INT_READ_FILE_WITH_HEAD)
        df = df[['pull_number', 'user_login', 'created_at', 'pull_request_review_id', 'commit_id',
                 'path']].copy(deep=True)
        df.columns = ['pr_number', 'reviewer_user_login', 'review_commented_at', 'review_id', 'commit_id', 'file']
        return df

    @staticmethod
    def GetPRCommitData(project):
        path = f'../dataset/prCommitData_edit/ALL_{project}_pr_commit_data.tsv'
        df = pandasHelper.readTSVFile(path, pandasHelper.INT_READ_FILE_WITH_HEAD)
        df = df[['pull_number', 'commit_author_login', 'commit_commit_author_date', 'sha']].copy(deep=True)
        df.columns = ['pr_number', 'commit_user_login', 'committed_at', 'commit_sha']
        return df

    @staticmethod
    def GetCommitFile(project):
        path = f'../dataset/commitFileRelation/ALL_{project}_data_commit_file_relation.tsv'
        df = pandasHelper.readTSVFile(path, pandasHelper.INT_READ_FILE_WITH_HEAD)
        df = df[['file_commit_sha', 'file_filename', 'file_status', 'file_additions', 'file_deletions']].copy(deep=True)
        df.columns = ['commit_sha', 'file', 'status', 'additions', 'deletions']
        df.drop_duplicates(inplace=True)
        counts_series = df['commit_sha'].value_counts()
        dict_count = {'commit_sha': counts_series.index, 'counts': counts_series.values}
        df_commit_file_counts = pd.DataFrame(dict_count)
        df = pd.merge(left=df, right=df_commit_file_counts, left_on='commit_sha', right_on='commit_sha')
        df = df.drop(df[df['counts'] >= Utils.MAX_CHANGE_FILES].index)
        return df

    @staticmethod
    def GetIssueComment(project):
        path = f'../dataset/issueCommentData/ALL_{project}_data_issuecomment.tsv'
        df = pandasHelper.readTSVFile(path, pandasHelper.INT_READ_FILE_WITH_HEAD)
        df = df[['pull_number', 'user_login', 'created_at']].copy(deep=True)
        df.columns = ['pr_number', 'comment_user_login', 'commented_at']
        return df

    @staticmethod
    def GetReviewTime(project, date):
        df_pr = Utils.GetPRData(project)
        dates = [f'{date[0]}-{Utils.MONTH[date[1] - 1]}-01 00:00:00',
                 f'{date[2]}-{Utils.MONTH[date[3] - 1]}-01 00:00:00']
        # df_pr = df_pr.loc[df_pr['pr_created_at'] >= dates[0]].copy(deep=True)

        df_pr = df_pr.loc[df_pr['pr_created_at'] < dates[1]].copy(deep=True)
        df_review = Utils.GetReviewData(project)
        df_pr_review = pd.merge(left=df_pr, right=df_review, left_on='pr_number', right_on='pr_number')
        df_pr_review = df_pr_review.loc[df_pr_review['reviewed_at'] <= df_pr_review['pr_closed_at']].copy(deep=True)
        df_commit = Utils.GetPRCommitData(project)
        df_pr_review = pd.merge(left=df_pr_review, right=df_commit, left_on=['pr_number', 'review_commit_id'],
                                right_on=['pr_number', 'commit_sha'])
        df_pr_review = df_pr_review[
            ['pr_number', 'author_user_login', 'pr_created_at', 'pr_closed_at', 'review_user_login', 'reviewed_at',
             'commit_sha',
             'commit_user_login', 'committed_at']]
        df_pr_review = df_pr_review.loc[df_pr_review['committed_at'] <= df_pr_review['pr_closed_at']].copy(deep=True)

        def get_gap(x):
            commit_time = int((datetime.strptime(x["pr_created_at"], "%Y-%m-%d %H:%M:%S") -
                               datetime.strptime(x["committed_at"], "%Y-%m-%d %H:%M:%S")).total_seconds())
            if commit_time >= 0:  # 代码提交是在拉取请求创建前
                gap = int((datetime.strptime(x["reviewed_at"], "%Y-%m-%d %H:%M:%S") -
                           datetime.strptime(x["pr_created_at"], "%Y-%m-%d %H:%M:%S")).total_seconds())
            else:
                gap = int((datetime.strptime(x["reviewed_at"], "%Y-%m-%d %H:%M:%S") -
                           datetime.strptime(x["committed_at"], "%Y-%m-%d %H:%M:%S")).total_seconds())
                if gap < 0:
                    gap = int((datetime.strptime(x["reviewed_at"], "%Y-%m-%d %H:%M:%S") -
                               datetime.strptime(x["pr_created_at"], "%Y-%m-%d %H:%M:%S")).total_seconds())
            return gap

        df_pr_review['review_load'] = df_pr_review[["reviewed_at", "committed_at", 'pr_created_at']].apply(
            lambda x: get_gap(x), axis=1)
        df_pr_review['review_load'] = df_pr_review['review_load'].apply(lambda x: int(x) / 60 / 60 / 24)
        return df_pr_review

    @staticmethod
    def GetCommitTime(project, date):
        df_pr = Utils.GetPRData(project)
        dates = [f'{date[0]}-{Utils.MONTH[date[1] - 1]}-01 00:00:00',
                 f'{date[2]}-{Utils.MONTH[date[3] - 1]}-01 00:00:00']

        # df_pr = df_pr.loc[df_pr['pr_created_at'] >= dates[0]].copy(deep=True)
        df_pr = df_pr.loc[df_pr['pr_created_at'] < dates[1]].copy(deep=True)

        # df_review = Utils.GetReviewData(project)
        # df_pr_review = pd.merge(left=df_pr, right=df_review, left_on='pr_number', right_on='pr_number')
        # df_pr_review = df_pr_review.loc[df_pr_review['reviewed_at'] <= df_pr_review['pr_closed_at']].copy(deep=True)
        df_commit = Utils.GetPRCommitData(project)
        # df_commit_file = Utils.GetCommitFile(project) df_pr_commit_file = pd.merge(left=df_commit,
        # right=df_commit_file, left_on='commit_sha', right_on='commit_sha') df_pr_commit_file = pd.merge(
        # left=df_pr_commit_file, right=df_pr, left_on='pr_number', right_on='pr_number') df_pr_commit_file =
        # df_pr_commit_file[ ['pr_number', 'author_user_login', 'pr_created_at', 'pr_closed_at', 'commit_sha',
        # 'commit_user_login', 'committed_at', 'file', 'status', 'additions', 'deletions', 'counts']].copy(deep=True)
        # df_pr_review = pd.merge(left=df_pr_review, right=df_commit, left_on='review_commit_id',
        # right_on='commit_sha')
        df_pr_commit = pd.merge(left=df_pr, right=df_commit, left_on='pr_number', right_on='pr_number')
        df_pr_commit = df_pr_commit[
            ['pr_number', 'author_user_login', 'pr_created_at', 'pr_closed_at', 'committed_at', 'commit_user_login',
             'commit_sha']].copy(deep=True)

        def get_temp_df(temp_df, gap=-1):
            if len(temp_df) == 1:
                temp_df['commit_load'] = [gap]
            else:
                temp_df.sort_values(by='committed_at', ascending=True, inplace=True)
                time_arr = temp_df['committed_at'].to_list()
                pr_arr = temp_df['pr_number'].to_list()
                time_df = [gap]
                for i in range(1, len(time_arr)):
                    gap = int((datetime.strptime(time_arr[i], "%Y-%m-%d %H:%M:%S") -
                               datetime.strptime(time_arr[i - 1],
                                                 "%Y-%m-%d %H:%M:%S")).total_seconds()) / 60 / 60 / 24
                    if gap >= 30 and pr_arr[i] != pr_arr[i - 1]:
                        gap = -1
                    time_df.append(gap)
                temp_df['commit_load'] = time_df
            return temp_df

        # def get_gap(x, arr):
        #     index = arr.index(x)
        #     y = arr[index - 1]
        #     gap = int((datetime.strptime(x, "%Y-%m-%d %H:%M:%S") -
        #                datetime.strptime(y, "%Y-%m-%d %H:%M:%S")).total_seconds())
        #     gap = gap / 60 / 60 / 24
        #     return gap

        # 先按照用户分组，再去考虑同时参与多个和两个很近的情况
        df_pr_commit.sort_values(by='committed_at', ascending=True, inplace=True)
        user_commit_dict = dict(list(df_pr_commit.groupby(by='commit_user_login')))
        df = None
        for user, temp_df in user_commit_dict.items():
            temp_df = get_temp_df(temp_df)
            if df is None:
                df = temp_df
            else:
                df = pd.concat([df, temp_df])
        return df

    @staticmethod
    def GetReviewFileTime(project, date):
        df = Utils.GetReviewTime(project, date)
        df_commit_file = Utils.GetCommitFile(project)
        df_review_commit_file = pd.merge(left=df, right=df_commit_file, left_on='commit_sha', right_on='commit_sha')
        df_review_commit_file = df_review_commit_file[
            ['pr_number', 'author_user_login', 'pr_created_at', 'pr_closed_at', 'review_user_login',
             'reviewed_at', 'commit_user_login', 'committed_at', 'review_load', 'file', 'status', 'additions',
             'deletions', 'counts', 'commit_sha']].copy(deep=True)

        commit_file_dict = dict(list(df_review_commit_file.groupby(by='commit_sha')))
        df_review_file_time = None
        for commit, temp_df in commit_file_dict.items():
            additions_lines = sum(temp_df['additions'].to_list())
            deletions_lines = sum(temp_df['deletions'].to_list())
            change_lines = additions_lines + deletions_lines
            if change_lines == 0:
                temp_df['file_review_load'] = temp_df["review_load"]
                temp_df['line_review_load'] = temp_df["review_load"].apply(lambda x: 0)
                temp_df['commit_additions'] = temp_df["review_load"].apply(lambda x: 0)
                temp_df['commit_deletions'] = temp_df["review_load"].apply(lambda x: 0)
                temp_df['commit_changes'] = temp_df["review_load"].apply(lambda x: 0)
            else:
                line_day = float(temp_df['review_load'].to_list()[0]) / change_lines
                temp_df['file_review_load'] = temp_df[["additions", "deletions"]].apply(
                    lambda x: (int(x['additions']) + int(x['deletions'])) * line_day, axis=1)
                temp_df['line_review_load'] = temp_df["review_load"].apply(lambda x: line_day)
                temp_df['commit_additions'] = temp_df["review_load"].apply(lambda x: additions_lines)
                temp_df['commit_deletions'] = temp_df["review_load"].apply(lambda x: deletions_lines)
                temp_df['commit_changes'] = temp_df["review_load"].apply(lambda x: additions_lines + deletions_lines)
            if df_review_file_time is None:
                df_review_file_time = temp_df
            else:
                df_review_file_time = pd.concat([df_review_file_time, temp_df])
        df_review_file_time = df_review_file_time.loc[df_review_file_time['review_load'] <= 90].copy(deep=True)
        df_review_file_time = df_review_file_time.loc[df_review_file_time['line_review_load'] <= 0.5].copy(deep=True)
        file_review_dict = dict(list(df_review_file_time.groupby(by='file')))
        file_lineLoad_dict = {}
        for file, temp_df in file_review_dict.items():
            line_load_avg = np.average(temp_df['line_review_load'].to_list())
            file_lineLoad_dict[file] = line_load_avg
        file_lineLoad_dict = dict(sorted(file_lineLoad_dict.items(), key=lambda d: d[1], reverse=True))
        file_lineload_avg = sum(file_lineLoad_dict.values()) / len(file_lineLoad_dict)
        return df_review_file_time, file_lineLoad_dict, file_lineload_avg

    @staticmethod
    def GetCommitFileTime(project, date):
        df_commit_time = Utils.GetCommitTime(project, date)
        df_commit_file = Utils.GetCommitFile(project)
        df_commit_file_time = pd.merge(left=df_commit_time, right=df_commit_file, left_on='commit_sha',
                                       right_on='commit_sha')
        df_commit_file_time['change_lines'] = df_commit_file_time[['additions', 'deletions']].apply(
            lambda x: int(x['additions']) + int(x['deletions']), axis=1)
        df_commit_file_time = df_commit_file_time.loc[df_commit_file_time['commit_load'] != 0].copy(deep=True)
        df_commit_file_time = df_commit_file_time.loc[df_commit_file_time['change_lines'] != 0].copy(deep=True)

        df_commit_file_no_time = df_commit_file_time.loc[df_commit_file_time['commit_load'] == -1].copy(deep=True)
        df_commit_file_time = df_commit_file_time.loc[df_commit_file_time['commit_load'] != -1].copy(deep=True)

        commit_file_dict = dict(list(df_commit_file_time.groupby(by='commit_sha')))
        df_commit_file_line_time = None
        for commit, temp_df in commit_file_dict.items():
            additions_lines = sum(temp_df['additions'].to_list())
            if additions_lines != 0:
                line_day = float(temp_df['commit_load'].to_list()[0]) / additions_lines
                temp_df['file_commit_load'] = temp_df["additions"].apply(
                    lambda x: x * line_day)
                temp_df['line_commit_load'] = temp_df["commit_load"].apply(lambda x: line_day)
            else:
                temp_df['file_commit_load'] = temp_df[["commit_load", 'counts']].apply(lambda x:
                                                                                       x['commit_load'] / x['counts'],
                                                                                       axis=1)
                temp_df['line_commit_load'] = temp_df["commit_load"].apply(lambda x: 0)
            if df_commit_file_line_time is None:
                df_commit_file_line_time = temp_df
            else:
                df_commit_file_line_time = pd.concat([df_commit_file_line_time, temp_df])
        df_commit_file_line_time = df_commit_file_line_time.loc[df_commit_file_line_time['line_commit_load'] <= 3]

        path_commit_time_dict = dict(list(df_commit_file_line_time.groupby(by='file')))
        file_lineLoad_dict = {}
        for file, temp_df in path_commit_time_dict.items():
            line_commit_load_avg = np.average(temp_df['line_commit_load'].to_list())
            file_lineLoad_dict[file] = line_commit_load_avg
        avg_line_load = np.average(list(file_lineLoad_dict.values()))
        commit_file_no_time_dict = dict(list(df_commit_file_no_time.groupby(by='file')))
        for file, temp_df in commit_file_no_time_dict.items():
            if file in file_lineLoad_dict:
                temp_df['line_commit_load'] = temp_df['additions'].apply(lambda x: file_lineLoad_dict[file])
                temp_df['file_commit_load'] = temp_df["additions"].apply(
                    lambda x: x * file_lineLoad_dict[file])
            else:
                temp_df['line_commit_load'] = temp_df['additions'].apply(lambda x: avg_line_load)
                temp_df['file_commit_load'] = temp_df["additions"].apply(
                    lambda x: x * avg_line_load)
            df_commit_file_line_time = pd.concat([df_commit_file_line_time, temp_df])
        file_load_dict = dict(list(df_commit_file_line_time.groupby(by='file')))
        file_lineLoad_dict = {}
        for file, temp_df in file_load_dict.items():
            line_load_avg = np.average(temp_df['line_commit_load'].to_list())
            file_lineLoad_dict[file] = line_load_avg
        file_lineLoad_dict = dict(sorted(file_lineLoad_dict.items(), key=lambda d: d[1], reverse=True))
        file_lineload_avg = np.average(file_lineLoad_dict.values())
        return df_commit_file_line_time, file_lineLoad_dict, file_lineload_avg

    @staticmethod
    def GetUserReviewAndCommit(project, date, gap=14):
        df_pr = Utils.GetPRData(project)
        date_end_time = f'{date[2]}-{Utils.MONTH[date[3] - 1]}-01 00:00:00'
        date_start_time = (datetime(date[2], date[3], 1, 0, 0) - timedelta(days=gap)).strftime("%Y"
                                                                                               "-%m-%d %H:%M:%S")
        df_pr = df_pr.loc[df_pr['pr_created_at'] >= date_start_time].copy(deep=True)
        df_pr = df_pr.loc[df_pr['pr_created_at'] < date_end_time].copy(deep=True)
        df_pr = df_pr.loc[df_pr['pr_closed_at'] <= date_end_time].copy(deep=True)
        df_review = Utils.GetReviewData(project)
        df_pr_review = pd.merge(left=df_pr, right=df_review, left_on='pr_number', right_on='pr_number')
        df_commit = Utils.GetPRCommitData(project)
        df_pr_commit = pd.merge(left=df_pr, right=df_commit, left_on='pr_number', right_on='pr_number')
        df_commit_file = Utils.GetCommitFile(project)
        df_pr_commit_file = pd.merge(left=df_pr_commit, right=df_commit_file, left_on='commit_sha',
                                     right_on='commit_sha')
        df_pr_commit_file = df_pr_commit_file[
            ['pr_number', 'pr_created_at', 'pr_closed_at', 'committed_at', 'commit_user_login', 'file', 'additions',
             'deletions']].copy(
            deep=True)
        df_pr_review_file = pd.merge(left=df_pr_review, right=df_commit_file, left_on='review_commit_id',
                                     right_on='commit_sha')
        df_pr_review_file = df_pr_review_file[['pr_number', 'pr_created_at', 'pr_closed_at', 'reviewed_at',
                                               'review_user_login', 'file', 'additions', 'deletions']].copy(deep=True)
        return df_pr_review_file, df_pr_commit_file

    @staticmethod
    def GetTrainSetFileReviewWorkload(project, date, gap=90):
        df_pr_review, df_pr_commit = Utils.GetUserReviewAndCommit(project, date, gap)
        df_review_file_line_time, file_review_lineLoad_dict, file_review_lineLoad_avg = Utils.GetReviewFileTime(project,
                                                                                                                date)

        def get_review_load(x):
            change_lines = x['additions'] + x['deletions']
            if x['file'] in file_review_lineLoad_dict:
                load = file_review_lineLoad_dict[x['file']] * change_lines
            else:
                avg_load = sum(file_review_lineLoad_dict.values()) / len(file_review_lineLoad_dict)
                load = avg_load * change_lines
            return load

        df_pr_review['load'] = df_pr_review[['file', 'additions', 'deletions']].apply(lambda x: get_review_load(x),
                                                                                      axis=1)
        df_pr_review = df_pr_review[['pr_number', 'review_user_login', 'file', 'load', 'additions', 'deletions']].copy(
            deep=True)
        df_pr_review['type'] = df_pr_review['load'].apply(lambda x: 'review')
        df_pr_review.rename(columns={'review_user_login': 'user_login'}, inplace=True)
        review_file_load_dict = dict(list(df_pr_review.groupby(by='user_login')))
        reviewer_load_dict = {}
        for reviewer, temp_df in review_file_load_dict.items():
            prs = list(set(temp_df['pr_number'].to_list()))
            pr_files = []
            pr_loads = []
            next_loads = []
            pr_temp_dict = dict(list(temp_df.groupby(by='pr_number')))
            for pr, pr_df in pr_temp_dict.items():
                pr_files.append(len(list(set(pr_df['file'].to_list()))))
                pr_loads.append(sum(pr_df['load'].to_list()))
                next_file_loads = 0
                path_dict = dict(list(pr_df.groupby(by='file')))
                for file, file_df in path_dict.items():
                    next_file_loads += np.average(file_df['load'].to_list())
                next_loads.append(next_file_loads)
            reviewer_load_dict[reviewer] = {
                'dfLoad': temp_df,
                'prNums': prs,
                'prCounts': len(prs),
                'prFiles': pr_files,
                'fileCounts': sum(pr_files),
                'prLoads': pr_loads,
                'loadSum': sum(pr_loads),
                'prNextLoads': next_loads,
                'nextLoadSum': sum(next_loads)
            }
        return {
                   'reviewerLoad': reviewer_load_dict,
                   'fileReviewLineLoad': file_review_lineLoad_dict,
                   'fileReviewLineLoadAvg': file_review_lineLoad_avg,
               }, df_review_file_line_time

    @staticmethod
    def GetTrainSetFileCommitWorkload(project, date, gap=90):
        df_pr_review, df_pr_commit = Utils.GetUserReviewAndCommit(project, date, gap)
        df_commit_file_line_time, file_commit_lineLoad_dict, file_commit_lineLoad_avg = Utils.GetCommitFileTime(project,
                                                                                                                date)

        def get_commit_load(x):
            change_lines = x['additions']
            if x['file'] in file_commit_lineLoad_dict:
                load = file_commit_lineLoad_dict[x['file']] * change_lines
            else:
                avg_load = sum(file_commit_lineLoad_dict.values()) / len(file_commit_lineLoad_dict)
                load = avg_load * change_lines
            return load

        df_pr_commit['load'] = df_pr_commit[['file', 'additions', 'deletions']].apply(lambda x: get_commit_load(x),
                                                                                      axis=1)
        df_pr_commit = df_pr_commit[['pr_number', 'commit_user_login', 'file', 'load', 'additions', 'deletions']].copy(
            deep=True)
        df_pr_commit.rename(columns={'commit_user_login': 'user_login'}, inplace=True)
        df_pr_commit['type'] = df_pr_commit['load'].apply(lambda x: 'commit')
        commit_file_load_dict = dict(list(df_pr_commit.groupby(by='user_login')))
        committer_load_dict = {}
        for committer, temp_df in commit_file_load_dict.items():
            prs = list(set(temp_df['pr_number'].to_list()))
            pr_files = []
            pr_loads = []
            next_loads = []
            pr_temp_dict = dict(list(temp_df.groupby(by='pr_number')))
            for pr, pr_df in pr_temp_dict.items():
                pr_files.append(len(list(set(pr_df['file'].to_list()))))
                pr_loads.append(sum(pr_df['load'].to_list()))
                next_file_loads = 0
                path_dict = dict(list(pr_df.groupby(by='file')))
                for file, file_df in path_dict.items():
                    next_file_loads += np.average(file_df['load'].to_list())
                next_loads.append(next_file_loads)
            committer_load_dict[committer] = {
                'dfLoad': temp_df,
                'prNums': prs,
                'prCounts': len(prs),
                'prFiles': pr_files,
                'fileCounts': sum(pr_files),
                'prLoads': pr_loads,
                'loadSum': sum(pr_loads),
                'prNextLoads': next_loads,
                'nextLoadSum': sum(next_loads)
            }
        return {
            'committerLoad': committer_load_dict,
            'df_commit_file_line_time': df_commit_file_line_time,
            'fileCommitLineLoad': file_commit_lineLoad_dict,
            'fileCommitLineLoadAvg': file_commit_lineLoad_avg,
        }

    @staticmethod
    def GetTargetPrReviewLoad(file_review_lineLoad_dict, file_review_lineLoad_avg, df_file, pr_num):
        df_file = df_file.loc[df_file['pull_number'] == pr_num].copy(deep=True)
        df_file = df_file.loc[df_file['create_before_pr']].copy(deep=True)

        def get_line_load(x):
            if x in file_review_lineLoad_dict:
                return file_review_lineLoad_dict[x]
            else:
                return file_review_lineLoad_avg

        df_file['line_load'] = df_file['file'].apply(lambda x: get_line_load(x))
        df_file['file_load'] = df_file[['line_load', 'file_changes']].apply(
            lambda x: x['line_load'] * x['file_changes'])
        return sum(df_file['file_load'].to_list())

    @staticmethod
    def GetReviewerLoadInfo(reviewerWorkLoad, df_review_file_line_time, convertDict, reviewer_login_list, date, gap):
        date_end_time = f'{date[2]}-{Utils.MONTH[date[3] - 1]}-01 00:00:00'
        date_start_time = (datetime(date[2], date[3], 1, 0, 0) - timedelta(days=gap)).strftime("%Y"
                                                                                               "-%m-%d %H:%M:%S")
        df_review_file_line_time = df_review_file_line_time.loc[
            df_review_file_line_time['pr_created_at'] >= date_start_time].copy(deep=True)
        df_review_file_line_time = df_review_file_line_time.loc[
            df_review_file_line_time['pr_created_at'] < date_end_time].copy(deep=True)
        df_review_file_line_time['review_user_login'] = df_review_file_line_time['review_user_login'].apply(
            lambda x: convertDict[x])
        reviewer_workload_dict = dict(list(df_review_file_line_time.groupby(by='review_user_login')))
        reviewer_load_sum = {}
        for user, temp_df in reviewer_workload_dict.items():
            reviewer_load_sum[user] = sum(temp_df['file_review_load'].to_list())
        reviewer_remain_load_dict = reviewerWorkLoad['reviewerLoad']
        reviewer_remain_pr_counts = {}
        reviewer_remain_file_counts = {}
        reviewer_remain_file_load = {}
        for user, obj in reviewer_remain_load_dict.items():
            reviewer_remain_pr_counts[convertDict[user]] = obj['prCounts']
            reviewer_remain_file_counts[convertDict[user]] = obj['fileCounts']
            reviewer_remain_file_load[convertDict[user]] = obj['']
        for user in reviewer_login_list:
            if user not in reviewer_load_sum:
                reviewer_load_sum[user] = 0
            if user not in reviewer_remain_pr_counts:
                reviewer_remain_pr_counts[user] = 0
            if user not in reviewer_remain_file_counts:
                reviewer_remain_file_counts[user] = 0
        return reviewer_load_sum, reviewer_remain_pr_counts, reviewer_remain_file_counts

