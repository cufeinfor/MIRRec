# coding=gbk
import heapq
import os
import random
import time
from datetime import datetime

import numpy
import pandas
# import scikit_posthocs
import pandas as pd
import scikit_posthocs
import scipy
import seaborn
import xlrd
import xlwt
from pandas import DataFrame
from scipy.stats import mannwhitneyu, ranksums, ttest_1samp, ttest_ind, wilcoxon

from source.config.projectConfig import projectConfig
from source.nlp.SplitWordHelper import SplitWordHelper
from source.scikit.service.BotUserRecognizer import BotUserRecognizer
from source.scikit.service.RecommendMetricUtils import RecommendMetricUtils
from source.utils.ExcelHelper import ExcelHelper
from source.utils.StringKeyUtils import StringKeyUtils
from source.utils.pandas.pandasHelper import pandasHelper
import matplotlib.pyplot as plt


class DataProcessUtils:

    COLUMN_NAME_ALL = ['pr_repo_full_name', 'pr_number', 'pr_id', 'pr_node_id',
                       'pr_state', 'pr_title', 'pr_user_login', 'pr_body',
                       'pr_created_at',
                       'pr_updated_at', 'pr_closed_at', 'pr_merged_at', 'pr_merge_commit_sha',
                       'pr_author_association', 'pr_merged', 'pr_comments', 'pr_review_comments',
                       'pr_commits', 'pr_additions', 'pr_deletions', 'pr_changed_files',
                       'pr_head_label', 'pr_base_label',
                       'review_repo_full_name', 'review_pull_number',
                       'review_id', 'review_user_login', 'review_body', 'review_state', 'review_author_association',
                       'review_submitted_at', 'review_commit_id', 'review_node_id',

                       'commit_sha',
                       'commit_node_id', 'commit_author_login', 'commit_committer_login', 'commit_commit_author_date',
                       'commit_commit_committer_date', 'commit_commit_message', 'commit_commit_comment_count',
                       'commit_status_total', 'commit_status_additions', 'commit_status_deletions',

                       'file_commit_sha',
                       'file_sha', 'file_filename', 'file_status', 'file_additions', 'file_deletions', 'file_changes',
                       'file_patch',

                       'review_comment_id', 'review_comment_user_login', 'review_comment_body',
                       'review_comment_pull_request_review_id', 'review_comment_diff_hunk', 'review_comment_path',
                       'review_comment_commit_id', 'review_comment_position', 'review_comment_original_position',
                       'review_comment_original_commit_id', 'review_comment_created_at', 'review_comment_updated_at',
                       'review_comment_author_association', 'review_comment_start_line',
                       'review_comment_original_start_line',
                       'review_comment_start_side', 'review_comment_line', 'review_comment_original_line',
                       'review_comment_side', 'review_comment_in_reply_to_id', 'review_comment_node_id',
                       'review_comment_change_trigger']


    COLUMN_NAME_PR_REVIEW_COMMIT_FILE = ['pr_repo_full_name', 'pr_number', 'pr_id', 'pr_node_id',
                                         'pr_state', 'pr_title', 'pr_user_login', 'pr_body',
                                         'pr_created_at',
                                         'pr_updated_at', 'pr_closed_at', 'pr_merged_at', 'pr_merge_commit_sha',
                                         'pr_author_association', 'pr_merged', 'pr_comments', 'pr_review_comments',
                                         'pr_commits', 'pr_additions', 'pr_deletions', 'pr_changed_files',
                                         'pr_head_label', 'pr_base_label',
                                         'review_repo_full_name', 'review_pull_number',
                                         'review_id', 'review_user_login', 'review_body', 'review_state',
                                         'review_author_association',
                                         'review_submitted_at', 'review_commit_id', 'review_node_id',

                                         'commit_sha',
                                         'commit_node_id', 'commit_author_login', 'commit_committer_login',
                                         'commit_commit_author_date',
                                         'commit_commit_committer_date', 'commit_commit_message',
                                         'commit_commit_comment_count',
                                         'commit_status_total', 'commit_status_additions', 'commit_status_deletions',

                                         'file_commit_sha',
                                         'file_sha', 'file_filename', 'file_status', 'file_additions', 'file_deletions',
                                         'file_changes',
                                         'file_patch']

    COLUMN_NAME_REVIEW_COMMENT = [
        'review_comment_id', 'review_comment_user_login', 'review_comment_body',
        'review_comment_pull_request_review_id', 'review_comment_diff_hunk', 'review_comment_path',
        'review_comment_commit_id', 'review_comment_position', 'review_comment_original_position',
        'review_comment_original_commit_id', 'review_comment_created_at', 'review_comment_updated_at',
        'review_comment_author_association', 'review_comment_start_line',
        'review_comment_original_start_line',
        'review_comment_start_side', 'review_comment_line', 'review_comment_original_line',
        'review_comment_side', 'review_comment_in_reply_to_id', 'review_comment_node_id',
        'review_comment_change_trigger']

    COLUMN_NAME_COMMIT_FILE = [
        'commit_sha',
        'commit_node_id', 'commit_author_login', 'commit_committer_login',
        'commit_commit_author_date',
        'commit_commit_committer_date', 'commit_commit_message',
        'commit_commit_comment_count',
        'commit_status_total', 'commit_status_additions', 'commit_status_deletions',
        'file_commit_sha',
        'file_sha', 'file_filename', 'file_status', 'file_additions', 'file_deletions',
        'file_changes',
        'file_patch'
    ]

    COLUMN_NAME_PR_COMMIT_RELATION = [
        'repo_full_name', 'pull_number', 'sha'
    ]

    MAX_CHANGE_FILES = 100

    @staticmethod
    def splitDataByMonth(filename, targetPath, targetFileName, dateCol, dataFrame=None, hasHead=False,
                         columnsName=None):

        df = None
        if dataFrame is not None:
            df = dataFrame
        elif not hasHead:
            df = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITHOUT_HEAD, low_memory=False)
            if columnsName is None:
                raise Exception("columnName is None without head")
            df.columns = columnsName
        else:
            df = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False)
        # print(df[dateCol])

        df['label'] = df[dateCol].apply(lambda x: (time.strptime(x, "%Y-%m-%d %H:%M:%S")))
        df['label_y'] = df['label'].apply(lambda x: x.tm_year)
        df['label_m'] = df['label'].apply(lambda x: x.tm_mon)
        print(max(df['label']), min(df['label']))

        maxYear = max(df['label']).tm_year
        maxMonth = max(df['label']).tm_mon
        minYear = min(df['label']).tm_year
        minMonth = min(df['label']).tm_mon
        print(maxYear, maxMonth, minYear, minMonth)

        # ����·���ж�
        if not os.path.isdir(targetPath):
            os.makedirs(targetPath)

        start = minYear * 12 + minMonth
        end = maxYear * 12 + maxMonth
        for i in range(start, end + 1):
            y = int((i - i % 12) / 12)
            m = i % 12
            if m == 0:
                m = 12
                y = y - 1
            print(y, m)
            subDf = df.loc[(df['label_y'] == y) & (df['label_m'] == m)].copy(deep=True)
            subDf.drop(columns=['label', 'label_y', 'label_m'], inplace=True)
            # print(subDf)
            print(subDf.shape)
            # targetFileName = filename.split(os.sep)[-1].split(".")[0]
            sub_filename = f'{targetFileName}_{y}_{m}_to_{y}_{m}.tsv'
            pandasHelper.writeTSVFile(os.path.join(targetPath, sub_filename), subDf
                                      , pandasHelper.STR_WRITE_STYLE_WRITE_TRUNC)

    @staticmethod
    def changeStringToNumber(data, columns):
        if isinstance(data, DataFrame):
            count = 0
            convertDict = {}
            for column in columns:
                pos = 0
                for item in data[column]:
                    if convertDict.get(item, None) is None:
                        count += 1
                        convertDict[item] = count
                    data.at[pos, column] = convertDict[item]
                    pos += 1
            return convertDict  # ��������ӳ���ֵ�

    @staticmethod
    def changeHGRecSelfStringToNumber(df, df_self, clomns):
        count = 0
        convertDict = {}  # ����ת�����ֵ�  ��ʼΪ1
        if isinstance(df, DataFrame):  # ע�⣺ dataframe֮ǰ��Ҫresetindex
            for column in clomns:
                pos = 0
                for item in df[column]:
                    if convertDict.get(item, None) is None:
                        count += 1
                        convertDict[item] = count
                    df.at[pos, column] = convertDict[item]
                    pos += 1
        if isinstance(df_self, DataFrame):  # ע�⣺ dataframe֮ǰ��Ҫresetindex
            for column in clomns:
                pos = 0
                for item in df_self[column]:
                    if convertDict.get(item, None) is None:
                        count += 1
                        convertDict[item] = count
                    df_self.at[pos, column] = convertDict[item]
                    pos += 1
        return convertDict

    @staticmethod
    def changeMyRecStringToNumber(df_review, df_commit, df_issue_comment, df_review_comment):
        clomns = [['author_user_login', 'review_user_login'],
                  ['author_user_login', 'commit_user_login'],
                  ['author_user_login', 'comment_user_login'],
                  ['author_user_login', 'review_comment_user_login']]
        count = 0
        convertDict = {}
        if isinstance(df_review, DataFrame):
            for column in clomns[0]:
                pos = 0
                for item in df_review[column]:
                    if pd.isna(item):
                        continue
                    if convertDict.get(item, None) is None:
                        count += 1
                        convertDict[item] = count
                    df_review.at[pos, column] = convertDict[item]
                    pos += 1
        if isinstance(df_commit, DataFrame):
            for column in clomns[1]:
                pos = 0
                for item in df_commit[column]:
                    if pd.isna(item):
                        continue
                    if convertDict.get(item, None) is None:
                        count += 1
                        convertDict[item] = count
                    df_commit.at[pos, column] = convertDict[item]
                    pos += 1
        if isinstance(df_issue_comment, DataFrame):
            for column in clomns[2]:
                pos = 0
                for item in df_issue_comment[column]:
                    if pd.isna(item):
                        continue
                    if convertDict.get(item, None) is None:
                        count += 1
                        convertDict[item] = count
                    df_issue_comment.at[pos, column] = convertDict[item]
                    pos += 1

        if isinstance(df_review_comment, DataFrame):
            for column in clomns[3]:
                pos = 0
                for item in df_review_comment[column]:
                    if pd.isna(item):
                        continue
                    if convertDict.get(item, None) is None:
                        count += 1
                        convertDict[item] = count
                    df_review_comment.at[pos, column] = convertDict[item]
                    pos += 1
        return convertDict

    @staticmethod
    def changeMyRecStringToNumber_v2(df_review, df_commit, df_issue_comment, df_review_comment):
        clomns = [['author_user_login', 'review_user_login'],
                  ['author_user_login', 'commit_user_login'],
                  ['author_user_login', 'comment_user_login'],
                  ['author_user_login', 'review_comment_user_login']]
        count = 0
        convertDict = {}
        reviewConvertDict = {}
        if isinstance(df_review, DataFrame):
            for column in clomns[0]:
                pos = 0
                for item in df_review[column]:
                    if pd.isna(item):
                        continue
                    if convertDict.get(item, None) is None:
                        count += 1
                        convertDict[item] = count
                        if column == 'review_user_login':
                            reviewConvertDict[item] = count
                    df_review.at[pos, column] = convertDict[item]
                    pos += 1
        if isinstance(df_commit, DataFrame):
            for column in clomns[1]:
                pos = 0
                for item in df_commit[column]:
                    if pd.isna(item):
                        continue
                    if convertDict.get(item, None) is None:
                        count += 1
                        convertDict[item] = count
                    df_commit.at[pos, column] = convertDict[item]
                    pos += 1
        if isinstance(df_issue_comment, DataFrame):
            for column in clomns[2]:
                pos = 0
                for item in df_issue_comment[column]:
                    if pd.isna(item):
                        continue
                    if convertDict.get(item, None) is None:
                        count += 1
                        convertDict[item] = count
                    df_issue_comment.at[pos, column] = convertDict[item]
                    pos += 1

        if isinstance(df_review_comment, DataFrame):
            for column in clomns[3]:
                pos = 0
                for item in df_review_comment[column]:
                    if pd.isna(item):
                        continue
                    if convertDict.get(item, None) is None:
                        count += 1
                        convertDict[item] = count
                    df_review_comment.at[pos, column] = convertDict[item]
                    pos += 1
        return convertDict, reviewConvertDict

    @staticmethod
    def judgeRecommend(recommendList, answer, recommendNum, convertDict=None):

        topk = RecommendMetricUtils.topKAccuracy(recommendList, answer, recommendNum)
        print("topk")
        print(topk)
        mrr = RecommendMetricUtils.MRR(recommendList, answer, recommendNum)
        print("mrr")
        print(mrr)
        precisionk, recallk, fmeasurek = RecommendMetricUtils.precisionK(recommendList, answer, recommendNum)
        print("precision:")
        print(precisionk)
        print("recall:")
        print(recallk)
        print("fmeasure:")
        print(fmeasurek)
        if convertDict is not None:
            rdk = RecommendMetricUtils.RD(recommendList, convertDict, k=recommendNum)
            print("rdk:")
            print(rdk)
            return topk, mrr, precisionk, recallk, fmeasurek, rdk

        return topk, mrr, precisionk, recallk, fmeasurek

    @staticmethod
    def errorAnalysis(recommendList, answer, filter_answer, recommendNum):

        # [recommend_positive_success_pr_ratio, recommend_positive_success_time_ratio] = RecommendMetricUtils. \
        #     positiveSuccess(recommendList, answer, filter_answer, recommendNum)
        #
        # [recommend_negative_success_pr_ratio, recommend_negative_success_time_ratio] = RecommendMetricUtils. \
        #     negativeSuccess(recommendList, answer, filter_answer, recommendNum)
        #
        # [recommend_positive_fail_pr_ratio, recommend_positive_fail_time_ratio] = RecommendMetricUtils. \
        #     positiveFail(recommendList, answer, filter_answer, recommendNum)
        #
        # [recommend_negative_fail_pr_ratio, recommend_negative_fail_time_ratio] = RecommendMetricUtils. \
        #     negativeFail(recommendList, answer, filter_answer, recommendNum)
        #
        # return [recommend_positive_success_pr_ratio, recommend_positive_success_time_ratio,
        #         recommend_negative_success_pr_ratio, recommend_negative_success_time_ratio,
        #         recommend_positive_fail_pr_ratio, recommend_positive_fail_time_ratio,
        #         recommend_negative_fail_pr_ratio, recommend_negative_fail_time_ratio]

        recommend_positive_success_pr_ratio = RecommendMetricUtils. \
            positiveSuccess2(recommendList, answer, filter_answer, recommendNum)

        recommend_negative_success_pr_ratio = RecommendMetricUtils. \
            negativeSuccess2(recommendList, answer, filter_answer, recommendNum)

        recommend_positive_fail_pr_ratio = RecommendMetricUtils. \
            positiveFail2(recommendList, answer, filter_answer, recommendNum)

        recommend_negative_fail_pr_ratio = RecommendMetricUtils. \
            negativeFail2(recommendList, answer, filter_answer, recommendNum)

        # return [recommend_positive_success_pr_ratio, recommend_positive_success_time_ratio,
        #         recommend_negative_success_pr_ratio, recommend_negative_success_time_ratio,
        #         recommend_positive_fail_pr_ratio, recommend_positive_fail_time_ratio,
        #         recommend_negative_fail_pr_ratio, recommend_negative_fail_time_ratio]

        return [recommend_positive_success_pr_ratio, recommend_negative_success_pr_ratio,
                recommend_positive_fail_pr_ratio, recommend_negative_fail_pr_ratio]

    @staticmethod
    def saveResult(filename, sheetName, topk, mrr, precisionk, recallk, fmeasurek, date, error_analysis_data=None,
                   rdk=None):
        content = None
        if date[3] == 1:
            content = [f"{date[2]}.{date[3]}", f"{date[0]}.{date[1]} - {date[2] - 1}.{12}", "TopKAccuracy"]
        else:
            content = [f"{date[2]}.{date[3]}", f"{date[0]}.{date[1]} - {date[2]}.{date[3] - 1}", "TopKAccuracy"]

        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 1, 2, 3, 4, 5]
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', ''] + topk
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 'MRR']
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 1, 2, 3, 4, 5]
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', ''] + mrr
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 'precisionK']
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 1, 2, 3, 4, 5]
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', ''] + precisionk
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 'recallk']
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 1, 2, 3, 4, 5]
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', ''] + recallk
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 'F-Measure']
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 1, 2, 3, 4, 5]
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', ''] + fmeasurek
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        if error_analysis_data is not None:
            # label = ['recommend_positive_success_pr_ratio', 'recommend_positive_success_time_ratio',
            #          'recommend_negative_success_pr_ratio', 'recommend_negative_success_time_ratio',
            #          'recommend_positive_fail_pr_ratio', 'recommend_positive_fail_time_ratio',
            #          'recommend_negative_fail_pr_ratio', 'recommend_negative_fail_time_ratio']
            label = ['recommend_positive_success_pr_ratio',
                     'recommend_negative_success_pr_ratio',
                     'recommend_positive_fail_pr_ratio',
                     'recommend_negative_fail_pr_ratio']
            for index, data in enumerate(error_analysis_data):
                content = ['', '', label[index]]
                ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
                content = ['', '', 1, 2, 3, 4, 5]
                ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
                content = ['', ''] + data
                ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        if rdk is not None:
            content = ['', '', 'rdk']
            ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
            content = ['', '', 1, 2, 3, 4, 5]
            ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
            content = ['', ''] + rdk
            ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())

    @staticmethod
    def saveResult_Community_Version(filename, sheetName, communities_data, date):
        content = None
        if date[3] == 1:
            content = [f"{date[2]}.{date[3]}", f"{date[0]}.{date[1]} - {date[2] - 1}.{12}"]
        else:
            content = [f"{date[2]}.{date[3]}", f"{date[0]}.{date[1]} - {date[2]}.{date[3] - 1}"]
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())

        communitiesTuple = sorted(communities_data.items(), key=lambda x: x[0])
        for cid, data in communitiesTuple:
            if cid != 'whole':
                continue
            content = ['', '�������', '������С', '��������', 'Modularity', 'entropy', 'avg_variance']
            ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
            content = ['', cid, data['size'], data['community_count'], data['modularity'], data['entropy'],
                       data['avg_variance']]
            ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
            content = ['', '', 'TopKAccuracy']
            ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
            content = ['', '', 1, 2, 3, 4, 5]
            ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
            content = ['', ''] + data['topk']
            ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
            content = ['', '', 'MRR']
            ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
            content = ['', '', 1, 2, 3, 4, 5]
            ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
            content = ['', ''] + data['mrr']
            ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
            content = ['', '', 'precisionK']
            ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
            content = ['', '', 1, 2, 3, 4, 5]
            ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
            content = ['', ''] + data['precisionk']
            ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
            content = ['', '', 'recallk']
            ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
            content = ['', '', 1, 2, 3, 4, 5]
            ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
            content = ['', ''] + data['recallk']
            ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
            content = ['', '', 'F-Measure']
            ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
            content = ['', '', 1, 2, 3, 4, 5]
            ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
            content = ['', ''] + data['fmeasurek']
            ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())

    @staticmethod
    def saveFinallyResult(filename, sheetName, topks, mrrs, precisionks, recallks, fmeasureks,
                          error_analysis_datas=None, rdks=None):

        content = ['', '', "AVG_TopKAccuracy"]
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 1, 2, 3, 4, 5]
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', ''] + DataProcessUtils.getAvgScore(topks)
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 'AVG_MRR']
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 1, 2, 3, 4, 5]
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', ''] + DataProcessUtils.getAvgScore(mrrs)
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 'AVG_precisionK']
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 1, 2, 3, 4, 5]
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', ''] + DataProcessUtils.getAvgScore(precisionks)
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 'AVG_recallk']
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 1, 2, 3, 4, 5]
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', ''] + DataProcessUtils.getAvgScore(recallks)
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 'AVG_F-Measure']
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 1, 2, 3, 4, 5]
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', ''] + DataProcessUtils.getAvgScore(fmeasureks)
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        if rdks is not None:
            content = ['', '', 'AVG_RD']
            ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
            content = ['', '', 1, 2, 3, 4, 5]
            ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
            content = ['', ''] + DataProcessUtils.getAvgScore(rdks)
            ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['']
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        if error_analysis_datas is not None:
            # label = ['recommend_positive_success_pr_ratio', 'recommend_positive_success_time_ratio',
            #          'recommend_negative_success_pr_ratio', 'recommend_negative_success_time_ratio',
            #          'recommend_positive_fail_pr_ratio', 'recommend_positive_fail_time_ratio',
            #          'recommend_negative_fail_pr_ratio', 'recommend_negative_fail_time_ratio']
            label = ['recommend_positive_success_pr_ratio',
                     'recommend_negative_success_pr_ratio',
                     'recommend_positive_fail_pr_ratio',
                     'recommend_negative_fail_pr_ratio']
            for index, data in enumerate(error_analysis_datas):
                content = ['', '', label[index]]
                ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
                content = ['', '', 1, 2, 3, 4, 5]
                ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
                content = ['', ''] + DataProcessUtils.getAvgScore(data)
                ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())

    @staticmethod
    def getAvgScore(scores):

        avg = []
        for i in range(0, scores[0].__len__()):
            avg.append(0)
        for score in scores:
            for i in range(0, score.__len__()):
                avg[i] += score[i]
        for i in range(0, scores[0].__len__()):
            avg[i] /= scores.__len__()
        return avg

    @staticmethod
    def convertFeatureDictToDataFrame(dicts, featureNum):
        ar = numpy.zeros((dicts.__len__(), featureNum))
        result = pandas.DataFrame(ar)
        pos = 0
        for d in dicts:
            for key in d.keys():
                result.loc[pos, key] = d[key]
            pos = pos + 1

        return result

    @staticmethod
    def contactReviewCommentData(projectName):

        pr_review_file_name = os.path.join(projectConfig.getRootPath() + os.sep + 'data' + os.sep + 'train'
                                           , f'ALL_{projectName}_data_pr_review_commit_file.tsv')
        review_comment_file_name = os.path.join(projectConfig.getRootPath() + os.sep + 'data' + os.sep + 'train'
                                                , f'ALL_data_review_comment.tsv')

        out_put_file_name = os.path.join(projectConfig.getRootPath() + os.sep + 'data' + os.sep + 'train'
                                         , f'ALL_{projectName}_data.tsv')

        reviewData = pandasHelper.readTSVFile(pr_review_file_name, pandasHelper.INT_READ_FILE_WITHOUT_HEAD)
        reviewData.columns = DataProcessUtils.COLUMN_NAME_PR_REVIEW_COMMIT_FILE
        print(reviewData.shape)

        commentData = pandasHelper.readTSVFile(review_comment_file_name, pandasHelper.INT_READ_FILE_WITHOUT_HEAD)
        commentData.columns = DataProcessUtils.COLUMN_NAME_REVIEW_COMMENT
        print(commentData.shape)

        result = reviewData.join(other=commentData.set_index('review_comment_pull_request_review_id')
                                 , on='review_id', how='left')

        print(result.loc[result['review_comment_id'].isna()].shape)
        pandasHelper.writeTSVFile(out_put_file_name, result, pandasHelper.STR_WRITE_STYLE_WRITE_TRUNC)

    @staticmethod
    def splitProjectCommitFileData(projectName):

        time1 = datetime.now()
        data_train_path = projectConfig.getDataTrainPath()
        target_file_path = projectConfig.getCommitFilePath()
        pr_commit_relation_path = projectConfig.getPrCommitRelationPath()
        target_file_name = f'ALL_{projectName}_data_commit_file.tsv'

        prReviewData = pandasHelper.readTSVFile(
            os.path.join(data_train_path, f'ALL_{projectName}_data_pr_review_commit_file.tsv'),
            pandasHelper.INT_READ_FILE_WITHOUT_HEAD, low_memory=False)
        print(prReviewData.shape)
        prReviewData.columns = DataProcessUtils.COLUMN_NAME_PR_REVIEW_COMMIT_FILE

        commitFileData = pandasHelper.readTSVFile(
            os.path.join(data_train_path, 'ALL_data_commit_file.tsv'), pandasHelper.INT_READ_FILE_WITHOUT_HEAD
            , low_memory=False)
        commitFileData.columns = DataProcessUtils.COLUMN_NAME_COMMIT_FILE
        print(commitFileData.shape)

        commitPRRelationData = pandasHelper.readTSVFile(
            os.path.join(pr_commit_relation_path, f'ALL_{projectName}_data_pr_commit_relation.tsv'),
            pandasHelper.INT_READ_FILE_WITHOUT_HEAD, low_memory=False
        )
        print(commitPRRelationData.shape)
        print("read file cost time:", datetime.now() - time1)

        commitPRRelationData.columns = ['repo_full_name', 'pull_number', 'sha']
        commitPRRelationData = commitPRRelationData['sha'].copy(deep=True)
        commitPRRelationData.drop_duplicates(inplace=True)
        print(commitPRRelationData.shape)

        prReviewData = prReviewData['commit_sha'].copy(deep=True)
        prReviewData.drop_duplicates(inplace=True)
        print(prReviewData.shape)

        needCommits = prReviewData.append(commitPRRelationData)
        print("before drop duplicates:", needCommits.shape)
        needCommits.drop_duplicates(inplace=True)
        print("actually need commit:", needCommits.shape)
        needCommits = list(needCommits)

        print(commitFileData.columns)
        commitFileData = commitFileData.loc[commitFileData['commit_sha'].
            apply(lambda x: x in needCommits)].copy(deep=True)
        print(commitFileData.shape)

        pandasHelper.writeTSVFile(os.path.join(target_file_path, target_file_name), commitFileData
                                  , pandasHelper.STR_WRITE_STYLE_WRITE_TRUNC)
        print(f"write over: {target_file_name}, cost time:", datetime.now() - time1)

    @staticmethod
    def contactFPSData(projectName, label=StringKeyUtils.STR_LABEL_REVIEW_COMMENT,
                       filter_change_trigger=True):


        time1 = datetime.now()
        data_train_path = projectConfig.getDataTrainPath()
        issue_comment_path = projectConfig.getIssueCommentPath()
        pull_request_path = projectConfig.getPullRequestPath()

        review_path = projectConfig.getReviewDataPath()
        change_trigger_path = projectConfig.getPRTimeLineDataPath()

        if label == StringKeyUtils.STR_LABEL_REVIEW_COMMENT:
            prReviewData = pandasHelper.readTSVFile(
                os.path.join(data_train_path, f'ALL_{projectName}_data_pr_review_commit_file.tsv'), low_memory=False)
            prReviewData.columns = DataProcessUtils.COLUMN_NAME_PR_REVIEW_COMMIT_FILE
            print("raw pr review :", prReviewData.shape)

            commit_file_data_path = projectConfig.getCommitFilePath()

            commitFileData = pandasHelper.readTSVFile(
                os.path.join(commit_file_data_path, f'ALL_{projectName}_data_commit_file.tsv'), low_memory=False,
                header=pandasHelper.INT_READ_FILE_WITH_HEAD)
            print("raw commit file :", commitFileData.shape)

            pr_commit_relation_path = projectConfig.getPrCommitRelationPath()
            commitPRRelationData = pandasHelper.readTSVFile(
                os.path.join(pr_commit_relation_path, f'ALL_{projectName}_data_pr_commit_relation.tsv'),
                pandasHelper.INT_READ_FILE_WITHOUT_HEAD, low_memory=False
            )
            commitPRRelationData.columns = DataProcessUtils.COLUMN_NAME_PR_COMMIT_RELATION
            print("pr_commit_relation:", commitPRRelationData.shape)


        issueCommentData = pandasHelper.readTSVFile(
            os.path.join(issue_comment_path, f'ALL_{projectName}_data_issuecomment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pull_request_path, f'ALL_{projectName}_data_pullrequest.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_path, f'ALL_{projectName}_data_review.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        pr_change_file_path = projectConfig.getPRChangeFilePath()

        prChangeFileData = pandasHelper.readTSVFile(
            os.path.join(pr_change_file_path, f'ALL_{projectName}_data_pr_change_file.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        print("read file cost time:", datetime.now() - time1)


        if label == StringKeyUtils.STR_LABEL_REVIEW_COMMENT:
            prReviewData = prReviewData.loc[prReviewData['pr_state'] == 'closed'].copy(deep=True)
            print("after fliter closed pr:", prReviewData.shape)
        elif label == StringKeyUtils.STR_LABEL_ISSUE_COMMENT or label == StringKeyUtils.STR_LABEL_ALL_COMMENT:
            pullRequestData = pullRequestData.loc[pullRequestData['state'] == 'closed'].copy(deep=True)
            print("after fliter closed pr:", pullRequestData.shape)

        if label == StringKeyUtils.STR_LABEL_REVIEW_COMMENT:

            prReviewData = prReviewData.loc[prReviewData['pr_user_login']
                                            != prReviewData['review_user_login']].copy(deep=True)
            print("after fliter author:", prReviewData.shape)


            prReviewData = prReviewData[['pr_number', 'review_user_login', 'pr_created_at']].copy(deep=True)
            prReviewData.drop_duplicates(inplace=True)
            prReviewData.reset_index(drop=True, inplace=True)
            print("after fliter pr_review:", prReviewData.shape)

            commitFileData = commitFileData[['commit_sha', 'file_filename']].copy(deep=True)
            commitFileData.drop_duplicates(inplace=True)
            commitFileData.reset_index(drop=True, inplace=True)
            print("after fliter commit_file:", commitFileData.shape)

        pullRequestData = pullRequestData[['number', 'created_at', 'closed_at', 'user_login', 'node_id']].copy(
            deep=True)
        reviewData = reviewData[["pull_number", "id", "user_login", 'submitted_at']].copy(deep=True)

        targetFileName = f'FPS_{projectName}_data'
        if label == StringKeyUtils.STR_LABEL_ISSUE_COMMENT:
            targetFileName = f'FPS_ISSUE_{projectName}_data'
        elif label == StringKeyUtils.STR_LABEL_ALL_COMMENT and not filter_change_trigger:
            targetFileName = f'FPS_ALL_{projectName}_data'
        elif label == StringKeyUtils.STR_LABEL_ALL_COMMENT and filter_change_trigger:
            targetFileName = f'FPS_ALL_{projectName}_data_change_trigger'


        if label == StringKeyUtils.STR_LABEL_REVIEW_COMMENT:
            data = pandas.merge(prReviewData, commitPRRelationData, left_on='pr_number', right_on='pull_number')
            print("merge relation:", data.shape)
            data = pandas.merge(data, commitFileData, left_on='sha', right_on='commit_sha')
            data.reset_index(drop=True, inplace=True)
            data.drop(columns=['sha'], inplace=True)
            data.drop(columns=['pr_number'], inplace=True)

            order = ['repo_full_name', 'pull_number', 'pr_created_at', 'review_user_login', 'commit_sha',
                     'file_filename']
            data = data[order]
        elif label == StringKeyUtils.STR_LABEL_ISSUE_COMMENT:
            # data = pandas.merge(pullRequestData, commitPRRelationData, left_on='number', right_on='pull_number')
            # data = pandas.merge(data, commitFileData, left_on='sha', right_on='commit_sha')
            data = pandas.merge(pullRequestData, prChangeFileData, left_on='number', right_on='pull_number')
            data = pandas.merge(data, issueCommentData, left_on='number', right_on='pull_number')

            data = data.loc[data['user_login_x'] != data['user_login_y']].copy(deep=True)
            data.drop(columns=['user_login_x'], axis=1, inplace=True)

            data.drop_duplicates(['number', 'filename', 'user_login_y'], inplace=True)
            data.reset_index(drop=True, inplace=True)
            data['commit_sha'] = None

            data = data[['repo_full_name_x', 'pull_number_x', 'created_at_x', 'user_login_y', 'commit_sha',
                         'filename']].copy(deep=True)
            data.drop_duplicates(inplace=True)
            data.columns = ['repo_full_name', 'pull_number', 'pr_created_at', 'review_user_login', 'commit_sha',
                            'file_filename']
            data.reset_index(drop=True)
        elif label == StringKeyUtils.STR_LABEL_ALL_COMMENT:

            data_issue = pandas.merge(pullRequestData, issueCommentData, left_on='number', right_on='pull_number')
            data_issue = data_issue.loc[data_issue['closed_at'] >= data_issue['created_at_y']].copy(deep=True)
            data_issue = data_issue.loc[data_issue['user_login_x'] != data_issue['user_login_y']].copy(deep=True)

            data_issue.dropna(subset=['user_login_y'], inplace=True)


            data_issue['isBot'] = data_issue['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
            data_issue = data_issue.loc[data_issue['isBot'] == False].copy(deep=True)
            data_issue = data_issue[['node_id_x', 'number', 'created_at_x', 'user_login_y']].copy(deep=True)
            data_issue.columns = ['node_id_x', 'number', 'created_at', 'user_login']
            data_issue.drop_duplicates(inplace=True)

            data_review = pandas.merge(pullRequestData, reviewData, left_on='number', right_on='pull_number')
            data_review = data_review.loc[data_review['user_login_x'] != data_review['user_login_y']].copy(deep=True)


            data_review = data_review.loc[data_review['closed_at'] >= data_review['submitted_at']].copy(deep=True)


            data_review.dropna(subset=['user_login_y'], inplace=True)


            data_review['isBot'] = data_review['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
            data_review = data_review.loc[data_review['isBot'] == False].copy(deep=True)
            data_review = data_review[['node_id', 'number', 'created_at', 'user_login_y']].copy(deep=True)
            data_review.columns = ['node_id', 'number', 'created_at', 'user_login']
            data_review.rename(columns={'node_id': 'node_id_x'}, inplace=True)
            data_review.drop_duplicates(inplace=True)

            # data = pandas.concat([data_issue, data_review], axis=0)  # 0 ��ϲ�
            data = data_review
            data.drop_duplicates(inplace=True)
            data.reset_index(drop=True)
            print(data.shape)



            data = pandas.merge(data, prChangeFileData, left_on='number', right_on='pull_number')


            data['commit_sha'] = 0
            data = data[
                ['repo_full_name', 'number', 'node_id_x', 'created_at', 'user_login', 'commit_sha', 'filename']].copy(
                deep=True)
            data.drop_duplicates(inplace=True)

            # unuseful_review_idx = [] for index, row in data.iterrows(): change_trigger_records =
            # changeTriggerData.loc[(changeTriggerData['pullrequest_node'] == row['node_id_x']) & (changeTriggerData[
            # 'user_login'] == row['user_login'])] if change_trigger_records.empty: unuseful_review_idx.append(index)
            # data = data.drop(labels=unuseful_review_idx, axis=0)
            if filter_change_trigger:

                changeTriggerData = pandasHelper.readTSVFile(
                    os.path.join(change_trigger_path, f'ALL_{projectName}_data_pr_change_trigger.tsv'),
                    pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
                )


                changeTriggerData['label'] = changeTriggerData.apply(
                    lambda x: (x['comment_type'] == 'label_issue_comment' and x['change_trigger'] == 1) or (
                            x['comment_type'] == 'label_review_comment' and x['change_trigger'] == 0), axis=1)
                changeTriggerData = changeTriggerData.loc[changeTriggerData['label'] == True].copy(deep=True)
                # changeTriggerData = changeTriggerData.loc[changeTriggerData['change_trigger'] >= 0].copy(deep=True)
                changeTriggerData = changeTriggerData[['pullrequest_node', 'user_login']].copy(deep=True)
                changeTriggerData.drop_duplicates(inplace=True)
                changeTriggerData.rename(columns={'pullrequest_node': 'node_id_x'}, inplace=True)
                data = pandas.merge(data, changeTriggerData, how='inner')
            data = data.drop(labels='node_id_x', axis=1)

            data.columns = ['repo_full_name', 'pull_number', 'pr_created_at', 'review_user_login', 'commit_sha',
                            'file_filename']
            data.sort_values(by='pull_number', ascending=False, inplace=True)
            data.reset_index(drop=True, inplace=True)

        print("after merge:", data.shape)



        DataProcessUtils.splitDataByMonth(filename=None,
                                          targetPath=os.path.join(projectConfig.getFPSDataPath(), projectName),
                                          targetFileName=targetFileName, dateCol='pr_created_at',
                                          dataFrame=data)

    @staticmethod
    def contactFPSSelfData(projectName, label=StringKeyUtils.STR_LABEL_REVIEW_COMMENT,
                           filter_change_trigger=True):
        issue_comment_path = projectConfig.getIssueCommentPath()
        pull_request_path = projectConfig.getPullRequestPath()
        review_path = projectConfig.getReviewDataPath()



        # issueCommentData = pandasHelper.readTSVFile(
        #     os.path.join(issue_comment_path, f'ALL_{projectName}_data_issuecomment.tsv'),
        #     pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        # )

        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pull_request_path, f'ALL_{projectName}_data_pullrequest.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_path, f'ALL_{projectName}_data_review.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )
        pr_change_file_path = projectConfig.getPRChangeFilePath()


        prChangeFileData = pandasHelper.readTSVFile(
            os.path.join(pr_change_file_path, f'ALL_{projectName}_data_pr_change_file.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        if label == StringKeyUtils.STR_LABEL_ALL_COMMENT:
            pullRequestData = pullRequestData.loc[pullRequestData['state'] == 'closed'].copy(deep=True)
            print("after fliter closed pr:", pullRequestData.shape)

        pullRequestData = pullRequestData[['number', 'created_at', 'closed_at', 'user_login', 'node_id']].copy(
            deep=True)
        reviewData = reviewData[["pull_number", "id", "user_login", 'submitted_at']].copy(deep=True)

        targetFileName = f'FPS_{projectName}_data'
        if label == StringKeyUtils.STR_LABEL_ALL_COMMENT and not filter_change_trigger:
            targetFileName = f'FPS_ALL_{projectName}_data'


        if label == StringKeyUtils.STR_LABEL_ALL_COMMENT:
            #

            # data_issue = pandas.merge(pullRequestData, issueCommentData, left_on='number', right_on='pull_number')
            #

            # data_issue = data_issue.loc[data_issue['closed_at'] >= data_issue['created_at_y']].copy(deep=True)
            # data_issue = data_issue.loc[data_issue['user_login_x'] != data_issue['user_login_y']].copy(deep=True)
            #
            # data_issue.dropna(subset=['user_login_y'], inplace=True)
            #

            # data_issue['isBot'] = data_issue['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
            # data_issue = data_issue.loc[data_issue['isBot'] == False].copy(deep=True)
            # data_issue = data_issue[['node_id_x', 'number', 'created_at_x', 'user_login_y']].copy(deep=True)
            # data_issue.columns = ['node_id_x', 'number', 'created_at', 'user_login']
            # data_issue.drop_duplicates(inplace=True)

            data_review = pandas.merge(pullRequestData, reviewData, left_on='number', right_on='pull_number')

            data_review = data_review.loc[data_review['user_login_x'] != data_review['user_login_y']].copy(deep=True)



            data_review = data_review.loc[data_review['closed_at'] >= data_review['submitted_at']].copy(deep=True)


            data_review.dropna(subset=['user_login_y'], inplace=True)


            data_review['isBot'] = data_review['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
            data_review = data_review.loc[data_review['isBot'] == False].copy(deep=True)
            data_review = data_review[['node_id', 'number', 'created_at', 'user_login_y']].copy(deep=True)
            data_review.columns = ['node_id', 'number', 'created_at', 'user_login']
            data_review.rename(columns={'node_id': 'node_id_x'}, inplace=True)
            data_review.drop_duplicates(inplace=True)

            data = data_review
            data.drop_duplicates(inplace=True)
            data.reset_index(drop=True)

            data = pandas.merge(data, prChangeFileData, left_on='number', right_on='pull_number')


            data['commit_sha'] = 0
            data = data[
                ['repo_full_name', 'number', 'node_id_x', 'created_at', 'user_login', 'commit_sha', 'filename']].copy(
                deep=True)
            data.drop_duplicates(inplace=True)


            data = data.drop(labels='node_id_x', axis=1)
            data.columns = ['repo_full_name', 'pull_number', 'pr_created_at', 'review_user_login', 'commit_sha',
                            'file_filename']
            data.sort_values(by='pull_number', ascending=False, inplace=True)
            data.reset_index(drop=True, inplace=True)



        DataProcessUtils.splitDataByMonth(filename=None,
                                          targetPath=os.path.join(projectConfig.getFPSDataPath(),
                                                                  projectName + '_self'),
                                          targetFileName=targetFileName, dateCol='pr_created_at',
                                          dataFrame=data)

    @staticmethod
    def contactFPS_ACData(projectName, filter_change_trigger=True):

        time1 = datetime.now()
        issue_comment_path = projectConfig.getIssueCommentPath()
        pull_request_path = projectConfig.getPullRequestPath()
        pr_change_file_path = projectConfig.getPRChangeFilePath()
        review_path = projectConfig.getReviewDataPath()



        issueCommentData = pandasHelper.readTSVFile(
            os.path.join(issue_comment_path, f'ALL_{projectName}_data_issuecomment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pull_request_path, f'ALL_{projectName}_data_pullrequest.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        prChangeFileData = pandasHelper.readTSVFile(
            os.path.join(pr_change_file_path, f'ALL_{projectName}_data_pr_change_file.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_path, f'ALL_{projectName}_data_review.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        print("read file cost time:", datetime.now() - time1)

        pullRequestData = pullRequestData.loc[pullRequestData['state'] == 'closed'].copy(deep=True)
        print("after fliter closed pr:", pullRequestData.shape)

        pullRequestData = pullRequestData[['number', 'created_at', 'closed_at', 'user_login', 'node_id']].copy(
            deep=True)
        reviewData = reviewData[["pull_number", "id", "user_login", 'submitted_at']].copy(deep=True)

        if filter_change_trigger:
            targetFileName = f'FPS_AC_ALL_{projectName}_data_change_trigger'
        else:
            targetFileName = f'FPS_AC_ALL_{projectName}_data'



        data_issue = pandas.merge(pullRequestData, issueCommentData, left_on='number', right_on='pull_number')


        data_issue = data_issue.loc[data_issue['closed_at'] >= data_issue['created_at_y']].copy(deep=True)
        data_issue = data_issue.loc[data_issue['user_login_x'] != data_issue['user_login_y']].copy(deep=True)

        data_issue.dropna(subset=['user_login_y'], inplace=True)


        data_issue['isBot'] = data_issue['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_issue = data_issue.loc[data_issue['isBot'] == False].copy(deep=True)
        data_issue = data_issue[['node_id_x', 'number', 'created_at_x', 'user_login_y']].copy(deep=True)
        data_issue.columns = ['node_id_x', 'number', 'created_at', 'user_login']
        data_issue.drop_duplicates(inplace=True)

        data_review = pandas.merge(pullRequestData, reviewData, left_on='number', right_on='pull_number')
        data_review = data_review.loc[data_review['user_login_x'] != data_review['user_login_y']].copy(deep=True)


        data_review = data_review.loc[data_review['closed_at'] >= data_review['submitted_at']].copy(deep=True)


        data_review.dropna(subset=['user_login_y'], inplace=True)


        data_review['isBot'] = data_review['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_review = data_review.loc[data_review['isBot'] == False].copy(deep=True)
        data_review = data_review[['node_id', 'number', 'created_at', 'user_login_y']].copy(deep=True)
        data_review.columns = ['node_id', 'number', 'created_at', 'user_login']
        data_review.rename(columns={'node_id': 'node_id_x'}, inplace=True)
        data_review.drop_duplicates(inplace=True)

        data = pandas.concat([data_issue, data_review], axis=0)  # 0 ��ϲ�
        data.drop_duplicates(inplace=True)
        data.reset_index(drop=True)
        print(data.shape)



        data = pandas.merge(data, prChangeFileData, left_on='number', right_on='pull_number')


        data['commit_sha'] = 0
        data = data[
            ['repo_full_name', 'number', 'node_id_x', 'created_at', 'user_login', 'commit_sha', 'filename']].copy(
            deep=True)
        data.drop_duplicates(inplace=True)

        if filter_change_trigger:
            change_trigger_path = projectConfig.getPRTimeLineDataPath()

            changeTriggerData = pandasHelper.readTSVFile(
                os.path.join(change_trigger_path, f'ALL_{projectName}_data_pr_change_trigger.tsv'),
                pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
            )


            changeTriggerData['label'] = changeTriggerData.apply(
                lambda x: (x['comment_type'] == 'label_issue_comment' and x['change_trigger'] == 1) or (
                        x['comment_type'] == 'label_review_comment' and x['change_trigger'] == 0), axis=1)
            changeTriggerData = changeTriggerData.loc[changeTriggerData['label'] == True].copy(deep=True)
            # changeTriggerData = changeTriggerData.loc[changeTriggerData['change_trigger'] >= 0].copy(deep=True)
            changeTriggerData = changeTriggerData[['pullrequest_node', 'user_login']].copy(deep=True)
            changeTriggerData.drop_duplicates(inplace=True)
            changeTriggerData.rename(columns={'pullrequest_node': 'node_id_x'}, inplace=True)
            data = pandas.merge(data, changeTriggerData, how='inner')
        data = data.drop(labels='node_id_x', axis=1)

        data.columns = ['repo_full_name', 'pull_number', 'pr_created_at', 'review_user_login', 'commit_sha',
                        'file_filename']
        data.sort_values(by='pull_number', ascending=False, inplace=True)
        data.reset_index(drop=True, inplace=True)

        print("after merge:", data.shape)



        DataProcessUtils.splitDataByMonth(filename=None,
                                          targetPath=os.path.join(projectConfig.getFPS_ACDataPath(), projectName),
                                          targetFileName=targetFileName, dateCol='pr_created_at',
                                          dataFrame=data)

    @staticmethod
    def contactIR_ACData(projectName, filter_change_trigger=True):

        if filter_change_trigger:
            targetFileName = f'IR_AC_ALL_{projectName}_data_change_trigger'
        else:
            targetFileName = f'IR_AC_ALL_{projectName}_data'

        issue_comment_path = projectConfig.getIssueCommentPath()
        pull_request_path = projectConfig.getPullRequestPath()
        review_path = projectConfig.getReviewDataPath()



        issueCommentData = pandasHelper.readTSVFile(
            os.path.join(issue_comment_path, f'ALL_{projectName}_data_issuecomment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pull_request_path, f'ALL_{projectName}_data_pullrequest.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_path, f'ALL_{projectName}_data_review.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        data_issue = pandas.merge(pullRequestData, issueCommentData, left_on='number', right_on='pull_number')


        data_issue = data_issue.loc[data_issue['closed_at'] >= data_issue['created_at_y']].copy(deep=True)
        data_issue = data_issue.loc[data_issue['user_login_x'] != data_issue['user_login_y']].copy(deep=True)

        data_issue.dropna(subset=['user_login_y'], inplace=True)


        data_issue['isBot'] = data_issue['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_issue = data_issue.loc[data_issue['isBot'] == False].copy(deep=True)
        "IR�����У� pr_number, review_user_login, pr_title, pr_body, pr_created_at"
        data_issue = data_issue[['number', 'title', 'created_at_x', 'user_login_y', 'node_id_x']].copy(
            deep=True)
        data_issue.columns = ['pr_number', 'pr_title', 'pr_created_at', 'review_user_login', 'pullrequest_node']
        data_issue.drop_duplicates(inplace=True)

        data_review = pandas.merge(pullRequestData, reviewData, left_on='number', right_on='pull_number')
        data_review = data_review.loc[data_review['user_login_x'] != data_review['user_login_y']].copy(deep=True)


        data_review = data_review.loc[data_review['closed_at'] >= data_review['submitted_at']].copy(deep=True)


        data_review.dropna(subset=['user_login_y'], inplace=True)


        data_review['isBot'] = data_review['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_review = data_review.loc[data_review['isBot'] == False].copy(deep=True)
        data_review = data_review[['number', 'title', 'created_at', 'user_login_y', 'node_id_x']].copy(
            deep=True)
        data_review.columns = ['pr_number', 'pr_title', 'pr_created_at', 'review_user_login', 'pullrequest_node']
        data_review.drop_duplicates(inplace=True)

        # data = pandas.concat([data_issue, data_review], axis=0)  # 0 ��ϲ�

        data = data_review

        data.drop_duplicates(inplace=True)
        data.reset_index(drop=True, inplace=True)
        print(data.shape)

        if filter_change_trigger:
            change_trigger_path = projectConfig.getPRTimeLineDataPath()

            changeTriggerData = pandasHelper.readTSVFile(
                os.path.join(change_trigger_path, f'ALL_{projectName}_data_pr_change_trigger.tsv'),
                pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
            )


            changeTriggerData['label'] = changeTriggerData.apply(
                lambda x: (x['comment_type'] == 'label_issue_comment' and x['change_trigger'] == 1) or (
                        x['comment_type'] == 'label_review_comment' and x['change_trigger'] == 0), axis=1)
            changeTriggerData = changeTriggerData.loc[changeTriggerData['label'] == True].copy(deep=True)
            changeTriggerData = changeTriggerData[['pullrequest_node', 'user_login']].copy(deep=True)
            changeTriggerData.drop_duplicates(inplace=True)
            changeTriggerData.rename(columns={'user_login': 'review_user_login'}, inplace=True)
            changeTriggerData.reset_index(inplace=True, drop=True)
            data = pandas.merge(data, changeTriggerData, how='inner')



        data = data[['pr_number', 'review_user_login', 'pr_title', 'pr_created_at']].copy(deep=True)
        data.sort_values(by='pr_number', ascending=False, inplace=True)
        data.reset_index(drop=True)



        DataProcessUtils.splitDataByMonth(filename=None,
                                          targetPath=os.path.join(projectConfig.getIR_ACDataPath(), projectName),
                                          targetFileName=targetFileName, dateCol='pr_created_at',
                                          dataFrame=data)

    @staticmethod
    def convertStringTimeToTimeStrip(s):
        return int(time.mktime(time.strptime(s, "%Y-%m-%d %H:%M:%S")))

    @staticmethod
    def contactMLData(projectName, label=StringKeyUtils.STR_LABEL_REVIEW_COMMENT, filter_change_trigger=False):

        time1 = datetime.now()
        data_train_path = projectConfig.getDataTrainPath()
        commit_file_data_path = projectConfig.getCommitFilePath()
        pr_commit_relation_path = projectConfig.getPrCommitRelationPath()
        issue_comment_path = projectConfig.getIssueCommentPath()
        pull_request_path = projectConfig.getPullRequestPath()
        review_path = projectConfig.getReviewDataPath()

        if label == StringKeyUtils.STR_LABEL_REVIEW_COMMENT:
            prReviewData = pandasHelper.readTSVFile(
                os.path.join(data_train_path, f'ALL_{projectName}_data_pr_review_commit_file.tsv'), low_memory=False)
            prReviewData.columns = DataProcessUtils.COLUMN_NAME_PR_REVIEW_COMMIT_FILE
            print("raw pr review :", prReviewData.shape)



        issueCommentData = pandasHelper.readTSVFile(
            os.path.join(issue_comment_path, f'ALL_{projectName}_data_issuecomment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pull_request_path, f'ALL_{projectName}_data_pullrequest.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_path, f'ALL_{projectName}_data_review.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        targetFileName = f'ML_{projectName}_data'
        if label == StringKeyUtils.STR_LABEL_ISSUE_COMMENT:
            targetFileName = f'ML_ISSUE_{projectName}_data'
        elif label == StringKeyUtils.STR_LABEL_ALL_COMMENT and filter_change_trigger:
            targetFileName = f'ML_ALL_{projectName}_data_change_trigger'
        elif label == StringKeyUtils.STR_LABEL_ALL_COMMENT and not filter_change_trigger:
            targetFileName = f'ML_ALL_{projectName}_data'

        print("read file cost time:", datetime.now() - time1)

        if label == StringKeyUtils.STR_LABEL_REVIEW_COMMENT:



            prReviewData = prReviewData.loc[prReviewData['pr_state'] == 'closed'].copy(deep=True)
            print("after fliter closed pr:", prReviewData.shape)


            prReviewData = prReviewData.loc[prReviewData['pr_user_login']
                                            != prReviewData['review_user_login']].copy(deep=True)
            print("after fliter author:", prReviewData.shape)


            prReviewData = prReviewData[['pr_number', 'review_user_login', 'pr_created_at',
                                         'pr_commits', 'pr_additions', 'pr_deletions',
                                         'pr_changed_files', 'pr_head_label', 'pr_base_label', 'pr_user_login']].copy(
                deep=True)
            prReviewData.drop_duplicates(inplace=True)
            prReviewData.reset_index(drop=True, inplace=True)
            print("after fliter pr_review:", prReviewData.shape)

            author_push_count = []
            author_submit_gap = []
            author_review_count = []
            pos = 0
            for data in prReviewData.itertuples(index=False):
                pullNumber = getattr(data, 'pr_number')
                author = getattr(data, 'pr_user_login')
                temp = prReviewData.loc[prReviewData['pr_user_login'] == author].copy(deep=True)
                temp = temp.loc[temp['pr_number'] < pullNumber].copy(deep=True)
                push_num = temp['pr_number'].drop_duplicates().shape[0]
                author_push_count.append(push_num)

                gap = DataProcessUtils.convertStringTimeToTimeStrip(prReviewData.loc[prReviewData.shape[0] - 1,
                                                                                     'pr_created_at']) - DataProcessUtils.convertStringTimeToTimeStrip(
                    prReviewData.loc[0, 'pr_created_at'])
                if push_num != 0:
                    last_num = list(temp['pr_number'])[-1]
                    this_created_time = getattr(data, 'pr_created_at')
                    last_created_time = list(prReviewData.loc[prReviewData['pr_number'] == last_num]['pr_created_at'])[
                        0]
                    gap = int(time.mktime(time.strptime(this_created_time, "%Y-%m-%d %H:%M:%S"))) - int(
                        time.mktime(time.strptime(last_created_time, "%Y-%m-%d %H:%M:%S")))
                author_submit_gap.append(gap)

                temp = prReviewData.loc[prReviewData['review_user_login'] == author].copy(deep=True)
                temp = temp.loc[temp['pr_number'] < pullNumber].copy(deep=True)
                review_num = temp.shape[0]
                author_review_count.append(review_num)
            prReviewData['author_push_count'] = author_push_count
            prReviewData['author_review_count'] = author_review_count
            prReviewData['author_submit_gap'] = author_submit_gap

            data = prReviewData

        elif label == StringKeyUtils.STR_LABEL_ALL_COMMENT:

            data_issue = pandas.merge(pullRequestData, issueCommentData, left_on='number', right_on='pull_number')


            data_issue = data_issue.loc[data_issue['closed_at'] >= data_issue['created_at_y']].copy(deep=True)
            data_issue = data_issue.loc[data_issue['user_login_x'] != data_issue['user_login_y']].copy(deep=True)

            data_issue.dropna(subset=['user_login_y'], inplace=True)

            data_issue.dropna(subset=['head_label'], inplace=True)


            data_issue['isBot'] = data_issue['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
            data_issue = data_issue.loc[data_issue['isBot'] == False].copy(deep=True)
            data_issue = data_issue[['number', 'node_id_x', 'user_login_y',
                                     'created_at_x', 'commits', 'additions', 'deletions',
                                     'changed_files', 'head_label', 'base_label', 'user_login_x']].copy(deep=True)

            data_issue.columns = ['pr_number', 'node_id_x', 'review_user_login', 'pr_created_at', 'pr_commits',
                                  'pr_additions', 'pr_deletions', 'pr_changed_files', 'pr_head_label',
                                  'pr_base_label', 'pr_user_login']
            data_issue.drop_duplicates(inplace=True)

            data_review = pandas.merge(pullRequestData, reviewData, left_on='number', right_on='pull_number')
            data_review = data_review.loc[data_review['user_login_x'] != data_review['user_login_y']].copy(deep=True)


            data_review = data_review.loc[data_review['closed_at'] >= data_review['submitted_at']].copy(deep=True)


            data_review.dropna(subset=['user_login_y'], inplace=True)

            data_review.dropna(subset=['head_label'], inplace=True)


            data_review['isBot'] = data_review['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
            data_review = data_review.loc[data_review['isBot'] == False].copy(deep=True)
            data_review = data_review[['number', 'node_id_x', 'user_login_y',
                                       'created_at', 'commits', 'additions', 'deletions',
                                       'changed_files', 'head_label', 'base_label', 'user_login_x']].copy(deep=True)

            data_review.columns = ['pr_number', 'node_id_x', 'review_user_login', 'pr_created_at', 'pr_commits',
                                   'pr_additions', 'pr_deletions', 'pr_changed_files', 'pr_head_label',
                                   'pr_base_label', 'pr_user_login']
            data_review.drop_duplicates(inplace=True)

            rawData = pandas.concat([data_issue, data_review], axis=0)  # 0 ��ϲ�
            rawData.drop_duplicates(inplace=True)
            rawData.reset_index(drop=True, inplace=True)
            print(rawData.shape)

            if filter_change_trigger:
                change_trigger_path = projectConfig.getPRTimeLineDataPath()

                changeTriggerData = pandasHelper.readTSVFile(
                    os.path.join(change_trigger_path, f'ALL_{projectName}_data_pr_change_trigger.tsv'),
                    pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
                )


                changeTriggerData['label'] = changeTriggerData.apply(
                    lambda x: (x['comment_type'] == 'label_issue_comment' and x['change_trigger'] == 1) or (
                            x['comment_type'] == 'label_review_comment' and x['change_trigger'] == 0), axis=1)
                changeTriggerData = changeTriggerData.loc[changeTriggerData['label'] == True].copy(deep=True)
                # changeTriggerData = changeTriggerData.loc[changeTriggerData['change_trigger'] >= 0].copy(deep=True)
                changeTriggerData = changeTriggerData[['pullrequest_node', 'user_login']].copy(deep=True)
                changeTriggerData.drop_duplicates(inplace=True)
                changeTriggerData.rename(columns={'pullrequest_node': 'node_id_x',
                                                  'user_login': "review_user_login"}, inplace=True)
                rawData = pandas.merge(rawData, changeTriggerData, how='inner')
            rawData = rawData.drop(labels='node_id_x', axis=1)

            "pr_number, review_user_login, pr_created_at, pr_commits, pr_additions, pr_deletions" \
            "pr_changed_files, pr_head_label, pr_base_label, pr_user_login, author_push_count," \
            "author_review_count, author_submit_gap"

            author_push_count = []
            author_submit_gap = []
            author_review_count = []
            pos = 0
            for data in rawData.itertuples(index=False):
                pullNumber = getattr(data, 'pr_number')
                author = getattr(data, 'pr_user_login')
                temp = rawData.loc[rawData['pr_user_login'] == author].copy(deep=True)
                temp = temp.loc[temp['pr_number'] < pullNumber].copy(deep=True)
                push_num = temp['pr_number'].drop_duplicates().shape[0]
                author_push_count.append(push_num)

                gap = DataProcessUtils.convertStringTimeToTimeStrip(rawData.loc[rawData.shape[0] - 1,
                                                                                'pr_created_at']) - DataProcessUtils.convertStringTimeToTimeStrip(
                    rawData.loc[0, 'pr_created_at'])
                if push_num != 0:
                    last_num = list(temp['pr_number'])[-1]
                    this_created_time = getattr(data, 'pr_created_at')
                    last_created_time = list(rawData.loc[rawData['pr_number'] == last_num]['pr_created_at'])[
                        0]
                    gap = int(time.mktime(time.strptime(this_created_time, "%Y-%m-%d %H:%M:%S"))) - int(
                        time.mktime(time.strptime(last_created_time, "%Y-%m-%d %H:%M:%S")))
                author_submit_gap.append(gap)

                temp = rawData.loc[rawData['review_user_login'] == author].copy(deep=True)
                temp = temp.loc[temp['pr_number'] < pullNumber].copy(deep=True)
                review_num = temp.shape[0]
                author_review_count.append(review_num)
            rawData['author_push_count'] = author_push_count
            rawData['author_review_count'] = author_review_count
            rawData['author_submit_gap'] = author_submit_gap
            data = rawData



        DataProcessUtils.splitDataByMonth(filename=None,
                                          targetPath=os.path.join(projectConfig.getMLDataPath(), projectName),
                                          targetFileName=targetFileName, dateCol='pr_created_at',
                                          dataFrame=data)

    @staticmethod
    def contactCAData(projectName):

        time1 = datetime.now()
        data_train_path = projectConfig.getDataTrainPath()
        commit_file_data_path = projectConfig.getCommitFilePath()
        pr_commit_relation_path = projectConfig.getPrCommitRelationPath()
        prReviewData = pandasHelper.readTSVFile(
            os.path.join(data_train_path, f'ALL_{projectName}_data_pr_review_commit_file.tsv'), low_memory=False)
        prReviewData.columns = DataProcessUtils.COLUMN_NAME_PR_REVIEW_COMMIT_FILE
        print("raw pr review :", prReviewData.shape)

        """commit file ��Ϣ��ƴ�ӳ����� ������̧ͷ"""
        commitFileData = pandasHelper.readTSVFile(
            os.path.join(commit_file_data_path, f'ALL_{projectName}_data_commit_file.tsv'), low_memory=False,
            header=pandasHelper.INT_READ_FILE_WITH_HEAD)
        print("raw commit file :", commitFileData.shape)

        commitPRRelationData = pandasHelper.readTSVFile(
            os.path.join(pr_commit_relation_path, f'ALL_{projectName}_data_pr_commit_relation.tsv'),
            pandasHelper.INT_READ_FILE_WITHOUT_HEAD, low_memory=False
        )
        commitPRRelationData.columns = DataProcessUtils.COLUMN_NAME_PR_COMMIT_RELATION
        print("pr_commit_relation:", commitPRRelationData.shape)

        print("read file cost time:", datetime.now() - time1)



        prReviewData = prReviewData.loc[prReviewData['pr_state'] == 'closed'].copy(deep=True)
        print("after fliter closed pr:", prReviewData.shape)


        prReviewData = prReviewData.loc[prReviewData['pr_user_login']
                                        != prReviewData['review_user_login']].copy(deep=True)
        print("after fliter author:", prReviewData.shape)


        prReviewData = prReviewData[['pr_number', 'review_user_login', 'pr_created_at']].copy(deep=True)
        prReviewData.drop_duplicates(inplace=True)
        prReviewData.reset_index(drop=True, inplace=True)
        print("after fliter pr_review:", prReviewData.shape)

        commitFileData = commitFileData[['commit_sha', 'file_filename']].copy(deep=True)
        commitFileData.drop_duplicates(inplace=True)
        commitFileData.reset_index(drop=True, inplace=True)
        print("after fliter commit_file:", commitFileData.shape)

        data = pandas.merge(prReviewData, commitPRRelationData, left_on='pr_number', right_on='pull_number')
        print("merge relation:", data.shape)
        data = pandas.merge(data, commitFileData, left_on='sha', right_on='commit_sha')
        data.reset_index(drop=True, inplace=True)
        data.drop(columns=['sha'], inplace=True)
        data.drop(columns=['pr_number'], inplace=True)

        order = ['repo_full_name', 'pull_number', 'pr_created_at', 'review_user_login', 'commit_sha', 'file_filename']
        data = data[order]
        # print(data.columns)
        print("after merge:", data.shape)



        DataProcessUtils.splitDataByMonth(filename=None, targetPath=os.path.join(projectConfig.getCADataPath(),
                                                                                 projectName),
                                          targetFileName=f'CA_{projectName}_data', dateCol='pr_created_at',
                                          dataFrame=data)

    @staticmethod
    def contactACData(projectName, filter_change_trigger=False):


        if filter_change_trigger:
            targetFileName = f'AC_ALL_{projectName}_data_change_trigger'
        else:
            targetFileName = f'AC_ALL_{projectName}_data'

        """��ȡ��Ϣ  ACֻ��Ҫpr ��created_time_at����Ϣ"""
        issue_comment_path = projectConfig.getIssueCommentPath()
        pull_request_path = projectConfig.getPullRequestPath()
        review_path = projectConfig.getReviewDataPath()
        change_trigger_path = projectConfig.getPRTimeLineDataPath()



        issueCommentData = pandasHelper.readTSVFile(
            os.path.join(issue_comment_path, f'ALL_{projectName}_data_issuecomment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pull_request_path, f'ALL_{projectName}_data_pullrequest.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_path, f'ALL_{projectName}_data_review.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        if filter_change_trigger:

            changeTriggerData = pandasHelper.readTSVFile(
                os.path.join(change_trigger_path, f'ALL_{projectName}_data_pr_change_trigger.tsv'),
                pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
            )



        data_issue = pandas.merge(pullRequestData, issueCommentData, left_on='number', right_on='pull_number')


        data_issue = data_issue.loc[data_issue['closed_at'] >= data_issue['created_at_y']].copy(deep=True)
        data_issue = data_issue.loc[data_issue['user_login_x'] != data_issue['user_login_y']].copy(deep=True)

        data_issue.dropna(subset=['user_login_y'], inplace=True)


        data_issue['isBot'] = data_issue['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_issue = data_issue.loc[data_issue['isBot'] == False].copy(deep=True)
        "AC�����У� pr_number, review_user_login, pr_created_at"
        data_issue = data_issue[['number', 'created_at_x', 'user_login_y', 'node_id_x']].copy(deep=True)
        data_issue.columns = ['pr_number', 'pr_created_at', 'review_user_login', 'pullrequest_node']
        data_issue.drop_duplicates(inplace=True)

        data_review = pandas.merge(pullRequestData, reviewData, left_on='number', right_on='pull_number')
        data_review = data_review.loc[data_review['user_login_x'] != data_review['user_login_y']].copy(deep=True)


        data_review = data_review.loc[data_review['closed_at'] >= data_review['submitted_at']].copy(deep=True)


        data_review.dropna(subset=['user_login_y'], inplace=True)


        data_review['isBot'] = data_review['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_review = data_review.loc[data_review['isBot'] == False].copy(deep=True)
        data_review = data_review[['number', 'created_at', 'user_login_y', 'node_id_x']].copy(deep=True)
        data_review.columns = ['pr_number', 'pr_created_at', 'review_user_login', 'pullrequest_node']
        data_review.drop_duplicates(inplace=True)

        # data = pandas.concat([data_issue, data_review], axis=0)  # 0 ��ϲ�
        data = data_review
        data.drop_duplicates(inplace=True)
        data.reset_index(drop=True, inplace=True)
        print(data.shape)

        if filter_change_trigger:


            changeTriggerData['label'] = changeTriggerData.apply(
                lambda x: (x['comment_type'] == 'label_issue_comment' and x['change_trigger'] == 1) or (
                        x['comment_type'] == 'label_review_comment' and x['change_trigger'] == 0), axis=1)
            changeTriggerData = changeTriggerData.loc[changeTriggerData['label'] == True].copy(deep=True)
            changeTriggerData = changeTriggerData[['pullrequest_node', 'user_login']].copy(deep=True)
            changeTriggerData.drop_duplicates(inplace=True)
            changeTriggerData.rename(columns={'user_login': 'review_user_login'}, inplace=True)
            changeTriggerData.reset_index(inplace=True, drop=True)
            data = pandas.merge(data, changeTriggerData, how='inner')



        data = data[['pr_number', 'review_user_login', 'pr_created_at']].copy(deep=True)
        data.sort_values(by='pr_number', ascending=False, inplace=True)
        data.reset_index(drop=True)



        DataProcessUtils.splitDataByMonth(filename=None,
                                          targetPath=os.path.join(projectConfig.getACDataPath(), projectName),
                                          targetFileName=targetFileName, dateCol='pr_created_at',
                                          dataFrame=data)

    @staticmethod
    def contactACSelfData(projectName, filter_change_trigger=False):

        targetFileName = f'AC_ALL_{projectName}_data'

        pull_request_path = projectConfig.getPullRequestPath()
        review_path = projectConfig.getReviewDataPath()

        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pull_request_path, f'ALL_{projectName}_data_pullrequest.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_path, f'ALL_{projectName}_data_review.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        data_review = pandas.merge(pullRequestData, reviewData, left_on='number', right_on='pull_number')
        data_review = data_review.loc[data_review['user_login_x'] == data_review['user_login_y']].copy(deep=True)


        data_review = data_review.loc[data_review['closed_at'] >= data_review['submitted_at']].copy(deep=True)


        data_review.dropna(subset=['user_login_y'], inplace=True)


        data_review['isBot'] = data_review['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_review = data_review.loc[data_review['isBot'] == False].copy(deep=True)
        data_review = data_review[['number', 'created_at', 'user_login_y', 'node_id_x']].copy(deep=True)
        data_review.columns = ['pr_number', 'pr_created_at', 'review_user_login', 'pullrequest_node']
        data_review.drop_duplicates(inplace=True)

        data = data_review
        data.drop_duplicates(inplace=True)
        data.reset_index(drop=True, inplace=True)
        print(data.shape)



        data = data[['pr_number', 'review_user_login', 'pr_created_at']].copy(deep=True)
        data.sort_values(by='pr_number', ascending=False, inplace=True)
        data.reset_index(drop=True)



        DataProcessUtils.splitDataByMonth(filename=None,
                                          targetPath=os.path.join(projectConfig.getACDataPath(), projectName + '_self'),
                                          targetFileName=targetFileName, dateCol='pr_created_at',
                                          dataFrame=data)

    @staticmethod
    def contactCNData(projectName, filter_change_trigger=False):

        start_time = datetime.now()
        data_train_path = projectConfig.getDataTrainPath()
        issue_comment_file_path = projectConfig.getIssueCommentPath()
        review_comment_file_path = projectConfig.getReviewCommentDataPath()
        review_file_path = projectConfig.getReviewDataPath()
        pullrequest_file_path = projectConfig.getPullRequestPath()


        issueCommentData = pandasHelper.readTSVFile(
            os.path.join(issue_comment_file_path, f'ALL_{projectName}_data_issuecomment.tsv'), low_memory=False,
            header=pandasHelper.INT_READ_FILE_WITH_HEAD)
        print("raw issue_comment file: ", issueCommentData.shape)


        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_file_path, f'ALL_{projectName}_data_review.tsv'), low_memory=False,
            header=pandasHelper.INT_READ_FILE_WITH_HEAD)
        print("raw review file: ", reviewData.shape)


        reviewCommentData = pandasHelper.readTSVFile(
            os.path.join(review_comment_file_path, f'ALL_{projectName}_data_review_comment.tsv'), low_memory=False,
            header=pandasHelper.INT_READ_FILE_WITH_HEAD)
        print("raw review_comment file: ", reviewCommentData.shape)

        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pullrequest_file_path, f'ALL_{projectName}_data_pullrequest.tsv'), low_memory=False,
            header=pandasHelper.INT_READ_FILE_WITH_HEAD)
        print("raw pr file:", pullRequestData.shape)

        if filter_change_trigger:
            change_trigger_path = projectConfig.getPRTimeLineDataPath()

            changeTriggerData = pandasHelper.readTSVFile(
                os.path.join(change_trigger_path, f'ALL_{projectName}_data_pr_change_trigger.tsv'),
                pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
            )

        print("read file cost time:", datetime.now() - start_time)



        pullRequestData = pullRequestData.loc[pullRequestData['state'] == 'closed'].copy(deep=True)
        print("after fliter closed pr:", pullRequestData.shape)



        pullRequestData = pullRequestData[
            ['repo_full_name', 'number', 'node_id', 'user_login', 'created_at', 'author_association',
             'closed_at']].copy(deep=True)
        pullRequestData.columns = ['repo_full_name', 'pull_number', 'pullrequest_node', 'pr_author', 'pr_created_at',
                                   'pr_author_association', 'closed_at']
        pullRequestData.drop_duplicates(inplace=True)
        pullRequestData.reset_index(drop=True, inplace=True)
        print("after fliter pr:", pullRequestData.shape)

        #

        # issueCommentData = issueCommentData[
        #     ['pull_number', 'node_id', 'user_login', 'created_at', 'author_association']].copy(deep=True)
        # issueCommentData.columns = ['pull_number', 'comment_node', 'reviewer', 'commented_at', 'reviewer_association']
        # issueCommentData.drop_duplicates(inplace=True)
        # issueCommentData.reset_index(drop=True, inplace=True)
        # issueCommentData['comment_type'] = StringKeyUtils.STR_LABEL_ISSUE_COMMENT
        # print("after fliter issue comment:", issueCommentData.shape)



        reviewData = reviewData[['pull_number', 'id', 'user_login', 'submitted_at', 'author_association']].copy(
            deep=True)
        # reviewData.columns = ['pull_number', 'pull_request_review_id', 'reviewer', 'submitted_at']
        reviewData.columns = ['pull_number', 'comment_node', 'reviewer', 'commented_at',
                              'reviewer_association']
        reviewData.drop_duplicates(inplace=True)
        reviewData.reset_index(drop=True, inplace=True)

        reviewData['comment_type'] = StringKeyUtils.STR_LABEL_REVIEW
        print("after fliter review:", reviewData.shape)

        # reviewCommentData = reviewCommentData[
        #     ['pull_request_review_id', 'node_id', 'user_login', 'created_at', 'author_association']].copy(deep=True)
        # reviewCommentData.columns = ['pull_request_review_id', 'comment_node', 'reviewer', 'commented_at',
        #                              'reviewer_association']
        # reviewCommentData.drop_duplicates(inplace=True)
        # reviewCommentData.reset_index(drop=True, inplace=True)
        # reviewCommentData['comment_type'] = StringKeyUtils.STR_LABEL_REVIEW_COMMENT
        # print("after fliter review comment:", reviewCommentData.shape)
        #
        # """���ӱ�"""
        # """����û���������۵�reviewҲ����"""
        # reviewCommentData = pandas.merge(reviewData, reviewCommentData, on='pull_request_review_id', how='left')
        # reviewCommentData['reviewer'] = reviewCommentData.apply(
        #     lambda row: row['reviewer_x'] if pandas.isna(row['reviewer_y']) else row['reviewer_y'], axis=1)
        # reviewCommentData['commented_at'] = reviewCommentData.apply(
        #     lambda row: row['submitted_at'] if pandas.isna(row['commented_at']) else row['commented_at'], axis=1)
        # reviewCommentData.drop(columns=['pull_request_review_id', 'submitted_at', 'reviewer_x', 'reviewer_y'],
        #                        inplace=True)
        #
        # data = pandas.concat([issueCommentData, reviewCommentData])
        data = reviewData

        data.reset_index(drop=True, inplace=True)
        data = pandas.merge(pullRequestData, data, left_on='pull_number', right_on='pull_number')
        print("contact review & issue comment:", data.shape)

        data = data.loc[data['closed_at'] >= data['commented_at']].copy(deep=True)
        data.drop(columns=['closed_at'], inplace=True)
        print("after filter comment after pr closed:", data.shape)

        data = data[data['reviewer'] != data['pr_author']]
        data.reset_index(drop=True, inplace=True)
        print("after filter self reviewer:", data.shape)



        data.dropna(subset=['reviewer', 'pr_author'], inplace=True)



        data['isBot'] = data['reviewer'].apply(lambda x: BotUserRecognizer.isBot(x))
        data = data.loc[data['isBot'] == False].copy(deep=True)
        data.drop(columns=['isBot'], inplace=True)
        print("after filter robot reviewer:", data.shape)

        if filter_change_trigger:


            # changeTriggerData = changeTriggerData.loc[changeTriggerData['change_trigger'] >= 0].copy(deep=True)
            changeTriggerData['label'] = changeTriggerData.apply(
                lambda x: (x['comment_type'] == 'label_issue_comment' and x['change_trigger'] == 1) or (
                        x['comment_type'] == 'label_review_comment' and x['change_trigger'] == 0), axis=1)
            changeTriggerData = changeTriggerData.loc[changeTriggerData['label'] == True].copy(deep=True)
            changeTriggerData = changeTriggerData[['comment_node']].copy(deep=True)
            changeTriggerData.drop_duplicates(inplace=True)
            data = pandas.merge(data, changeTriggerData, how='inner')
            print("after filter by change_trigger:", data.shape)

        data = data.drop(labels='pullrequest_node', axis=1)

        if filter_change_trigger:
            targetFileName = f'CN_{projectName}_data_change_trigger'
        else:
            targetFileName = f'CN_{projectName}_data'



        DataProcessUtils.splitDataByMonth(filename=None,
                                          targetPath=os.path.join(projectConfig.getCNDataPath(), projectName),
                                          targetFileName=targetFileName, dateCol='pr_created_at',
                                          dataFrame=data)

    @staticmethod
    def contactCNSelfData(projectName, filter_change_trigger=False):
        start_time = datetime.now()
        review_file_path = projectConfig.getReviewDataPath()
        pullrequest_file_path = projectConfig.getPullRequestPath()


        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_file_path, f'ALL_{projectName}_data_review.tsv'), low_memory=False,
            header=pandasHelper.INT_READ_FILE_WITH_HEAD)
        print("raw review file: ", reviewData.shape)

        """��ȡpullRequest"""
        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pullrequest_file_path, f'ALL_{projectName}_data_pullrequest.tsv'), low_memory=False,
            header=pandasHelper.INT_READ_FILE_WITH_HEAD)
        print("raw pr file:", pullRequestData.shape)

        print("read file cost time:", datetime.now() - start_time)



        pullRequestData = pullRequestData.loc[pullRequestData['state'] == 'closed'].copy(deep=True)
        print("after fliter closed pr:", pullRequestData.shape)



        pullRequestData = pullRequestData[
            ['repo_full_name', 'number', 'node_id', 'user_login', 'created_at', 'author_association',
             'closed_at']].copy(deep=True)
        pullRequestData.columns = ['repo_full_name', 'pull_number', 'pullrequest_node', 'pr_author', 'pr_created_at',
                                   'pr_author_association', 'closed_at']
        pullRequestData.drop_duplicates(inplace=True)
        pullRequestData.reset_index(drop=True, inplace=True)
        print("after fliter pr:", pullRequestData.shape)



        reviewData = reviewData[['pull_number', 'id', 'user_login', 'submitted_at', 'author_association']].copy(
            deep=True)
        reviewData.columns = ['pull_number', 'comment_node', 'reviewer', 'commented_at',
                              'reviewer_association']
        reviewData.drop_duplicates(inplace=True)
        reviewData.reset_index(drop=True, inplace=True)

        reviewData['comment_type'] = StringKeyUtils.STR_LABEL_REVIEW
        print("after fliter review:", reviewData.shape)
        data = reviewData

        data.reset_index(drop=True, inplace=True)
        data = pandas.merge(pullRequestData, data, left_on='pull_number', right_on='pull_number')
        print("contact review & issue comment:", data.shape)

        """����comment��closed֮��ĳ���"""
        data = data.loc[data['closed_at'] >= data['commented_at']].copy(deep=True)
        data.drop(columns=['closed_at'], inplace=True)
        print("after filter comment after pr closed:", data.shape)

        """ȥ���Լ���reviewer�����"""
        data = data[data['reviewer'] == data['pr_author']]
        data.reset_index(drop=True, inplace=True)
        print("after filter self reviewer:", data.shape)



        data.dropna(subset=['reviewer', 'pr_author'], inplace=True)



        data['isBot'] = data['reviewer'].apply(lambda x: BotUserRecognizer.isBot(x))
        data = data.loc[data['isBot'] == False].copy(deep=True)
        data.drop(columns=['isBot'], inplace=True)
        print("after filter robot reviewer:", data.shape)
        data = data.drop(labels='pullrequest_node', axis=1)

        if filter_change_trigger:
            targetFileName = f'CN_{projectName}_data_change_trigger'
        else:
            targetFileName = f'CN_{projectName}_data'



        DataProcessUtils.splitDataByMonth(filename=None,
                                          targetPath=os.path.join(projectConfig.getCNDataPath(), projectName + '_self'),
                                          targetFileName=targetFileName, dateCol='pr_created_at',
                                          dataFrame=data)

    @staticmethod
    def contactCN_IR_Data(projectName, filter_change_trigger=False):

        start_time = datetime.now()
        issue_comment_file_path = projectConfig.getIssueCommentPath()
        review_comment_file_path = projectConfig.getReviewCommentDataPath()
        review_file_path = projectConfig.getReviewDataPath()
        pullrequest_file_path = projectConfig.getPullRequestPath()


        issueCommentData = pandasHelper.readTSVFile(
            os.path.join(issue_comment_file_path, f'ALL_{projectName}_data_issuecomment.tsv'), low_memory=False,
            header=pandasHelper.INT_READ_FILE_WITH_HEAD)
        print("raw issue_comment file: ", issueCommentData.shape)


        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_file_path, f'ALL_{projectName}_data_review.tsv'), low_memory=False,
            header=pandasHelper.INT_READ_FILE_WITH_HEAD)
        print("raw review file: ", reviewData.shape)


        reviewCommentData = pandasHelper.readTSVFile(
            os.path.join(review_comment_file_path, f'ALL_{projectName}_data_review_comment.tsv'), low_memory=False,
            header=pandasHelper.INT_READ_FILE_WITH_HEAD)
        print("raw review_comment file: ", reviewCommentData.shape)


        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pullrequest_file_path, f'ALL_{projectName}_data_pullrequest.tsv'), low_memory=False,
            header=pandasHelper.INT_READ_FILE_WITH_HEAD)
        print("raw pr file:", pullRequestData.shape)

        print("read file cost time:", datetime.now() - start_time)



        pullRequestData = pullRequestData.loc[pullRequestData['state'] == 'closed'].copy(deep=True)
        print("after fliter closed pr:", pullRequestData.shape)



        pullRequestData = pullRequestData[
            ['repo_full_name', 'number', 'title', 'body', 'node_id', 'user_login', 'created_at', 'author_association',
             'closed_at']].copy(deep=True)
        pullRequestData.columns = ['repo_full_name', 'pull_number', 'pr_title', 'pr_body', 'pullrequest_node',
                                   'pr_author', 'pr_created_at',
                                   'pr_author_association', 'closed_at']
        pullRequestData.drop_duplicates(inplace=True)
        pullRequestData.reset_index(drop=True, inplace=True)
        print("after fliter pr:", pullRequestData.shape)

        #

        # issueCommentData = issueCommentData[
        #     ['pull_number', 'node_id', 'user_login', 'created_at', 'author_association']].copy(deep=True)
        # issueCommentData.columns = ['pull_number', 'comment_node', 'reviewer', 'commented_at', 'reviewer_association']
        # issueCommentData.drop_duplicates(inplace=True)
        # issueCommentData.reset_index(drop=True, inplace=True)
        # issueCommentData['comment_type'] = StringKeyUtils.STR_LABEL_ISSUE_COMMENT
        # print("after fliter issue comment:", issueCommentData.shape)



        reviewData = reviewData[['pull_number', 'id', 'user_login', 'submitted_at', 'author_association']].copy(
            deep=True)
        # reviewData.columns = ['pull_number', 'pull_request_review_id', 'reviewer', 'submitted_at']
        reviewData.columns = ['pull_number', 'comment_node', 'reviewer', 'commented_at', 'reviewer_association']
        reviewData.drop_duplicates(inplace=True)
        reviewData.reset_index(drop=True, inplace=True)
        reviewData['comment_type'] = StringKeyUtils.STR_LABEL_REVIEW
        print("after fliter review:", reviewData.shape)

        # reviewCommentData = reviewCommentData[
        #     ['pull_request_review_id', 'node_id', 'user_login', 'created_at', 'author_association']].copy(deep=True)
        # reviewCommentData.columns = ['pull_request_review_id', 'comment_node', 'reviewer', 'commented_at',
        #                              'reviewer_association']
        # reviewCommentData.drop_duplicates(inplace=True)
        # reviewCommentData.reset_index(drop=True, inplace=True)
        # reviewCommentData['comment_type'] = StringKeyUtils.STR_LABEL_REVIEW_COMMENT
        # print("after fliter review comment:", reviewCommentData.shape)
        #
        # """���ӱ�"""
        # """����û���������۵�reviewҲ����"""
        # reviewCommentData = pandas.merge(reviewData, reviewCommentData, on='pull_request_review_id', how='left')
        # reviewCommentData['reviewer'] = reviewCommentData.apply(
        #     lambda row: row['reviewer_x'] if pandas.isna(row['reviewer_y']) else row['reviewer_y'], axis=1)
        # reviewCommentData['commented_at'] = reviewCommentData.apply(
        #     lambda row: row['submitted_at'] if pandas.isna(row['commented_at']) else row['commented_at'], axis=1)
        # reviewCommentData.drop(columns=['pull_request_review_id', 'submitted_at', 'reviewer_x', 'reviewer_y'],
        #                        inplace=True)
        #
        # data = pandas.concat([issueCommentData, reviewCommentData])

        data = reviewData

        data.reset_index(drop=True, inplace=True)
        data = pandas.merge(pullRequestData, data, left_on='pull_number', right_on='pull_number')
        print("contact review & issue comment:", data.shape)

        """����comment��closed֮��ĳ���"""
        data = data.loc[data['closed_at'] >= data['commented_at']].copy(deep=True)
        data.drop(columns=['closed_at'], inplace=True)
        print("after filter comment after pr closed:", data.shape)

        """ȥ���Լ���reviewer�����"""
        data = data[data['reviewer'] != data['pr_author']]
        data.reset_index(drop=True, inplace=True)
        print("after filter self reviewer:", data.shape)



        data.dropna(subset=['reviewer', 'pr_author'], inplace=True)



        data['isBot'] = data['reviewer'].apply(lambda x: BotUserRecognizer.isBot(x))
        data = data.loc[data['isBot'] == False].copy(deep=True)
        data.drop(columns=['isBot'], inplace=True)
        print("after filter robot reviewer:", data.shape)

        if filter_change_trigger:
            change_trigger_path = projectConfig.getPRTimeLineDataPath()

            changeTriggerData = pandasHelper.readTSVFile(
                os.path.join(change_trigger_path, f'ALL_{projectName}_data_pr_change_trigger.tsv'),
                pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
            )


            # changeTriggerData = changeTriggerData.loc[changeTriggerData['change_trigger'] >= 0].copy(deep=True)
            changeTriggerData['label'] = changeTriggerData.apply(
                lambda x: (x['comment_type'] == 'label_issue_comment' and x['change_trigger'] == 1) or (
                        x['comment_type'] == 'label_review_comment' and x['change_trigger'] == 0), axis=1)
            changeTriggerData = changeTriggerData.loc[changeTriggerData['label'] == True].copy(deep=True)
            changeTriggerData = changeTriggerData[['comment_node']].copy(deep=True)
            changeTriggerData.drop_duplicates(inplace=True)
            data = pandas.merge(data, changeTriggerData, how='inner')
            print("after filter by change_trigger:", data.shape)

        data = data.drop(labels='pullrequest_node', axis=1)

        if filter_change_trigger:
            targetFileName = f'CN_IR_{projectName}_data_change_trigger'
        else:
            targetFileName = f'CN_IR_{projectName}_data'



        DataProcessUtils.splitDataByMonth(filename=None,
                                          targetPath=os.path.join(projectConfig.getCN_IRDataPath(), projectName),
                                          targetFileName=targetFileName, dateCol='pr_created_at',
                                          dataFrame=data)

    @staticmethod
    def contactEARECData(projectName, filter_change_trigger=False):

        start_time = datetime.now()
        issue_comment_file_path = projectConfig.getIssueCommentPath()
        review_comment_file_path = projectConfig.getReviewCommentDataPath()
        review_file_path = projectConfig.getReviewDataPath()
        pullrequest_file_path = projectConfig.getPullRequestPath()


        issueCommentData = pandasHelper.readTSVFile(
            os.path.join(issue_comment_file_path, f'ALL_{projectName}_data_issuecomment.tsv'), low_memory=False,
            header=pandasHelper.INT_READ_FILE_WITH_HEAD)
        print("raw issue_comment file: ", issueCommentData.shape)


        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_file_path, f'ALL_{projectName}_data_review.tsv'), low_memory=False,
            header=pandasHelper.INT_READ_FILE_WITH_HEAD)
        print("raw review file: ", reviewData.shape)


        reviewCommentData = pandasHelper.readTSVFile(
            os.path.join(review_comment_file_path, f'ALL_{projectName}_data_review_comment.tsv'), low_memory=False,
            header=pandasHelper.INT_READ_FILE_WITH_HEAD)
        print("raw review_comment file: ", reviewCommentData.shape)

        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pullrequest_file_path, f'ALL_{projectName}_data_pullrequest.tsv'), low_memory=False,
            header=pandasHelper.INT_READ_FILE_WITH_HEAD)
        print("raw pr file:", pullRequestData.shape)

        if filter_change_trigger:
            change_trigger_path = projectConfig.getPRTimeLineDataPath()

            changeTriggerData = pandasHelper.readTSVFile(
                os.path.join(change_trigger_path, f'ALL_{projectName}_data_pr_change_trigger.tsv'),
                pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
            )

        print("read file cost time:", datetime.now() - start_time)



        pullRequestData = pullRequestData.loc[pullRequestData['state'] == 'closed'].copy(deep=True)
        print("after fliter closed pr:", pullRequestData.shape)



        pullRequestData = pullRequestData[
            ['repo_full_name', 'number', 'title', 'body', 'node_id', 'user_login', 'created_at', 'author_association',
             'closed_at']].copy(deep=True)
        pullRequestData.columns = ['repo_full_name', 'pull_number', 'pr_title', 'pr_body', 'pullrequest_node',
                                   'pr_author', 'pr_created_at',
                                   'pr_author_association', 'closed_at']
        pullRequestData.drop_duplicates(inplace=True)
        pullRequestData.reset_index(drop=True, inplace=True)
        print("after fliter pr:", pullRequestData.shape)

        #

        # issueCommentData = issueCommentData[
        #     ['pull_number', 'node_id', 'user_login', 'created_at', 'author_association']].copy(deep=True)
        # issueCommentData.columns = ['pull_number', 'comment_node', 'reviewer', 'commented_at', 'reviewer_association']
        # issueCommentData.drop_duplicates(inplace=True)
        # issueCommentData.reset_index(drop=True, inplace=True)
        # issueCommentData['comment_type'] = StringKeyUtils.STR_LABEL_ISSUE_COMMENT
        # print("after fliter issue comment:", issueCommentData.shape)



        reviewData = reviewData[['pull_number', 'id', 'user_login', 'submitted_at', 'author_association']].copy(
            deep=True)
        reviewData.columns = ['pull_number', 'comment_node', 'reviewer', 'commented_at', 'reviewer_association']
        reviewData.drop_duplicates(inplace=True)
        reviewData.reset_index(drop=True, inplace=True)
        reviewData['comment_type'] = StringKeyUtils.STR_LABEL_REVIEW
        print("after fliter review:", reviewData.shape)

        # reviewCommentData = reviewCommentData[
        #     ['pull_request_review_id', 'node_id', 'user_login', 'created_at', 'author_association']].copy(deep=True)
        # reviewCommentData.columns = ['pull_request_review_id', 'comment_node', 'reviewer', 'commented_at',
        #                              'reviewer_association']
        # reviewCommentData.drop_duplicates(inplace=True)
        # reviewCommentData.reset_index(drop=True, inplace=True)
        # reviewCommentData['comment_type'] = StringKeyUtils.STR_LABEL_REVIEW_COMMENT
        # print("after fliter review comment:", reviewCommentData.shape)

        # """���ӱ�"""
        # """����û���������۵�reviewҲ����"""
        # reviewCommentData = pandas.merge(reviewData, reviewCommentData, on='pull_request_review_id', how='left')
        # reviewCommentData['reviewer'] = reviewCommentData.apply(
        #     lambda row: row['reviewer_x'] if pandas.isna(row['reviewer_y']) else row['reviewer_y'], axis=1)
        # reviewCommentData['commented_at'] = reviewCommentData.apply(
        #     lambda row: row['submitted_at'] if pandas.isna(row['commented_at']) else row['commented_at'], axis=1)
        # reviewCommentData.drop(columns=['pull_request_review_id', 'submitted_at', 'reviewer_x', 'reviewer_y'],
        #                        inplace=True)
        #
        # data = pandas.concat([issueCommentData, reviewCommentData])

        data = reviewData

        data.reset_index(drop=True, inplace=True)
        data = pandas.merge(pullRequestData, data, left_on='pull_number', right_on='pull_number')
        print("contact review & issue comment:", data.shape)

        data = data.loc[data['closed_at'] >= data['commented_at']].copy(deep=True)
        data.drop(columns=['closed_at'], inplace=True)
        print("after filter comment after pr closed:", data.shape)

        data = data[data['reviewer'] != data['pr_author']]
        data.reset_index(drop=True, inplace=True)
        print("after filter self reviewer:", data.shape)



        data.dropna(subset=['reviewer', 'pr_author'], inplace=True)



        data['isBot'] = data['reviewer'].apply(lambda x: BotUserRecognizer.isBot(x))
        data = data.loc[data['isBot'] == False].copy(deep=True)
        data.drop(columns=['isBot'], inplace=True)
        print("after filter robot reviewer:", data.shape)

        if filter_change_trigger:
            """change_triggerֻȡ��comment_node��dataȡ����"""
            # changeTriggerData = changeTriggerData.loc[changeTriggerData['change_trigger'] >= 0].copy(deep=True)
            changeTriggerData['label'] = changeTriggerData.apply(
                lambda x: (x['comment_type'] == 'label_issue_comment' and x['change_trigger'] == 1) or (
                        x['comment_type'] == 'label_review_comment' and x['change_trigger'] == 0), axis=1)
            changeTriggerData = changeTriggerData.loc[changeTriggerData['label'] == True].copy(deep=True)
            changeTriggerData = changeTriggerData[['comment_node']].copy(deep=True)
            changeTriggerData.drop_duplicates(inplace=True)
            data = pandas.merge(data, changeTriggerData, how='inner')
            print("after filter by change_trigger:", data.shape)

        data = data.drop(labels='pullrequest_node', axis=1)

        if filter_change_trigger:
            targetFileName = f'EAREC_{projectName}_data_change_trigger'
        else:
            targetFileName = f'EAREC_{projectName}_data'



        DataProcessUtils.splitDataByMonth(filename=None,
                                          targetPath=os.path.join(projectConfig.getEARECDataPath(), projectName),
                                          targetFileName=targetFileName, dateCol='pr_created_at',
                                          dataFrame=data)

    @staticmethod
    def contactEARECSelfData(projectName, filter_change_trigger=False):

        start_time = datetime.now()
        issue_comment_file_path = projectConfig.getIssueCommentPath()
        review_comment_file_path = projectConfig.getReviewCommentDataPath()
        review_file_path = projectConfig.getReviewDataPath()
        pullrequest_file_path = projectConfig.getPullRequestPath()


        issueCommentData = pandasHelper.readTSVFile(
            os.path.join(issue_comment_file_path, f'ALL_{projectName}_data_issuecomment.tsv'), low_memory=False,
            header=pandasHelper.INT_READ_FILE_WITH_HEAD)
        print("raw issue_comment file: ", issueCommentData.shape)


        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_file_path, f'ALL_{projectName}_data_review.tsv'), low_memory=False,
            header=pandasHelper.INT_READ_FILE_WITH_HEAD)
        print("raw review file: ", reviewData.shape)


        reviewCommentData = pandasHelper.readTSVFile(
            os.path.join(review_comment_file_path, f'ALL_{projectName}_data_review_comment.tsv'), low_memory=False,
            header=pandasHelper.INT_READ_FILE_WITH_HEAD)
        print("raw review_comment file: ", reviewCommentData.shape)

        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pullrequest_file_path, f'ALL_{projectName}_data_pullrequest.tsv'), low_memory=False,
            header=pandasHelper.INT_READ_FILE_WITH_HEAD)
        print("raw pr file:", pullRequestData.shape)

        print("read file cost time:", datetime.now() - start_time)



        pullRequestData = pullRequestData.loc[pullRequestData['state'] == 'closed'].copy(deep=True)
        print("after fliter closed pr:", pullRequestData.shape)



        pullRequestData = pullRequestData[
            ['repo_full_name', 'number', 'title', 'body', 'node_id', 'user_login', 'created_at', 'author_association',
             'closed_at']].copy(deep=True)
        pullRequestData.columns = ['repo_full_name', 'pull_number', 'pr_title', 'pr_body', 'pullrequest_node',
                                   'pr_author', 'pr_created_at',
                                   'pr_author_association', 'closed_at']
        pullRequestData.drop_duplicates(inplace=True)
        pullRequestData.reset_index(drop=True, inplace=True)
        print("after fliter pr:", pullRequestData.shape)



        reviewData = reviewData[['pull_number', 'id', 'user_login', 'submitted_at', 'author_association']].copy(
            deep=True)
        reviewData.columns = ['pull_number', 'comment_node', 'reviewer', 'commented_at', 'reviewer_association']
        reviewData.drop_duplicates(inplace=True)
        reviewData.reset_index(drop=True, inplace=True)
        reviewData['comment_type'] = StringKeyUtils.STR_LABEL_REVIEW
        print("after fliter review:", reviewData.shape)
        data = reviewData

        data.reset_index(drop=True, inplace=True)
        data = pandas.merge(pullRequestData, data, left_on='pull_number', right_on='pull_number')
        print("contact review & issue comment:", data.shape)

        data = data.loc[data['closed_at'] >= data['commented_at']].copy(deep=True)
        data.drop(columns=['closed_at'], inplace=True)
        print("after filter comment after pr closed:", data.shape)

        data = data[data['reviewer'] == data['pr_author']]
        data.reset_index(drop=True, inplace=True)
        print("after filter self reviewer:", data.shape)



        data.dropna(subset=['reviewer', 'pr_author'], inplace=True)



        data['isBot'] = data['reviewer'].apply(lambda x: BotUserRecognizer.isBot(x))
        data = data.loc[data['isBot'] == False].copy(deep=True)
        data.drop(columns=['isBot'], inplace=True)
        print("after filter robot reviewer:", data.shape)

        data = data.drop(labels='pullrequest_node', axis=1)

        targetFileName = f'EAREC_{projectName}_data'



        DataProcessUtils.splitDataByMonth(filename=None,
                                          targetPath=os.path.join(projectConfig.getEARECDataPath(),
                                                                  projectName + '_self'),
                                          targetFileName=targetFileName, dateCol='pr_created_at',
                                          dataFrame=data)

    @staticmethod
    def contactCFData(projectName, filter_change_trigger=False):

        time1 = datetime.now()
        pull_request_path = projectConfig.getPullRequestPath()
        pr_change_file_path = projectConfig.getPRChangeFilePath()
        review_path = projectConfig.getReviewDataPath()
        issue_comment_path = projectConfig.getIssueCommentPath()
        review_comment_file_path = projectConfig.getReviewCommentDataPath()


        issueCommentData = pandasHelper.readTSVFile(
            os.path.join(issue_comment_path, f'ALL_{projectName}_data_issuecomment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pull_request_path, f'ALL_{projectName}_data_pullrequest.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        prChangeFileData = pandasHelper.readTSVFile(
            os.path.join(pr_change_file_path, f'ALL_{projectName}_data_pr_change_file.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_path, f'ALL_{projectName}_data_review.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        reviewCommentData = pandasHelper.readTSVFile(
            os.path.join(review_comment_file_path, f'ALL_{projectName}_data_review_comment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        if filter_change_trigger:

            change_trigger_path = projectConfig.getPRTimeLineDataPath()
            changeTriggerData = pandasHelper.readTSVFile(
                os.path.join(change_trigger_path, f'ALL_{projectName}_data_pr_change_trigger.tsv'),
                pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
            )
        print("read file cost time:", datetime.now() - time1)

        # �ļ�����
        reviewData = reviewData[["pull_number", "id", "user_login", 'submitted_at']].copy(deep=True)
        reviewData.columns = ['pull_number', 'comment_node', 'reviewer', 'comment_at']
        reviewData['comment_type'] = StringKeyUtils.STR_LABEL_REVIEW

        reviewData.drop_duplicates(inplace=True)
        reviewData.reset_index(drop=True, inplace=True)
        print("after fliter review:", reviewData.shape)

        pullRequestData = pullRequestData.loc[pullRequestData['state'] == 'closed'].copy(deep=True)
        pullRequestData = pullRequestData[
            ['repo_full_name', 'number', 'created_at', 'closed_at', 'user_login']].copy(deep=True)
        pullRequestData.columns = ['repo_full_name', 'pull_number', 'pr_created_at', 'pr_closed_at', 'pr_author']
        pullRequestData.drop_duplicates(inplace=True)
        pullRequestData.reset_index(drop=True, inplace=True)
        print("after fliter pr:", pullRequestData.shape)

        prChangeFileData = prChangeFileData[
            ['pull_number', 'filename']].copy(deep=True)
        prChangeFileData.drop_duplicates(inplace=True)
        prChangeFileData.reset_index(drop=True, inplace=True)
        print("after fliter change_file data:", prChangeFileData.shape)

        # issueCommentData = issueCommentData[
        #     ['pull_number', 'node_id', 'user_login', 'created_at']].copy(deep=True)
        # issueCommentData.columns = ['pull_number', 'comment_node', 'reviewer', 'comment_at']
        # issueCommentData['comment_type'] = StringKeyUtils.STR_LABEL_ISSUE_COMMENT
        # issueCommentData.drop_duplicates(inplace=True)
        # issueCommentData.reset_index(drop=True, inplace=True)
        # print("after fliter issue comment data:", issueCommentData.shape)

        #
        # reviewCommentData = reviewCommentData[
        #     ['pull_request_review_id', 'node_id', 'user_login', 'created_at']].copy(deep=True)
        # reviewCommentData.columns = ['pull_request_review_id', 'comment_node', 'reviewer', 'comment_at']
        # reviewCommentData = pandas.merge(reviewData, reviewCommentData, on='pull_request_review_id', how='left')
        # reviewCommentData['reviewer'] = reviewCommentData.apply(
        #     lambda row: row['reviewer_x'] if pandas.isna(row['reviewer_y']) else row['reviewer_y'], axis=1)
        # reviewCommentData['comment_at'] = reviewCommentData.apply(
        #     lambda row: row['submitted_at'] if pandas.isna(row['comment_at']) else row['comment_at'], axis=1)
        # reviewCommentData = reviewCommentData[
        #     ['pull_number', 'comment_node', 'reviewer', 'comment_at']].copy(deep=True)
        # reviewCommentData.columns = ['pull_number', 'comment_node', 'reviewer', 'comment_at']
        # reviewCommentData['comment_type'] = StringKeyUtils.STR_LABEL_REVIEW_COMMENT
        # reviewCommentData.drop_duplicates(inplace=True)
        # reviewCommentData.reset_index(drop=True, inplace=True)
        # print("after fliter review comment data:", reviewCommentData.shape)

        # data = pandas.concat([issueCommentData, reviewCommentData], axis=0)

        data = reviewData

        data = pandas.merge(pullRequestData, data, on='pull_number')


        data = data.loc[data['pr_closed_at'] >= data['comment_at']].copy(deep=True)

        data = data.loc[data['pr_author'] != data['reviewer']].copy(deep=True)

        data.dropna(subset=['reviewer'], inplace=True)


        data['isBot'] = data['reviewer'].apply(lambda x: BotUserRecognizer.isBot(x))
        data = data.loc[data['isBot'] == False].copy(deep=True)

        data = data[
            ['repo_full_name', 'pull_number', 'pr_author', 'pr_created_at', 'reviewer', 'comment_type', 'comment_node',
             'comment_at']].copy(deep=True)


        data = pandas.merge(data, prChangeFileData, on='pull_number')
        data.drop_duplicates(inplace=True)
        data.reset_index(drop=True)
        print(data.shape)
        print("after merge file change:", data.shape)

        if filter_change_trigger:

            changeTriggerData['label'] = changeTriggerData.apply(
                lambda x: (x['comment_type'] == 'label_issue_comment' and x['change_trigger'] == 1) or (
                        x['comment_type'] == 'label_review_comment' and x['change_trigger'] >= 0), axis=1)
            changeTriggerData = changeTriggerData.loc[changeTriggerData['label'] == True].copy(deep=True)
            changeTriggerData = changeTriggerData[['comment_node']].copy(deep=True)
            changeTriggerData.drop_duplicates(inplace=True)
            data = pandas.merge(data, changeTriggerData, how='inner')
            print("after fliter by change_trigger:", data.shape)

        data.sort_values(by='pull_number', ascending=False, inplace=True)
        data.reset_index(drop=True, inplace=True)



        DataProcessUtils.splitDataByMonth(filename=None,
                                          targetPath=os.path.join(projectConfig.getCFDataPath(), projectName),
                                          targetFileName=f'CF_{projectName}_data', dateCol='pr_created_at',
                                          dataFrame=data)

    @staticmethod
    def convertLabelListToDataFrame(label_data, pull_list, maxNum):

        ar = numpy.zeros((label_data.__len__(), maxNum), dtype=int)
        pos = 0
        for pull_num in pull_list:
            labels = label_data[pull_num]
            for label in labels:
                if label <= maxNum:
                    ar[pos][label - 1] = 1
            pos += 1
        return ar

    @staticmethod
    def convertLabelListToListArray(label_data, pull_list):

        answerList = []
        for pull_num in pull_list:
            answer = []
            labels = label_data[pull_num]
            for label in labels:
                answer.append(label)
            answerList.append(answer)
        return answerList

    @staticmethod
    def getListFromProbable(probable, classList, k):  # �Ƽ�k��
        recommendList = []
        for case in probable:
            max_index_list = list(map(lambda x: numpy.argwhere(case == x), heapq.nlargest(k, case)))
            caseList = []
            pos = 0
            while pos < k:
                item = max_index_list[pos]
                for i in item:
                    caseList.append(classList[i[0]])
                pos += item.shape[0]
            recommendList.append(caseList)
        return recommendList

    @staticmethod
    def convertMultilabelProbaToDataArray(probable):  # �Ƽ�k��

        result = numpy.empty((probable[0].shape[0], probable.__len__()))
        y = 0
        for pro in probable:
            x = 0
            for p in pro[:, 1]:
                result[x][y] = p
                x += 1
            y += 1
        return result

    @staticmethod
    def contactIRData(projectName, label=StringKeyUtils.STR_LABEL_REVIEW_COMMENT, filter_change_trigger=False):


        targetFileName = f'IR_{projectName}_data'
        if label == StringKeyUtils.STR_LABEL_ISSUE_COMMENT:
            targetFileName = f'IR_ISSUE_{projectName}_data'
        elif label == StringKeyUtils.STR_LABEL_ALL_COMMENT and filter_change_trigger:
            targetFileName = f'IR_ALL_{projectName}_data_change_trigger'
        elif label == StringKeyUtils.STR_LABEL_ALL_COMMENT and not filter_change_trigger:
            targetFileName = f'IR_ALL_{projectName}_data'


        data_train_path = projectConfig.getDataTrainPath()
        issue_comment_path = projectConfig.getIssueCommentPath()
        pull_request_path = projectConfig.getPullRequestPath()
        review_path = projectConfig.getReviewDataPath()

        if label == StringKeyUtils.STR_LABEL_REVIEW_COMMENT:
            prReviewData = pandasHelper.readTSVFile(
                os.path.join(data_train_path, f'ALL_{projectName}_data_pr_review_commit_file.tsv'), low_memory=False)
            prReviewData.columns = DataProcessUtils.COLUMN_NAME_PR_REVIEW_COMMIT_FILE
            print("raw pr review :", prReviewData.shape)



        issueCommentData = pandasHelper.readTSVFile(
            os.path.join(issue_comment_path, f'ALL_{projectName}_data_issuecomment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pull_request_path, f'ALL_{projectName}_data_pullrequest.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_path, f'ALL_{projectName}_data_review.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        if filter_change_trigger:
            change_trigger_path = projectConfig.getPRTimeLineDataPath()

            changeTriggerData = pandasHelper.readTSVFile(
                os.path.join(change_trigger_path, f'ALL_{projectName}_data_pr_change_trigger.tsv'),
                pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
            )

        if filter_change_trigger:

            changeTriggerData = pandasHelper.readTSVFile(
                os.path.join(change_trigger_path, f'ALL_{projectName}_data_pr_change_trigger.tsv'),
                pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
            )

        if label == StringKeyUtils.STR_LABEL_REVIEW_COMMENT:


            prReviewData = prReviewData.loc[prReviewData['pr_state'] == 'closed'].copy(deep=True)
            print("after fliter closed pr:", prReviewData.shape)


            prReviewData = prReviewData.loc[prReviewData['pr_user_login']
                                            != prReviewData['review_user_login']].copy(deep=True)
            print("after fliter author:", prReviewData.shape)


            prReviewData = prReviewData[
                ['pr_number', 'review_user_login', 'pr_title', 'pr_body', 'pr_created_at']].copy(
                deep=True)
            prReviewData.drop_duplicates(inplace=True)
            prReviewData.reset_index(drop=True, inplace=True)
            print("after fliter pr_review:", prReviewData.shape)
            data = prReviewData
        elif label == StringKeyUtils.STR_LABEL_ALL_COMMENT:


            data_issue = pandas.merge(pullRequestData, issueCommentData, left_on='number', right_on='pull_number')


            data_issue = data_issue.loc[data_issue['closed_at'] >= data_issue['created_at_y']].copy(deep=True)
            data_issue = data_issue.loc[data_issue['user_login_x'] != data_issue['user_login_y']].copy(deep=True)

            data_issue.dropna(subset=['user_login_y'], inplace=True)


            data_issue['isBot'] = data_issue['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
            data_issue = data_issue.loc[data_issue['isBot'] == False].copy(deep=True)
            "IR�����У� pr_number, review_user_login, pr_title, pr_body, pr_created_at"
            data_issue = data_issue[['number', 'title', 'body_x', 'created_at_x', 'user_login_y', 'node_id_x']].copy(
                deep=True)
            data_issue.columns = ['pr_number', 'pr_title', 'pr_body', 'pr_created_at', 'review_user_login',
                                  'pullrequest_node']
            data_issue.drop_duplicates(inplace=True)

            data_review = pandas.merge(pullRequestData, reviewData, left_on='number', right_on='pull_number')
            data_review = data_review.loc[data_review['user_login_x'] != data_review['user_login_y']].copy(deep=True)


            data_review = data_review.loc[data_review['closed_at'] >= data_review['submitted_at']].copy(deep=True)


            data_review.dropna(subset=['user_login_y'], inplace=True)


            data_review['isBot'] = data_review['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
            data_review = data_review.loc[data_review['isBot'] == False].copy(deep=True)
            data_review = data_review[['number', 'title', 'body_x', 'created_at', 'user_login_y', 'node_id_x']].copy(
                deep=True)
            data_review.columns = ['pr_number', 'pr_title', 'pr_body', 'pr_created_at', 'review_user_login',
                                   'pullrequest_node']
            data_review.drop_duplicates(inplace=True)

            # data = pandas.concat([data_issue, data_review], axis=0)  # 0 ��ϲ�
            data = data_review

            data.drop_duplicates(inplace=True)
            data.reset_index(drop=True, inplace=True)
            print(data.shape)

            if filter_change_trigger:


                changeTriggerData['label'] = changeTriggerData.apply(
                    lambda x: (x['comment_type'] == 'label_issue_comment' and x['change_trigger'] == 1) or (
                            x['comment_type'] == 'label_review_comment' and x['change_trigger'] == 0), axis=1)
                changeTriggerData = changeTriggerData.loc[changeTriggerData['label'] == True].copy(deep=True)
                changeTriggerData = changeTriggerData[['pullrequest_node', 'user_login']].copy(deep=True)
                changeTriggerData.drop_duplicates(inplace=True)
                changeTriggerData.rename(columns={'user_login': 'review_user_login'}, inplace=True)
                changeTriggerData.reset_index(inplace=True, drop=True)
                data = pandas.merge(data, changeTriggerData, how='inner')



            data = data[['pr_number', 'review_user_login', 'pr_title', 'pr_body', 'pr_created_at']].copy(deep=True)
            data.sort_values(by='pr_number', ascending=False, inplace=True)
            data.reset_index(drop=True)



        DataProcessUtils.splitDataByMonth(filename=None,
                                          targetPath=os.path.join(projectConfig.getIRDataPath(), projectName),
                                          targetFileName=targetFileName, dateCol='pr_created_at',
                                          dataFrame=data)

    @staticmethod
    def contactIRSelfData(projectName, label=StringKeyUtils.STR_LABEL_REVIEW_COMMENT, filter_change_trigger=False):
        targetFileName = f'IR_{projectName}_data'
        if label == StringKeyUtils.STR_LABEL_ISSUE_COMMENT:
            targetFileName = f'IR_ISSUE_{projectName}_data'
        elif label == StringKeyUtils.STR_LABEL_ALL_COMMENT and filter_change_trigger:
            targetFileName = f'IR_ALL_{projectName}_data_change_trigger'
        elif label == StringKeyUtils.STR_LABEL_ALL_COMMENT and not filter_change_trigger:
            targetFileName = f'IR_ALL_{projectName}_data'


        data_train_path = projectConfig.getDataTrainPath()
        issue_comment_path = projectConfig.getIssueCommentPath()
        pull_request_path = projectConfig.getPullRequestPath()
        review_path = projectConfig.getReviewDataPath()

        if label == StringKeyUtils.STR_LABEL_REVIEW_COMMENT:
            prReviewData = pandasHelper.readTSVFile(
                os.path.join(data_train_path, f'ALL_{projectName}_data_pr_review_commit_file.tsv'), low_memory=False)
            prReviewData.columns = DataProcessUtils.COLUMN_NAME_PR_REVIEW_COMMIT_FILE
            print("raw pr review :", prReviewData.shape)



        issueCommentData = pandasHelper.readTSVFile(
            os.path.join(issue_comment_path, f'ALL_{projectName}_data_issuecomment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pull_request_path, f'ALL_{projectName}_data_pullrequest.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_path, f'ALL_{projectName}_data_review.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        if label == StringKeyUtils.STR_LABEL_REVIEW_COMMENT:


            prReviewData = prReviewData.loc[prReviewData['pr_state'] == 'closed'].copy(deep=True)
            print("after fliter closed pr:", prReviewData.shape)


            prReviewData = prReviewData.loc[prReviewData['pr_user_login']
                                            != prReviewData['review_user_login']].copy(deep=True)
            print("after fliter author:", prReviewData.shape)


            prReviewData = prReviewData[
                ['pr_number', 'review_user_login', 'pr_title', 'pr_body', 'pr_created_at']].copy(
                deep=True)
            prReviewData.drop_duplicates(inplace=True)
            prReviewData.reset_index(drop=True, inplace=True)
            print("after fliter pr_review:", prReviewData.shape)
            data = prReviewData
        elif label == StringKeyUtils.STR_LABEL_ALL_COMMENT:


            data_issue = pandas.merge(pullRequestData, issueCommentData, left_on='number', right_on='pull_number')


            data_issue = data_issue.loc[data_issue['closed_at'] >= data_issue['created_at_y']].copy(deep=True)
            data_issue = data_issue.loc[data_issue['user_login_x'] != data_issue['user_login_y']].copy(deep=True)

            data_issue.dropna(subset=['user_login_y'], inplace=True)


            data_issue['isBot'] = data_issue['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
            data_issue = data_issue.loc[data_issue['isBot'] == False].copy(deep=True)
            "IR�����У� pr_number, review_user_login, pr_title, pr_body, pr_created_at"
            data_issue = data_issue[['number', 'title', 'body_x', 'created_at_x', 'user_login_y', 'node_id_x']].copy(
                deep=True)
            data_issue.columns = ['pr_number', 'pr_title', 'pr_body', 'pr_created_at', 'review_user_login',
                                  'pullrequest_node']
            data_issue.drop_duplicates(inplace=True)

            data_review = pandas.merge(pullRequestData, reviewData, left_on='number', right_on='pull_number')
            data_review = data_review.loc[data_review['user_login_x'] == data_review['user_login_y']].copy(deep=True)


            data_review = data_review.loc[data_review['closed_at'] >= data_review['submitted_at']].copy(deep=True)


            data_review.dropna(subset=['user_login_y'], inplace=True)


            data_review['isBot'] = data_review['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
            data_review = data_review.loc[data_review['isBot'] == False].copy(deep=True)
            data_review = data_review[['number', 'title', 'body_x', 'created_at', 'user_login_y', 'node_id_x']].copy(
                deep=True)
            data_review.columns = ['pr_number', 'pr_title', 'pr_body', 'pr_created_at', 'review_user_login',
                                   'pullrequest_node']
            data_review.drop_duplicates(inplace=True)

            # data = pandas.concat([data_issue, data_review], axis=0)  # 0 ��ϲ�
            data = data_review

            data.drop_duplicates(inplace=True)
            data.reset_index(drop=True, inplace=True)
            print(data.shape)



            data = data[['pr_number', 'review_user_login', 'pr_title', 'pr_body', 'pr_created_at']].copy(deep=True)
            data.sort_values(by='pr_number', ascending=False, inplace=True)
            data.reset_index(drop=True)



        DataProcessUtils.splitDataByMonth(filename=None,
                                          targetPath=os.path.join(projectConfig.getIRDataPath(), projectName + '_self'),
                                          targetFileName=targetFileName, dateCol='pr_created_at',
                                          dataFrame=data)

    @staticmethod
    def contactSVM_CData(projectName, filter_change_trigger=False):

        if filter_change_trigger:
            targetFileName = f'SVM_C_ALL_{projectName}_data_change_trigger'
        else:
            targetFileName = f'SVM_C_ALL_{projectName}_data'

        """��ȡ��Ϣ  SVM_C ֻ��Ҫpr ��title��body����Ϣ"""
        data_train_path = projectConfig.getDataTrainPath()
        issue_comment_path = projectConfig.getIssueCommentPath()
        pull_request_path = projectConfig.getPullRequestPath()
        review_path = projectConfig.getReviewDataPath()



        issueCommentData = pandasHelper.readTSVFile(
            os.path.join(issue_comment_path, f'ALL_{projectName}_data_issuecomment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pull_request_path, f'ALL_{projectName}_data_pullrequest.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_path, f'ALL_{projectName}_data_review.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        if filter_change_trigger:
            change_trigger_path = projectConfig.getPRTimeLineDataPath()

            changeTriggerData = pandasHelper.readTSVFile(
                os.path.join(change_trigger_path, f'ALL_{projectName}_data_pr_change_trigger.tsv'),
                pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
            )



        data_issue = pandas.merge(pullRequestData, issueCommentData, left_on='number', right_on='pull_number')


        data_issue = data_issue.loc[data_issue['closed_at'] >= data_issue['created_at_y']].copy(deep=True)
        data_issue = data_issue.loc[data_issue['user_login_x'] != data_issue['user_login_y']].copy(deep=True)

        data_issue.dropna(subset=['user_login_y'], inplace=True)


        data_issue['isBot'] = data_issue['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_issue = data_issue.loc[data_issue['isBot'] == False].copy(deep=True)
        "IR�����У� pr_number, review_user_login, pr_title, pr_body, pr_created_at"
        data_issue = data_issue[['number', 'title', 'body_x', 'created_at_x', 'user_login_y', 'node_id_x']].copy(
            deep=True)
        data_issue.columns = ['pr_number', 'pr_title', 'pr_body', 'pr_created_at', 'review_user_login',
                              'pullrequest_node']
        data_issue.drop_duplicates(inplace=True)

        data_review = pandas.merge(pullRequestData, reviewData, left_on='number', right_on='pull_number')
        data_review = data_review.loc[data_review['user_login_x'] != data_review['user_login_y']].copy(deep=True)


        data_review = data_review.loc[data_review['closed_at'] >= data_review['submitted_at']].copy(deep=True)


        data_review.dropna(subset=['user_login_y'], inplace=True)


        data_review['isBot'] = data_review['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_review = data_review.loc[data_review['isBot'] == False].copy(deep=True)
        data_review = data_review[['number', 'title', 'body_x', 'created_at', 'user_login_y', 'node_id_x']].copy(
            deep=True)
        data_review.columns = ['pr_number', 'pr_title', 'pr_body', 'pr_created_at', 'review_user_login',
                               'pullrequest_node']
        data_review.drop_duplicates(inplace=True)

        data = pandas.concat([data_issue, data_review], axis=0)  # 0 ��ϲ�
        data.drop_duplicates(inplace=True)
        data.reset_index(drop=True, inplace=True)
        print(data.shape)

        if filter_change_trigger:


            changeTriggerData['label'] = changeTriggerData.apply(
                lambda x: (x['comment_type'] == 'label_issue_comment' and x['change_trigger'] == 1) or (
                        x['comment_type'] == 'label_review_comment' and x['change_trigger'] == 0), axis=1)
            changeTriggerData = changeTriggerData.loc[changeTriggerData['label'] == True].copy(deep=True)
            changeTriggerData = changeTriggerData[['pullrequest_node', 'user_login']].copy(deep=True)
            changeTriggerData.drop_duplicates(inplace=True)
            changeTriggerData.rename(columns={'user_login': 'review_user_login'}, inplace=True)
            changeTriggerData.reset_index(inplace=True, drop=True)
            data = pandas.merge(data, changeTriggerData, how='inner')



        data = data[['pr_number', 'review_user_login', 'pr_title', 'pr_body', 'pr_created_at']].copy(deep=True)
        data.sort_values(by='pr_number', ascending=False, inplace=True)
        data.reset_index(drop=True)



        DataProcessUtils.splitDataByMonth(filename=None,
                                          targetPath=os.path.join(projectConfig.getSVM_CDataPath(), projectName),
                                          targetFileName=targetFileName, dateCol='pr_created_at',
                                          dataFrame=data)

    @staticmethod
    def contactPBData(projectName, label=StringKeyUtils.STR_LABEL_REVIEW_COMMENT):
        """
        ���� label == review comment and issue comment
             ALL_{projectName}_data_pullrequest
             ALL_{projectName}_data_issuecomment
             ALL_{projectName}_data_review
             ALL_{projectName}_data_review_comment
             �����ļ�ƴ��PB�������Ϣ���ļ�
        """

        targetFileName = f'PB_{projectName}_data'
        if label == StringKeyUtils.STR_LABEL_ISSUE_COMMENT:
            targetFileName = f'PB_ISSUE_{projectName}_data'
        elif label == StringKeyUtils.STR_LABEL_ALL_COMMENT:
            targetFileName = f'PB_ALL_{projectName}_data'



        data_train_path = projectConfig.getDataTrainPath()
        issue_comment_path = projectConfig.getIssueCommentPath()
        pull_request_path = projectConfig.getPullRequestPath()
        review_path = projectConfig.getReviewDataPath()
        review_comment_path = projectConfig.getReviewCommentDataPath()



        issueCommentData = pandasHelper.readTSVFile(
            os.path.join(issue_comment_path, f'ALL_{projectName}_data_issuecomment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pull_request_path, f'ALL_{projectName}_data_pullrequest.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_path, f'ALL_{projectName}_data_review.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        reviewCommentData = pandasHelper.readTSVFile(
            os.path.join(review_comment_path, f'ALL_{projectName}_data_review_comment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        if label == StringKeyUtils.STR_LABEL_ALL_COMMENT:


            data_issue = pandas.merge(pullRequestData, issueCommentData, left_on='number', right_on='pull_number')


            data_issue = data_issue.loc[data_issue['closed_at'] >= data_issue['created_at_y']].copy(deep=True)
            data_issue = data_issue.loc[data_issue['user_login_x'] != data_issue['user_login_y']].copy(deep=True)

            data_issue.dropna(subset=['user_login_y'], inplace=True)


            data_issue['isBot'] = data_issue['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
            data_issue = data_issue.loc[data_issue['isBot'] == False].copy(deep=True)
            "PR�����У� repo_full_name, number, review_user_login, pr_title, pr_body, pr_created_at, comment_body"
            data_issue = data_issue[['repo_full_name_x', 'number', 'node_id_x', 'title', 'body_x',
                                     'created_at_x', 'node_id_y', 'user_login_y', 'body_y']].copy(deep=True)
            data_issue.columns = ['repo_full_name', 'pr_number', 'node_id', 'pr_title', 'pr_body',
                                  'pr_created_at', 'comment_node_id', 'review_user_login', 'comment_body']
            data_issue.drop_duplicates(inplace=True)

            data_review = pandas.merge(pullRequestData, reviewData, left_on='number', right_on='pull_number')
            data_review = pandas.merge(data_review, reviewCommentData, left_on='id_y',
                                       right_on='pull_request_review_id')
            data_review = data_review.loc[data_review['user_login_x'] != data_review['user_login_y']].copy(deep=True)


            data_review = data_review.loc[data_review['closed_at'] >= data_review['submitted_at']].copy(deep=True)


            data_review.dropna(subset=['user_login_y'], inplace=True)


            data_review['isBot'] = data_review['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
            data_review = data_review.loc[data_review['isBot'] == False].copy(deep=True)
            "PR�����У� repo_full_name, number, review_user_login, pr_title, pr_body, pr_created_at, comment_body"
            data_review = data_review[['repo_full_name_x', 'number', 'node_id_x', 'title', 'body_x',
                                       'created_at_x', 'node_id', 'user_login_y', 'body']].copy(deep=True)
            data_review.columns = ['repo_full_name', 'pr_number', 'node_id', 'pr_title', 'pr_body',
                                   'pr_created_at', 'comment_node_id', 'review_user_login', 'comment_body']
            data_review.drop_duplicates(inplace=True)

            data = pandas.concat([data_issue, data_review], axis=0)  # 0 ��ϲ�
            data.drop_duplicates(inplace=True)
            data.reset_index(drop=True, inplace=True)
            print(data.shape)

            change_trigger_path = projectConfig.getPRTimeLineDataPath()

            changeTriggerData = pandasHelper.readTSVFile(
                os.path.join(change_trigger_path, f'ALL_{projectName}_data_pr_change_trigger.tsv'),
                pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
            )


            changeTriggerData['label'] = changeTriggerData.apply(
                lambda x: (x['comment_type'] == 'label_issue_comment' and x['change_trigger'] == 1) or (
                        x['comment_type'] == 'label_review_comment' and x['change_trigger'] == 0), axis=1)
            changeTriggerData = changeTriggerData.loc[changeTriggerData['label'] == True].copy(deep=True)
            # changeTriggerData = changeTriggerData[['pullrequest_node', 'user_login']].copy(deep=True)
            changeTriggerData = changeTriggerData[['comment_node']].copy(deep=True)
            changeTriggerData.drop_duplicates(inplace=True)
            # changeTriggerData.rename(columns={'pullrequest_node': 'node_id', "user_login": 'review_user_login'}
            #                          , inplace=True)
            changeTriggerData.rename(columns={'comment_node': 'comment_node_id'}
                                     , inplace=True)
            data = pandas.merge(data, changeTriggerData, how='inner')
            # data = data.drop(labels='node_id', axis=1)
            data = data.drop(labels='comment_node_id', axis=1)



            data = data[['repo_full_name', 'pr_number', 'review_user_login', 'pr_title',
                         'pr_body', 'pr_created_at', 'comment_body']].copy(deep=True)
            data.drop_duplicates(inplace=True)
            data.sort_values(by='pr_number', ascending=False, inplace=True)
            data.reset_index(drop=True)



        DataProcessUtils.splitDataByMonth(filename=None,
                                          targetPath=os.path.join(projectConfig.getPBDataPath(), projectName),
                                          targetFileName=targetFileName, dateCol='pr_created_at',
                                          dataFrame=data)

    @staticmethod
    def contactTCData(projectName, label=StringKeyUtils.STR_LABEL_REVIEW_COMMENT):


        targetFileName = f'TC_{projectName}_data'
        if label == StringKeyUtils.STR_LABEL_ISSUE_COMMENT:
            targetFileName = f'TC_ISSUE_{projectName}_data'
        elif label == StringKeyUtils.STR_LABEL_ALL_COMMENT:
            targetFileName = f'TC_ALL_{projectName}_data'



        data_train_path = projectConfig.getDataTrainPath()
        issue_comment_path = projectConfig.getIssueCommentPath()
        pull_request_path = projectConfig.getPullRequestPath()
        review_path = projectConfig.getReviewDataPath()
        review_comment_path = projectConfig.getReviewCommentDataPath()



        issueCommentData = pandasHelper.readTSVFile(
            os.path.join(issue_comment_path, f'ALL_{projectName}_data_issuecomment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pull_request_path, f'ALL_{projectName}_data_pullrequest.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_path, f'ALL_{projectName}_data_review.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        reviewCommentData = pandasHelper.readTSVFile(
            os.path.join(review_comment_path, f'ALL_{projectName}_data_review_comment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        if label == StringKeyUtils.STR_LABEL_ALL_COMMENT:


            data_issue = pandas.merge(pullRequestData, issueCommentData, left_on='number', right_on='pull_number')


            data_issue = data_issue.loc[data_issue['closed_at'] >= data_issue['created_at_y']].copy(deep=True)
            data_issue = data_issue.loc[data_issue['user_login_x'] != data_issue['user_login_y']].copy(deep=True)

            data_issue.dropna(subset=['user_login_y'], inplace=True)


            data_issue['isBot'] = data_issue['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
            data_issue = data_issue.loc[data_issue['isBot'] == False].copy(deep=True)
            "PR�����У� repo_full_name, number, review_user_login, pr_title, pr_body, pr_created_at, comment_body"
            data_issue = data_issue[['repo_full_name_x', 'number', 'node_id_x', 'title', 'body_x',
                                     'created_at_x', 'node_id_y', 'user_login_y', 'body_y']].copy(deep=True)
            data_issue.columns = ['repo_full_name', 'pr_number', 'node_id', 'pr_title', 'pr_body',
                                  'pr_created_at', 'comment_node_id', 'review_user_login', 'comment_body']
            data_issue.drop_duplicates(inplace=True)

            data_review = pandas.merge(pullRequestData, reviewData, left_on='number', right_on='pull_number')
            data_review = pandas.merge(data_review, reviewCommentData, left_on='id_y',
                                       right_on='pull_request_review_id')
            data_review = data_review.loc[data_review['user_login_x'] != data_review['user_login_y']].copy(deep=True)


            data_review = data_review.loc[data_review['closed_at'] >= data_review['submitted_at']].copy(deep=True)


            data_review.dropna(subset=['user_login_y'], inplace=True)


            data_review['isBot'] = data_review['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
            data_review = data_review.loc[data_review['isBot'] == False].copy(deep=True)
            "TC�����У� repo_full_name, number, review_user_login, pr_title, pr_body, pr_created_at, comment_body"
            data_review = data_review[['repo_full_name_x', 'number', 'node_id_x', 'title', 'body_x',
                                       'created_at_x', 'node_id', 'user_login_y', 'body']].copy(deep=True)
            data_review.columns = ['repo_full_name', 'pr_number', 'node_id', 'pr_title', 'pr_body',
                                   'pr_created_at', 'comment_node_id', 'review_user_login', 'comment_body']
            data_review.drop_duplicates(inplace=True)

            data = pandas.concat([data_issue, data_review], axis=0)  # 0 ��ϲ�
            data.drop_duplicates(inplace=True)
            data.reset_index(drop=True, inplace=True)
            print(data.shape)

            change_trigger_path = projectConfig.getPRTimeLineDataPath()

            changeTriggerData = pandasHelper.readTSVFile(
                os.path.join(change_trigger_path, f'ALL_{projectName}_data_pr_change_trigger.tsv'),
                pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
            )


            changeTriggerData['label'] = changeTriggerData.apply(
                lambda x: (x['comment_type'] == 'label_issue_comment' and x['change_trigger'] == 1) or (
                        x['comment_type'] == 'label_review_comment' and x['change_trigger'] == 0), axis=1)
            changeTriggerData = changeTriggerData.loc[changeTriggerData['label'] == True].copy(deep=True)
            # changeTriggerData = changeTriggerData.loc[changeTriggerData['change_trigger'] >= 0].copy(deep=True)
            # changeTriggerData = changeTriggerData[['pullrequest_node', 'user_login']].copy(deep=True)
            changeTriggerData = changeTriggerData[['comment_node']].copy(deep=True)
            changeTriggerData.drop_duplicates(inplace=True)
            # changeTriggerData.rename(columns={'pullrequest_node': 'node_id', "user_login": 'review_user_login'}
            #                          , inplace=True)
            changeTriggerData.rename(columns={'comment_node': 'comment_node_id'}
                                     , inplace=True)
            data = pandas.merge(data, changeTriggerData, how='inner')
            # data = data.drop(labels='node_id', axis=1)
            data = data.drop(labels='comment_node_id', axis=1)



            data = data[['repo_full_name', 'pr_number', 'review_user_login', 'pr_title',
                         'pr_body', 'pr_created_at', 'comment_body']].copy(deep=True)
            data.drop_duplicates(inplace=True)
            data.sort_values(by='pr_number', ascending=False, inplace=True)
            data.reset_index(drop=True)



        DataProcessUtils.splitDataByMonth(filename=None,
                                          targetPath=os.path.join(projectConfig.getTCDataPath(), projectName),
                                          targetFileName=targetFileName, dateCol='pr_created_at',
                                          dataFrame=data)

    @staticmethod
    def contactGAData(projectName, label=StringKeyUtils.STR_LABEL_REVIEW_COMMENT, filter_change_trigger=False):

        targetFileName = f'GA_{projectName}_data'
        if label == StringKeyUtils.STR_LABEL_ISSUE_COMMENT:
            targetFileName = f'GA_ISSUE_{projectName}_data'
        elif label == StringKeyUtils.STR_LABEL_ALL_COMMENT:
            targetFileName = f'GA_ALL_{projectName}_data'



        data_train_path = projectConfig.getDataTrainPath()
        issue_comment_path = projectConfig.getIssueCommentPath()
        pull_request_path = projectConfig.getPullRequestPath()
        review_path = projectConfig.getReviewDataPath()
        review_comment_path = projectConfig.getReviewCommentDataPath()

        pr_change_file_path = projectConfig.getPRChangeFilePath()



        issueCommentData = pandasHelper.readTSVFile(
            os.path.join(issue_comment_path, f'ALL_{projectName}_data_issuecomment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pull_request_path, f'ALL_{projectName}_data_pullrequest.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_path, f'ALL_{projectName}_data_review.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        reviewCommentData = pandasHelper.readTSVFile(
            os.path.join(review_comment_path, f'ALL_{projectName}_data_review_comment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        prChangeFileData = pandasHelper.readTSVFile(
            os.path.join(pr_change_file_path, f'ALL_{projectName}_data_pr_change_file.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        if label == StringKeyUtils.STR_LABEL_ALL_COMMENT:


            data_issue = pandas.merge(pullRequestData, issueCommentData, left_on='number', right_on='pull_number')


            data_issue = data_issue.loc[data_issue['closed_at'] >= data_issue['created_at_y']].copy(deep=True)
            data_issue = data_issue.loc[data_issue['user_login_x'] != data_issue['user_login_y']].copy(deep=True)

            data_issue.dropna(subset=['user_login_y'], inplace=True)


            data_issue['isBot'] = data_issue['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
            data_issue = data_issue.loc[data_issue['isBot'] == False].copy(deep=True)

            "GA�����У� repo_full_name, number, review_user_login, pr_created_at, review_created_at, filename"
            data_issue = data_issue[['repo_full_name_x', 'number', 'node_id_x', 'created_at_x',
                                     'node_id_y', 'user_login_y', 'created_at_y']].copy(deep=True)
            data_issue.columns = ['repo_full_name', 'pr_number', 'node_id', 'pr_created_at', 'comment_node_id',
                                  'review_user_login', 'review_created_at']
            data_issue.drop_duplicates(inplace=True)

            data_review = pandas.merge(pullRequestData, reviewData, left_on='number', right_on='pull_number')
            data_review = pandas.merge(data_review, reviewCommentData, left_on='id_y',
                                       right_on='pull_request_review_id', how='left')
            data_review = data_review.loc[data_review['user_login_x'] != data_review['user_login_y']].copy(deep=True)


            data_review = data_review.loc[data_review['closed_at'] >= data_review['submitted_at']].copy(deep=True)


            data_review.dropna(subset=['user_login_y'], inplace=True)


            data_review['isBot'] = data_review['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
            data_review = data_review.loc[data_review['isBot'] == False].copy(deep=True)
            "GA�����У� repo_full_name, number, review_user_login, pr_created_at, review_created_at, filename"



            def fill_created_at(x):
                if not isinstance(x['node_id'], str):
                    return x['submitted_at']
                else:
                    return x['created_at_y']

            data_review['created_at_y'] = data_review.apply(lambda x: fill_created_at(x), axis=1)
            data_review = data_review[['repo_full_name_x', 'number', 'node_id_x', 'created_at_x',
                                       'node_id', 'user_login_y', 'created_at_y']].copy(deep=True)
            data_review.columns = ['repo_full_name', 'pr_number', 'node_id', 'pr_created_at', 'comment_node_id',
                                   'review_user_login', 'review_created_at']
            data_review.drop_duplicates(inplace=True)

            data = pandas.concat([data_issue, data_review], axis=0)  # 0 ��ϲ�
            data.drop_duplicates(inplace=True)
            data.reset_index(drop=True, inplace=True)
            print(data.shape)

            if filter_change_trigger:
                change_trigger_path = projectConfig.getPRTimeLineDataPath()

                changeTriggerData = pandasHelper.readTSVFile(
                    os.path.join(change_trigger_path, f'ALL_{projectName}_data_pr_change_trigger.tsv'),
                    pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
                )


                changeTriggerData['label'] = changeTriggerData.apply(
                    lambda x: (x['comment_type'] == 'label_issue_comment' and x['change_trigger'] == 1) or (
                            x['comment_type'] == 'label_review_comment' and x['change_trigger'] == 0), axis=1)
                changeTriggerData = changeTriggerData.loc[changeTriggerData['label'] == True].copy(deep=True)
                # changeTriggerData = changeTriggerData[['pullrequest_node', 'user_login']].copy(deep=True)
                changeTriggerData = changeTriggerData[['comment_node']].copy(deep=True)
                changeTriggerData.drop_duplicates(inplace=True)
                # changeTriggerData.rename(columns={'pullrequest_node': 'node_id', "user_login": 'review_user_login'}
                #                          , inplace=True)
                changeTriggerData.rename(columns={'comment_node': 'comment_node_id'}
                                         , inplace=True)
                data = pandas.merge(data, changeTriggerData, how='inner')
                # data = data.drop(labels='node_id', axis=1)
            data = data.drop(labels='comment_node_id', axis=1)


            data.sort_values(by=['pr_number', 'review_user_login', 'review_created_at'],
                             ascending=[True, True, True], inplace=True)
            data.drop_duplicates(subset=['pr_number', 'review_user_login'], keep='first', inplace=True)



            data = pandas.merge(data, prChangeFileData, left_on='pr_number', right_on='pull_number')
            data.rename(columns={'repo_full_name_x': 'repo_full_name'}, inplace=True)



            data = data[['repo_full_name', 'pr_number', 'review_user_login', 'pr_created_at',
                         'review_created_at', 'filename']].copy(deep=True)
            data.drop_duplicates(inplace=True)
            data.sort_values(by='pr_number', ascending=False, inplace=True)
            data.reset_index(drop=True)



        DataProcessUtils.splitDataByMonth(filename=None,
                                          targetPath=os.path.join(projectConfig.getGADataPath(), projectName),
                                          targetFileName=targetFileName, dateCol='pr_created_at',
                                          dataFrame=data)

    @staticmethod
    def contactCHREVData(projectName, filter_change_trigger=False):


        targetFileName = None
        if filter_change_trigger:
            targetFileName = f'CHREV_ALL_{projectName}_data_change_trigger'
        else:
            targetFileName = f'CHREV_ALL_{projectName}_data'



        data_train_path = projectConfig.getDataTrainPath()
        issue_comment_path = projectConfig.getIssueCommentPath()
        pull_request_path = projectConfig.getPullRequestPath()
        review_path = projectConfig.getReviewDataPath()
        review_comment_path = projectConfig.getReviewCommentDataPath()
        pr_change_file_path = projectConfig.getPRChangeFilePath()



        issueCommentData = pandasHelper.readTSVFile(
            os.path.join(issue_comment_path, f'ALL_{projectName}_data_issuecomment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pull_request_path, f'ALL_{projectName}_data_pullrequest.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_path, f'ALL_{projectName}_data_review.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        reviewCommentData = pandasHelper.readTSVFile(
            os.path.join(review_comment_path, f'ALL_{projectName}_data_review_comment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        #
        # change_trigger_path = projectConfig.getPRTimeLineDataPath()
        # changeTriggerData = pandasHelper.readTSVFile(
        #     os.path.join(change_trigger_path, f'ALL_{projectName}_data_pr_change_trigger.tsv'),
        #     pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        # )



        prChangeFileData = pandasHelper.readTSVFile(
            os.path.join(pr_change_file_path, f'ALL_{projectName}_data_pr_change_file.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        data_issue = pandas.merge(pullRequestData, issueCommentData, left_on='number', right_on='pull_number')


        data_issue = data_issue.loc[data_issue['closed_at'] >= data_issue['created_at_y']].copy(deep=True)
        data_issue = data_issue.loc[data_issue['user_login_x'] != data_issue['user_login_y']].copy(deep=True)

        data_issue.dropna(subset=['user_login_y'], inplace=True)


        data_issue['isBot'] = data_issue['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_issue = data_issue.loc[data_issue['isBot'] == False].copy(deep=True)
        "PR�����У� number, review_user_login, pr_created_at, comment_at, filename"
        data_issue = data_issue[['number', 'created_at_x', 'node_id_y', 'user_login_y', 'created_at_y']].copy(deep=True)
        data_issue.columns = ['pr_number', 'pr_created_at', 'comment_node_id', 'review_user_login', 'comment_at']
        data_issue.drop_duplicates(inplace=True)

        data_review = pandas.merge(pullRequestData, reviewData, left_on='number', right_on='pull_number')
        # data_review = pandas.merge(data_review, reviewCommentData, left_on='id_y', right_on='pull_request_review_id',
        #                            how='left')

        # data_review['created_at_y'] = data_review.apply(
        #     lambda row: row['submitted_at'] if pandas.isna(row['created_at_y']) else row['created_at_y'], axis=1)

        data_review['created_at_y'] = data_review.apply(
            lambda row: row['submitted_at'], axis=1)

        data_review = data_review.loc[data_review['user_login_x'] != data_review['user_login_y']].copy(deep=True)


        data_review = data_review.loc[data_review['closed_at'] >= data_review['created_at_y']].copy(deep=True)


        data_review.dropna(subset=['user_login_y'], inplace=True)


        data_review['isBot'] = data_review['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_review = data_review.loc[data_review['isBot'] == False].copy(deep=True)
        "HG�����У� number, review_user_login, pr_created_at, comment_at, filename"
        data_review = data_review[['number', 'created_at', 'node_id_y', 'user_login_y', 'created_at_y']].copy(deep=True)
        data_review.columns = ['pr_number', 'pr_created_at', 'comment_node_id', 'review_user_login', 'comment_at']
        data_review.drop_duplicates(inplace=True)

        # data = pandas.concat([data_issue, data_review], axis=0)  # 0 ��ϲ�
        data = data_review

        data.drop_duplicates(inplace=True)
        data.reset_index(drop=True, inplace=True)
        print(data.shape)

        if filter_change_trigger:

            change_trigger_path = projectConfig.getPRTimeLineDataPath()
            changeTriggerData = pandasHelper.readTSVFile(
                os.path.join(change_trigger_path, f'ALL_{projectName}_data_pr_change_trigger.tsv'),
                pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
            )


            changeTriggerData['label'] = changeTriggerData.apply(
                lambda x: (x['comment_type'] == 'label_issue_comment' and x['change_trigger'] == 1) or (
                        x['comment_type'] == 'label_review_comment' and x['change_trigger'] == 0), axis=1)
            changeTriggerData = changeTriggerData.loc[changeTriggerData['label'] == True].copy(deep=True)
            changeTriggerData = changeTriggerData[['comment_node']].copy(deep=True)
            changeTriggerData.drop_duplicates(inplace=True)
            changeTriggerData.rename(columns={'comment_node': 'comment_node_id'}
                                     , inplace=True)
            data = pandas.merge(data, changeTriggerData, how='inner')
        # data = data.drop(labels='comment_node_id', axis=1)




        data = pandas.merge(data, prChangeFileData, left_on='pr_number', right_on='pull_number')



        data = data[
            ['pr_number', 'pr_created_at', 'review_user_login', 'comment_node_id', 'comment_at', 'filename']].copy(
            deep=True)
        data.drop_duplicates(inplace=True)
        data.sort_values(by='pr_number', ascending=False, inplace=True)
        data.reset_index(drop=True)



        DataProcessUtils.splitDataByMonth(filename=None,
                                          targetPath=os.path.join(projectConfig.getCHREVDataPath(), projectName),
                                          targetFileName=targetFileName, dateCol='pr_created_at',
                                          dataFrame=data)

    @staticmethod
    def contactCHREVSelfData(projectName, filter_change_trigger=False):
        targetFileName = f'CHREV_ALL_{projectName}_data'



        issue_comment_path = projectConfig.getIssueCommentPath()
        pull_request_path = projectConfig.getPullRequestPath()
        review_path = projectConfig.getReviewDataPath()
        pr_change_file_path = projectConfig.getPRChangeFilePath()



        issueCommentData = pandasHelper.readTSVFile(
            os.path.join(issue_comment_path, f'ALL_{projectName}_data_issuecomment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pull_request_path, f'ALL_{projectName}_data_pullrequest.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_path, f'ALL_{projectName}_data_review.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        prChangeFileData = pandasHelper.readTSVFile(
            os.path.join(pr_change_file_path, f'ALL_{projectName}_data_pr_change_file.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        data_issue = pandas.merge(pullRequestData, issueCommentData, left_on='number', right_on='pull_number')


        data_issue = data_issue.loc[data_issue['closed_at'] >= data_issue['created_at_y']].copy(deep=True)
        data_issue = data_issue.loc[data_issue['user_login_x'] != data_issue['user_login_y']].copy(deep=True)

        data_issue.dropna(subset=['user_login_y'], inplace=True)


        data_issue['isBot'] = data_issue['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_issue = data_issue.loc[data_issue['isBot'] == False].copy(deep=True)
        "PR�����У� number, review_user_login, pr_created_at, comment_at, filename"
        data_issue = data_issue[['number', 'created_at_x', 'node_id_y', 'user_login_y', 'created_at_y']].copy(deep=True)
        data_issue.columns = ['pr_number', 'pr_created_at', 'comment_node_id', 'review_user_login', 'comment_at']
        data_issue.drop_duplicates(inplace=True)

        data_review = pandas.merge(pullRequestData, reviewData, left_on='number', right_on='pull_number')

        data_review['created_at_y'] = data_review.apply(
            lambda row: row['submitted_at'], axis=1)


        data_review = data_review.loc[data_review['user_login_x'] == data_review['user_login_y']].copy(deep=True)



        data_review = data_review.loc[data_review['closed_at'] >= data_review['created_at_y']].copy(deep=True)


        data_review.dropna(subset=['user_login_y'], inplace=True)


        data_review['isBot'] = data_review['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_review = data_review.loc[data_review['isBot'] == False].copy(deep=True)
        "HG�����У� number, review_user_login, pr_created_at, comment_at, filename"
        data_review = data_review[['number', 'created_at', 'node_id_y', 'user_login_y', 'created_at_y']].copy(deep=True)
        data_review.columns = ['pr_number', 'pr_created_at', 'comment_node_id', 'review_user_login', 'comment_at']
        data_review.drop_duplicates(inplace=True)
        data = data_review

        data.drop_duplicates(inplace=True)
        data.reset_index(drop=True, inplace=True)
        print(data.shape)



        data = pandas.merge(data, prChangeFileData, left_on='pr_number', right_on='pull_number')



        data = data[
            ['pr_number', 'pr_created_at', 'review_user_login', 'comment_node_id', 'comment_at', 'filename']].copy(
            deep=True)
        data.drop_duplicates(inplace=True)
        data.sort_values(by='pr_number', ascending=False, inplace=True)
        data.reset_index(drop=True)



        DataProcessUtils.splitDataByMonth(filename=None,
                                          targetPath=os.path.join(projectConfig.getCHREVDataPath(),
                                                                  projectName + '_self'),
                                          targetFileName=targetFileName, dateCol='pr_created_at',
                                          dataFrame=data)

    @staticmethod
    def contactXFData(projectName, filter_change_trigger=False):


        targetFileName = None
        if filter_change_trigger:
            targetFileName = f'XF_ALL_{projectName}_data_change_trigger'
        else:
            targetFileName = f'XF_ALL_{projectName}_data'



        data_train_path = projectConfig.getDataTrainPath()
        issue_comment_path = projectConfig.getIssueCommentPath()
        pull_request_path = projectConfig.getPullRequestPath()
        review_path = projectConfig.getReviewDataPath()
        review_comment_path = projectConfig.getReviewCommentDataPath()

        pr_change_file_path = projectConfig.getPRChangeFilePath()



        issueCommentData = pandasHelper.readTSVFile(
            os.path.join(issue_comment_path, f'ALL_{projectName}_data_issuecomment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pull_request_path, f'ALL_{projectName}_data_pullrequest.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_path, f'ALL_{projectName}_data_review.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        reviewCommentData = pandasHelper.readTSVFile(
            os.path.join(review_comment_path, f'ALL_{projectName}_data_review_comment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        prChangeFileData = pandasHelper.readTSVFile(
            os.path.join(pr_change_file_path, f'ALL_{projectName}_data_pr_change_file.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        data_issue = pandas.merge(pullRequestData, issueCommentData, left_on='number', right_on='pull_number')


        data_issue = data_issue.loc[data_issue['closed_at'] >= data_issue['created_at_y']].copy(deep=True)
        data_issue = data_issue.loc[data_issue['user_login_x'] != data_issue['user_login_y']].copy(deep=True)

        data_issue.dropna(subset=['user_login_y'], inplace=True)


        data_issue['isBot'] = data_issue['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_issue = data_issue.loc[data_issue['isBot'] == False].copy(deep=True)
        "PR�����У� number, review_user_login, pr_created_at, comment_at, filename"
        data_issue = data_issue[
            ['number', 'user_login_x', 'created_at_x', 'node_id_y', 'user_login_y', 'created_at_y']].copy(deep=True)
        data_issue.columns = ['pr_number', 'author_user_login', 'pr_created_at', 'comment_node_id', 'review_user_login',
                              'comment_at']
        data_issue.drop_duplicates(inplace=True)

        data_review = pandas.merge(pullRequestData, reviewData, left_on='number', right_on='pull_number')
        data_review = pandas.merge(data_review, reviewCommentData, left_on='id_y', right_on='pull_request_review_id',
                                   how='left')

        data_review['created_at_y'] = data_review.apply(
            lambda row: row['submitted_at'] if pandas.isna(row['created_at_y']) else row['created_at_y'], axis=1)

        data_review = data_review.loc[data_review['user_login_x'] != data_review['user_login_y']].copy(deep=True)


        data_review = data_review.loc[data_review['closed_at'] >= data_review['created_at_y']].copy(deep=True)


        data_review.dropna(subset=['user_login_y'], inplace=True)


        data_review['isBot'] = data_review['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_review = data_review.loc[data_review['isBot'] == False].copy(deep=True)
        "HG�����У� number, review_user_login, pr_created_at, comment_at, filename"
        data_review = data_review[
            ['number', 'user_login_x', 'created_at_x', 'node_id', 'user_login_y', 'created_at_y']].copy(deep=True)
        data_review.columns = ['pr_number', 'author_user_login', 'pr_created_at', 'comment_node_id',
                               'review_user_login', 'comment_at']
        data_review.drop_duplicates(inplace=True)

        data = pandas.concat([data_issue, data_review], axis=0)  # 0 ��ϲ�
        data.drop_duplicates(inplace=True)
        data.reset_index(drop=True, inplace=True)
        print(data.shape)

        if filter_change_trigger:
            change_trigger_path = projectConfig.getPRTimeLineDataPath()

            changeTriggerData = pandasHelper.readTSVFile(
                os.path.join(change_trigger_path, f'ALL_{projectName}_data_pr_change_trigger.tsv'),
                pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
            )


            changeTriggerData['label'] = changeTriggerData.apply(
                lambda x: (x['comment_type'] == 'label_issue_comment' and x['change_trigger'] == 1) or (
                        x['comment_type'] == 'label_review_comment' and x['change_trigger'] == 0), axis=1)
            changeTriggerData = changeTriggerData.loc[changeTriggerData['label'] == True].copy(deep=True)
            changeTriggerData = changeTriggerData[['comment_node']].copy(deep=True)
            changeTriggerData.drop_duplicates(inplace=True)
            changeTriggerData.rename(columns={'comment_node': 'comment_node_id'}
                                     , inplace=True)
            data = pandas.merge(data, changeTriggerData, how='inner')
        # data = data.drop(labels='comment_node_id', axis=1)




        data = pandas.merge(data, prChangeFileData, left_on='pr_number', right_on='pull_number')



        data = data[
            ['pr_number', 'author_user_login', 'pr_created_at', 'review_user_login', 'comment_node_id',
             'comment_at', 'filename']].copy(deep=True)
        data.drop_duplicates(inplace=True)
        data.sort_values(by='pr_number', ascending=False, inplace=True)
        data.reset_index(drop=True)



        DataProcessUtils.splitDataByMonth(filename=None,
                                          targetPath=os.path.join(projectConfig.getXFDataPath(), projectName),
                                          targetFileName=targetFileName, dateCol='pr_created_at',
                                          dataFrame=data)

    @staticmethod
    def contactHGData(projectName, filter_change_trigger=False):


        targetFileName = f'HG_ALL_{projectName}_data'



        data_train_path = projectConfig.getDataTrainPath()
        issue_comment_path = projectConfig.getIssueCommentPath()
        pull_request_path = projectConfig.getPullRequestPath()
        review_path = projectConfig.getReviewDataPath()
        review_comment_path = projectConfig.getReviewCommentDataPath()
        pr_change_file_path = projectConfig.getPRChangeFilePath()



        issueCommentData = pandasHelper.readTSVFile(
            os.path.join(issue_comment_path, f'ALL_{projectName}_data_issuecomment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pull_request_path, f'ALL_{projectName}_data_pullrequest.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_path, f'ALL_{projectName}_data_review.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        reviewCommentData = pandasHelper.readTSVFile(
            os.path.join(review_comment_path, f'ALL_{projectName}_data_review_comment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        prChangeFileData = pandasHelper.readTSVFile(
            os.path.join(pr_change_file_path, f'ALL_{projectName}_data_pr_change_file.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        data_issue = pandas.merge(pullRequestData, issueCommentData, left_on='number', right_on='pull_number')


        data_issue = data_issue.loc[data_issue['closed_at'] >= data_issue['created_at_y']].copy(deep=True)
        data_issue = data_issue.loc[data_issue['user_login_x'] != data_issue['user_login_y']].copy(deep=True)

        data_issue.dropna(subset=['user_login_y'], inplace=True)


        data_issue['isBot'] = data_issue['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_issue = data_issue.loc[data_issue['isBot'] == False].copy(deep=True)

        "HG�����У�repo_full_name, number, author_user_login, review_user_login, comment_node_id, pr_created_at, filename"
        data_issue = data_issue[['repo_full_name_x', 'number', 'node_id_x', 'user_login_x', 'created_at_x',
                                 'node_id_y', 'user_login_y', 'created_at_y']].copy(deep=True)
        data_issue.columns = ['repo_full_name', 'pr_number', 'node_id', 'author_user_login', 'pr_created_at',
                              'comment_node_id', 'review_user_login', 'review_created_at']
        data_issue.drop_duplicates(inplace=True)

        data_review = pandas.merge(pullRequestData, reviewData, left_on='number', right_on='pull_number')
        # data_review = pandas.merge(data_review, reviewCommentData, left_on='id_y', right_on='pull_request_review_id',
        #                            how='left')

        data_review['created_at_y'] = data_review.apply(
            lambda row: row['submitted_at'], axis=1)

        data_review = data_review.loc[data_review['user_login_x'] != data_review['user_login_y']].copy(deep=True)


        data_review = data_review.loc[data_review['closed_at'] >= data_review['created_at_y']].copy(deep=True)


        data_review.dropna(subset=['user_login_y'], inplace=True)


        data_review['isBot'] = data_review['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_review = data_review.loc[data_review['isBot'] == False].copy(deep=True)
        "HG�����У� repo_full_name, number, author_user_login, review_user_login, comment_node_id, pr_created_at, filename"
        data_review = data_review[['repo_full_name_x', 'number', 'node_id_x', 'user_login_x', 'created_at',
                                   'node_id_y', 'user_login_y', 'created_at_y']].copy(deep=True)
        data_review.columns = ['repo_full_name', 'pr_number', 'node_id', 'author_user_login', 'pr_created_at',
                               'comment_node_id', 'review_user_login', 'review_created_at']
        data_review.drop_duplicates(inplace=True)

        # data = pandas.concat([data_issue, data_review], axis=0)  # 0 ��ϲ�
        data = data_review


        data.dropna(subset=['author_user_login', 'review_user_login'], inplace=True)
        data.drop_duplicates(inplace=True)
        data.reset_index(drop=True, inplace=True)
        print(data.shape)

        if filter_change_trigger:
            change_trigger_path = projectConfig.getPRTimeLineDataPath()

            changeTriggerData = pandasHelper.readTSVFile(
                os.path.join(change_trigger_path, f'ALL_{projectName}_data_pr_change_trigger.tsv'),
                pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
            )


            changeTriggerData['label'] = changeTriggerData.apply(
                lambda x: (x['comment_type'] == 'label_issue_comment' and x['change_trigger'] == 1) or (
                        x['comment_type'] == 'label_review_comment' and x['change_trigger'] == 0), axis=1)
            changeTriggerData = changeTriggerData.loc[changeTriggerData['label'] == True].copy(deep=True)
            changeTriggerData = changeTriggerData[['comment_node']].copy(deep=True)
            changeTriggerData.drop_duplicates(inplace=True)
            changeTriggerData.rename(columns={'comment_node': 'comment_node_id'}, inplace=True)
            data = pandas.merge(data, changeTriggerData, how='inner')

            data.sort_values(by=['pr_number', 'review_user_login'], ascending=[True, True], inplace=True)
            data.drop_duplicates(subset=['pr_number', 'review_user_login', 'comment_node_id'], keep='first',
                                 inplace=True)



        data = pandas.merge(data, prChangeFileData, left_on='pr_number', right_on='pull_number')
        data.rename(columns={'repo_full_name_x': 'repo_full_name'}, inplace=True)



        data = data[['repo_full_name', 'pr_number', 'author_user_login', 'review_user_login', 'comment_node_id',
                     'pr_created_at', 'review_created_at', 'filename']].copy(deep=True)
        data.drop_duplicates(inplace=True)
        data.sort_values(by='pr_number', ascending=False, inplace=True)
        data.reset_index(drop=True)



        DataProcessUtils.splitDataByMonth(filename=None, targetPath=os.path.join(projectConfig.getHGDataPath(),
                                                                                 projectName),
                                          targetFileName=targetFileName, dateCol='pr_created_at',
                                          dataFrame=data)

    @staticmethod
    def contactHGRecFilterData(projectName):


        targetFileName = f'HGRecFilter_ALL_{projectName}_data'



        data_train_path = projectConfig.getDataTrainPath()
        issue_comment_path = projectConfig.getIssueCommentPath()
        pull_request_path = projectConfig.getPullRequestPath()
        review_path = projectConfig.getReviewDataPath()
        pr_commit_path = projectConfig.getPrCommitEditPath()
        commit_file_path = projectConfig.getCommitFilePath()



        issueCommentData = pandasHelper.readTSVFile(
            os.path.join(issue_comment_path, f'ALL_{projectName}_data_issuecomment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pull_request_path, f'ALL_{projectName}_data_pullrequest.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_path, f'ALL_{projectName}_data_review.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        CommitFileData = pandasHelper.readTSVFile(
            os.path.join(commit_file_path, f'ALL_{projectName}_data_commit_file_relation.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )
        CommitFileData = CommitFileData[['file_commit_sha', 'file_filename']].copy(deep=True)
        CommitFileData.columns = ['sha', 'filename']
        CommitFileData.drop_duplicates(inplace=True)
        counts_series = CommitFileData['sha'].value_counts()
        dict_count = {'sha': counts_series.index, 'counts': counts_series.values}
        df_commit_file_counts = pd.DataFrame(dict_count)
        CommitFileData = pd.merge(left=CommitFileData, right=df_commit_file_counts, left_on='sha', right_on='sha')
        CommitFileData = CommitFileData.drop(
            CommitFileData[CommitFileData['counts'] >= DataProcessUtils.MAX_CHANGE_FILES].index)


        """pr_commit"""
        PrCommitData = pandasHelper.readTSVFile(
            os.path.join(pr_commit_path, f'ALL_{projectName}_pr_commit_data.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )
        commit_sha_arr = list(set(CommitFileData['sha'].values.tolist()))
        PrCommitData['label'] = PrCommitData['sha'].apply(lambda x: x in commit_sha_arr)
        PrCommitData = PrCommitData.loc[PrCommitData['label']].copy(deep=True)
        PrCommitData.drop(columns=['label'], inplace=True)


        data_commit = pandas.merge(pullRequestData, PrCommitData, left_on='number', right_on='pull_number')
        data_commit = data_commit.loc[
            data_commit['closed_at'] >= data_commit['commit_commit_author_date']].copy(deep=True)
        data_commit['create_before_pr'] = data_commit.apply(lambda x: x['created_at'] >= x['commit_commit_author_date'],
                                                            axis=1)


        df_commit_temp = data_commit[['repo_full_name_x', 'number', 'sha', 'create_before_pr']]
        df_commit_temp.columns = ['repo_full_name', 'pull_number', 'sha', 'create_before_pr']
        prChangeFileData = pd.merge(df_commit_temp, CommitFileData, left_on='sha', right_on='sha')
        prChangeFileData = prChangeFileData[['repo_full_name', 'pull_number', 'filename', 'create_before_pr']].copy(
            deep=True)
        prChangeFileData.columns = ['repo_full_name', 'pull_number', 'filename', 'create_before_pr']

        data_review = pandas.merge(pullRequestData, reviewData, left_on='number', right_on='pull_number')


        data_review['created_at_y'] = data_review.apply(
            lambda row: row['submitted_at'], axis=1)

        data_review = data_review.loc[data_review['user_login_x'] != data_review['user_login_y']].copy(deep=True)


        data_review = data_review.loc[data_review['closed_at'] >= data_review['created_at_y']].copy(deep=True)


        data_review.dropna(subset=['user_login_y'], inplace=True)


        data_review['isBot'] = data_review['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_review = data_review.loc[data_review['isBot'] == False].copy(deep=True)
        "HG�����У� repo_full_name, number, author_user_login, review_user_login, comment_node_id, pr_created_at, filename"
        data_review = data_review[['repo_full_name_x', 'number', 'node_id_x', 'user_login_x', 'created_at',
                                   'node_id_y', 'user_login_y', 'created_at_y']].copy(deep=True)
        data_review.columns = ['repo_full_name', 'pr_number', 'node_id', 'author_user_login', 'pr_created_at',
                               'comment_node_id', 'review_user_login', 'review_created_at']
        data_review.drop_duplicates(inplace=True)

        data = data_review


        data.dropna(subset=['author_user_login', 'review_user_login'], inplace=True)
        data.drop_duplicates(inplace=True)
        data.reset_index(drop=True, inplace=True)
        print(data.shape)



        data = pandas.merge(data, prChangeFileData, left_on='pr_number', right_on='pull_number')
        data.rename(columns={'repo_full_name_x': 'repo_full_name'}, inplace=True)



        data = data[['repo_full_name', 'pr_number', 'author_user_login', 'review_user_login', 'comment_node_id',
                     'pr_created_at', 'review_created_at', 'filename', 'create_before_pr']].copy(deep=True)
        data.drop_duplicates(inplace=True)
        data.sort_values(by='pr_number', ascending=False, inplace=True)
        data.reset_index(drop=True)



        DataProcessUtils.splitDataByMonth(filename=None, targetPath=os.path.join(projectConfig.getHGRecFilterDataPath(),
                                                                                 projectName),
                                          targetFileName=targetFileName, dateCol='pr_created_at',
                                          dataFrame=data)

    @staticmethod
    def contactHGRecFilterSelfData(projectName):
        targetFileName = f'HGRecFilter_ALL_{projectName}_data'


        pull_request_path = projectConfig.getPullRequestPath()
        review_path = projectConfig.getReviewDataPath()
        pr_commit_path = projectConfig.getPrCommitEditPath()
        commit_file_path = projectConfig.getCommitFilePath()

        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pull_request_path, f'ALL_{projectName}_data_pullrequest.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_path, f'ALL_{projectName}_data_review.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )
        CommitFileData = pandasHelper.readTSVFile(
            os.path.join(commit_file_path, f'ALL_{projectName}_data_commit_file_relation.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )
        CommitFileData = CommitFileData[['file_commit_sha', 'file_filename']].copy(deep=True)
        CommitFileData.columns = ['sha', 'filename']
        CommitFileData.drop_duplicates(inplace=True)
        counts_series = CommitFileData['sha'].value_counts()
        dict_count = {'sha': counts_series.index, 'counts': counts_series.values}
        df_commit_file_counts = pd.DataFrame(dict_count)
        CommitFileData = pd.merge(left=CommitFileData, right=df_commit_file_counts, left_on='sha', right_on='sha')
        CommitFileData = CommitFileData.drop(
            CommitFileData[CommitFileData['counts'] >= DataProcessUtils.MAX_CHANGE_FILES].index)

        """pr_commit"""
        PrCommitData = pandasHelper.readTSVFile(
            os.path.join(pr_commit_path, f'ALL_{projectName}_pr_commit_data.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )
        commit_sha_arr = list(set(CommitFileData['sha'].values.tolist()))
        PrCommitData['label'] = PrCommitData['sha'].apply(lambda x: x in commit_sha_arr)
        PrCommitData = PrCommitData.loc[PrCommitData['label']].copy(deep=True)
        PrCommitData.drop(columns=['label'], inplace=True)

        data_commit = pandas.merge(pullRequestData, PrCommitData, left_on='number', right_on='pull_number')
        data_commit = data_commit.loc[
            data_commit['closed_at'] >= data_commit['commit_commit_author_date']].copy(deep=True)
        data_commit['create_before_pr'] = data_commit.apply(lambda x: x['created_at'] >= x['commit_commit_author_date'],
                                                            axis=1)

        df_commit_temp = data_commit[['repo_full_name_x', 'number', 'sha', 'create_before_pr']]
        df_commit_temp.columns = ['repo_full_name', 'pull_number', 'sha', 'create_before_pr']
        prChangeFileData = pd.merge(df_commit_temp, CommitFileData, left_on='sha', right_on='sha')
        prChangeFileData = prChangeFileData[['repo_full_name', 'pull_number', 'filename', 'create_before_pr']].copy(
            deep=True)
        prChangeFileData.columns = ['repo_full_name', 'pull_number', 'filename', 'create_before_pr']

        data_review = pandas.merge(pullRequestData, reviewData, left_on='number', right_on='pull_number')

        data_review['created_at_y'] = data_review.apply(
            lambda row: row['submitted_at'], axis=1)
        ### �޸�
        data_review = data_review.loc[data_review['user_login_x'] == data_review['user_login_y']].copy(deep=True)


        data_review = data_review.loc[data_review['closed_at'] >= data_review['created_at_y']].copy(deep=True)


        data_review.dropna(subset=['user_login_y'], inplace=True)


        data_review['isBot'] = data_review['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_review = data_review.loc[data_review['isBot'] == False].copy(deep=True)
        "HG�����У� repo_full_name, number, author_user_login, review_user_login, comment_node_id, pr_created_at, filename"
        data_review = data_review[['repo_full_name_x', 'number', 'node_id_x', 'user_login_x', 'created_at',
                                   'node_id_y', 'user_login_y', 'created_at_y']].copy(deep=True)
        data_review.columns = ['repo_full_name', 'pr_number', 'node_id', 'author_user_login', 'pr_created_at',
                               'comment_node_id', 'review_user_login', 'review_created_at']
        data_review.drop_duplicates(inplace=True)
        data = data_review


        data.dropna(subset=['author_user_login', 'review_user_login'], inplace=True)
        data.drop_duplicates(inplace=True)
        data.reset_index(drop=True, inplace=True)
        print(data.shape)


        data = pandas.merge(data, prChangeFileData, left_on='pr_number', right_on='pull_number')
        data.rename(columns={'repo_full_name_x': 'repo_full_name'}, inplace=True)


        data = data[['repo_full_name', 'pr_number', 'author_user_login', 'review_user_login', 'comment_node_id',
                     'pr_created_at', 'review_created_at', 'filename', 'create_before_pr']].copy(deep=True)
        data.drop_duplicates(inplace=True)
        data.sort_values(by='pr_number', ascending=False, inplace=True)
        data.reset_index(drop=True)



        DataProcessUtils.splitDataByMonth(filename=None, targetPath=os.path.join(projectConfig.getHGRecFilterDataPath(),
                                                                                 projectName + '_self'),
                                          targetFileName=targetFileName, dateCol='pr_created_at',
                                          dataFrame=data)


    @staticmethod
    def contactRF_AData(projectName, filter_change_trigger=False):


        time1 = datetime.now()
        issue_comment_path = projectConfig.getIssueCommentPath()
        pull_request_path = projectConfig.getPullRequestPath()
        review_path = projectConfig.getReviewDataPath()



        issueCommentData = pandasHelper.readTSVFile(
            os.path.join(issue_comment_path, f'ALL_{projectName}_data_issuecomment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pull_request_path, f'ALL_{projectName}_data_pullrequest.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_path, f'ALL_{projectName}_data_review.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        targetFileName = None
        if filter_change_trigger:
            targetFileName = f'RF_A_ALL_{projectName}_data_change_trigger'
        else:
            targetFileName = f'RF_A_ALL_{projectName}_data'

        print("read file cost time:", datetime.now() - time1)


        data_issue = pandas.merge(pullRequestData, issueCommentData, left_on='number', right_on='pull_number')


        data_issue = data_issue.loc[data_issue['closed_at'] >= data_issue['created_at_y']].copy(deep=True)
        data_issue = data_issue.loc[data_issue['user_login_x'] != data_issue['user_login_y']].copy(deep=True)

        data_issue.dropna(subset=['user_login_y'], inplace=True)

        data_issue.dropna(subset=['head_label'], inplace=True)


        data_issue['isBot'] = data_issue['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_issue = data_issue.loc[data_issue['isBot'] == False].copy(deep=True)
        """number,review_user_login, pr_created_at, pr_user_login, author_association, 
        commits, additions, deletions, changed_files, merged"""
        data_issue = data_issue[['number', 'node_id_x', 'user_login_y',
                                 'created_at_x', 'user_login_x', 'author_association_x', 'commits',
                                 'additions', 'deletions', 'changed_files', 'merged']].copy(deep=True)

        data_issue.columns = ['pr_number', 'node_id_x', 'review_user_login', 'pr_created_at', 'author_user_login',
                              'author_association', 'commits', 'additions', 'deletions', 'changed_files', 'merged']
        data_issue.drop_duplicates(inplace=True)

        data_review = pandas.merge(pullRequestData, reviewData, left_on='number', right_on='pull_number')
        data_review = data_review.loc[data_review['user_login_x'] != data_review['user_login_y']].copy(deep=True)


        data_review = data_review.loc[data_review['closed_at'] >= data_review['submitted_at']].copy(deep=True)


        data_review.dropna(subset=['user_login_y'], inplace=True)


        data_review['isBot'] = data_review['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_review = data_review.loc[data_review['isBot'] == False].copy(deep=True)
        """number,review_user_login, pr_created_at, pr_user_login, author_association, 
            commits, additions, deletions, changed_files, merged"""
        data_review = data_review[['number', 'node_id_x', 'user_login_y',
                                   'created_at', 'user_login_x', 'author_association_x', 'commits',
                                   'additions', 'deletions', 'changed_files', 'merged']].copy(deep=True)

        data_review.columns = ['pr_number', 'node_id_x', 'review_user_login', 'pr_created_at', 'author_user_login',
                               'author_association', 'commits', 'additions', 'deletions', 'changed_files', 'merged']
        data_review.drop_duplicates(inplace=True)

        # rawData = pandas.concat([data_issue, data_review], axis=0)  # 0 ��ϲ�

        rawData = data_review

        rawData.drop_duplicates(inplace=True)
        rawData.reset_index(drop=True, inplace=True)
        print(rawData.shape)

        if filter_change_trigger:
            change_trigger_path = projectConfig.getPRTimeLineDataPath()

            changeTriggerData = pandasHelper.readTSVFile(
                os.path.join(change_trigger_path, f'ALL_{projectName}_data_pr_change_trigger.tsv'),
                pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
            )


            changeTriggerData['label'] = changeTriggerData.apply(
                lambda x: (x['comment_type'] == 'label_issue_comment' and x['change_trigger'] == 1) or (
                        x['comment_type'] == 'label_review_comment' and x['change_trigger'] == 0), axis=1)
            changeTriggerData = changeTriggerData.loc[changeTriggerData['label'] == True].copy(deep=True)
            changeTriggerData = changeTriggerData[['pullrequest_node', 'user_login']].copy(deep=True)
            changeTriggerData.drop_duplicates(inplace=True)
            changeTriggerData.rename(columns={'pullrequest_node': 'node_id_x',
                                              'user_login': "review_user_login"}, inplace=True)
            rawData = pandas.merge(rawData, changeTriggerData, how='inner')

        rawData = rawData.drop(labels='node_id_x', axis=1)
        rawData.sort_values(['pr_number', 'review_user_login'], ascending=[True, True],
                            inplace=True)
        rawData.drop_duplicates(subset=['pr_number', 'review_user_login'], inplace=True, keep='first')

        data = rawData



        DataProcessUtils.splitDataByMonth(filename=None,
                                          targetPath=os.path.join(projectConfig.getRF_ADataPath(), projectName),
                                          targetFileName=targetFileName, dateCol='pr_created_at',
                                          dataFrame=data)

    @staticmethod
    def contactCDRData(projectName, filter_change_trigger=False):


        time1 = datetime.now()
        issue_comment_path = projectConfig.getIssueCommentPath()
        pull_request_path = projectConfig.getPullRequestPath()
        review_path = projectConfig.getReviewDataPath()



        issueCommentData = pandasHelper.readTSVFile(
            os.path.join(issue_comment_path, f'ALL_{projectName}_data_issuecomment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pull_request_path, f'ALL_{projectName}_data_pullrequest.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_path, f'ALL_{projectName}_data_review.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        targetFileName = None
        if filter_change_trigger:
            targetFileName = f'CDR_ALL_{projectName}_data_change_trigger'
        else:
            targetFileName = f'CDR_ALL_{projectName}_data'

        print("read file cost time:", datetime.now() - time1)


        data_issue = pandas.merge(pullRequestData, issueCommentData, left_on='number', right_on='pull_number')


        data_issue = data_issue.loc[data_issue['closed_at'] >= data_issue['created_at_y']].copy(deep=True)
        data_issue = data_issue.loc[data_issue['user_login_x'] != data_issue['user_login_y']].copy(deep=True)

        data_issue.dropna(subset=['user_login_y'], inplace=True)

        data_issue.dropna(subset=['head_label'], inplace=True)


        data_issue['isBot'] = data_issue['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_issue = data_issue.loc[data_issue['isBot'] == False].copy(deep=True)
        data_issue = data_issue[['number', 'node_id_x', 'user_login_y',
                                 'created_at_x', 'user_login_x', 'created_at_y']].copy(deep=True)

        data_issue.columns = ['pr_number', 'node_id_x', 'review_user_login', 'pr_created_at', 'pr_user_login',
                              'comment_at']
        data_issue.drop_duplicates(inplace=True)

        data_review = pandas.merge(pullRequestData, reviewData, left_on='number', right_on='pull_number')
        data_review = data_review.loc[data_review['user_login_x'] != data_review['user_login_y']].copy(deep=True)


        data_review = data_review.loc[data_review['closed_at'] >= data_review['submitted_at']].copy(deep=True)


        data_review.dropna(subset=['user_login_y'], inplace=True)


        data_review['isBot'] = data_review['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_review = data_review.loc[data_review['isBot'] == False].copy(deep=True)
        data_review = data_review[['number', 'node_id_x', 'user_login_y',
                                   'created_at', 'user_login_x', 'submitted_at']].copy(deep=True)

        data_review.columns = ['pr_number', 'node_id_x', 'review_user_login', 'pr_created_at', 'pr_user_login',
                               'comment_at']
        data_review.drop_duplicates(inplace=True)

        rawData = pandas.concat([data_issue, data_review], axis=0)  # 0 ��ϲ�
        rawData.drop_duplicates(inplace=True)
        rawData.reset_index(drop=True, inplace=True)
        print(rawData.shape)

        if filter_change_trigger:
            change_trigger_path = projectConfig.getPRTimeLineDataPath()

            changeTriggerData = pandasHelper.readTSVFile(
                os.path.join(change_trigger_path, f'ALL_{projectName}_data_pr_change_trigger.tsv'),
                pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
            )


            changeTriggerData['label'] = changeTriggerData.apply(
                lambda x: (x['comment_type'] == 'label_issue_comment' and x['change_trigger'] == 1) or (
                        x['comment_type'] == 'label_review_comment' and x['change_trigger'] == 0), axis=1)
            changeTriggerData = changeTriggerData.loc[changeTriggerData['label'] == True].copy(deep=True)
            changeTriggerData = changeTriggerData[['pullrequest_node', 'user_login']].copy(deep=True)
            changeTriggerData.drop_duplicates(inplace=True)
            changeTriggerData.rename(columns={'pullrequest_node': 'node_id_x',
                                              'user_login': "review_user_login"}, inplace=True)
            rawData = pandas.merge(rawData, changeTriggerData, how='inner')

        rawData = rawData.drop(labels='node_id_x', axis=1)
        rawData.sort_values(['pr_number', 'review_user_login', 'comment_at'], ascending=[True, True, True],
                            inplace=True)
        rawData.drop_duplicates(subset=['pr_number', 'review_user_login'], inplace=True, keep='first')

        data = rawData



        DataProcessUtils.splitDataByMonth(filename=None,
                                          targetPath=os.path.join(projectConfig.getCDRDataPath(), projectName),
                                          targetFileName=targetFileName, dateCol='pr_created_at',
                                          dataFrame=data)

    @staticmethod
    def getReviewerFrequencyDict(projectName, date):

        issue_comment_path = projectConfig.getIssueCommentPath()
        pull_request_path = projectConfig.getPullRequestPath()
        review_path = projectConfig.getReviewDataPath()
        review_comment_path = projectConfig.getReviewCommentDataPath()
        change_trigger_path = projectConfig.getPRTimeLineDataPath()



        issueCommentData = pandasHelper.readTSVFile(
            os.path.join(issue_comment_path, f'ALL_{projectName}_data_issuecomment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pull_request_path, f'ALL_{projectName}_data_pullrequest.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_path, f'ALL_{projectName}_data_review.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        changeTriggerData = pandasHelper.readTSVFile(
            os.path.join(change_trigger_path, f'ALL_{projectName}_data_pr_change_trigger.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        reviewCommentData = pandasHelper.readTSVFile(
            os.path.join(review_comment_path, f'ALL_{projectName}_data_review_comment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        data_issue = pandas.merge(pullRequestData, issueCommentData, left_on='number', right_on='pull_number')


        data_issue = data_issue.loc[data_issue['closed_at'] >= data_issue['created_at_y']].copy(deep=True)
        data_issue = data_issue.loc[data_issue['user_login_x'] != data_issue['user_login_y']].copy(deep=True)

        data_issue.dropna(subset=['user_login_y'], inplace=True)


        data_issue['isBot'] = data_issue['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_issue = data_issue.loc[data_issue['isBot'] == False].copy(deep=True)
        data_issue = data_issue[['number', 'node_id_x', 'created_at_x', 'node_id_y', 'user_login_y']].copy(deep=True)
        data_issue.columns = ['pr_number', 'node_id', 'pr_created_at', 'comment_node_id', 'review_user_login']
        data_issue.drop_duplicates(inplace=True)

        data_review = pandas.merge(pullRequestData, reviewData, left_on='number', right_on='pull_number')
        data_review = pandas.merge(data_review, reviewCommentData, left_on='id_y', right_on='pull_request_review_id')
        data_review = data_review.loc[data_review['user_login_x'] != data_review['user_login_y']].copy(deep=True)


        data_review = data_review.loc[data_review['closed_at'] >= data_review['submitted_at']].copy(deep=True)


        data_review.dropna(subset=['user_login_y'], inplace=True)


        data_review['isBot'] = data_review['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_review = data_review.loc[data_review['isBot'] == False].copy(deep=True)
        "PR�����У� repo_full_name, number, review_user_login, pr_title, pr_body, pr_created_at, comment_body"
        data_review = data_review[['number', 'node_id_x', 'created_at_x', 'node_id', 'user_login_y']].copy(deep=True)
        data_review.columns = ['pr_number', 'node_id', 'pr_created_at', 'comment_node_id', 'review_user_login']
        data_review.drop_duplicates(inplace=True)

        data = pandas.concat([data_issue, data_review], axis=0)  # 0 ��ϲ�
        data.drop_duplicates(inplace=True)
        data.reset_index(drop=True, inplace=True)
        print(data.shape)



        changeTriggerData['label'] = changeTriggerData.apply(
            lambda x: (x['comment_type'] == 'label_issue_comment' and x['change_trigger'] == 1) or (
                    x['comment_type'] == 'label_review_comment' and x['change_trigger'] == 0), axis=1)
        changeTriggerData = changeTriggerData.loc[changeTriggerData['label'] == True].copy(deep=True)
        # changeTriggerData = changeTriggerData[['pullrequest_node', 'user_login']].copy(deep=True)
        changeTriggerData = changeTriggerData[['comment_node']].copy(deep=True)
        changeTriggerData.drop_duplicates(inplace=True)
        # changeTriggerData.rename(columns={'pullrequest_node': 'node_id', "user_login": 'review_user_login'}
        #                          , inplace=True)
        changeTriggerData.rename(columns={'comment_node': 'comment_node_id'}
                                 , inplace=True)
        data = pandas.merge(data, changeTriggerData, how='inner')
        data = data.drop(labels='comment_node_id', axis=1)

        minYear, minMonth, maxYear, maxMonth = date

        start = minYear * 12 + minMonth
        end = maxYear * 12 + maxMonth
        data['label'] = data['pr_created_at'].apply(lambda x: (time.strptime(x, "%Y-%m-%d %H:%M:%S")))
        data['label_y'] = data['label'].apply(lambda x: x.tm_year)
        data['label_m'] = data['label'].apply(lambda x: x.tm_mon)
        tempdData = None
        for i in range(start, end):
            y = int((i - i % 12) / 12)
            m = i % 12
            if m == 0:
                m = 12
                y = y - 1
            print(y, m)
            subDf = data.loc[(data['label_y'] == y) & (data['label_m'] == m)].copy(deep=True)
            if tempdData is None:
                tempdData = subDf
            else:
                tempdData = pandas.concat([tempdData, subDf])

        # print(data)
        reviewers = tempdData['review_user_login'].value_counts()
        return dict(reviewers)

    @staticmethod
    def getStopWordList():
        stopwords = SplitWordHelper().getEnglishStopList()  # ��ȡͨ��Ӣ��ͣ�ô�
        allStr = '['
        len = 0
        for word in stopwords:
            len += 1
            allStr += '\"'
            allStr += word
            allStr += '\"'
            if len != stopwords.__len__():
                allStr += ','
        allStr += ']'
        print(allStr)

    @staticmethod
    def dunn():
        # dunn ����
        filename = os.path.join(projectConfig.getDataPath(), "compare.xlsx")
        data = pandas.read_excel(filename, sheet_name="Top-1_1")
        print(data)
        data.columns = [0, 1, 2, 3, 4]
        data.index = [0, 1, 2, 3, 4, 5, 6, 7]
        print(data)
        x = [[1, 2, 3, 5, 1], [12, 31, 54, 12], [10, 12, 6, 74, 11]]
        print(data.values.T)
        result = scikit_posthocs.posthoc_nemenyi_friedman(data.values)
        print(result)
        print(data.values.T[1])
        print(data.values.T[3])
        data1 = []
        for i in range(0, 5):
            data1.append([])
            for j in range(0, 5):
                if i == j:
                    data1[i].append(numpy.nan)
                    continue
                statistic, pvalue = wilcoxon(data.values.T[i], data.values.T[j])
                print(pvalue)
                data1[i].append(pvalue)
        data1 = pandas.DataFrame(data1)
        print(data1)
        import matplotlib.pyplot as plt
        name = ['FPS', 'IR', 'SVM', 'RF', 'CB']
        # scikit_posthocs.sign_plot(result, g=name)
        # plt.show()
        for i in range(0, 5):
            data1[i][i] = numpy.nan
        ax = seaborn.heatmap(data1, annot=True, vmax=1, square=True, yticklabels=name, xticklabels=name, cmap='GnBu_r')
        ax.set_title("Mann Whitney U test")
        plt.show()

    @staticmethod
    def compareDataFrameByPullNumber():

        file2 = 'FPS_ALL_opencv_data_2017_10_to_2017_10.tsv'
        file1 = 'FPS_SEAA_opencv_data_2017_10_to_2017_10.tsv'
        df1 = pandasHelper.readTSVFile(projectConfig.getFPSDataPath() + os.sep + file1,
                                       header=pandasHelper.INT_READ_FILE_WITH_HEAD)
        df1.drop(columns=['pr_created_at', 'commit_sha'], inplace=True)
        df2 = pandasHelper.readTSVFile(projectConfig.getFPSDataPath() + os.sep + file2,
                                       header=pandasHelper.INT_READ_FILE_WITH_HEAD)
        df2.drop(columns=['pr_created_at', 'commit_sha'], inplace=True)

        df1 = pandas.concat([df1, df2])
        df1 = pandas.concat([df1, df2])
        df1.drop_duplicates(inplace=True, keep=False)
        print(df1)

    @staticmethod
    def changeTriggerAnalyzer(repo):


        change_trigger_filename = projectConfig.getPRTimeLineDataPath() + os.sep + f'ALL_{repo}_data_pr_change_trigger.tsv'
        change_trigger_df = pandasHelper.readTSVFile(fileName=change_trigger_filename, header=0)

        # timeline_filename = projectConfig.getPRTimeLineDataPath() + os.sep + f'ALL_{repo}_data_prtimeline.tsv'
        # timeline_df = pandasHelper.readTSVFile(fileName=timeline_filename, header=0)
        # timeline_df = timeline_df.loc[(timeline_df['typename'] == 'IssueComment')\
        #                               |(timeline_df['typename'] == 'PullRequestReview')].copy(deep=True)
        # timeline_useful_prs = list(set(timeline_df['pullrequest_node']))

        prs = list(set(change_trigger_df['pullrequest_node']))
        print("prs nums:", prs.__len__())


        df_issue = change_trigger_df.loc[change_trigger_df['comment_type'] == 'label_issue_comment']
        print("issue all:", df_issue.shape[0])
        issue_is_change_count = df_issue.loc[df_issue['change_trigger'] == 1].shape[0]
        issue_not_change_count = df_issue.loc[df_issue['change_trigger'] == -1].shape[0]
        print("issue is count:", issue_is_change_count, " not count:", issue_not_change_count)
        plt.subplot(121)
        x = ['useful', 'useless']
        plt.bar(x=x, height=[issue_is_change_count, issue_not_change_count])
        plt.title(f'issue comment({repo})')
        for a, b in zip(x, [issue_is_change_count, issue_not_change_count]):
            plt.text(a, b, '%.0f' % b, ha='center', va='bottom', fontsize=11)
        # plt.show()

        df_review = change_trigger_df.loc[change_trigger_df['comment_type'] == 'label_review_comment']
        print("review all:", df_review.shape[0])
        x = range(-1, 11)
        y = []
        for i in x:
            y.append(df_review.loc[df_review['change_trigger'] == i].shape[0])
        plt.subplot(122)
        plt.bar(x=x, height=y)
        plt.title(f'review comment({repo})')
        for a, b in zip(x, y):
            plt.text(a, b, '%.0f' % b, ha='center', va='bottom', fontsize=11)

        print("review comment useful:", df_review.shape[0] - df_review.loc[df_review['change_trigger'] == -1].shape[0])
        plt.show()

    @staticmethod
    def changeTriggerAnalyzer_v3(repo):


        change_trigger_filename = projectConfig.getPRTimeLineDataPath() + os.sep + f'ALL_{repo}_data_pr_change_trigger.tsv'
        change_trigger_df = pandasHelper.readTSVFile(fileName=change_trigger_filename, header=0)

        # timeline_filename = projectConfig.getPRTimeLineDataPath() + os.sep + f'ALL_{repo}_data_prtimeline.tsv'
        # timeline_df = pandasHelper.readTSVFile(fileName=timeline_filename, header=0)
        # timeline_df = timeline_df.loc[(timeline_df['typename'] == 'IssueComment')\
        #                               |(timeline_df['typename'] == 'PullRequestReview')].copy(deep=True)
        # timeline_useful_prs = list(set(timeline_df['pullrequest_node']))

        prs = list(set(change_trigger_df['pullrequest_node']))
        print("prs nums:", prs.__len__())


        df_review_with_no_comment = change_trigger_df.loc[
            change_trigger_df['comment_type'] == 'label_review_with_on_comment']
        print("review with no comment all:", df_review_with_no_comment.shape[0])
        review_with_no_comment_is_change_count = \
            df_review_with_no_comment.loc[df_review_with_no_comment['change_trigger'] == 1].shape[0]
        review_with_no_comment_not_change_count = \
            df_review_with_no_comment[df_review_with_no_comment['change_trigger'] == -1].shape[0]
        print("review_with_no_comment_is_change_count is count:",
              review_with_no_comment_is_change_count,
              " not count:", review_with_no_comment_not_change_count)
        plt.subplot(121)
        x = ['useful', 'useless']
        plt.bar(x=x, height=[review_with_no_comment_is_change_count, review_with_no_comment_not_change_count])
        plt.title(f'review with no comment({repo})')
        for a, b in zip(x, [review_with_no_comment_is_change_count, review_with_no_comment_not_change_count]):
            plt.text(a, b, '%.0f' % b, ha='center', va='bottom', fontsize=11)
        # plt.show()

        df_review = change_trigger_df.loc[change_trigger_df['comment_type'] == 'label_review_comment']
        print("review_with_no_comment all:", df_review.shape[0])
        x = []
        y = []

        specialEventMap = {StringKeyUtils.STR_CHANGE_TRIGGER_REVIEW_COMMENT_AUTHOR: "author",
                           StringKeyUtils.STR_CHANGE_TRIGGER_REVIEW_COMMENT_OUT_PR: "out_pr",
                           StringKeyUtils.STR_CHANGE_TRIGGER_REVIEW_COMMENT_BETWEEN_REOPEN: "in_reopen",
                           StringKeyUtils.STR_CHANGE_TRIGGER_REVIEW_COMMENT_FILE_MOVE: "file_move",
                           StringKeyUtils.STR_CHANGE_TRIGGER_REVIEW_COMMENT_ERROR: "error"}

        for k, v in specialEventMap.items():
            x.append(v)
            y.append(df_review.loc[df_review['change_trigger'] == k].shape[0])
        x.reverse()
        y.reverse()
        x.extend([str(xx) for xx in range(-1, 11)])

        normal_range = range(-1, 11)
        for i in normal_range:
            y.append(df_review.loc[df_review['change_trigger'] == i].shape[0])
        plt.subplot(122)
        fake_x = range(0, y.__len__())
        plt.bar(x=fake_x, height=y)
        plt.xticks(fake_x, x, rotation=45)
        plt.title(f'review comment({repo})')
        for a, b in zip(fake_x, y):
            plt.text(a, b, '%.0f' % b, ha='center', va='bottom', fontsize=11)

        print("review comment useful:", df_review.loc[df_review['change_trigger'] >= 0].shape[0])
        plt.show()

    @staticmethod
    def changeTriggerResponseTimeAnalyzer(repo):
        """��change trigger pair������ͳ��  �����Ը���pair�Ļ�Ӧʱ����ͳ��"""
        change_trigger_filename = projectConfig.getPRTimeLineDataPath() + os.sep + f'ALL_{repo}_data_pr_change_trigger.tsv'
        change_trigger_df = pandasHelper.readTSVFile(fileName=change_trigger_filename, header=0)

        timeline_filename = projectConfig.getPRTimeLineDataPath() + os.sep + f'ALL_{repo}_data_prtimeline.tsv'
        timeline_df = pandasHelper.readTSVFile(fileName=timeline_filename, header=0)

        # review_change_filename = projectConfig.getReviewChangeDataPath() + os.sep + f'ALL_data_review_change.tsv'
        # review_change_df = pandasHelper.readTSVFile(fileName=review_change_filename, header=0)

        review_comment_filename = projectConfig.getReviewCommentDataPath() + os.sep + f'ALL_{repo}_data_review_comment.tsv'
        review_comment_df = pandasHelper.readTSVFile(fileName=review_comment_filename, header=0)

        review_filename = projectConfig.getReviewDataPath() + os.sep + f'ALL_{repo}_data_review.tsv'
        review_df = pandasHelper.readTSVFile(fileName=review_filename, header=0)

        review_df = pandas.merge(review_df, review_comment_df, left_on='id', right_on='pull_request_review_id')

        pr_filename = projectConfig.getPullRequestPath() + os.sep + f'ALL_{repo}_data_pullrequest.tsv'
        pull_request_df = pandasHelper.readTSVFile(pr_filename, pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False)

        prs = list(set(change_trigger_df['pullrequest_node']))
        print("prs nums:", prs.__len__())

        # """�� review_change_df ��ɸѡ"""
        # temp_pr_node = change_trigger_df['pullrequest_node'].copy(deep=True)
        # temp_pr_node.drop_duplicates(inplace=True)
        # review_change_df = pandas.merge(review_change_df, temp_pr_node, left_on='pull_request_node_id',
        #                                 right_on='pullrequest_node')

        change_trigger_df['label'] = change_trigger_df.apply(
            lambda x: (x['comment_type'] == 'label_issue_comment' and x['change_trigger'] == 1) or (
                    x['comment_type'] == 'label_review_comment' and x['change_trigger'] == 0), axis=1)
        change_trigger_df = change_trigger_df.loc[change_trigger_df['label'] == True].copy(deep=True)

        # [(reviewer, node_id, gap), (), ()]
        comment_response_list = []
        errorCount = 0
        mergeCommentCase = 0
        tooLongCase = 0
        replyCommentCase = 0
        issueCommentCase = 0


        filter_change_trigger_df = change_trigger_df.copy(deep=True)
        filter_change_trigger_df['response_time'] = -1

        for index, row in change_trigger_df.iterrows():
            # print(row)

            pr_node = getattr(row, 'pullrequest_node')
            comment_type = getattr(row, 'comment_type')
            pr_df = pull_request_df.loc[pull_request_df['node_id'] == pr_node].copy(deep=True)
            pr_df.reset_index(drop=True, inplace=True)
            author = pr_df.at[0, 'user_login']

            comment_node_id = getattr(row, 'comment_node')
            reviewer = getattr(row, 'user_login')

            # """�������ߺ�reviewerͬһ�˵ĳ���"""
            # if author == reviewer:
            #     continue

            comment_index = None

            if comment_type == 'label_issue_comment':
                try:
                    comment_df = timeline_df.loc[timeline_df['timelineitem_node'] == comment_node_id]
                    comment_index = comment_df.index[0]
                except Exception as e:
                    print("issue comment case:", issueCommentCase)
                    issueCommentCase += 1
                    continue
            else:
                try:
                    """��reivew_df ��review��node"""
                    review_comment_temp_df = review_df.loc[review_df['node_id_y'] == comment_node_id]
                    review_comment_temp_df.reset_index(inplace=True, drop=True)
                    replay_to_id = review_comment_temp_df.at[0, 'in_reply_to_id']
                    if not numpy.isnan(replay_to_id):
                        review_comment_temp_df = review_df.loc[review_df['id_y'] == replay_to_id]
                    review_comment_temp_df.reset_index(inplace=True, drop=True)
                    review_node = review_comment_temp_df.at[0, 'node_id_x']
                    comment_df = timeline_df.loc[timeline_df['timelineitem_node'] == review_node]
                    comment_index = comment_df.index[0]
                except Exception as e:
                    print(e)
                    print("reply to id is none", replyCommentCase)
                    replyCommentCase += 1
                    continue

            change_index = comment_index + 1
            comment_time = timeline_df.at[comment_index, 'created_at']
            change_time = timeline_df.at[change_index, 'created_at']
            change_type = timeline_df.at[change_index, 'typename']
            try:
                comment_time = datetime.strptime(comment_time, "%Y-%m-%d %H:%M:%S")

                #     """change time ��commit_dfӳ��"""
                #     typename = timeline_df.at[change_index, 'typename']
                #     if typename == 'PullRequestCommit':
                #         origin = timeline_df.at[change_index, 'origin']
                #         item = json.loads(origin)
                #         commit = item.get(StringKeyUtils.STR_KEY_COMMIT)
                #         if commit is not None and isinstance(commit, dict):
                #             oid = commit.get('oid')
                #             if oid is None:
                #                 raise Exception("oid is None!")
                #             commit_temp_df = commit_df.loc[commit_df['sha'] == oid]
                #             commit_temp_df.reset_index(inplace=True, drop=True)
                #             change_time = commit_temp_df.at[0, 'commit_committer_date']

                change_time = datetime.strptime(change_time, "%Y-%m-%d %H:%M:%S")
                review_gap_second = (change_time - comment_time).total_seconds()
                if review_gap_second < 0:
                    print(review_gap_second)
                    raise Exception('gap is < 0' + pr_node, ' ', comment_node_id)
                time_gap = review_gap_second / 60

                """�� ISSUE COMMENT��һ�ι���  ����issue comment��merge�ĳ��� ���ҷ�ӳ��5�����ڵ�������"""
                if comment_type == 'label_issue_comment' and change_type == 'MergedEvent' and time_gap < 5:
                    mergeCommentCase += 1
                    continue
                """�Գ���ʱ�� pair������  ������Ϊ3��"""
                if time_gap > 60 * 24 * 3:
                    tooLongCase += 1
                    continue
                comment_response_list.append((reviewer, comment_node_id, comment_type, change_type, time_gap))
                filter_change_trigger_df.loc[index, 'response_time'] = time_gap
            except Exception as e:
                print("created time not found in timeline :", errorCount)
                errorCount += 1
        print(comment_response_list.__len__())
        print("merge change case :", mergeCommentCase)
        print("too long pair case:", tooLongCase)


        X = [x[-1] for x in comment_response_list]
        r = [x[0] for x in comment_response_list]
        # plt.hist(X, 100)
        # plt.show()
        #
        # plt.hist(r, 100)
        # plt.show()
        #
        # plt.plot(X, r, 'ro')
        # plt.show()


        filter_change_trigger_df = filter_change_trigger_df.loc[filter_change_trigger_df['response_time'] > 0].copy(
            deep=True)
        filter_change_trigger_df.drop(columns=['label'], inplace=True)
        filter_change_trigger_df.reset_index(drop=True, inplace=True)
        print(filter_change_trigger_df.shape)

        PR_CHANGE_TRIGGER_COLUMNS = ["pullrequest_node", "user_login", "comment_node",
                                     "comment_type", "change_trigger", "filepath", 'response_time']

        target_filename = projectConfig.getPRTimeLineDataPath() + os.sep + f'ALL_{repo}_data_pr_change_trigger_filter.tsv'
        target_content = DataFrame(columns=PR_CHANGE_TRIGGER_COLUMNS)
        pandasHelper.writeTSVFile(target_filename, target_content, pandasHelper.STR_WRITE_STYLE_APPEND_NEW,
                                  header=pandasHelper.INT_WRITE_WITH_HEADER)

        pandasHelper.writeTSVFile(target_filename, filter_change_trigger_df, pandasHelper.STR_WRITE_STYLE_APPEND_NEW,
                                  header=pandasHelper.INT_WRITE_WITHOUT_HEADER)

    @staticmethod
    def recoverName(recommendList, answerList, convertDict):

        tempDict = {k: v for v, k in convertDict.items()}
        recommendList = [[tempDict[i] for i in x] for x in recommendList]
        answerList = [[tempDict[i] for i in x] for x in answerList]
        return recommendList, answerList

    @staticmethod
    def saveRecommendList(recommendListPath,prList, recommendList, answerList, convertDict, authorList=None, typeList=None, key=None,
                          filter_answer_list=None):

        recommendList, answerList = DataProcessUtils.recoverName(recommendList, answerList, convertDict)
        tempDict = {k: v for v, k in convertDict.items()}
        if authorList is not None:
            authorList = [tempDict[x] for x in authorList]
        if filter_answer_list is not None:
            filter_answer_list = [[tempDict[i] for i in x] for x in filter_answer_list]
        col = ['pr', 'r1', 'r2', 'r3', 'r4', 'r5', 'a1', 'a2', 'a3', 'a4', 'a5', 'author', 'type', 'fa1', 'fa2', 'fa3',
               'fa4', 'fa5']
        data = DataFrame(columns=col)
        for index, pr in enumerate(prList):
            d = {'pr': pr}
            for i, r in enumerate(recommendList[index]):
                d[f'r{i + 1}'] = r
            for i, a in enumerate(answerList[index]):
                d[f'a{i + 1}'] = a
            if authorList is not None:
                d['author'] = authorList[index]
            if typeList is not None:
                d['type'] = typeList[index]
            if filter_answer_list is not None:
                for i, a in enumerate(filter_answer_list[index]):
                    d[f'fa{i + 1}'] = a
            data = data.append(d, ignore_index=True)
        # pandasHelper.writeTSVFile('temp.tsv', data
        #                           , pandasHelper.STR_WRITE_STYLE_WRITE_TRUNC)
        project = key.split('(', 1)[0]
        if not os.path.exists(f'{recommendListPath}/{project}'):
            os.makedirs(f'{recommendListPath}/{project}')
        data.to_excel(f'{recommendListPath}/{project}/recommendList_{key}.xls', encoding='utf-8', index=False,
                      header=True)

    @staticmethod
    def combineBestResult(prList, answerList, recommendLists, recommendNum=5):
        formatRecommendLists = [[] * (prList.__len__() - 1)]
        for algoRes in recommendLists:
            for idx in range(0, algoRes.__len__()):
                if idx == formatRecommendLists.__len__():
                    formatRecommendLists.append([])
                formatRecommendLists[idx].append(algoRes[idx])


        bestRecommendList = []
        for pr_idx in range(0, prList.__len__()):
            answerCase = answerList[pr_idx]
            recommendCases = formatRecommendLists[pr_idx]
            bestRecommendCase = []
            for top_idx in range(0, recommendNum):
                # ȡ��ÿ���㷨��top_idx�𰸲��ϲ�
                top_all_answer = list(map(lambda x: x[top_idx], recommendCases))
                find_best = False
                for rev in top_all_answer:
                    if rev in answerCase and rev not in bestRecommendCase:
                        bestRecommendCase.append(rev)
                        find_best = True
                        break
                # ���������ȷ������һ�����
                if find_best:
                    continue
                bestRecommendCase.append('case-' + str(top_idx + 1))
            bestRecommendList.append(bestRecommendCase)
        return bestRecommendList

    @staticmethod
    def getSplitFilePath(path, sep=StringKeyUtils.STR_SPLIT_SEP_TWO):
        return path.split(sep)

    @staticmethod
    def LCS_2(list1, list2):

        suf = 0
        length = min(list1.__len__(), list2.__len__())
        for i in range(0, length):
            if list1[list1.__len__() - 1 - i] == list2[list2.__len__() - 1 - i]:
                suf += 1
            else:
                break
        score = suf / max(list1.__len__(), list2.__len__())
        return score

    @staticmethod
    def LCP_2(list1, list2):

        pre = 0
        length = min(list1.__len__(), list2.__len__())
        for i in range(0, length):
            if list1[i] == list2[i]:
                pre += 1
            else:
                break
        # if configPraser.getPrintMode():
        #     print("Longest common pre:", pre)
        return pre / max(list1.__len__(), list2.__len__())

    @staticmethod
    def LCSubseq_2(list1, list2):


        com = 0
        dp = [[0 for i in range(0, list2.__len__() + 1)] for i in range(0, list1.__len__() + 1)]
        for i in range(1, list1.__len__() + 1):
            for j in range(1, list2.__len__() + 1):
                if list1[i - 1] == list2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        com = dp[list1.__len__()][list2.__len__()]
        # if configPraser.getPrintMode():
        #     print("Longest common subString", com)
        return com / max(list1.__len__(), list2.__len__())

    @staticmethod
    def LCSubstr_2(list1, list2):

        com = 0
        dp = [[0 for i in range(0, list2.__len__() + 1)] for i in range(0, list1.__len__() + 1)]
        for i in range(1, list1.__len__() + 1):
            for j in range(1, list2.__len__() + 1):
                if list1[i - 1] == list2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    com = max(com, dp[i][j])
                else:
                    dp[i][j] = 0
        # if configPraser.getPrintMode():
        #     print("Longest common subString", com)
        return com / max(list1.__len__(), list2.__len__())

    @staticmethod
    def caculatePrDistance(projectName, date, filter_change_trigger=True):

        time1 = datetime.now()
        pull_request_path = projectConfig.getPullRequestPath()
        pr_change_file_path = projectConfig.getPRChangeFilePath()

        minYear, minMonth, maxYear, maxMonth = date


        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pull_request_path, f'ALL_{projectName}_data_pullrequest.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        prChangeFileData = pandasHelper.readTSVFile(
            os.path.join(pr_change_file_path, f'ALL_{projectName}_data_pr_change_file.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        pullRequestData = pullRequestData.loc[pullRequestData['is_pr'] == 1].copy(deep=True)

        if filter_change_trigger:
            change_trigger_path = projectConfig.getPRTimeLineDataPath()

            changeTriggerData = pandasHelper.readTSVFile(
                os.path.join(change_trigger_path, f'ALL_{projectName}_data_pr_change_trigger.tsv'),
                pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
            )
            changeTriggerData = changeTriggerData[['pullrequest_node']].copy(deep=True)
            changeTriggerData.drop_duplicates(inplace=True)
            changeTriggerData.reset_index(inplace=True, drop=True)
            pullRequestData = pandas.merge(pullRequestData, changeTriggerData, left_on='node_id',
                                           right_on='pullrequest_node')

        pullRequestData['label'] = pullRequestData['created_at'].apply(
            lambda x: (time.strptime(x, "%Y-%m-%d %H:%M:%S")))
        pullRequestData['label_y'] = pullRequestData['label'].apply(lambda x: x.tm_year)
        pullRequestData['label_m'] = pullRequestData['label'].apply(lambda x: x.tm_mon)

        def isInTimeGap(x):
            d = x['label_y'] * 12 + x['label_m']
            d1 = minYear * 12 + minMonth
            d2 = maxYear * 12 + maxMonth
            return d1 <= d <= d2



        pullRequestData['target'] = pullRequestData.apply(lambda x: isInTimeGap(x), axis=1)
        pullRequestData = pullRequestData.loc[pullRequestData['target'] == 1]
        pullRequestData.reset_index(drop=True, inplace=True)


        data = pandas.merge(pullRequestData, prChangeFileData, left_on='number', right_on='pull_number')
        prList = list(set(data['number']))
        prList.sort()
        prFileDict = dict(list(data.groupby('number')))


        prFileMap = {}
        for pr in prList:
            prFileMap[pr] = list(set(prFileDict[pr]['filename']))

        """(p1, p2, s1)  p1 < p2"""

        cols = ['pr1', 'pr2', 'distance']
        df_LCS = DataFrame(columns=cols)  # ���׺
        df_LCP = DataFrame(columns=cols)  # �ǰ׺
        df_LCSubseq = DataFrame(columns=cols)  # ��󹫹��Ӵ�
        df_LCSubstr = DataFrame(columns=cols)  # ���������Ӵ�
        df = DataFrame(columns=cols)  # ���������Ӵ�

        for index, p1 in enumerate(prList):
            print("now:", index, " all:", prList.__len__())
            for p2 in prList:
                if p1 < p2:
                    files1 = prFileMap[p1]
                    files2 = prFileMap[p2]

                    score_LCS = 0
                    score_LCSubseq = 0
                    score_LCP = 0
                    score_LCSubstr = 0

                    for filename1 in files1:
                        for filename2 in files2:
                            score_LCS += DataProcessUtils.LCS_2(filename1, filename2)
                            score_LCSubseq += DataProcessUtils.LCSubseq_2(filename1, filename2)
                            score_LCP += DataProcessUtils.LCP_2(filename1, filename2)
                            score_LCSubstr += DataProcessUtils.LCSubstr_2(filename1, filename2)

                    score_LCS /= files1.__len__() * files2.__len__()
                    score_LCSubseq /= files1.__len__() * files2.__len__()
                    score_LCP /= files1.__len__() * files2.__len__()
                    score_LCSubstr /= files1.__len__() * files2.__len__()

                    df_LCS = df_LCS.append({'pr1': p1, 'pr2': p2, 'distance': score_LCS}, ignore_index=True)
                    df_LCSubseq = df_LCSubseq.append({'pr1': p1, 'pr2': p2, 'distance': score_LCSubseq},
                                                     ignore_index=True)
                    df_LCP = df_LCP.append({'pr1': p1, 'pr2': p2, 'distance': score_LCP}, ignore_index=True)
                    df_LCSubstr = df_LCSubstr.append({'pr1': p1, 'pr2': p2, 'distance': score_LCSubstr},
                                                     ignore_index=True)
                    score_All = score_LCS + score_LCSubseq + score_LCP + score_LCSubstr
                    df = df.append({'pr1': p1, 'pr2': p2, 'distance': score_All},
                                   ignore_index=True)

        targetPath = os.path.join(os.path.join(projectConfig.getPullRequestDistancePath(), projectName),
                                  f'{date[0]}_{date[1]}_to_{date[2]}_{date[3]}')
        if not os.path.exists(targetPath):
            os.makedirs(targetPath)
        pandasHelper.writeTSVFile(os.path.join(targetPath, f"pr_distance_{projectName}_LCS.tsv"),
                                  df_LCS, pandasHelper.STR_WRITE_STYLE_WRITE_TRUNC)
        pandasHelper.writeTSVFile(os.path.join(targetPath, f"pr_distance_{projectName}_LCSubseq.tsv"),
                                  df_LCSubseq, pandasHelper.STR_WRITE_STYLE_WRITE_TRUNC)
        pandasHelper.writeTSVFile(os.path.join(targetPath, f"pr_distance_{projectName}_LCP.tsv"),
                                  df_LCP, pandasHelper.STR_WRITE_STYLE_WRITE_TRUNC)
        pandasHelper.writeTSVFile(os.path.join(targetPath, f"pr_distance_{projectName}_LCSubstr.tsv"),
                                  df_LCSubstr, pandasHelper.STR_WRITE_STYLE_WRITE_TRUNC)
        pandasHelper.writeTSVFile(os.path.join(targetPath, f"pr_distance_{projectName}.tsv"),
                                  df, pandasHelper.STR_WRITE_STYLE_WRITE_TRUNC)

    @staticmethod
    def caculatePrDistanceByIncrement(projectName, date, filter_change_trigger=False):

        time1 = datetime.now()
        pull_request_path = projectConfig.getPullRequestPath()
        pr_change_file_path = projectConfig.getPRChangeFilePath()
        change_trigger_path = projectConfig.getPRTimeLineDataPath()
        minYear, minMonth, maxYear, maxMonth = date


        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pull_request_path, f'ALL_{projectName}_data_pullrequest.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        prChangeFileData = pandasHelper.readTSVFile(
            os.path.join(pr_change_file_path, f'ALL_{projectName}_data_pr_change_file.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        changeTriggerData = pandasHelper.readTSVFile(
            os.path.join(change_trigger_path, f'ALL_{projectName}_data_pr_change_trigger.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        pullRequestData = pullRequestData.loc[pullRequestData['is_pr'] == 1].copy(deep=True)

        if filter_change_trigger:
            changeTriggerData = changeTriggerData[['pullrequest_node']].copy(deep=True)
            changeTriggerData.drop_duplicates(inplace=True)
            changeTriggerData.reset_index(inplace=True, drop=True)
            pullRequestData = pandas.merge(pullRequestData, changeTriggerData, left_on='node_id',
                                           right_on='pullrequest_node')

        pullRequestData['label'] = pullRequestData['created_at'].apply(
            lambda x: (time.strptime(x, "%Y-%m-%d %H:%M:%S")))
        pullRequestData['label_y'] = pullRequestData['label'].apply(lambda x: x.tm_year)
        pullRequestData['label_m'] = pullRequestData['label'].apply(lambda x: x.tm_mon)

        pullRequestData['data_value'] = pullRequestData.apply(lambda x: x['label_y'] * 12 + x['label_m'], axis=1)

        def isInTimeGap(x):
            d = x['label_y'] * 12 + x['label_m']
            d1 = minYear * 12 + minMonth
            d2 = maxYear * 12 + maxMonth
            return d1 <= d <= d2



        pullRequestData['target'] = pullRequestData.apply(lambda x: isInTimeGap(x), axis=1)
        pullRequestData = pullRequestData.loc[pullRequestData['target'] == 1]
        pullRequestData.reset_index(drop=True, inplace=True)

        data = pandas.merge(pullRequestData, prChangeFileData, left_on='number', right_on='pull_number')
        prList = list(set(data['number']))
        prList.sort()
        prFileDict = dict(list(data.groupby('number')))

        prFileMap = {}
        for pr in prList:
            prFileMap[pr] = tuple(set(prFileDict[pr]['filename']))

        list_p1 = []
        list_p2 = []
        list_dis_FPS = []

        fileListMap = {}
        for pr in prList:
            for f in prFileMap[pr]:
                if fileListMap.get(f) is None:
                    fileListMap[f] = tuple(DataProcessUtils.getSplitFilePath(f))

        cols = ['pr1', 'pr2', 'distance']

        min_data = minYear * 12 + minMonth
        max_data = maxYear * 12 + maxMonth

        for i in range(min_data + 1, max_data + 1, 1):
            pr_df_1 = data.loc[data['data_value'] == i]
            list_pr_1 = list(set(pr_df_1['number']))
            pr_df_2 = data.loc[data['data_value'] < i]
            list_pr_2 = list(set(pr_df_2['number']))

            for index, p1 in enumerate(list_pr_1):
                print("now:", index, " all:", list_pr_1.__len__())
                for p2 in list_pr_2:
                    if p1 > p2:  # ��� �ұ�С
                        files1 = prFileMap[p1]
                        files2 = prFileMap[p2]

                        score_FPS = 0

                        for filename1 in files1:
                            for filename2 in files2:
                                score_FPS += DataProcessUtils.LCS_2(filename1, filename2)
                                score_FPS += DataProcessUtils.LCSubseq_2(filename1, filename2)
                                score_FPS += DataProcessUtils.LCP_2(filename1, filename2)
                                score_FPS += DataProcessUtils.LCSubstr_2(filename1, filename2)

                        score_FPS /= files1.__len__() * files2.__len__()

                        list_p1.append(p1)
                        list_p2.append(p2)
                        list_dis_FPS.append(score_FPS)

        df_FPS = DataFrame({'pr1': list_p1, 'pr2': list_p2, 'distance': list_dis_FPS})

        targetPath = projectConfig.getPullRequestDistancePath()
        pandasHelper.writeTSVFile(os.path.join(targetPath, f"pr_distance_{projectName}_FPS.tsv"),
                                  df_FPS, pandasHelper.STR_WRITE_STYLE_WRITE_TRUNC)

    @staticmethod
    def fillAlgorithmResultExcelHelper(filter_train=False, filter_test=False, error_analysis=True):

        projectList = ['opencv', 'cakephp', 'xbmc', 'symfony', 'akka', 'babel',
                       'django', 'brew', 'netty', 'scikit-learn', 'moby', 'metasploit-framework',
                       'Baystation12', 'react', 'pandas', 'angular', 'next.js']
        algorithmList = ['FPS', 'IR', 'RF', 'CN', 'AC', 'CHREV', 'XF', 'RF_A']

        algorithmFileLabelMap = {'FPS': 'FPS', 'IR': 'IR', 'RF': 'ML_0',
                                 'CN': 'CN', 'AC': 'AC', 'CHREV': 'CHREV', 'XF': 'XF', 'RF_A': 'RF_A'}
        sheetMap = ['TopK', 'PT', 'NT', 'PF', 'NF']


        metricList = ['AVG_TopKAccuracy', 'recommend_positive_success_pr_ratio', 'recommend_negative_success_pr_ratio',
                      'recommend_positive_fail_pr_ratio', 'recommend_negative_fail_pr_ratio']


        readPath = r'C:\Users\ThinkPad\Desktop\ͳ������\����'

        recommendNumList = [1, 3, 5]


        excelName = f'�Ƽ��㷨����ͳ��_{filter_train}_{filter_test}_{error_analysis}.xlsx'


        ExcelHelper().initExcelFile(fileName=excelName, sheetName=sheetMap[0])
        for sheetName in sheetMap:
            ExcelHelper().addSheet(filename=excelName, sheetName=sheetName)


        for metric_index, sheetName in enumerate(sheetMap):
            metric = metricList[metric_index]


            inital_line = []
            inital_line.append("")
            for i in range(0, 3):
                for m in algorithmList:
                    inital_line.append(m)
                inital_line.append("")
                inital_line.append("")

            ExcelHelper().appendExcelRow(excelName, sheetName, inital_line, style=ExcelHelper.getNormalStyle())

            for project in projectList:

                line = []
                line.append(project)
                for recommendNum in recommendNumList:
                    for algorithm in algorithmList:
                        line.append(DataProcessUtils.readSingleResultFromExcel(path=readPath,
                                                                               algorithm_label=algorithmFileLabelMap[
                                                                                   algorithm],
                                                                               project=project,
                                                                               filter_train=filter_train,
                                                                               filter_test=filter_test,
                                                                               error_analysis=error_analysis,
                                                                               metric=metric,
                                                                               recommendNum=recommendNum))
                    line.append("")
                    line.append("")
                ExcelHelper().appendExcelRow(excelName, sheetName, line, style=ExcelHelper.getNormalStyle())

    @staticmethod
    def readSingleResultFromExcel(path, algorithm_label, project, filter_train, filter_test, error_analysis,
                                  metric, recommendNum):
        sheetName = 'result'

        isFind = False
        fileName = f'output{algorithm_label}_{project}_{filter_train}_{filter_test}_{error_analysis}.xlsx'
        fileName = os.path.join(path, fileName)
        if not os.path.exists(fileName):
            return "NA"
        sheet = ExcelHelper().readExcelSheet(fileName, sheetName)
        rowNum = sheet.nrows - 1
        while not isFind:
            # print(rowNum)
            line = sheet.row_values(rowNum)
            if isinstance(line, list) and line.__len__() > 0:
                if line[2] == metric:
                    isFind = True
                    break
            rowNum -= 1

        rowNum += 2
        content = sheet.row_values(rowNum)
        result = content[1 + recommendNum]
        return result

    @staticmethod
    def selectPeriodChangeTrigger(projectName, start, end):
        # ��ȡpull request�ļ�
        pull_request_path = projectConfig.getPullRequestPath()
        change_trigger_path = projectConfig.getPRTimeLineDataPath()
        #
        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pull_request_path, f'ALL_{projectName}_data_pullrequest.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        changeTriggerData = pandasHelper.readTSVFile(
            os.path.join(change_trigger_path, f'ALL_{projectName}_data_pr_change_trigger.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        changeTriggerData.drop_duplicates(inplace=True)


        pullRequestData = pullRequestData.loc[pullRequestData['is_pr'] == 1].copy(deep=True)
        pullRequestData = pullRequestData[(pullRequestData['number'] >= start) & (pullRequestData['number'] <= end)]
        pullRequestData = pullRequestData[['node_id', 'user_login']]
        pullRequestData.columns = ['pullrequest_node', 'pr_author']


        data = pandas.merge(changeTriggerData, pullRequestData, on=['pullrequest_node'])
        #
        # """����na"""
        # data.dropna(subset=['user_login'], inplace=True)
        #
        # """ȥ���Լ������Լ���case"""
        # data = data[data['user_login'] != data['pr_author']]
        # data.drop(columns=['pr_author'], inplace=True)
        #
        #

        # data['isBot'] = data['user_login'].apply(lambda x: BotUserRecognizer.isBot(x))
        # data = data.loc[data['isBot'] == False].copy(deep=True)
        # data.drop(columns=['isBot'], inplace=True)

        pandasHelper.writeTSVFile(os.path.join(change_trigger_path, f'ALL_{projectName}_data_pr_change_trigger.tsv'),
                                  data)

    @staticmethod
    def findUnUsefulUser(projectName, algorithm, dates, filter_train=False, filter_test=False):
        """
        Ѱ������עˮ�϶���û�
        """
        excelName = f'{projectName}_user_statistics_{algorithm}_{filter_train}_{filter_test}.xls'
        sheetName = 'result'
        content = ['�û���', '��Ч������', '�Ƽ���Ч��', '��Ч������', '��������', '��Ч�Ƽ���', '�ܱ��Ƽ���', 'Ӧ���Ƽ���']
        ExcelHelper().initExcelFile(fileName=excelName, sheetName=sheetName, excel_key_list=content)

        change_trigger_path = projectConfig.getPRTimeLineDataPath()

        changeTriggerData = pandasHelper.readTSVFile(
            os.path.join(change_trigger_path, f'ALL_{projectName}_data_pr_change_trigger.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False,
        )

        users = []
        rev_total_cnt_dict = {}
        rev_useful_cnt_dict = {}
        grouped_change_trigger = changeTriggerData.groupby((['user_login']))
        for user, group in grouped_change_trigger:
            group['label'] = group.apply(
                lambda x: (x['comment_type'] == 'label_issue_comment' and x['change_trigger'] == 1) or (
                        x['comment_type'] == 'label_review_comment' and x['change_trigger'] >= 0), axis=1)
            useful_cnt = group.loc[group['label'] == True].shape[0]
            rev_useful_cnt_dict[user] = useful_cnt
            rev_total_cnt_dict[user] = group.shape[0]
            users.append(user)

        rev_recommend_total_cnt_dict = {}
        rev_recommend_useful_cnt_dict = {}
        rev_recommend_should_cnt_dict = {}

        for date in dates:
            recommend_list_file_path = projectConfig.getAlgorithmPath() + os.sep + f'{algorithm}/recommendList_{projectName}{str(date)}{filter_train}{filter_test}.xls'
            row_idx = 1
            recommend_case = ExcelHelper().readExcelRow(recommend_list_file_path, "Sheet1", startRow=row_idx)
            while recommend_case[0] != '':
                recommendList = recommend_case[1:6]
                answerList = recommend_case[6:11]
                for reviewer in recommendList:
                    if rev_recommend_total_cnt_dict.get(reviewer, None) is None:
                        rev_recommend_total_cnt_dict[reviewer] = 0
                    rev_recommend_total_cnt_dict[reviewer] += 1
                    if reviewer in answerList:
                        if rev_recommend_useful_cnt_dict.get(reviewer, None) is None:
                            rev_recommend_useful_cnt_dict[reviewer] = 0
                        rev_recommend_useful_cnt_dict[reviewer] += 1
                for answer in answerList:
                    if rev_recommend_should_cnt_dict.get(answer, None) is None:
                        rev_recommend_should_cnt_dict[answer] = 0
                    rev_recommend_should_cnt_dict[answer] += 1
                row_idx += 1
                try:
                    recommend_case = ExcelHelper().readExcelRow(recommend_list_file_path, "Sheet1", startRow=row_idx)
                except IndexError:
                    break

        for user in users:

            if rev_total_cnt_dict[user] > 10:
                if rev_recommend_total_cnt_dict.get(user, 0) == 0:
                    rev_rec_ratio = 0
                else:
                    rev_rec_ratio = rev_recommend_useful_cnt_dict.get(user, 0) * 100 / rev_recommend_total_cnt_dict[
                        user]
                content = [user,
                           rev_useful_cnt_dict.get(user, 0) * 100 / rev_total_cnt_dict[user],
                           rev_rec_ratio,
                           rev_useful_cnt_dict.get(user, 0),
                           rev_total_cnt_dict[user],
                           rev_recommend_useful_cnt_dict.get(user, 0),
                           rev_recommend_total_cnt_dict.get(user, 0),
                           rev_recommend_should_cnt_dict.get(user, 0)]
                ExcelHelper().appendExcelRow(excelName, sheetName, content, style=ExcelHelper.getNormalStyle())

    @staticmethod
    def getAnswerListFromChangeTriggerData(project, date, prList, convertDict, filename, review_key, pr_key):

        y = date[2]
        m = date[3]
        df = None

        df = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITH_HEAD)
        df.reset_index(inplace=True, drop=True)


        df['label'] = df['pr_created_at'].apply(
            lambda x: (time.strptime(x, "%Y-%m-%d %H:%M:%S").tm_year == y and
                       time.strptime(x, "%Y-%m-%d %H:%M:%S").tm_mon == m))


        tagDict = dict(list(df.groupby(pr_key)))


        test_data = df.loc[df['label']].copy(deep=True)

        test_data.drop(columns=['label'], inplace=True)

        test_data_y = []
        for pull_number in prList:
            if pull_number not in tagDict.keys():
                test_data_y.append([])
            else:
                reviewers = list(tagDict[pull_number].drop_duplicates([review_key])[review_key])
                reviewers = [convertDict[x] for x in reviewers]
                test_data_y.append(reviewers)

        return test_data_y

    @staticmethod
    def getAnswerListFromChangeTriggerDataByIncrement(project, prList, convertDict, filenameList, review_key, pr_key):

        df = None
        for filename in filenameList:
            if df is None:
                df = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITH_HEAD)
            else:
                temp = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITH_HEAD)
                df = df.append(temp)  # �ϲ�

        df.reset_index(inplace=True, drop=True)

        tagDict = dict(list(df.groupby(pr_key)))


        test_data = df.copy(deep=True)

        test_data_y = []
        for pull_number in prList:
            if pull_number not in tagDict.keys():
                test_data_y.append([])
            else:
                reviewers = list(tagDict[pull_number].drop_duplicates([review_key])[review_key])
                reviewers = [convertDict[x] for x in reviewers]
                test_data_y.append(reviewers)

        return test_data_y

    @staticmethod
    def recommendErrorAnalyzer(error_analysis_data, project, label=None):
        recommend_positive_success_pr_ratio, recommend_positive_success_time_ratio, \
        recommend_negative_success_pr_ratio, recommend_negative_success_time_ratio, \
        recommend_positive_fail_pr_ratio, recommend_positive_fail_time_ratio, \
        recommend_negative_fail_pr_ratio, recommend_negative_fail_time_ratio = error_analysis_data


        for i in range(0, 5):
            plt.subplot(3, 5, i + 1)
            labels = ['PT', 'NT', 'PF', 'NF']
            x = [DataProcessUtils.getAvgScore(recommend_positive_success_time_ratio)[i],
                 DataProcessUtils.getAvgScore(recommend_negative_success_time_ratio)[i],
                 DataProcessUtils.getAvgScore(recommend_positive_fail_time_ratio)[i],
                 DataProcessUtils.getAvgScore(recommend_negative_fail_time_ratio)[i]]

            plt.pie(x, labels=labels, autopct='%3.2f%%')
            plt.title(f"k={i + 1} (time)")

            plt.axis('equal')


        labels = ['PT', 'NT', 'PF', 'NF']
        for i in range(0, 4):
            x = range(1, 6)
            y = []
            for m in x:
                y.append(DataProcessUtils.getAvgScore(error_analysis_data[2 * i])[m - 1] * 100)
            plt.subplot(3, 2, i + 3)
            plt.bar(x=x, height=y)
            plt.title(f"{labels[i]} (pr)")
            for a, b in zip(x, y):
                plt.text(a, b, '%3.4f%%' % b, ha='center', va='bottom', fontsize=9)

        plt.suptitle(project)
        plt.savefig(f"{project}_{label}.png")
        # plt.show()
        plt.close()

    @staticmethod
    def recommendErrorAnalyzer2(error_analysis_data, project, label=None):
        recommend_positive_success_pr_ratio, recommend_negative_success_pr_ratio, \
        recommend_positive_fail_pr_ratio, recommend_negative_fail_pr_ratio = error_analysis_data


        for i in range(0, 5):
            plt.subplot(2, 3, i + 1)
            labels = ['PT', 'NT', 'PF', 'NF']
            x = [DataProcessUtils.getAvgScore(recommend_positive_success_pr_ratio)[i],
                 DataProcessUtils.getAvgScore(recommend_negative_success_pr_ratio)[i],
                 DataProcessUtils.getAvgScore(recommend_positive_fail_pr_ratio)[i],
                 DataProcessUtils.getAvgScore(recommend_negative_fail_pr_ratio)[i]]

            plt.pie(x, labels=labels, autopct='%3.2f%%')
            plt.title(f"k={i + 1} (pr)")

            plt.axis('equal')

        plt.suptitle(project)
        print(projectConfig.getHGResultPath() + os.sep + f"output_error/{label}.png")
        plt.savefig(projectConfig.getHGResultPath() + os.sep + f"output_error/{label}.png")
        # plt.show()
        plt.close()

    @staticmethod
    def getUserListFromProject(projectName):


        time1 = datetime.now()
        issue_comment_path = projectConfig.getIssueCommentPath()
        pull_request_path = projectConfig.getPullRequestPath()
        review_path = projectConfig.getReviewDataPath()
        change_trigger_path = projectConfig.getPRTimeLineDataPath()



        issueCommentData = pandasHelper.readTSVFile(
            os.path.join(issue_comment_path, f'ALL_{projectName}_data_issuecomment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pull_request_path, f'ALL_{projectName}_data_pullrequest.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_path, f'ALL_{projectName}_data_review.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        userList = []
        userList.extend(list(reviewData['user_login']))
        userList.extend(list(pullRequestData['user_login']))
        userList.extend(list(issueCommentData['user_login']))
        userList = list(set(userList))
        return userList

    @staticmethod
    def genBaseData(projects, date):
        """����ʱ������"""
        excelName = "basic_data.xls"
        sheetName = "result"
        content = ['Repository', 'PRs', 'commenter_cnt', 'comment_ratio',
                   'issueComment_cnt', 'issueComment_ratio',
                   'reviewComment_cnt', 'reviewComment_ratio', 'contributor_cnt']
        ExcelHelper().initExcelFile(fileName=excelName, sheetName=sheetName, excel_key_list=content)
        issue_comment_file_path = projectConfig.getIssueCommentPath()
        review_comment_file_path = projectConfig.getReviewCommentDataPath()
        pullrequest_file_path = projectConfig.getPullRequestPath()
        change_trigger_path = projectConfig.getPRTimeLineDataPath()
        review_file_path = projectConfig.getReviewDataPath()

        for projectName in projects:


            issueCommentData = pandasHelper.readTSVFile(
                os.path.join(issue_comment_file_path, f'ALL_{projectName}_data_issuecomment.tsv'), low_memory=False,
                header=pandasHelper.INT_READ_FILE_WITH_HEAD)
            print("raw issue_comment file: ", issueCommentData.shape)

            reviewData = pandasHelper.readTSVFile(
                os.path.join(review_file_path, f'ALL_{projectName}_data_review.tsv'), low_memory=False,
                header=pandasHelper.INT_READ_FILE_WITH_HEAD)
            print("raw review file: ", reviewData.shape)


            reviewCommentData = pandasHelper.readTSVFile(
                os.path.join(review_comment_file_path, f'ALL_{projectName}_data_review_comment.tsv'), low_memory=False,
                header=pandasHelper.INT_READ_FILE_WITH_HEAD)
            print("raw review_comment file: ", reviewCommentData.shape)


            pullRequestData = pandasHelper.readTSVFile(
                os.path.join(pullrequest_file_path, f'ALL_{projectName}_data_pullrequest.tsv'), low_memory=False,
                header=pandasHelper.INT_READ_FILE_WITH_HEAD)
            print("raw pr file:", pullRequestData.shape)


            changeTriggerData = pandasHelper.readTSVFile(
                os.path.join(change_trigger_path, f'ALL_{projectName}_data_pr_change_trigger.tsv'),
                pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
            )


            changeTriggerData['label'] = changeTriggerData.apply(
                lambda x: (x['comment_type'] == 'label_issue_comment' and x['change_trigger'] == 1) or (
                        x['comment_type'] == 'label_review_comment' and x['change_trigger'] == 0), axis=1)
            changeTriggerData = changeTriggerData.loc[changeTriggerData['label'] == True].copy(deep=True)
            changeTriggerData = changeTriggerData[['comment_node']].copy(deep=True)
            changeTriggerData.drop_duplicates(inplace=True)



            pullRequestData = pullRequestData.loc[pullRequestData['state'] == 'closed'].copy(deep=True)
            print("after fliter closed pr:", pullRequestData.shape)



            minYear, minMonth, maxYear, maxMonth = date

            pullRequestData['label'] = pullRequestData['created_at'].apply(
                lambda x: (time.strptime(x, "%Y-%m-%d %H:%M:%S")))
            pullRequestData['label_y'] = pullRequestData['label'].apply(lambda x: x.tm_year)
            pullRequestData['label_m'] = pullRequestData['label'].apply(lambda x: x.tm_mon)

            pullRequestData['data_value'] = pullRequestData.apply(lambda x: x['label_y'] * 12 + x['label_m'], axis=1)

            def isInTimeGap(x):
                d = x['label_y'] * 12 + x['label_m']
                d1 = minYear * 12 + minMonth
                d2 = maxYear * 12 + maxMonth
                return d1 <= d <= d2



            pullRequestData['target'] = pullRequestData.apply(lambda x: isInTimeGap(x), axis=1)
            pullRequestData = pullRequestData.loc[pullRequestData['target'] == 1]


            pullRequestData = pullRequestData[
                ['repo_full_name', 'number', 'node_id', 'user_login', 'created_at', 'author_association',
                 'closed_at']].copy(deep=True)
            pullRequestData.columns = ['repo_full_name', 'pull_number', 'pullrequest_node', 'pr_author',
                                       'pr_created_at',
                                       'pr_author_association', 'closed_at']
            pullRequestData.drop_duplicates(inplace=True)
            pullRequestData.reset_index(drop=True, inplace=True)
            print("after fliter pr:", pullRequestData.shape)

            issueCommentData = issueCommentData[
                ['pull_number', 'node_id', 'user_login', 'created_at', 'author_association']].copy(deep=True)
            issueCommentData.columns = ['pull_number', 'comment_node', 'reviewer', 'commented_at',
                                        'reviewer_association']
            issueCommentData.drop_duplicates(inplace=True)
            issueCommentData.reset_index(drop=True, inplace=True)
            issueCommentData['comment_type'] = StringKeyUtils.STR_LABEL_ISSUE_COMMENT
            print("after fliter issue comment:", issueCommentData.shape)

            reviewData = reviewData[['pull_number', 'id', 'user_login', 'submitted_at']].copy(deep=True)
            reviewData.columns = ['pull_number', 'pull_request_review_id', 'reviewer', 'submitted_at']
            reviewData.drop_duplicates(inplace=True)
            reviewData.reset_index(drop=True, inplace=True)
            print("after fliter review:", reviewData.shape)


            reviewCommentData = reviewCommentData[
                ['pull_request_review_id', 'node_id', 'user_login', 'created_at', 'author_association']].copy(deep=True)
            reviewCommentData.columns = ['pull_request_review_id', 'comment_node', 'reviewer', 'commented_at',
                                         'reviewer_association']
            reviewCommentData.drop_duplicates(inplace=True)
            reviewCommentData.reset_index(drop=True, inplace=True)
            reviewCommentData['comment_type'] = StringKeyUtils.STR_LABEL_REVIEW_COMMENT
            print("after fliter review comment:", reviewCommentData.shape)
            reviewCommentData = pandas.merge(reviewData, reviewCommentData, on='pull_request_review_id', how='left')
            reviewCommentData['reviewer'] = reviewCommentData.apply(
                lambda row: row['reviewer_x'] if pandas.isna(row['reviewer_y']) else row['reviewer_y'], axis=1)
            reviewCommentData['commented_at'] = reviewCommentData.apply(
                lambda row: row['submitted_at'] if pandas.isna(row['commented_at']) else row['commented_at'], axis=1)
            reviewCommentData.drop(columns=['pull_request_review_id', 'submitted_at', 'reviewer_x', 'reviewer_y'],
                                   inplace=True)

            issueCommentData = pandas.merge(issueCommentData, pullRequestData, how="inner")
            reviewCommentData = pandas.merge(reviewCommentData, pullRequestData, how="inner")

            pr_cnt = list(set(pullRequestData['pullrequest_node'])).__len__()
            contributor_cnt = list(set(pullRequestData['pr_author'])).__len__()
            issue_comment_cnt = list(set(issueCommentData['comment_node'])).__len__()
            review_comment_cnt = list(set(reviewCommentData['comment_node'])).__len__()
            commenter_cnt = set(list(issueCommentData['reviewer']) + list(reviewCommentData['reviewer'])).__len__()

            issueCommentData = pandas.merge(issueCommentData, changeTriggerData, how="inner")
            reviewCommentData = pandas.merge(reviewCommentData, changeTriggerData, how="inner")

            filter_issue_comment_cnt = list(set(issueCommentData['comment_node'])).__len__()
            filter_review_comment_cnt = list(set(reviewCommentData['comment_node'])).__len__()
            filter_commenter_cnt = set(
                list(issueCommentData['reviewer']) + list(reviewCommentData['reviewer'])).__len__()

            content = [projectName, pr_cnt, commenter_cnt, filter_commenter_cnt / commenter_cnt,
                       issue_comment_cnt, filter_issue_comment_cnt / issue_comment_cnt,
                       review_comment_cnt, filter_review_comment_cnt / review_comment_cnt, contributor_cnt]

            ExcelHelper().appendExcelRow(excelName, sheetName, content, style=ExcelHelper.getNormalStyle())

    @staticmethod
    def genNotMergePrRatio(projects, startYear, endYear):
        excelName = "open_pr_ratio.xls"
        sheetName = "result"
        content = ['year', 'all pr', 'open pr', 'ratio']
        ExcelHelper().initExcelFile(fileName=excelName, sheetName=sheetName, excel_key_list=content)
        issue_comment_file_path = projectConfig.getIssueCommentPath()
        review_comment_file_path = projectConfig.getReviewCommentDataPath()
        pullrequest_file_path = projectConfig.getPullRequestPath()
        timeline_path = projectConfig.getPRTimeLineDataPath()
        review_file_path = projectConfig.getReviewDataPath()

        count = 0

        all_map = {}
        open_map = {}
        reject_map = {}
        for i in range(startYear, endYear + 1):
            all_map[i] = 0
            open_map[i] = 0
            reject_map[i] = 0

        for projectName in projects:
            pullRequestData = pandasHelper.readTSVFile(
                os.path.join(pullrequest_file_path, f'ALL_{projectName}_data_pullrequest.tsv'), low_memory=False,
                header=pandasHelper.INT_READ_FILE_WITH_HEAD)
            print("raw pr file:", pullRequestData.shape)

            timelineData = pandasHelper.readTSVFile(
                os.path.join(timeline_path, f'ALL_{projectName}_data_prtimeline.tsv'), low_memory=False,
                header=pandasHelper.INT_READ_FILE_WITH_HEAD)
            print("raw pr file:", timelineData.shape)

            pullRequestData = pullRequestData.loc[pullRequestData['is_pr'] == True].copy(deep=True)

            pullRequestData['label'] = pullRequestData['created_at'].apply(
                lambda x: (time.strptime(x, "%Y-%m-%d %H:%M:%S")))
            pullRequestData['label_y'] = pullRequestData['label'].apply(lambda x: x.tm_year)
            pullRequestData['label_m'] = pullRequestData['label'].apply(lambda x: x.tm_mon)

            group = dict(list(timelineData.groupby('pullrequest_node')))
            for index, df in group.items():
                if df.shape[0] > 5:
                    count += 1
            print(count)

            for i in range(startYear, endYear + 1):
                temp = pullRequestData.loc[pullRequestData['label_y'] == i]
                all_map[i] += temp.shape[0]
                temp_open = temp.loc[temp['state'] == 'open']
                open_map[i] += temp_open.shape[0]

                temp2 = temp.loc[temp['merged'] == 0]
                reject_map[i] += temp2.shape[0]

        for i in range(startYear, endYear + 1):
            content = [i, all_map[i], open_map[i], open_map[i] / all_map[i]]
            ExcelHelper().appendExcelRow(excelName, sheetName, content, style=ExcelHelper.getNormalStyle())
            content = [i, all_map[i], reject_map[i], reject_map[i] / all_map[i]]
            ExcelHelper().appendExcelRow(excelName, sheetName, content, style=ExcelHelper.getNormalStyle())

    @staticmethod
    def genOpenPrRatio(projects, startYear, endYear):
        """����ʱ������"""
        excelName = "open_pr_ratio.xls"
        sheetName = "result"
        content = ['year', 'all pr', 'open pr', 'ratio']
        ExcelHelper().initExcelFile(fileName=excelName, sheetName=sheetName, excel_key_list=content)
        issue_comment_file_path = projectConfig.getIssueCommentPath()
        review_comment_file_path = projectConfig.getReviewCommentDataPath()
        pullrequest_file_path = projectConfig.getPullRequestPath()
        change_trigger_path = projectConfig.getPRTimeLineDataPath()
        review_file_path = projectConfig.getReviewDataPath()

        all_map = {}
        open_map = {}
        for i in range(startYear, endYear + 1):
            all_map[i] = 0
            open_map[i] = 0

        for projectName in projects:
            pullRequestData = pandasHelper.readTSVFile(
                os.path.join(pullrequest_file_path, f'ALL_{projectName}_data_pullrequest.tsv'), low_memory=False,
                header=pandasHelper.INT_READ_FILE_WITH_HEAD)
            print("raw pr file:", pullRequestData.shape)

            pullRequestData = pullRequestData.loc[pullRequestData['is_pr'] == True].copy(deep=True)

            pullRequestData['label'] = pullRequestData['created_at'].apply(
                lambda x: (time.strptime(x, "%Y-%m-%d %H:%M:%S")))
            pullRequestData['label_y'] = pullRequestData['label'].apply(lambda x: x.tm_year)
            pullRequestData['label_m'] = pullRequestData['label'].apply(lambda x: x.tm_mon)

            for i in range(startYear, endYear + 1):
                temp = pullRequestData.loc[pullRequestData['label_y'] == i]
                all_map[i] += temp.shape[0]
                temp_open = temp.loc[temp['state'] == 'open']
                open_map[i] += temp_open.shape[0]

        for i in range(startYear, endYear + 1):
            content = [i, all_map[i], open_map[i], open_map[i] / all_map[i]]
            ExcelHelper().appendExcelRow(excelName, sheetName, content, style=ExcelHelper.getNormalStyle())

    @staticmethod
    def changeTriggerAnalyzerALL(repos, date):
        disDict = {}
        for p in repos:
            pullrequest_file_path = projectConfig.getPullRequestPath()
            pullRequestData = pandasHelper.readTSVFile(
                os.path.join(pullrequest_file_path, f'ALL_{p}_data_pullrequest.tsv'), low_memory=False,
                header=pandasHelper.INT_READ_FILE_WITH_HEAD)
            print("raw pr file:", pullRequestData.shape)



            pullRequestData = pullRequestData.loc[pullRequestData['state'] == 'closed'].copy(deep=True)
            print("after fliter closed pr:", pullRequestData.shape)

            minYear, minMonth, maxYear, maxMonth = date

            pullRequestData['label'] = pullRequestData['created_at'].apply(
                lambda x: (time.strptime(x, "%Y-%m-%d %H:%M:%S")))
            pullRequestData['label_y'] = pullRequestData['label'].apply(lambda x: x.tm_year)
            pullRequestData['label_m'] = pullRequestData['label'].apply(lambda x: x.tm_mon)

            pullRequestData['data_value'] = pullRequestData.apply(lambda x: x['label_y'] * 12 + x['label_m'], axis=1)

            def isInTimeGap(x):
                d = x['label_y'] * 12 + x['label_m']
                d1 = minYear * 12 + minMonth
                d2 = maxYear * 12 + maxMonth
                return d1 <= d <= d2



            pullRequestData['target'] = pullRequestData.apply(lambda x: isInTimeGap(x), axis=1)
            pullRequestData = pullRequestData.loc[pullRequestData['target'] == 1]



            pullRequestData = pullRequestData.loc[pullRequestData['state'] == 'closed'].copy(deep=True)
            print("after fliter closed pr:", pullRequestData.shape)



            change_trigger_filename = projectConfig.getPRTimeLineDataPath() + os.sep + f'ALL_{p}_data_pr_change_trigger.tsv'
            change_trigger_df = pandasHelper.readTSVFile(fileName=change_trigger_filename, header=0)

            change_trigger_df = pandas.merge(change_trigger_df, pullRequestData, left_on='pullrequest_node',
                                             right_on='node_id')

            df_review = change_trigger_df.loc[change_trigger_df['comment_type'] == 'label_review_comment']
            print("review all:", df_review.shape[0])
            x = range(-1, 11)
            y = []
            for i in x:
                count = df_review.loc[df_review['change_trigger'] == i].shape[0]
                if disDict.get(i, None) is None:
                    disDict[i] = 0
                disDict[i] += count
        return disDict

    @staticmethod
    def wilcoxon():
        filename = os.path.join("top5_new.xlsx")
        data = pandas.read_excel(filename, sheet_name="result")
        print(data)
        # data.columns = [0, 1, 2, 3, 4]
        # data.index = [0, 1, 2, 3, 4, 5, 6, 7]
        # print(data)
        # x = [[1, 2, 3, 5, 1], [12, 31, 54, 12], [10, 12, 6, 74, 11]]
        # print(data.values.T)
        # result = scikit_posthocs.posthoc_nemenyi_friedman(data.values)
        # print(result)
        # print(data.values.T[1])
        # print(data.values.T[3])
        data1 = []
        for i in range(0, 7):
            data1.append([])
            for j in range(0, 7):
                if i == j:
                    data1[i].append(numpy.nan)
                    continue
                statistic, pvalue = wilcoxon(data.values.T[i], data.values.T[j], alternative="two-sided")
                print(pvalue)
                data1[i].append(pvalue)
        data1 = pandas.DataFrame(data1)
        print(data1)
        data1.to_excel('a.xls')
        import matplotlib.pyplot as plt
        name = ['FPS', 'IR', 'AC', 'CHREV', 'XF', 'RF_A', 'CN']
        # scikit_posthocs.sign_plot(result, g=name)
        # plt.show()
        for i in range(0, 7):
            data1[i][i] = numpy.nan
        ax = seaborn.heatmap(data1, annot=True, vmax=1, square=True, yticklabels=name, xticklabels=name, cmap='GnBu_r')
        ax.set_title("Wilcoxon signed-rank test_top1_origin")
        plt.show()

    @staticmethod
    def change_trigger_v3_test(repo):


        change_trigger_filename = projectConfig.getPRTimeLineDataPath() + os.sep + f'ALL_{repo}_data_pr_change_trigger.tsv'
        change_trigger_df = pandasHelper.readTSVFile(fileName=change_trigger_filename, header=0)

        prs = list(set(change_trigger_df['pullrequest_node']))
        print("prs nums:", prs.__len__())

        specialCase = 0
        group = dict(list(change_trigger_df.groupby('pullrequest_node')))
        for node, temp_df in group.items():
            lastFileName = None
            hasValidComment = False
            for index, row in temp_df.iterrows():
                commentType = row['comment_type']
                change_trigger = row['change_trigger']
                if commentType == 'label_review_comment' and change_trigger >= 0:
                    hasValidComment = True
                    lastFileName = row['filepath']
                    continue
                if commentType == 'label_review_comment' and hasValidComment and change_trigger == -1:
                    if row['filepath'] != lastFileName:
                        specialCase += 1
                        print(node)
                        break
        print("all:", prs.__len__(), "  special:", specialCase)

    @staticmethod
    def saveAllProjectFinnalyResult(filename, projectName, sheetName, way, topks, mrrs, precisionks, recallks,
                                    fmeasureks, error_analysis_datas):
        content = [projectName, way]
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['top-k', 1, 2, 3, 4, 5]
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = [''] + DataProcessUtils.getAvgScore(topks)
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['mrr-k', 1, 2, 3, 4, 5]
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = [''] + DataProcessUtils.getAvgScore(mrrs)
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['precision-k', 1, 2, 3, 4, 5]
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = [''] + DataProcessUtils.getAvgScore(precisionks)
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['recall-k', 1, 2, 3, 4, 5]
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = [''] + DataProcessUtils.getAvgScore(recallks)
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['F-Measure', 1, 2, 3, 4, 5]
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = [''] + DataProcessUtils.getAvgScore(fmeasureks)
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['']
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        if error_analysis_datas is not None:
            # label = ['recommend_positive_success_pr_ratio', 'recommend_positive_success_time_ratio',
            #          'recommend_negative_success_pr_ratio', 'recommend_negative_success_time_ratio',
            #          'recommend_positive_fail_pr_ratio', 'recommend_positive_fail_time_ratio',
            #          'recommend_negative_fail_pr_ratio', 'recommend_negative_fail_time_ratio']
            label = ['recommend_positive_success_pr_ratio',
                     'recommend_negative_success_pr_ratio',
                     'recommend_positive_fail_pr_ratio',
                     'recommend_negative_fail_pr_ratio']
            for index, data in enumerate(error_analysis_datas):
                content = [label[index], 1, 2, 3, 4, 5]
                ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
                content = [''] + DataProcessUtils.getAvgScore(data)
                ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())

    @staticmethod
    def contactMyRecData(projectName):


        targetFileName_review = f'MyRec_ALL_{projectName}_data_review'
        targetFileName_issue_comment = f'MyRec_ALL_{projectName}_data_issue_comment'
        targetFileName_review_comment = f'MyRec_ALL_{projectName}_data_review_comment'
        targetFileName_commit = f'MyRec_ALL_{projectName}_data_commit'



        issue_comment_path = projectConfig.getIssueCommentPath()
        pull_request_path = projectConfig.getPullRequestPath()
        review_path = projectConfig.getReviewDataPath()
        review_comment_path = projectConfig.getReviewCommentDataPath()
        pr_change_file_path = projectConfig.getPRChangeFilePath()
        pr_commit_path = projectConfig.getPrCommitPath()



        issueCommentData = pandasHelper.readTSVFile(
            os.path.join(issue_comment_path, f'ALL_{projectName}_data_issuecomment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pull_request_path, f'ALL_{projectName}_data_pullrequest.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_path, f'ALL_{projectName}_data_review.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        reviewCommentData = pandasHelper.readTSVFile(
            os.path.join(review_comment_path, f'ALL_{projectName}_data_review_comment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        prChangeFileData = pandasHelper.readTSVFile(
            os.path.join(pr_change_file_path, f'ALL_{projectName}_data_pr_change_file.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )
        """pr_commit"""
        PrCommitData = pandasHelper.readTSVFile(
            os.path.join(pr_commit_path, f'ALL_{projectName}_pr_commit_data.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        data_review = pandas.merge(pullRequestData, reviewData, left_on='number', right_on='pull_number')
        data_review = data_review.loc[data_review['user_login_x'] != data_review['user_login_y']].copy(deep=True)


        data_review = data_review.loc[data_review['closed_at'] >= data_review['submitted_at']].copy(deep=True)


        data_review.dropna(subset=['user_login_y'], inplace=True)


        data_review['isBot'] = data_review['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_review = data_review.loc[data_review['isBot'] == False].copy(deep=True)
        "HG�����У� repo_full_name, number, author_user_login, review_user_login, comment_node_id, pr_created_at, filename"
        data_review = data_review[
            ['number', 'user_login_x', 'created_at', 'user_login_y', 'submitted_at']].copy(
            deep=True)
        data_review.columns = ['pr_number', 'author_user_login', 'pr_created_at', 'review_user_login',
                               'review_created_at']

        data_review.drop_duplicates(inplace=True)


        data_review = pandas.merge(data_review, prChangeFileData, left_on='pr_number', right_on='pull_number')


        data_review.dropna(subset=['author_user_login', 'review_user_login'],
                           inplace=True)


        data_review = data_review[['pr_number', 'author_user_login', 'pr_created_at', 'review_user_login',
                                   'review_created_at', 'filename']].copy(
            deep=True)
        data_review.drop_duplicates(inplace=True)
        data_review.sort_values(by='pr_number', ascending=False, inplace=True)
        data_review.reset_index(drop=True)
        review_pr_number = list(set(data_review['pr_number'].values.tolist()))

        # issue����


        data_issue = pandas.merge(pullRequestData, issueCommentData, left_on='number', right_on='pull_number')


        data_issue = data_issue.loc[data_issue['closed_at'] >= data_issue['created_at_y']].copy(deep=True)
        data_issue = data_issue.loc[data_issue['user_login_x'] != data_issue['user_login_y']].copy(deep=True)

        data_issue.dropna(subset=['user_login_x', 'user_login_y'], inplace=True)


        data_issue['isBot'] = data_issue['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_issue = data_issue.loc[data_issue['isBot'] == False].copy(deep=True)
        data_issue = data_issue[
            ['number', 'user_login_x', 'created_at_x', 'user_login_y', 'created_at_y']].copy(
            deep=True)
        data_issue.columns = ['pr_number', 'author_user_login', 'pr_created_at', 'comment_user_login',
                              'comment_created_at']

        data_issue.drop_duplicates(inplace=True)

        issue_pr_number = list(set(data_issue['pr_number'].values.tolist()))
        data_review["has_issue_comment"] = data_review['pr_number'].apply(lambda x: x in issue_pr_number)
        data_issue["is_review_issue_comment"] = data_issue['pr_number'].apply(lambda x: x in review_pr_number)
        data_issue = data_issue.loc[data_issue['is_review_issue_comment'] == True].copy(deep=True)

        data_no_issue_review = data_review.loc[data_review['has_issue_comment'] == False].copy(deep=True)
        data_empty_issue = DataFrame(columns=data_issue.columns)
        data_empty_issue["pr_number"] = data_no_issue_review["pr_number"]
        data_empty_issue["is_review_issue_comment"] = data_no_issue_review["has_issue_comment"]
        data_empty_issue["pr_created_at"] = data_no_issue_review["pr_created_at"]
        data_empty_issue.drop_duplicates(inplace=True)
        data_issue = pandas.concat([data_issue, data_empty_issue], axis=0)
        data_issue.sort_values(by='pr_number', ascending=False, inplace=True)
        data_issue.drop_duplicates(inplace=True)


        DataProcessUtils.splitDataByMonth(filename=None, targetPath=os.path.join(projectConfig.getMyRecDataPath(),
                                                                                 projectName),
                                          targetFileName=targetFileName_issue_comment, dateCol='pr_created_at',
                                          dataFrame=data_issue)
        # review comment
        data_review_comment = pandas.merge(pullRequestData, reviewCommentData, left_on='number', right_on='pull_number')
        data_review_comment = data_review_comment.loc[
            data_review_comment['user_login_x'] != data_review_comment['user_login_y']].copy(deep=True)
        data_review_comment = data_review_comment.loc[
            data_review_comment['closed_at'] >= data_review_comment['created_at_y']].copy(deep=True)


        data_review_comment.dropna(subset=['user_login_x', 'user_login_y'], inplace=True)


        data_review_comment['isBot'] = data_review_comment['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_review_comment = data_review_comment.loc[data_review_comment['isBot'] == False].copy(deep=True)
        data_review_comment = data_review_comment[
            ['number', 'user_login_x', 'created_at_x', 'user_login_y', 'created_at_y']].copy(
            deep=True)
        data_review_comment.columns = ['pr_number', 'author_user_login', 'pr_created_at', 'review_comment_user_login',
                                       'review_comment_created_at']
        data_review_comment.drop_duplicates(inplace=True)

        review_comment_pr_number = list(set(data_review_comment['pr_number'].values.tolist()))
        data_review_comment["is_review_review_comment"] = data_review_comment['pr_number'].apply(
            lambda x: x in review_pr_number)
        data_review["has_review_comment"] = data_review['pr_number'].apply(lambda x: x in review_comment_pr_number)
        data_review_comment = data_review_comment.loc[data_review_comment['is_review_review_comment'] == True].copy(
            deep=True)

        data_no_review_review = data_review.loc[data_review['has_review_comment'] == False].copy(deep=True)

        data_empty_review = DataFrame(columns=data_review_comment.columns)
        data_empty_review["pr_number"] = data_no_review_review["pr_number"]
        data_empty_review["is_review_issue_comment"] = data_no_review_review["has_review_comment"]
        data_empty_review["pr_created_at"] = data_no_review_review["pr_created_at"]
        data_empty_review.drop_duplicates(inplace=True)
        data_review_comment = pandas.concat([data_review_comment, data_empty_review], axis=0)
        data_review_comment.sort_values(by='pr_number', ascending=False, inplace=True)
        data_review_comment.drop_duplicates(inplace=True)
        data_review_comment.drop(columns='is_review_issue_comment', inplace=True)


        DataProcessUtils.splitDataByMonth(filename=None, targetPath=os.path.join(projectConfig.getMyRecDataPath(),
                                                                                 projectName),
                                          targetFileName=targetFileName_review_comment, dateCol='pr_created_at',
                                          dataFrame=data_review_comment)

        # commit
        data_commit = pandas.merge(pullRequestData, PrCommitData, left_on='number', right_on='pull_number')
        data_commit = data_commit.loc[
            data_commit['closed_at'] >= data_commit['commit_commit_author_date']].copy(deep=True)


        data_commit.dropna(subset=['user_login', 'commit_author_login'], inplace=True)


        data_commit['isBot'] = data_commit['commit_author_login'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_commit = data_commit.loc[data_commit['isBot'] == False].copy(deep=True)
        data_commit = data_commit[
            ['number', 'user_login', 'created_at', 'commit_author_login', 'commit_commit_author_date',
             'commit_status_total', 'commit_status_additions', 'commit_status_deletions']].copy(
            deep=True)
        data_commit.columns = ['pr_number', 'author_user_login', 'pr_created_at', 'commit_user_login',
                               'commit_created_at', 'commit_status_total', 'commit_status_additions',
                               'commit_status_deletions']
        data_commit.sort_values(by='pr_number', ascending=False, inplace=True)
        data_commit.drop_duplicates(inplace=True)

        commit_pr_number = list(set(data_commit['pr_number'].values.tolist()))
        data_commit["is_review_commit"] = data_commit['pr_number'].apply(
            lambda x: x in review_pr_number)
        data_review["has_commit"] = data_review['pr_number'].apply(lambda x: x in commit_pr_number)
        data_commit = data_commit.loc[data_commit['is_review_commit'] == True].copy(
            deep=True)

        data_no_commit_review = data_review.loc[data_review['has_commit'] == False].copy(deep=True)

        data_empty_commit = DataFrame(columns=data_commit.columns)
        data_empty_commit["pr_number"] = data_no_commit_review["pr_number"]
        data_empty_commit["is_review_commit"] = data_no_commit_review["has_commit"]
        data_empty_commit["pr_created_at"] = data_no_commit_review["pr_created_at"]
        data_empty_commit.drop_duplicates(inplace=True)
        data_commit = pandas.concat([data_commit, data_empty_commit], axis=0)
        data_commit.sort_values(by='pr_number', ascending=False, inplace=True)
        data_commit.drop_duplicates(inplace=True)


        DataProcessUtils.splitDataByMonth(filename=None, targetPath=os.path.join(projectConfig.getMyRecDataPath(),
                                                                                 projectName),
                                          targetFileName=targetFileName_commit, dateCol='pr_created_at',
                                          dataFrame=data_commit)


        DataProcessUtils.splitDataByMonth(filename=None, targetPath=os.path.join(projectConfig.getMyRecDataPath(),
                                                                                 projectName),
                                          targetFileName=targetFileName_review, dateCol='pr_created_at',
                                          dataFrame=data_review)

    @staticmethod
    def contactIR_MyRecData(projectName):

        targetFileName_review = f'IR_MyRec_ALL_{projectName}_data_review'
        targetFileName_issue_comment = f'IR_MyRec_ALL_{projectName}_data_issue_comment'
        targetFileName_review_comment = f'IR_MyRec_ALL_{projectName}_data_review_comment'
        targetFileName_commit = f'IR_MyRec_ALL_{projectName}_data_commit'



        issue_comment_path = projectConfig.getIssueCommentPath()
        pull_request_path = projectConfig.getPullRequestPath()
        review_path = projectConfig.getReviewDataPath()
        review_comment_path = projectConfig.getReviewCommentDataPath()
        pr_change_file_path = projectConfig.getPRChangeFilePath()
        pr_commit_path = projectConfig.getPrCommitPath()



        issueCommentData = pandasHelper.readTSVFile(
            os.path.join(issue_comment_path, f'ALL_{projectName}_data_issuecomment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pull_request_path, f'ALL_{projectName}_data_pullrequest.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_path, f'ALL_{projectName}_data_review.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        reviewCommentData = pandasHelper.readTSVFile(
            os.path.join(review_comment_path, f'ALL_{projectName}_data_review_comment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        prChangeFileData = pandasHelper.readTSVFile(
            os.path.join(pr_change_file_path, f'ALL_{projectName}_data_pr_change_file.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )
        """pr_commit"""
        PrCommitData = pandasHelper.readTSVFile(
            os.path.join(pr_commit_path, f'ALL_{projectName}_pr_commit_data.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )
        data_review = pandas.merge(pullRequestData, reviewData, left_on='number', right_on='pull_number')
        data_review = data_review.loc[data_review['user_login_x'] != data_review['user_login_y']].copy(deep=True)


        data_review = data_review.loc[data_review['closed_at'] >= data_review['submitted_at']].copy(deep=True)


        data_review.dropna(subset=['user_login_y'], inplace=True)


        data_review['isBot'] = data_review['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_review = data_review.loc[data_review['isBot'] == False].copy(deep=True)
        "HG�����У� repo_full_name, number, author_user_login, review_user_login, comment_node_id, pr_created_at, filename"
        # data_review = data_review[
        #     ['number', 'user_login_x', 'created_at', 'user_login_y', 'submitted_at']].copy(
        #     deep=True)
        # data_review.columns = ['pr_number', 'author_user_login', 'pr_created_at', 'review_user_login',
        #                        'review_created_at']

        data_review = data_review[
            ['number', 'user_login_x', 'title', 'body_x', 'created_at', 'user_login_y', 'submitted_at']].copy(
            deep=True)
        data_review.columns = ['pr_number', 'author_user_login', 'pr_title', 'pr_body', 'pr_created_at',
                               'review_user_login',
                               'review_created_at']

        data_review.drop_duplicates(inplace=True)


        data_review = pandas.merge(data_review, prChangeFileData, left_on='pr_number', right_on='pull_number')


        data_review.dropna(subset=['author_user_login', 'review_user_login'],
                           inplace=True)


        data_review = data_review[
            ['pr_number', 'author_user_login', 'pr_title', 'pr_body', 'pr_created_at', 'review_user_login',
             'review_created_at', 'filename']].copy(
            deep=True)
        data_review.drop_duplicates(inplace=True)
        data_review.sort_values(by='pr_number', ascending=False, inplace=True)
        data_review.reset_index(drop=True)
        review_pr_number = list(set(data_review['pr_number'].values.tolist()))

        # issue����


        data_issue = pandas.merge(pullRequestData, issueCommentData, left_on='number', right_on='pull_number')


        data_issue = data_issue.loc[data_issue['closed_at'] >= data_issue['created_at_y']].copy(deep=True)
        data_issue = data_issue.loc[data_issue['user_login_x'] != data_issue['user_login_y']].copy(deep=True)

        data_issue.dropna(subset=['user_login_x', 'user_login_y'], inplace=True)


        data_issue['isBot'] = data_issue['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_issue = data_issue.loc[data_issue['isBot'] == False].copy(deep=True)
        # data_issue = data_issue[
        #     ['number', 'user_login_x', 'created_at_x', 'user_login_y', 'created_at_y']].copy(
        #     deep=True)
        # data_issue.columns = ['pr_number', 'author_user_login', 'pr_created_at', 'comment_user_login',
        #                       'comment_created_at']

        data_issue = data_issue[
            ['number', 'user_login_x', 'title', 'body_x', 'created_at_x', 'user_login_y', 'created_at_y']].copy(
            deep=True)
        data_issue.columns = ['pr_number', 'author_user_login', 'pr_title', 'pr_body', 'pr_created_at',
                              'comment_user_login',
                              'comment_created_at']

        data_issue.drop_duplicates(inplace=True)

        issue_pr_number = list(set(data_issue['pr_number'].values.tolist()))
        data_review["has_issue_comment"] = data_review['pr_number'].apply(lambda x: x in issue_pr_number)
        data_issue["is_review_issue_comment"] = data_issue['pr_number'].apply(lambda x: x in review_pr_number)
        data_issue = data_issue.loc[data_issue['is_review_issue_comment'] == True].copy(deep=True)

        data_no_issue_review = data_review.loc[data_review['has_issue_comment'] == False].copy(deep=True)
        data_empty_issue = DataFrame(columns=data_issue.columns)
        data_empty_issue["pr_number"] = data_no_issue_review["pr_number"]
        data_empty_issue["is_review_issue_comment"] = data_no_issue_review["has_issue_comment"]
        data_empty_issue["pr_created_at"] = data_no_issue_review["pr_created_at"]
        data_empty_issue.drop_duplicates(inplace=True)
        data_issue = pandas.concat([data_issue, data_empty_issue], axis=0)
        data_issue.sort_values(by='pr_number', ascending=False, inplace=True)
        data_issue.drop_duplicates(inplace=True)


        DataProcessUtils.splitDataByMonth(filename=None, targetPath=os.path.join(projectConfig.getIR_MyRecDataPath(),
                                                                                 projectName),
                                          targetFileName=targetFileName_issue_comment, dateCol='pr_created_at',
                                          dataFrame=data_issue)
        # review comment
        data_review_comment = pandas.merge(pullRequestData, reviewCommentData, left_on='number', right_on='pull_number')
        data_review_comment = data_review_comment.loc[
            data_review_comment['user_login_x'] != data_review_comment['user_login_y']].copy(deep=True)
        data_review_comment = data_review_comment.loc[
            data_review_comment['closed_at'] >= data_review_comment['created_at_y']].copy(deep=True)


        data_review_comment.dropna(subset=['user_login_x', 'user_login_y'], inplace=True)


        data_review_comment['isBot'] = data_review_comment['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_review_comment = data_review_comment.loc[data_review_comment['isBot'] == False].copy(deep=True)
        # data_review_comment = data_review_comment[
        #     ['number', 'user_login_x', 'created_at_x', 'user_login_y', 'created_at_y']].copy(
        #     deep=True)
        # data_review_comment.columns = ['pr_number', 'author_user_login', 'pr_created_at', 'review_comment_user_login',
        #                                'review_comment_created_at']

        data_review_comment = data_review_comment[
            ['number', 'user_login_x', 'title', 'body_x', 'created_at_x', 'user_login_y', 'created_at_y']].copy(
            deep=True)
        data_review_comment.columns = ['pr_number', 'author_user_login', 'pr_title', 'pr_body', 'pr_created_at',
                                       'review_comment_user_login',
                                       'review_comment_created_at']

        data_review_comment.drop_duplicates(inplace=True)

        review_comment_pr_number = list(set(data_review_comment['pr_number'].values.tolist()))
        data_review_comment["is_review_review_comment"] = data_review_comment['pr_number'].apply(
            lambda x: x in review_pr_number)
        data_review["has_review_comment"] = data_review['pr_number'].apply(lambda x: x in review_comment_pr_number)
        data_review_comment = data_review_comment.loc[data_review_comment['is_review_review_comment'] == True].copy(
            deep=True)

        data_no_review_review = data_review.loc[data_review['has_review_comment'] == False].copy(deep=True)

        data_empty_review = DataFrame(columns=data_review_comment.columns)
        data_empty_review["pr_number"] = data_no_review_review["pr_number"]
        data_empty_review["is_review_issue_comment"] = data_no_review_review["has_review_comment"]
        data_empty_review["pr_created_at"] = data_no_review_review["pr_created_at"]
        data_empty_review.drop_duplicates(inplace=True)
        data_review_comment = pandas.concat([data_review_comment, data_empty_review], axis=0)
        data_review_comment.sort_values(by='pr_number', ascending=False, inplace=True)
        data_review_comment.drop_duplicates(inplace=True)
        data_review_comment.drop(columns='is_review_issue_comment', inplace=True)


        DataProcessUtils.splitDataByMonth(filename=None, targetPath=os.path.join(projectConfig.getIR_MyRecDataPath(),
                                                                                 projectName),
                                          targetFileName=targetFileName_review_comment, dateCol='pr_created_at',
                                          dataFrame=data_review_comment)

        # commit
        data_commit = pandas.merge(pullRequestData, PrCommitData, left_on='number', right_on='pull_number')
        data_commit = data_commit.loc[
            data_commit['closed_at'] >= data_commit['commit_commit_author_date']].copy(deep=True)


        data_commit.dropna(subset=['user_login', 'commit_author_login'], inplace=True)


        data_commit['isBot'] = data_commit['commit_author_login'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_commit = data_commit.loc[data_commit['isBot'] == False].copy(deep=True)
        # data_commit = data_commit[
        #     ['number', 'user_login', 'created_at', 'commit_author_login', 'commit_commit_author_date',
        #      'commit_status_total', 'commit_status_additions', 'commit_status_deletions']].copy(
        #     deep=True)
        # data_commit.columns = ['pr_number', 'author_user_login', 'pr_created_at', 'commit_user_login',
        #                        'commit_created_at', 'commit_status_total', 'commit_status_additions',
        #                        'commit_status_deletions']
        data_commit = data_commit[
            ['number', 'user_login', 'title', 'body', 'created_at', 'commit_author_login', 'commit_commit_author_date',
             'commit_status_total', 'commit_status_additions', 'commit_status_deletions']].copy(
            deep=True)
        data_commit.columns = ['pr_number', 'author_user_login', 'pr_title', 'pr_body', 'pr_created_at',
                               'commit_user_login',
                               'commit_created_at', 'commit_status_total', 'commit_status_additions',
                               'commit_status_deletions']

        data_commit.sort_values(by='pr_number', ascending=False, inplace=True)
        data_commit.drop_duplicates(inplace=True)

        commit_pr_number = list(set(data_commit['pr_number'].values.tolist()))
        data_commit["is_review_commit"] = data_commit['pr_number'].apply(
            lambda x: x in review_pr_number)
        data_review["has_commit"] = data_review['pr_number'].apply(lambda x: x in commit_pr_number)
        data_commit = data_commit.loc[data_commit['is_review_commit'] == True].copy(
            deep=True)

        data_no_commit_review = data_review.loc[data_review['has_commit'] == False].copy(deep=True)

        data_empty_commit = DataFrame(columns=data_commit.columns)
        data_empty_commit["pr_number"] = data_no_commit_review["pr_number"]
        data_empty_commit["is_review_commit"] = data_no_commit_review["has_commit"]
        data_empty_commit["pr_created_at"] = data_no_commit_review["pr_created_at"]
        data_empty_commit.drop_duplicates(inplace=True)
        data_commit = pandas.concat([data_commit, data_empty_commit], axis=0)
        data_commit.sort_values(by='pr_number', ascending=False, inplace=True)
        data_commit.drop_duplicates(inplace=True)


        DataProcessUtils.splitDataByMonth(filename=None, targetPath=os.path.join(projectConfig.getIR_MyRecDataPath(),
                                                                                 projectName),
                                          targetFileName=targetFileName_commit, dateCol='pr_created_at',
                                          dataFrame=data_commit)


        DataProcessUtils.splitDataByMonth(filename=None, targetPath=os.path.join(projectConfig.getIR_MyRecDataPath(),
                                                                                 projectName),
                                          targetFileName=targetFileName_review, dateCol='pr_created_at',
                                          dataFrame=data_review)

    @staticmethod
    def helpGetResult(metric, table, rowStart=513):
        if metric == 'ACC':
            row = table.row(rowx=rowStart)
        elif metric == 'MRR':
            row = table.row(rowx=rowStart + 3)
        elif metric == 'Precision':
            row = table.row(rowx=rowStart + 6)
        elif metric == 'Recall':
            row = table.row(rowx=rowStart + 9)
        elif metric == 'F1_score':
            row = table.row(rowx=rowStart + 12)
        return row

    @staticmethod
    def getResult(method, projects, metric):
        result = []
        kArr = ['1', '3', '5']
        for p in projects:
            line = [p]
            path = os.path.join(os.path.join(projectConfig.getDataRootPath('source code'), method),'result' + os.sep + 'output')

            print(path, p)
            if method == 'AC' or method == 'IR' or method == 'IR_AC' or method == 'CHREV' or method == 'FPS':
                temp_path = os.path.join(path, f'output{method}_{p}_False_False_False.xlsx')
                table = xlrd.open_workbook(temp_path).sheets()[0]
                row = DataProcessUtils.helpGetResult(metric, table)
                for col in kArr:
                    line.append(round(float(str(row[int(col) + 1]).split(":")[1]), 3))
            if method == 'MC':
                temp_path = os.path.join(path, f'output{method}_{p}.xlsx')
                table = xlrd.open_workbook(temp_path).sheets()[0]
                row = DataProcessUtils.helpGetResult(metric, table)
                for col in kArr:
                    line.append(round(float(str(row[int(col) + 1]).split(":")[1]), 3))
            if method == 'FPS_AC':
                temp_path = os.path.join(path, f'output{method}_{p}_False_False_False_test_type_slide.xlsx')
                table = xlrd.open_workbook(temp_path).sheets()[0]
                row = DataProcessUtils.helpGetResult(metric, table)
                for col in kArr:
                    line.append(round(float(str(row[int(col) + 1]).split(":")[1]), 3))
            if method == 'CF':
                temp_path = os.path.join(path, f'output{method}_{p}.xls')
                table = xlrd.open_workbook(temp_path).sheets()[0]
                row = DataProcessUtils.helpGetResult(metric, table)
                for col in kArr:
                    line.append(round(float(str(row[int(col) + 1]).split(":")[1]), 3))
            if method == 'CN':
                temp_path = os.path.join(path, f'output{method}_{p}_False_False_False.xlsx')
                table = xlrd.open_workbook(temp_path).sheets()[0]
                row = DataProcessUtils.helpGetResult(metric, table, rowStart=1053)
                for col in kArr:
                    line.append(round(float(str(row[int(col) + 1]).split(":")[1]), 3))
            if method == 'CN_IR':
                temp_path = os.path.join(path, f'output{method}_{p}_False_False.xls')
                table = xlrd.open_workbook(temp_path).sheets()[0]
                row = DataProcessUtils.helpGetResult(metric, table, rowStart=1053)
                for col in kArr:
                    line.append(round(float(str(row[int(col) + 1]).split(":")[1]), 3))
            if method == 'EAREC':
                temp_path = os.path.join(path, f'output{method}_{p}_False_False.xls')
                table = xlrd.open_workbook(temp_path).sheets()[0]
                row = DataProcessUtils.helpGetResult(metric, table)
                for col in kArr:
                    line.append(round(float(str(row[int(col) + 1]).split(":")[1]), 3))
            # if method == 'HGRec':
            if method == 'HGRecFilter':
                temp_path = os.path.join(path, f'outputHG_{p}_0.9_10_0.8_False_False_False.xlsx')
                table = xlrd.open_workbook(temp_path).sheets()[0]
                row = DataProcessUtils.helpGetResult(metric, table)
                for col in kArr:
                    line.append(round(float(str(row[int(col) + 1]).split(":")[1]), 3))
            if method == 'MyRec':
                temp_path = os.path.join(path, f'outputMyRec_{p}_0.9_10_0.8_False_False_False.xlsx')
                table = xlrd.open_workbook(temp_path).sheets()[0]
                row = DataProcessUtils.helpGetResult(metric, table)
                for col in kArr:
                    line.append(round(float(str(row[int(col) + 1]).split(":")[1]), 3))
            if method == 'MyRecEdit':
                temp_path = os.path.join(path, f'outputMyRecEdit_{p}_0.9_10_0.8_6_1_1_0.4.xlsx')
                table = xlrd.open_workbook(temp_path).sheets()[0]
                row = DataProcessUtils.helpGetResult(metric, table)
                for col in kArr:
                    line.append(round(float(str(row[int(col) + 1]).split(":")[1]), 3))
            result.append(line)

        targetPath = os.path.join(projectConfig.getDataRootPath('source code'),
                                  'result' + os.sep + method + os.sep + f'result_{method}_{metric}.xlsx')
        if not os.path.exists(os.path.join(projectConfig.getDataRootPath('source code'),
                                           'result' + os.sep + method)):
            os.makedirs(os.path.join(projectConfig.getDataRootPath('source code'),
                                     'result' + os.sep + method))
        wbook = xlwt.Workbook()
        wsheet = wbook.add_sheet('sheet1')
        STR_STYLE_NORMAL = 'align: vertical center, horizontal center'
        style = xlwt.easyxf(STR_STYLE_NORMAL)
        wsheet.write_merge(0, 0, 0, 3, metric, style)
        wsheet.write_merge(1, 2, 0, 0, '���ݼ�', style)
        index = 3
        wsheet.write_merge(1, 1, index - 2, index, method, style)
        wsheet.write(2, 1, kArr[0], style)
        wsheet.write(2, 2, kArr[1], style)
        wsheet.write(2, 3, kArr[2], style)
        index_row = 3
        for l in result:
            index_col = 0
            for key in l:
                wsheet.write(index_row, index_col, key, style)
                index_col += 1
            index_row += 1
        try:
            wbook.save(targetPath)
        except Exception as e:
            print(e)


    @staticmethod
    def getResultKnowledge(method, projects, metric):
        result = []
        kArr = ['1', '3', '5']
        for p in projects:
            line = [p]
            path = os.path.join(os.path.join(projectConfig.getDataRootPath('source code'), method),
                                'result' + os.sep + 'output_knowledge')
            if method == 'Sofia':
                path = os.path.join(os.path.join(projectConfig.getDataRootPath('source code'), 'CHREV'),
                                    'result' + os.sep + 'output_knowledge')
            if method == 'Knowledge':
                path = os.path.join(os.path.join(projectConfig.getDataRootPath('source code'), 'KnowledgeRec'),
                                    'result' + os.sep + 'output')
            print(path, p)
            if method == 'AC' or method == 'IR' or method == 'IR_AC' or method == 'CHREV' or method == 'FPS' or method=='Sofia':
                temp_path = os.path.join(path, f'output{method}_{p}_False_False_False.xlsx')
                table = xlrd.open_workbook(temp_path).sheets()[0]
                if method == 'CHREV' or method == 'Sofia':
                    row = DataProcessUtils.helpGetResult(metric, table, rowStart=603)
                elif method == 'AC':
                    row = DataProcessUtils.helpGetResult(metric, table, rowStart=1022)
                else:
                    row = DataProcessUtils.helpGetResult(metric, table)
                for col in kArr:
                    line.append(round(float(str(row[int(col) + 1]).split(":")[1]), 3))
            if method == 'FPS_AC':
                temp_path = os.path.join(path, f'output{method}_{p}_False_False_False_test_type_slide.xlsx')
                table = xlrd.open_workbook(temp_path).sheets()[0]
                row = DataProcessUtils.helpGetResult(metric, table)
                for col in kArr:
                    line.append(round(float(str(row[int(col) + 1]).split(":")[1]), 3))
            if method == 'CF':
                temp_path = os.path.join(path, f'output{method}_{p}.xls')
                table = xlrd.open_workbook(temp_path).sheets()[0]
                row = DataProcessUtils.helpGetResult(metric, table)
                for col in kArr:
                    line.append(round(float(str(row[int(col) + 1]).split(":")[1]), 3))
            if method == 'CN':
                temp_path = os.path.join(path, f'output{method}_{p}_False_False_False.xlsx')
                table = xlrd.open_workbook(temp_path).sheets()[0]
                row = DataProcessUtils.helpGetResult(metric, table, rowStart=1053)
                for col in kArr:
                    line.append(round(float(str(row[int(col) + 1]).split(":")[1]), 3))
            if method == 'CN_IR':
                temp_path = os.path.join(path, f'output{method}_{p}_False_False.xls')
                table = xlrd.open_workbook(temp_path).sheets()[0]
                row = DataProcessUtils.helpGetResult(metric, table, rowStart=1053)
                for col in kArr:
                    line.append(round(float(str(row[int(col) + 1]).split(":")[1]), 3))
            if method == 'EAREC':
                temp_path = os.path.join(path, f'output{method}_{p}_False_False.xls')
                table = xlrd.open_workbook(temp_path).sheets()[0]
                row = DataProcessUtils.helpGetResult(metric, table)
                for col in kArr:
                    line.append(round(float(str(row[int(col) + 1]).split(":")[1]), 3))
            # if method == 'HGRec':
            if method == 'HGRecFilter':
                temp_path = os.path.join(path, f'outputHG_{p}_0.9_10_0.8_False_False_False.xlsx')
                table = xlrd.open_workbook(temp_path).sheets()[0]
                row = DataProcessUtils.helpGetResult(metric, table)
                for col in kArr:
                    line.append(round(float(str(row[int(col) + 1]).split(":")[1]), 3))
            if method == 'MyRec':
                temp_path = os.path.join(path, f'outputMyRec_{p}_0.9_10_0.8_False_False_False.xlsx')
                table = xlrd.open_workbook(temp_path).sheets()[0]
                row = DataProcessUtils.helpGetResult(metric, table)
                for col in kArr:
                    line.append(round(float(str(row[int(col) + 1]).split(":")[1]), 3))
            if method == 'MyRecEdit' or method == 'Knowledge':
                alpha = 0.9
                # temp_path = os.path.join(path, f'outputKnowledge_{p}_True_1_1.xlsx')
                temp_path = os.path.join(path, f'outputKnowledge_{p}_True_{alpha}_{1-alpha}.xlsx')
                table = xlrd.open_workbook(temp_path).sheets()[0]
                row = DataProcessUtils.helpGetResult(metric, table)
                for col in kArr:
                    line.append(round(float(str(row[int(col) + 1]).split(":")[1]), 3))
            result.append(line)

        # targetPath = os.path.join(projectConfig.getDataRootPath('source code'),
        #                           'result_knowledge' + os.sep + method + os.sep + f'result_{method}_{metric}.xlsx')

        targetPath = os.path.join(projectConfig.getDataRootPath('source code'),
                                  'result_knowledge' + os.sep + method + os.sep + f'result_{method}_{metric}_{0.9}.xlsx')
        if not os.path.exists(os.path.join(projectConfig.getDataRootPath('source code'),
                                           'result_knowledge' + os.sep + method)):
            os.makedirs(os.path.join(projectConfig.getDataRootPath('source code'),
                                     'result_knowledge' + os.sep + method))
        wbook = xlwt.Workbook()
        wsheet = wbook.add_sheet('sheet1')
        STR_STYLE_NORMAL = 'align: vertical center, horizontal center'
        style = xlwt.easyxf(STR_STYLE_NORMAL)
        wsheet.write_merge(0, 0, 0, 3, metric, style)
        wsheet.write_merge(1, 2, 0, 0, '���ݼ�', style)
        index = 3
        wsheet.write_merge(1, 1, index - 2, index, method, style)
        wsheet.write(2, 1, kArr[0], style)
        wsheet.write(2, 2, kArr[1], style)
        wsheet.write(2, 3, kArr[2], style)
        index_row = 3
        for l in result:
            index_col = 0
            for key in l:
                wsheet.write(index_row, index_col, key, style)
                index_col += 1
            index_row += 1
        try:
            wbook.save(targetPath)
        except Exception as e:
            print(e)

    @staticmethod
    def contactMyRecEditData(projectName):

        targetFileName_review = f'MyRecEdit_ALL_{projectName}_data_review'
        targetFileName_issue_comment = f'MyRecEdit_ALL_{projectName}_data_issue_comment'
        targetFileName_review_comment = f'MyRecEdit_ALL_{projectName}_data_review_comment'
        targetFileName_commit = f'MyRecEdit_ALL_{projectName}_data_commit'



        issue_comment_path = projectConfig.getIssueCommentPath()
        pull_request_path = projectConfig.getPullRequestPath()
        review_path = projectConfig.getReviewDataPath()
        review_comment_path = projectConfig.getReviewCommentDataPath()
        pr_change_file_path = projectConfig.getPRChangeFilePath()
        pr_commit_path = projectConfig.getPrCommitEditPath()
        commit_file_path = projectConfig.getCommitFilePath()


        issueCommentData = pandasHelper.readTSVFile(
            os.path.join(issue_comment_path, f'ALL_{projectName}_data_issuecomment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pull_request_path, f'ALL_{projectName}_data_pullrequest.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )



        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_path, f'ALL_{projectName}_data_review.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )


        reviewCommentData = pandasHelper.readTSVFile(
            os.path.join(review_comment_path, f'ALL_{projectName}_data_review_comment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        CommitFileData = pandasHelper.readTSVFile(
            os.path.join(commit_file_path, f'ALL_{projectName}_data_commit_file_relation.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )
        CommitFileData = CommitFileData[['file_commit_sha', 'file_filename']].copy(deep=True)
        CommitFileData.columns = ['sha', 'filename']
        CommitFileData.drop_duplicates(inplace=True)
        counts_series = CommitFileData['sha'].value_counts()
        dict_count = {'sha': counts_series.index, 'counts': counts_series.values}
        df_commit_file_counts = pd.DataFrame(dict_count)
        CommitFileData = pd.merge(left=CommitFileData, right=df_commit_file_counts, left_on='sha', right_on='sha')
        CommitFileData = CommitFileData.drop(
            CommitFileData[CommitFileData['counts'] >= DataProcessUtils.MAX_CHANGE_FILES].index)#�����޸��ļ���������100��commit


        """pr_commit"""
        PrCommitData = pandasHelper.readTSVFile(
            os.path.join(pr_commit_path, f'ALL_{projectName}_pr_commit_data.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )
        commit_sha_arr = list(set(CommitFileData['sha'].values.tolist()))
        PrCommitData['label'] = PrCommitData['sha'].apply(lambda x: x in commit_sha_arr)
        PrCommitData = PrCommitData.loc[PrCommitData['label']].copy(deep=True)
        PrCommitData.drop(columns=['label'], inplace=True)


        data_commit = pandas.merge(pullRequestData, PrCommitData, left_on='number', right_on='pull_number')
        data_commit = data_commit.loc[
            data_commit['closed_at'] >= data_commit['commit_commit_author_date']].copy(deep=True)#������ȡ����رպ��
        data_commit['create_before_pr'] = data_commit.apply(lambda x: x['created_at'] >= x['commit_commit_author_date'],
                                                            axis=1)

        df_commit_temp = data_commit[['repo_full_name_x', 'number', 'sha', 'create_before_pr']]
        df_commit_temp.columns = ['repo_full_name', 'pull_number', 'sha', 'create_before_pr']
        prChangeFileData = pd.merge(df_commit_temp, CommitFileData, left_on='sha', right_on='sha')
        prChangeFileData = prChangeFileData[['repo_full_name', 'pull_number', 'filename', 'create_before_pr']].copy(
            deep=True)
        prChangeFileData.columns = ['repo_full_name', 'pull_number', 'filename', 'create_before_pr']

        #

        # prChangeFileData = pandasHelper.readTSVFile(
        #     os.path.join(pr_change_file_path, f'ALL_{projectName}_data_pr_change_file.tsv'),
        #     pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        # )
        pr_file_arr = list(set(prChangeFileData['pull_number'].values.tolist()))
        pullRequestData['hasFileChange'] = pullRequestData['number'].apply(lambda x: x in pr_file_arr)
        pullRequestData = pullRequestData.loc[pullRequestData['hasFileChange']].copy(deep=True)
        pullRequestData.drop(columns=['hasFileChange'], inplace=True)

        # �ҵ�Y �ȷָ�ã�����pr��review��file
        data_review = pandas.merge(pullRequestData, reviewData, left_on='number', right_on='pull_number')
        data_review = data_review.loc[data_review['user_login_x'] != data_review['user_login_y']].copy(deep=True)


        data_review = data_review.loc[data_review['closed_at'] >= data_review['submitted_at']].copy(deep=True)


        data_review.dropna(subset=['user_login_y'], inplace=True)


        data_review['isBot'] = data_review['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_review = data_review.loc[data_review['isBot'] == False].copy(deep=True)
        "HG�����У� repo_full_name, number, author_user_login, review_user_login, comment_node_id, pr_created_at, filename"
        data_review = data_review[
            ['number', 'user_login_x', 'created_at', 'user_login_y', 'submitted_at']].copy(
            deep=True)
        data_review.columns = ['pr_number', 'author_user_login', 'pr_created_at', 'review_user_login',
                               'review_created_at']

        data_review.drop_duplicates(inplace=True)


        data_review = pandas.merge(data_review, prChangeFileData, left_on='pr_number', right_on='pull_number')


        data_review.dropna(subset=['author_user_login', 'review_user_login'],
                           inplace=True)
        data_review = data_review[['pr_number', 'author_user_login', 'pr_created_at', 'review_user_login',
                                   'review_created_at', 'filename', 'create_before_pr']].copy(
            deep=True)

        data_review.drop_duplicates(inplace=True)
        data_review.sort_values(by='pr_number', ascending=False, inplace=True)
        data_review.reset_index(drop=True)
        review_pr_number = list(set(data_review['pr_number'].values.tolist()))

        # issue����


        data_issue = pandas.merge(pullRequestData, issueCommentData, left_on='number', right_on='pull_number')


        data_issue = data_issue.loc[data_issue['closed_at'] >= data_issue['created_at_y']].copy(deep=True)
        data_issue = data_issue.loc[data_issue['user_login_x'] != data_issue['user_login_y']].copy(deep=True)

        data_issue.dropna(subset=['user_login_x', 'user_login_y'], inplace=True)


        data_issue['isBot'] = data_issue['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_issue = data_issue.loc[data_issue['isBot'] == False].copy(deep=True)
        data_issue = data_issue[
            ['number', 'user_login_x', 'created_at_x', 'user_login_y', 'created_at_y']].copy(
            deep=True)
        data_issue.columns = ['pr_number', 'author_user_login', 'pr_created_at', 'comment_user_login',
                              'comment_created_at']

        data_issue.drop_duplicates(inplace=True)

        issue_pr_number = list(set(data_issue['pr_number'].values.tolist()))
        data_review["has_issue_comment"] = data_review['pr_number'].apply(lambda x: x in issue_pr_number)
        data_issue["is_review_issue_comment"] = data_issue['pr_number'].apply(lambda x: x in review_pr_number)
        data_issue = data_issue.loc[data_issue['is_review_issue_comment'] == True].copy(deep=True)

        data_no_issue_review = data_review.loc[data_review['has_issue_comment'] == False].copy(deep=True)
        data_empty_issue = DataFrame(columns=data_issue.columns)
        data_empty_issue["pr_number"] = data_no_issue_review["pr_number"]
        data_empty_issue["is_review_issue_comment"] = data_no_issue_review["has_issue_comment"]
        data_empty_issue["pr_created_at"] = data_no_issue_review["pr_created_at"]
        data_empty_issue.drop_duplicates(inplace=True)
        data_issue = pandas.concat([data_issue, data_empty_issue], axis=0)
        data_issue.sort_values(by='pr_number', ascending=False, inplace=True)
        data_issue.drop_duplicates(inplace=True)



        DataProcessUtils.splitDataByMonth(filename=None, targetPath=os.path.join(projectConfig.getMyRecEditDataPath(),
                                                                                 projectName),
                                          targetFileName=targetFileName_issue_comment, dateCol='pr_created_at',
                                          dataFrame=data_issue)
        # review comment
        data_review_comment = pandas.merge(pullRequestData, reviewCommentData, left_on='number', right_on='pull_number')
        data_review_comment = data_review_comment.loc[
            data_review_comment['user_login_x'] != data_review_comment['user_login_y']].copy(deep=True)
        data_review_comment = data_review_comment.loc[
            data_review_comment['closed_at'] >= data_review_comment['created_at_y']].copy(deep=True)


        data_review_comment.dropna(subset=['user_login_x', 'user_login_y'], inplace=True)


        data_review_comment['isBot'] = data_review_comment['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_review_comment = data_review_comment.loc[data_review_comment['isBot'] == False].copy(deep=True)
        data_review_comment = data_review_comment[
            ['number', 'user_login_x', 'created_at_x', 'user_login_y', 'created_at_y']].copy(
            deep=True)
        data_review_comment.columns = ['pr_number', 'author_user_login', 'pr_created_at', 'review_comment_user_login',
                                       'review_comment_created_at']
        data_review_comment.drop_duplicates(inplace=True)

        review_comment_pr_number = list(set(data_review_comment['pr_number'].values.tolist()))
        data_review_comment["is_review_review_comment"] = data_review_comment['pr_number'].apply(
            lambda x: x in review_pr_number)
        data_review["has_review_comment"] = data_review['pr_number'].apply(lambda x: x in review_comment_pr_number)
        data_review_comment = data_review_comment.loc[data_review_comment['is_review_review_comment'] == True].copy(
            deep=True)

        data_no_review_review = data_review.loc[data_review['has_review_comment'] == False].copy(deep=True)

        data_empty_review = DataFrame(columns=data_review_comment.columns)
        data_empty_review["pr_number"] = data_no_review_review["pr_number"]
        data_empty_review["is_review_issue_comment"] = data_no_review_review["has_review_comment"]
        data_empty_review["pr_created_at"] = data_no_review_review["pr_created_at"]
        data_empty_review.drop_duplicates(inplace=True)
        data_review_comment = pandas.concat([data_review_comment, data_empty_review], axis=0)
        data_review_comment.sort_values(by='pr_number', ascending=False, inplace=True)
        data_review_comment.drop_duplicates(inplace=True)
        data_review_comment.drop(columns='is_review_issue_comment', inplace=True)


        DataProcessUtils.splitDataByMonth(filename=None, targetPath=os.path.join(projectConfig.getMyRecEditDataPath(),
                                                                                 projectName),
                                          targetFileName=targetFileName_review_comment, dateCol='pr_created_at',
                                          dataFrame=data_review_comment)

        # commit
        # data_commit = pandas.merge(pullRequestData, PrCommitData, left_on='number', right_on='pull_number')
        # data_commit = data_commit.loc[
        #     data_commit['created_at'] >= data_commit['commit_commit_author_date']].copy(deep=True)


        data_commit.dropna(subset=['user_login', 'commit_author_login'], inplace=True)


        data_commit['isBot'] = data_commit['commit_author_login'].apply(lambda x: BotUserRecognizer.isBot(x))
        data_commit = data_commit.loc[data_commit['isBot'] == False].copy(deep=True)
        data_commit = data_commit[
            ['number', 'user_login', 'created_at', 'commit_author_login', 'commit_commit_author_date',
             'commit_status_total', 'commit_status_additions', 'commit_status_deletions', 'create_before_pr']].copy(
            deep=True)
        data_commit.columns = ['pr_number', 'author_user_login', 'pr_created_at', 'commit_user_login',
                               'commit_created_at', 'commit_status_total', 'commit_status_additions',
                               'commit_status_deletions', 'create_before_pr']
        data_commit.sort_values(by='pr_number', ascending=False, inplace=True)
        data_commit.drop_duplicates(inplace=True)

        commit_pr_number = list(set(data_commit['pr_number'].values.tolist()))
        data_commit["is_review_commit"] = data_commit['pr_number'].apply(
            lambda x: x in review_pr_number)
        data_review["has_commit"] = data_review['pr_number'].apply(lambda x: x in commit_pr_number)
        data_commit = data_commit.loc[data_commit['is_review_commit'] == True].copy(
            deep=True)

        data_no_commit_review = data_review.loc[data_review['has_commit'] == False].copy(deep=True)

        data_empty_commit = DataFrame(columns=data_commit.columns)
        data_empty_commit["pr_number"] = data_no_commit_review["pr_number"]
        data_empty_commit["is_review_commit"] = data_no_commit_review["has_commit"]
        data_empty_commit["pr_created_at"] = data_no_commit_review["pr_created_at"]
        data_empty_commit.drop_duplicates(inplace=True)
        data_commit = pandas.concat([data_commit, data_empty_commit], axis=0)
        data_commit.sort_values(by='pr_number', ascending=False, inplace=True)
        data_commit.drop_duplicates(inplace=True)



        DataProcessUtils.splitDataByMonth(filename=None, targetPath=os.path.join(projectConfig.getMyRecEditDataPath(),
                                                                                 projectName),
                                          targetFileName=targetFileName_commit, dateCol='pr_created_at',
                                          dataFrame=data_commit)



        DataProcessUtils.splitDataByMonth(filename=None, targetPath=os.path.join(projectConfig.getMyRecEditDataPath(),
                                                                                 projectName),
                                          targetFileName=targetFileName_review, dateCol='pr_created_at',
                                          dataFrame=data_review)




if __name__ == '__main__':

    # DataProcessUtils.fillAlgorithmResultExcelHelper(False, False, True)
    projects = ['akka', 'angular', 'bitcoin', 'cakephp', 'django', 'joomla-cms', 'rails', 'scala',
                'scikit-learn', 'symfony', 'xbmc', ]
    date = (2017, 1, 2020, 6)
    for p in projects[3:4]:
        DataProcessUtils.caculatePrDistance(p, date, filter_change_trigger=False)
    #     DataProcessUtils.splitProjectCommitFileData(p)
    # # DataProcessUtils.contactTCData(p, label=StringKeyUtils.STR_LABEL_ALL_COMMENT)
    # # DataProcessUtils.contactPBData(p, label=StringKeyUtils.STR_LABEL_ALL_COMMENT)
    # # DataProcessUtils.fillAlgorithmResultExcelHelper(False, False, True)
