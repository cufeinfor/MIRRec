import math
import os
import time
from datetime import datetime

from source.config.projectConfig import projectConfig
from source.scikit.FPS.FPSAlgorithm import FPSAlgorithm
from source.utils.pandas.pandasHelper import pandasHelper


class WeightCalculate:
    @staticmethod
    def loadLocalPrDistance(project):

        path = projectConfig.getPullRequestDistancePath()

        prDisDf_LCP = pandasHelper.readTSVFile(path + os.sep +
                                               f"pr_distance_{project}_FPS.tsv",
                                               header=pandasHelper.INT_READ_FILE_WITH_HEAD)

        DisMapLCP = {}
        DisMapLCS = {}
        DisMapLCSubseq = {}
        DisMapLCSubstr = {}
        for row in prDisDf_LCP.itertuples(index=False, name='Pandas'):
            p1 = row[0]
            p2 = row[1]
            dis = row[2]
            DisMapLCP[(p1, p2)] = dis
            DisMapLCP[(p2, p1)] = dis

        return [DisMapLCS, DisMapLCP, DisMapLCSubseq, DisMapLCSubstr]

    @staticmethod
    def getTrainDataPrDistance(train_data, K, pathDict, date, prCreatedTimeMap):
        trainPrDis = {}

        start_time = time.strptime(str(date[0]) + "-" + str(date[1]) + "-" + "01 00:00:00", "%Y-%m-%d %H:%M:%S")
        start_time = int(time.mktime(start_time) - 86400)

        end_time = time.strptime(str(date[2]) + "-" + str(date[3]) + "-" + "01 00:00:00", "%Y-%m-%d %H:%M:%S")
        end_time = int(time.mktime(end_time) - 1)

        print(train_data.shape)
        data = train_data[['pr_number', 'filename']].copy(deep=True)
        data.drop_duplicates(inplace=True)
        data.reset_index(inplace=True, drop=True)
        prList = list(set(data['pr_number']))
        prList.sort()
        scoreMap = {}

        for index, p1 in enumerate(prList):
            scores = {}
            print("now pr:", index, " all:", prList.__len__())
            for p2 in prList:
                if p1 < p2:
                    score = 0
                    paths1 = list(pathDict[p1]['filename'])
                    paths2 = list(pathDict[p2]['filename'])
                    score = 0
                    for filename1 in paths1:
                        for filename2 in paths2:
                            score += FPSAlgorithm.LCP_2(filename1, filename2)
                    score /= paths1.__len__() * paths2.__len__()
                    t1 = prCreatedTimeMap[p1]
                    t2 = prCreatedTimeMap[p2]
                    t = math.fabs(t1 - t2) / (end_time - start_time)
                    score = score * math.exp(-t)

                    scores[p2] = score
                    scoreMap[(p1, p2)] = score
                    scoreMap[(p2, p1)] = score
                elif p1 > p2:
                    score = scoreMap[(p1, p2)]
                    scores[p2] = score

            KNN = [x[0] for x in sorted(scores.items(), key=lambda d: d[1], reverse=True)[0:K]]
            for p2 in KNN:
                trainPrDis[(p1, p2)] = scores[p2]
        return trainPrDis

    @staticmethod
    def buildPrToRevRelation(train_data):
        print("start building request -> reviewer relations....")
        start = datetime.now()

        pr_created_time_data = train_data['pr_created_at'].apply(
            lambda x: time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
        start_time = min(pr_created_time_data.to_list())

        pr_created_time_data = train_data['pr_created_at'].apply(
            lambda x: time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
        end_time = max(pr_created_time_data.to_list())
        prToRevMat = {}
        grouped_train_data = train_data.groupby([train_data['pr_number'], train_data['review_user_login']])
        max_weight = 0
        for relation, group in grouped_train_data:
            group.reset_index(drop=True, inplace=True)
            weight = WeightCalculate.caculateRevToPrWeight(group, start_time, end_time)
            max_weight = max(weight, max_weight)
            if not prToRevMat.__contains__(relation[0]):
                prToRevMat[relation[0]] = {}
            prToRevMat[relation[0]][relation[1]] = weight

        for pr, relations in prToRevMat.items():
            for rev, weight in relations.items():
                prToRevMat[pr][rev] = weight / max_weight
        return prToRevMat

    @staticmethod
    def caculateRevToPrWeight(comment_records, start_time, end_time):

        weight_lambda = 0.8
        weight = 0
        comment_records = comment_records.copy(deep=True)
        comment_records.drop(columns=['filename'], inplace=True)
        comment_records.drop_duplicates(inplace=True)
        comment_records.reset_index(inplace=True, drop=True)

        for cm_idx, cm_row in comment_records.iterrows():
            cm_timestamp = time.strptime(cm_row['review_created_at'], "%Y-%m-%d %H:%M:%S")
            cm_timestamp = int(time.mktime(cm_timestamp))

            t = (cm_timestamp - start_time) / (end_time - start_time)
            cm_weight = math.pow(weight_lambda, cm_idx) * math.exp(t - 1)
            weight += cm_weight
        return weight

    @staticmethod
    def buildAuthToPrRelation(train_data, date):
        start = datetime.now()

        start_time = time.strptime(str(date[0]) + "-" + str(date[1]) + "-" + "01 00:00:00", "%Y-%m-%d %H:%M:%S")
        start_time = int(time.mktime(start_time) - 86400)

        end_time = time.strptime(str(date[2]) + "-" + str(date[3]) + "-" + "01 00:00:00", "%Y-%m-%d %H:%M:%S")
        end_time = int(time.mktime(end_time) - 1)

        authToRrMat = {}
        grouped_train_data = train_data.groupby([train_data['author_user_login'], train_data['pr_number']])
        max_weight = 0
        for relation, group in grouped_train_data:
            group.reset_index(drop=True, inplace=True)
            weight = WeightCalculate.caculateAuthToPrWeight(group, start_time, end_time)
            max_weight = max(weight, max_weight)
            if not authToRrMat.__contains__(relation[0]):
                authToRrMat[relation[0]] = {}
            authToRrMat[relation[0]][relation[1]] = weight

        for auth, relations in authToRrMat.items():
            for rev, weight in relations.items():
                authToRrMat[auth][rev] = weight / max_weight
        return authToRrMat

    @staticmethod
    def caculateAuthToPrWeight(comment_records, start_time, end_time):

        weight_lambda = 0.8
        weight = 0

        comment_records = comment_records.copy(deep=True)
        comment_records.drop(columns=['filename'], inplace=True)
        comment_records.drop_duplicates(inplace=True)
        comment_records.reset_index(inplace=True, drop=True)

        for cm_idx, cm_row in comment_records.iterrows():
            cm_timestamp = time.strptime(cm_row['pr_created_at'], "%Y-%m-%d %H:%M:%S")
            cm_timestamp = int(time.mktime(cm_timestamp))

            t = (cm_timestamp - start_time) / (end_time - start_time)
            cm_weight = math.pow(weight_lambda, cm_idx) * t
            weight += cm_weight
            if weight > 1:
                print("some thing errors....")
            break

        return weight

    @staticmethod
    def buildCommitToPrRelation(train_data_commit, date):
        start = datetime.now()

        start_time = time.strptime(str(date[0]) + "-" + str(date[1]) + "-" + "01 00:00:00", "%Y-%m-%d %H:%M:%S")
        start_time = int(time.mktime(start_time) - 86400)

        end_time = time.strptime(str(date[2]) + "-" + str(date[3]) + "-" + "01 00:00:00", "%Y-%m-%d %H:%M:%S")
        end_time = int(time.mktime(end_time) - 1)

        commitToRrMat = {}
        grouped_train_data = train_data_commit.groupby(
            [train_data_commit['pr_number'], train_data_commit['commit_user_login']])
        max_weight = 0
        for relation, group in grouped_train_data:
            group.reset_index(drop=True, inplace=True)
            weight = WeightCalculate.caculateCommitToPrWeight(group, start_time, end_time)
            max_weight = max(weight, max_weight)
            if not commitToRrMat.__contains__(relation[0]):
                commitToRrMat[relation[0]] = {}
            commitToRrMat[relation[0]][relation[1]] = weight

        for auth, relations in commitToRrMat.items():
            for rev, weight in relations.items():
                commitToRrMat[auth][rev] = weight / max_weight
        return commitToRrMat

    @staticmethod
    def caculateCommitToPrWeight(commit_records, start_time, end_time):
        weight_lambda = 0.8
        weight = 0
        commit_records = commit_records.copy(deep=True)
        commit_records.drop_duplicates(inplace=True)
        commit_records.reset_index(inplace=True, drop=True)

        for cm_idx, cm_row in commit_records.iterrows():
            cm_timestamp = time.strptime(cm_row['commit_created_at'], "%Y-%m-%d %H:%M:%S")
            cm_timestamp = int(time.mktime(cm_timestamp))

            code_lines_total = int(cm_row['commit_status_total'])
            code_lines_total = max(1, code_lines_total)
            code_line_canshu = 1 / (1 + math.exp(-code_lines_total * 0.01))

            t = math.fabs(cm_timestamp - start_time) / (end_time - start_time)
            cm_weight = math.pow(weight_lambda, cm_idx) * t * code_line_canshu
            weight += cm_weight
        return weight

    @staticmethod
    def buildPrToIssueCommentRelation(train_data_issue_comment):

        start = datetime.now()

        pr_created_time_data = train_data_issue_comment['pr_created_at'].apply(
            lambda x: time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
        start_time = min(pr_created_time_data.to_list())

        pr_created_time_data = train_data_issue_comment['pr_created_at'].apply(
            lambda x: time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
        end_time = max(pr_created_time_data.to_list())
        commentToPrMat = {}
        train_data = train_data_issue_comment.loc[train_data_issue_comment['comment_user_login'] != ''].copy(deep=True)
        grouped_train_data = train_data.groupby([train_data['pr_number'], train_data['comment_user_login']])
        max_weight = 0
        for relation, group in grouped_train_data:
            group.reset_index(drop=True, inplace=True)
            weight = WeightCalculate.caculateIssueCommentToPrWeight(group, start_time, end_time)
            max_weight = max(weight, max_weight)
            if not commentToPrMat.__contains__(relation[0]):
                commentToPrMat[relation[0]] = {}
            commentToPrMat[relation[0]][relation[1]] = weight

        for pr, relations in commentToPrMat.items():
            for rev, weight in relations.items():
                commentToPrMat[pr][rev] = weight / max_weight
        return commentToPrMat

    @staticmethod
    def caculateIssueCommentToPrWeight(comment_records, start_time, end_time):

        weight_lambda = 0.8
        weight = 0
        comment_records = comment_records.copy(deep=True)
        comment_records.drop_duplicates(inplace=True)
        comment_records.reset_index(inplace=True, drop=True)

        for cm_idx, cm_row in comment_records.iterrows():
            cm_timestamp = time.strptime(cm_row['comment_created_at'], "%Y-%m-%d %H:%M:%S")
            cm_timestamp = int(time.mktime(cm_timestamp))

            t = (cm_timestamp - start_time) / (end_time - start_time)
            cm_weight = math.pow(weight_lambda, cm_idx) * math.exp(t - 1)
            weight += cm_weight
        return weight

    @staticmethod
    def buildPrToReviewCommentRelation(train_data_review_comment):
        start = datetime.now()

        pr_created_time_data = train_data_review_comment['pr_created_at'].apply(
            lambda x: time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
        start_time = min(pr_created_time_data.to_list())

        pr_created_time_data = train_data_review_comment['pr_created_at'].apply(
            lambda x: time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
        end_time = max(pr_created_time_data.to_list())
        commentToPrMat = {}
        train_data = train_data_review_comment.loc[train_data_review_comment['review_comment_user_login'] != ''].copy(
            deep=True)
        grouped_train_data = train_data.groupby([train_data['pr_number'], train_data['review_comment_user_login']])
        max_weight = 0
        for relation, group in grouped_train_data:
            group.reset_index(drop=True, inplace=True)
            weight = WeightCalculate.caculateReviewCommentToPrWeight(group, start_time, end_time)
            max_weight = max(weight, max_weight)
            if not commentToPrMat.__contains__(relation[0]):
                commentToPrMat[relation[0]] = {}
            commentToPrMat[relation[0]][relation[1]] = weight

        for pr, relations in commentToPrMat.items():
            for rev, weight in relations.items():
                commentToPrMat[pr][rev] = weight / max_weight
        return commentToPrMat

    @staticmethod
    def caculateReviewCommentToPrWeight(comment_records, start_time, end_time):

        weight_lambda = 0.8
        weight = 0
        comment_records = comment_records.copy(deep=True)
        comment_records.drop_duplicates(inplace=True)
        comment_records.reset_index(inplace=True, drop=True)

        for cm_idx, cm_row in comment_records.iterrows():
            cm_timestamp = time.strptime(cm_row['review_comment_created_at'], "%Y-%m-%d %H:%M:%S")
            cm_timestamp = int(time.mktime(cm_timestamp))

            t = (cm_timestamp - start_time) / (end_time - start_time)
            cm_weight = math.pow(weight_lambda, cm_idx) * math.exp(t - 1)
            weight += cm_weight
        return weight
