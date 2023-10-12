import math
import os
import pickle
import time
from datetime import datetime
from MyRecEdit.Method import Method
from MyRecEdit.Utils import Utils
from HyperGraphHelper import HyperGraphHelper
from Node import Node

from source.scikit.service.DataProcessUtils import DataProcessUtils
from source.utils.ExcelHelper import ExcelHelper


class MyRecEditTrain:

    @staticmethod
    def TestAlgorithm(project, dates, alpha=0.8, K=20, c=1, TrainPRDisIsComputed=False, HyperGraphIsCreated=False,
                      re=0.25, ct=0.25, ic=0.25, rc=0.25):
        start_time = datetime.now()
        recommendNum = 5
        excelName = f'test/outputMyRecEdit_{re}_{ct}_{ic}_{rc}.xls'
        sheetName = 'result'

        topks = []
        mrrs = []
        precisionks = []
        recallks = []
        fmeasureks = []
        error_analysis_datas = None

        ExcelHelper().initExcelFile(fileName=excelName, sheetName=sheetName, excel_key_list=['train', 'test'])
        for date in dates:
            startTime = datetime.now()
            recommendList, answerList, prList, convertDict, trainSize = MyRecEditTrain.algorithmBody(date, project,
                                                                                                     recommendNum,
                                                                                                     alpha=alpha, K=K,
                                                                                                     c=c,
                                                                                                     TrainPRDisIsComputed=TrainPRDisIsComputed,
                                                                                                     HyperGraphIsCreated=HyperGraphIsCreated,
                                                                                                     re=re, ct=ct,
                                                                                                     ic=ic,
                                                                                                     rc=rc)

            topk, mrr, precisionk, recallk, fmeasurek = \
                DataProcessUtils.judgeRecommend(recommendList, answerList, recommendNum)

            topks.append(topk)
            mrrs.append(mrr)
            precisionks.append(precisionk)
            recallks.append(recallk)
            fmeasureks.append(fmeasurek)

            error_analysis_data = None
            DataProcessUtils.saveResult(excelName, sheetName, topk, mrr, precisionk, recallk, fmeasurek, date,
                                        error_analysis_data)
            content = ['']
            ExcelHelper().appendExcelRow(excelName, sheetName, content, style=ExcelHelper.getNormalStyle())
            content = ['train', 'test']
            ExcelHelper().appendExcelRow(excelName, sheetName, content, style=ExcelHelper.getNormalStyle())
            endTime = datetime.now()
            print("cost time:", endTime - startTime)
            obj = {
                'start': startTime,
                'end': endTime,
                'sub': endTime - startTime,
                'project': project,
                'date': date,
                'K': K
            }
            with open(
                    f'time/{project}_{date[0]}_{date[1]}_{date[2]}_{date[3]}_{K}.pkl',
                    'wb') as f:
                pickle.dump(obj, f)

        DataProcessUtils.saveFinallyResult(excelName, sheetName, topks, mrrs, precisionks, recallks,
                                           fmeasureks, error_analysis_datas)
        end_time = datetime.now()
        print("cost time:", end_time - start_time)

    @staticmethod
    def algorithmBody(date, project, recommendNum=5, alpha=0.98, K=20, c=1, TrainPRDisIsComputed=False,
                      HyperGraphIsCreated=False,
                      re=0.25, ct=0.25, ic=0.25, rc=0.25):

        df_review, df_commit, df_issue_comment, df_review_comment = Utils.getOriginData(project, date)
        train_data, train_data_commit, train_data_issue_comment, train_data_review_comment, train_data_y, \
        train_data_y_workload, train_data_committer, train_data_issue_commenter, train_data_review_commenter, \
        test_data, test_data_commit, test_data_y, test_data_y_workload, test_data_committer, convertDict, \
        reviewConvertDict, df_review, df_review_comment, df_commit, df_issue_comment = \
            Utils.preProcess(df_review, df_commit, df_issue_comment, df_review_comment, date)

        prList = list(set(test_data['pr_number']))
        prList.sort()

        recommendList, answerList, authorList = MyRecEditTrain.RecommendByMyRecEdit(train_data,
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

        return recommendList, answerList, prList, convertDict, trainSize

    @staticmethod
    def RecommendByMyRecEdit(train_data, train_data_commit, train_data_issue_comment,
                             train_data_review_comment, train_data_y, train_data_y_workload,
                             train_data_committer, train_data_issue_commenter, train_data_review_commenter,
                             test_data, test_data_commit, test_data_y,
                             test_data_y_workload, test_data_committer, date,
                             project, convertDict, recommendNum=5,
                             K=20, alpha=0.8, c=1,
                             TrainPRDisIsComputed=False,
                             HyperGraphIsCreated=False
                             , re=0.25, ct=0.25, ic=0.25, rc=0.25):
        recommendList = []
        answerList = []
        testDict = dict(list(test_data.groupby('pr_number')))
        authorList = []
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

        print(" pr distance cost time:", datetime.now() - start)

        tempData = train_data[['pr_number', 'review_user_login']].copy(deep=True)
        tempData.drop_duplicates(inplace=True)
        reviewerFreqDict = {}
        for r, temp_df in dict(list(tempData.groupby('review_user_login'))).items():
            reviewerFreqDict[r] = temp_df.shape[0]
        prList = list(set(train_data['pr_number']))
        prList.sort()
        prList = tuple(prList)
        pr_created_time_data = train_data['pr_created_at'].apply(
            lambda x: time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
        start_time = min(pr_created_time_data.to_list())

        pr_created_time_data = train_data['pr_created_at'].apply(
            lambda x: time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
        end_time = max(pr_created_time_data.to_list())
        graph = HyperGraphHelper.SaveAndGetGraph(K, c, project, date, train_data, train_data_commit,
                                                 train_data_issue_comment,
                                                 train_data_review_comment, train_data_y, train_data_committer,
                                                 train_data_issue_commenter, train_data_review_commenter, pathDict,
                                                 prCreatedTimeMap,
                                                 HyperGraphIsCreated, TrainPRDisIsComputed, AddSelfChargeData=False)

        startTime = datetime.now()
        pos = 0
        now = datetime.now()
        inverseNodeMap = {k: v for v, k in graph.node_id_map.items()}

        reviewer_node_list = []
        reviewer_login_list = []
        for i in range(0, graph.num_nodes):
            node_id = graph.node_id_map[i]
            node = graph.get_node_by_key(node_id)
            if node.type == Node.STR_NODE_TYPE_REVIEWER and reviewerFreqDict[node.contentKey] >= 2:
                reviewer_node_list.append((node, inverseNodeMap[node.id]))
                reviewer_login_list.append(node.contentKey)
        reviewer_login_list = list(set(reviewer_login_list))

        committer_node_list = []
        issue_commenter_node_list = []
        review_commenter_node_list = []
        for i in range(0, graph.num_nodes):
            node_id = graph.node_id_map[i]
            node = graph.get_node_by_key(node_id)
            if node.contentKey in reviewer_login_list:
                if node.type == Node.STR_NODE_TYPE_COMMITTER:
                    committer_node_list.append((node, inverseNodeMap[node.id]))
                if node.type == Node.STR_NODE_TYPE_ISSUE_COMMENTER:
                    issue_commenter_node_list.append((node, inverseNodeMap[node.id]))
                if node.type == Node.STR_NODE_TYPE_REVIEW_COMMENTER:
                    review_commenter_node_list.append((node, inverseNodeMap[node.id]))
        prNumList = []
        for test_pull_number, test_df in testDict.items():
            print('*' * 200)
            print(project, date, f"now:{pos}/{testDict.items().__len__()}",
                  f'alpha:{alpha},re:{re},ct:{ct},ic:{ic},rc:{rc}', 'cost time:', datetime.now() - now)
            test_df.reset_index(drop=True, inplace=True)
            pos += 1

            pr_num = list(test_df['pr_number'])[0]
            prNumList.append(pr_num)
            paths2 = list(set(test_df['filename']))
            answer = test_data_y[pr_num]
            answerList.append(answer)
            author = test_df['author_user_login'][0]
            authorList.append(author)
            scores = Method.MultiRelationMethod(graph, pr_num, prList, pathDict, paths2, prCreatedTimeMap, start_time,
                                                end_time, author,
                                                reviewer_node_list, issue_commenter_node_list,
                                                review_commenter_node_list, committer_node_list, re, ct, ic, rc, K,
                                                alpha)

            reverseConvertDict = {v: k for k, v in convertDict.items()}
            recommend_list = [(reverseConvertDict[x[0]], x[1]) for x in
                              sorted(scores.items(), key=lambda d: d[1], reverse=True)]
            print(recommend_list)

            recommend_list = [x[0] for x in sorted(scores.items(),
                                                   key=lambda d: d[1], reverse=True)[0:recommendNum]]
            recommendList.append(recommend_list)

        return recommendList, answerList, authorList


