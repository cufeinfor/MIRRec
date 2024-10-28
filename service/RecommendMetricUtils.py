# coding=gbk
import math


class RecommendMetricUtils:
    """�����Ƽ���׼ȷ�ʵĹ�����"""

    @staticmethod
    def topKAccuracy(recommendCase, answerCase, k):
        """top k Accuracy  �������Ϊ���Ԥ���ʵ�ʵ��Ƽ��б���б�"""
        topK = [0 for i in range(k)]
        if recommendCase.__len__() != answerCase.__len__():
            raise Exception("case is not right")
        casePos = 0
        for recommendList in recommendCase:
            # print("recommend:", recommendList)
            answerList = answerCase[casePos]
            # print("answerList:", answerList)
            casePos += 1
            listPos = 0
            firstFind = False
            for recommend in recommendList:
                if firstFind:
                    break
                index = -1
                try:
                    index = answerList.index(recommend)
                except Exception as e:
                    pass
                if index != -1:
                    for i in range(listPos, k):
                        topK[i] += 1
                    firstFind = True
                    break
                else:
                    listPos += 1
        for i in range(0, topK.__len__()):
            if recommendCase.__len__() == 0:
                topK[i] = 0
            else:
                topK[i] /= recommendCase.__len__()
        return topK

    @staticmethod
    def MRR(recommendCase, answerCase, k=5):
        """MRR �������Ϊ���Ԥ���ʵ�ʵ��Ƽ��б���б�"""
        MMR = [0 for x in range(0, k)]
        for i in range(0, k):
            totalScore = 0
            if recommendCase.__len__() != answerCase.__len__():
                raise Exception("case is not right")
            casePos = 0
            for recommendList in recommendCase:
                # print("recommend:", recommendList)
                answerList = answerCase[casePos]
                # print("answerList:", answerList)
                recommendList = recommendList[0:i + 1]
                casePos += 1
                listPos = 0
                firstFind = False
                for recommend in recommendList:
                    if firstFind:
                        break
                    index = -1
                    try:
                        index = answerList.index(recommend)
                    except Exception as e:
                        pass
                    if index != -1:
                        totalScore += 1.0 / (listPos + 1)  # ��һ����ȷ��Ա����������
                        firstFind = True
                        break
                    else:
                        listPos += 1
                # print("score:", totalScore)
            if recommendCase.__len__() == 0:
                MMR[i] = 0
            else:
                MMR[i] = totalScore / recommendCase.__len__()
            # MMR[i] = totalScore / recommendCase.__len__()
        return MMR

    @staticmethod
    def precisionK(recommendCase, answerCase, k=5):
        """top k precision
           top k recall
           top k f-measure
           �������Ϊ���Ԥ���ʵ�ʵ��Ƽ��б���б�"""
        precisonk = [0 for x in range(0, k)]
        recallk = [0 for x in range(0, k)]
        fmeasurek = [0 for x in range(0, k)]
        if recommendCase.__len__() != answerCase.__len__():
            raise Exception("case is not right")

        for i in range(0, k):
            totalPrecisionScore = 0
            totalRecallScore = 0
            casePos = 0
            for recommendList in recommendCase:
                answerList = answerCase[casePos]
                recommendList = recommendList[0:i + 1]
                casePos += 1

                precision = [x for x in recommendList if x in answerList].__len__() / recommendList.__len__()
                recall = [x for x in answerList if x in recommendList].__len__() / answerList.__len__()
                totalPrecisionScore += precision
                totalRecallScore += recall
            totalPrecisionScore /= recommendCase.__len__()
            totalRecallScore /= recommendCase.__len__()
            if totalPrecisionScore + totalRecallScore > 0:
                fmeasure = (2 * totalPrecisionScore * totalRecallScore) / (totalRecallScore + totalPrecisionScore)
            else:
                fmeasure = 0
            precisonk[i] = totalPrecisionScore
            recallk[i] = totalRecallScore
            fmeasurek[i] = fmeasure
        return precisonk, recallk, fmeasurek

    @staticmethod
    def positiveSuccess(recommendCase, answerCase, filterAnswerCase, k=5):
        """�������Ϊ���Ԥ���ʵ�ʵ��Ƽ��б���б�"""
        positive_success_pr_ratio_k = [0 for x in range(0, k)]
        positive_success_time_ratio_k = [0 for x in range(0, k)]
        if recommendCase.__len__() != answerCase.__len__() and answerCase.__len__() != filterAnswerCase.__len__():
            raise Exception("case is not right")

        for i in range(0, k):
            positive_success_pr_ratio = 0
            positive_success_time_ratio = 0
            for index, recommendList in enumerate(recommendCase):
                answerList = answerCase[index]
                filterAnswerList = filterAnswerCase[index]
                recommendList = recommendList[0:i + 1]
                positive_success_list = [x for x in recommendList if (x in answerList and x in filterAnswerList)]
                positive_success_pr_ratio += (positive_success_list.__len__() >= 1)
                positive_success_time_ratio += positive_success_list.__len__()

            positive_success_pr_ratio /= recommendCase.__len__()
            positive_success_time_ratio /= recommendCase.__len__() * (i + 1)
            positive_success_pr_ratio_k[i] = positive_success_pr_ratio
            positive_success_time_ratio_k[i] = positive_success_time_ratio
        return positive_success_pr_ratio_k, positive_success_time_ratio_k

    @staticmethod
    def negativeSuccess(recommendCase, answerCase, filterAnswerCase, k=5):
        """�������Ϊ���Ԥ���ʵ�ʵ��Ƽ��б���б�"""
        negative_success_pr_ratio_k = [0 for x in range(0, k)]
        negative_success_time_ratio_k = [0 for x in range(0, k)]
        if recommendCase.__len__() != answerCase.__len__() and answerCase.__len__() != filterAnswerCase.__len__():
            raise Exception("case is not right")

        for i in range(0, k):
            negative_success_pr_ratio = 0
            negative_success_time_ratio = 0
            for index, recommendList in enumerate(recommendCase):
                answerList = answerCase[index]
                filterAnswerList = filterAnswerCase[index]
                recommendList = recommendList[0:i + 1]
                negative_success_list = [x for x in recommendList if (x in answerList and x not in filterAnswerList)]
                negative_success_pr_ratio += (negative_success_list.__len__() >= 1)
                negative_success_time_ratio += negative_success_list.__len__()

            negative_success_pr_ratio /= recommendCase.__len__()
            negative_success_time_ratio /= recommendCase.__len__() * (i + 1)
            negative_success_pr_ratio_k[i] = negative_success_pr_ratio
            negative_success_time_ratio_k[i] = negative_success_time_ratio
        return negative_success_pr_ratio_k, negative_success_time_ratio_k

    @staticmethod
    def positiveFail(recommendCase, answerCase, filterAnswerCase, k=5):
        """�������Ϊ���Ԥ���ʵ�ʵ��Ƽ��б���б�"""
        positive_fail_pr_ratio_k = [0 for x in range(0, k)]
        positive_fail_time_ratio_k = [0 for x in range(0, k)]
        if recommendCase.__len__() != answerCase.__len__() and answerCase.__len__() != filterAnswerCase.__len__():
            raise Exception("case is not right")

        for i in range(0, k):
            positive_fail_pr_ratio = 0
            positive_fail_time_ratio = 0
            for index, recommendList in enumerate(recommendCase):
                answerList = answerCase[index]
                filterAnswerList = filterAnswerCase[index]
                recommendList = recommendList[0:i + 1]
                positive_fail_list = [x for x in recommendList if
                                      (x not in answerList and filterAnswerList.__len__() > 0)]
                positive_fail_pr_ratio += (positive_fail_list.__len__() >= 1)
                positive_fail_time_ratio += positive_fail_list.__len__()

            positive_fail_pr_ratio /= recommendCase.__len__()
            positive_fail_time_ratio /= recommendCase.__len__() * (i + 1)
            positive_fail_pr_ratio_k[i] = positive_fail_pr_ratio
            positive_fail_time_ratio_k[i] = positive_fail_time_ratio
        return positive_fail_pr_ratio_k, positive_fail_time_ratio_k

    @staticmethod
    def negativeFail(recommendCase, answerCase, filterAnswerCase, k=5):
        """�������Ϊ���Ԥ���ʵ�ʵ��Ƽ��б���б�"""
        negative_fail_pr_ratio_k = [0 for x in range(0, k)]
        negative_fail_time_ratio_k = [0 for x in range(0, k)]
        if recommendCase.__len__() != answerCase.__len__() and answerCase.__len__() != filterAnswerCase.__len__():
            raise Exception("case is not right")

        for i in range(0, k):
            negative_fail_pr_ratio = 0
            negative_fail_time_ratio = 0
            for index, recommendList in enumerate(recommendCase):
                answerList = answerCase[index]
                filterAnswerList = filterAnswerCase[index]
                recommendList = recommendList[0:i + 1]
                negative_fail_list = [x for x in recommendList if
                                      (x not in answerList and filterAnswerList.__len__() == 0)]
                negative_fail_pr_ratio += (negative_fail_list.__len__() >= 1)
                negative_fail_time_ratio += negative_fail_list.__len__()

            negative_fail_pr_ratio /= recommendCase.__len__()
            negative_fail_time_ratio /= recommendCase.__len__() * (i + 1)
            negative_fail_pr_ratio_k[i] = negative_fail_pr_ratio
            negative_fail_time_ratio_k[i] = negative_fail_time_ratio
        return negative_fail_pr_ratio_k, negative_fail_time_ratio_k

    @staticmethod
    def positiveSuccess2(recommendCase, answerCase, filterAnswerCase, k=5):
        """�������Ϊ���Ԥ���ʵ�ʵ��Ƽ��б���б�
           �ж������޸�
        """
        positive_success_pr_ratio_k = [0 for x in range(0, k)]
        if recommendCase.__len__() != answerCase.__len__() and answerCase.__len__() != filterAnswerCase.__len__():
            raise Exception("case is not right")

        for i in range(0, k):
            positive_success_pr_ratio = 0
            for index, recommendList in enumerate(recommendCase):
                answerList = answerCase[index]
                filterAnswerList = filterAnswerCase[index]
                recommendList = recommendList[0:i + 1]
                positive_success_list = [x for x in recommendList if (x in answerList and x in filterAnswerList)]
                positive_success_pr_ratio += (positive_success_list.__len__() >= 1)

            positive_success_pr_ratio /= recommendCase.__len__()
            positive_success_pr_ratio_k[i] = positive_success_pr_ratio
        return positive_success_pr_ratio_k

    @staticmethod
    def negativeSuccess2(recommendCase, answerCase, filterAnswerCase, k=5):
        """�������Ϊ���Ԥ���ʵ�ʵ��Ƽ��б���б�"""
        negative_success_pr_ratio_k = [0 for x in range(0, k)]
        if recommendCase.__len__() != answerCase.__len__() and answerCase.__len__() != filterAnswerCase.__len__():
            raise Exception("case is not right")

        for i in range(0, k):
            negative_success_pr_ratio = 0
            for index, recommendList in enumerate(recommendCase):
                answerList = answerCase[index]
                filterAnswerList = filterAnswerCase[index]
                recommendList = recommendList[0:i + 1]
                # ����  �Ƽ��б����������ϴ���
                # ����
                if [x for x in recommendList if x in answerList].__len__() > 0:
                    negative_success_list = [x for x in recommendList if (x in answerList and x in filterAnswerList)]
                    negative_success_pr_ratio += (negative_success_list.__len__() == 0)

            negative_success_pr_ratio /= recommendCase.__len__()
            negative_success_pr_ratio_k[i] = negative_success_pr_ratio
        return negative_success_pr_ratio_k

    @staticmethod
    def positiveFail2(recommendCase, answerCase, filterAnswerCase, k=5):
        """�������Ϊ���Ԥ���ʵ�ʵ��Ƽ��б���б�"""
        positive_fail_pr_ratio_k = [0 for x in range(0, k)]
        if recommendCase.__len__() != answerCase.__len__() and answerCase.__len__() != filterAnswerCase.__len__():
            raise Exception("case is not right")

        for i in range(0, k):
            positive_fail_pr_ratio = 0
            for index, recommendList in enumerate(recommendCase):
                answerList = answerCase[index]
                filterAnswerList = filterAnswerCase[index]
                recommendList = recommendList[0:i + 1]
                positive_fail_list = [x for x in recommendList if x in answerList]
                """һ����û����"""
                if positive_fail_list.__len__() == 0 and filterAnswerList.__len__() > 0:
                    positive_fail_pr_ratio += 1

            positive_fail_pr_ratio /= recommendCase.__len__()
            positive_fail_pr_ratio_k[i] = positive_fail_pr_ratio
        return positive_fail_pr_ratio_k

    @staticmethod
    def negativeFail2(recommendCase, answerCase, filterAnswerCase, k=5):
        """�������Ϊ���Ԥ���ʵ�ʵ��Ƽ��б���б�"""
        negative_fail_pr_ratio_k = [0 for x in range(0, k)]
        if recommendCase.__len__() != answerCase.__len__() and answerCase.__len__() != filterAnswerCase.__len__():
            raise Exception("case is not right")

        for i in range(0, k):
            negative_fail_pr_ratio = 0
            for index, recommendList in enumerate(recommendCase):
                answerList = answerCase[index]
                filterAnswerList = filterAnswerCase[index]
                recommendList = recommendList[0:i + 1]
                """һ����û����"""
                negative_fail_list = [x for x in recommendList if x in answerList]
                if negative_fail_list.__len__() == 0 and filterAnswerList.__len__() == 0:
                    negative_fail_pr_ratio += 1

            negative_fail_pr_ratio /= recommendCase.__len__()
            negative_fail_pr_ratio_k[i] = negative_fail_pr_ratio
        return negative_fail_pr_ratio_k

    @staticmethod
    def RD(recommendCase, convertDict, k=5):
        RDS = [0 for x in range(k)]
        n = len(convertDict)

        for i in range(0, k):
            arr = [x[0:i + 1] for x in recommendCase]
            temp = {}
            for recommand in arr:
                for r in recommand:
                    if temp.get(r, None) is None:
                        temp[r] = 0
                    temp[r] += 1
            RDS[i] = -1 / (math.log2(n)) * sum(
                [math.log2(x / ((i + 1) * len(recommendCase))) * (x / ((i + 1) * len(recommendCase))) for x in
                 list(temp.values())])

        return RDS


if __name__ == '__main__':
    topk = RecommendMetricUtils.topKAccuracy([[32, 29, 16, 22, 13]], [[0, 16]], 5)
    print(topk)
