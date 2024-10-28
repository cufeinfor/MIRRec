# coding=gbk
import os


class projectConfig:
    projectName = 'dataset'
    workloadInfo = 'workload_info'
    resultName = 'result'
    candidates = 'candidates'
    PATH_CONFIG = 'source' + os.sep + 'config' + os.sep + 'config.txt'
    PATH_TEST_INPUT_EXCEL = 'data' + os.sep + 'Test200.xlsx'
    PATH_TEST_OUTPUT_EXCEL = 'data' + os.sep + 'output.xlsx'
    PATH_TEST_OUTPUT_PATH = 'data'
    PATH_STOP_WORD_HGD = 'data' + os.sep + 'HGDStopWord.txt'
    PATH_SPLIT_WORD_EXCEL = 'data' + os.sep + 'output_splitword.xlsx'
    PATH_USER_DICT_PATH = 'data' + os.sep + 'user_dict.utf8'
    PATH_TEST_CRF_INPUT = 'data' + os.sep + 'people-daily.txt'
    PATH_TEST_CRF_TEST_RESULT = 'data' + os.sep + 'test.rst'
    PATH_TEST_REVIEW_COMMENT = 'data' + os.sep + 'reviewComments.tsv'
    PATH_TEST_WINE_RED = 'data' + os.sep + 'winequality-red.xlsx'
    PATH_TEST_REVHELPER_DATA = 'data' + os.sep + 'revhelperDemoData.csv'
    PATH_TEST_FPS_DATA = 'data' + os.sep + 'FPSDemoData.tsv'
    PATH_STOP_WORD_ENGLISH = 'source' + os.sep + 'nlp' + os.sep + 'stop-words_english_1_en.txt'
    PATH_RUBY_KEY_WORD = 'data' + os.sep + 'rubyKeyWord.txt'
    PATH_CHANGE_TRIGGER = 'data' + os.sep + 'pullrequest_rails.tsv'

    PATH_COMMIT_RELATION = 'prCommitRelation'
    PATH_ISSUE_COMMENT_PATH = 'issueCommentData'
    PATH_DATA_TRAIN = ''
    PATH_COMMIT_FILE = 'commitFileRelation'
    PATH_PULL_REQUEST = 'pullRequestData'
    PATH_PR_COMMIT = 'prCommitData'
    PATH_PR_COMMIT_EDIT = 'prCommitData_edit'
    PATH_PR_CHANGE_FILE = 'prChangeFileData'
    PATH_REVIEW = 'reviewData'
    PATH_TIMELINE = 'prTimeLineData'
    PATH_REVIEW_COMMENT = 'reviewCommentData'
    PATH_REVIEW_CHANGE = 'reviewChangeData'
    PATH_PULL_REQUEST_DISTANCE = 'prDistance'
    PATH_USER_FOLLOW_RELATION = 'userFollowRelation'
    PATH_USER_WATCH_REPO_RELATION = 'userFollowRelation'
    PATH_SEAA = 'data' + os.sep + 'SEAA'

    PATH_FPS_DATA = 'dataset' + os.sep + 'FPS'
    PATH_ML_DATA = 'dataset' + os.sep + 'ML'
    PATH_IR_DATA = 'dataset' + os.sep + 'IR'
    PATH_CA_DATA = 'dataset' + os.sep + 'CA'
    PATH_PB_DATA = 'dataset' + os.sep + 'PB'
    PATH_TC_DATA = 'dataset' + os.sep + 'TC'
    PATH_CN_DATA = 'dataset' + os.sep + 'CN'
    PATH_GA_DATA = 'dataset' + os.sep + 'GA'
    PATH_CF_DATA = 'dataset' + os.sep + 'CF'
    PATH_HG_DATA = 'dataset' + os.sep + 'HG'
    PATH_AC_DATA = 'dataset' + os.sep + 'AC'
    PATH_CN_IR_DATA = 'dataset' + os.sep + 'CN_IR'
    PATH_CHREV_DATA = 'dataset' + os.sep + 'CHREV'
    PATH_XF_DATA = 'dataset' + os.sep + 'XF'
    PATH_SVM_C_DATA = 'dataset' + os.sep + 'SVM_C'
    PATH_FPS_AC_DATA = 'dataset' + os.sep + 'FPS_AC'
    PATH_IR_AC_DATA = 'dataset' + os.sep + 'IR_AC'
    PATH_CN_AC_DATA = 'dataset' + os.sep + 'CN_AC'
    PATH_RF_A_DATA = 'dataset' + os.sep + 'RF_A'
    PATH_EAREC_DATA = 'dataset' + os.sep + 'EAREC'
    PATH_CDR_DATA = 'dataset' + os.sep + 'CDR'

    PATH_ALGORITHM = 'source' + os.sep + 'scikit' + os.sep

    TEST_OUT_PUT_SHEET_NAME = 'sheet1'

    @staticmethod
    def getRootPath():
        curPath = os.path.abspath(os.path.dirname(__file__))
        projectName = projectConfig.projectName
        rootPath = os.path.join(curPath.split(projectName)[0], projectName)  # 获取myProject，也就是项目的根路径
        return rootPath

    @staticmethod
    def getCandidatesPath():
        return projectConfig.getDataRootPath('candidates')

    @staticmethod
    def getDataRootPath(projectName):
        upupup_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        rootPath = os.path.join(upupup_path.split(projectName)[0], projectName)  # 获取myProject，也就是项目的根路径
        return rootPath

    @staticmethod
    def getConfigPath():
        return os.path.join(projectConfig.getDataRootPath('source code'), projectConfig.PATH_CONFIG)

    @staticmethod
    def getDataPath():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_TEST_OUTPUT_PATH)

    @staticmethod
    def getTestInputExcelPath():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_TEST_INPUT_EXCEL)

    @staticmethod
    def getTestoutputExcelPath():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_TEST_OUTPUT_EXCEL)

    @staticmethod
    def getStopWordHGDPath():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_STOP_WORD_HGD)

    @staticmethod
    def getSplitWordExcelPath():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_SPLIT_WORD_EXCEL)

    @staticmethod
    def getUserDictPath():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_USER_DICT_PATH)

    @staticmethod
    def getCRFInputData():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_TEST_CRF_INPUT)

    @staticmethod
    def getCRFTestDataResult():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_TEST_CRF_TEST_RESULT)

    @staticmethod
    def getReviewCommentTestData():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_TEST_REVIEW_COMMENT)

    @staticmethod
    def getRandomForestTestData():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_TEST_REVHELPER_DATA)

    @staticmethod
    def getFPSTestData():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_TEST_FPS_DATA)

    @staticmethod
    def getStopWordEnglishPath():
        return os.path.join(projectConfig.getDataRootPath('source code'), projectConfig.PATH_STOP_WORD_ENGLISH)

    @staticmethod
    def getRubyKeyWordPath():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_RUBY_KEY_WORD)

    @staticmethod
    def getChangeTriggerPRPath():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_CHANGE_TRIGGER)

    @staticmethod
    def getPrCommitRelationPath():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_COMMIT_RELATION)

    @staticmethod
    def getIssueCommentPath():
        return os.path.join(projectConfig.getDataRootPath('dataset'), projectConfig.PATH_ISSUE_COMMENT_PATH)

    @staticmethod
    def getDataTrainPath():
        return os.path.join(projectConfig.getDataRootPath('dataset'), projectConfig.PATH_DATA_TRAIN)

    @staticmethod
    def getCommitFilePath():
        return os.path.join(projectConfig.getDataRootPath('dataset'), projectConfig.PATH_COMMIT_FILE)

    @staticmethod
    def getReviewChangeDataPath():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_REVIEW_CHANGE)

    @staticmethod
    def getPrCommitPath():
        return os.path.join(projectConfig.getDataRootPath('dataset'), projectConfig.PATH_PR_COMMIT)

    @staticmethod
    def getPrCommitEditPath():
        return os.path.join(projectConfig.getDataRootPath('dataset'), projectConfig.PATH_PR_COMMIT_EDIT)

    @staticmethod
    def getPullRequestPath():
        return os.path.join(projectConfig.getDataRootPath('dataset'), projectConfig.PATH_PULL_REQUEST)

    @staticmethod
    def getPullRequestDistancePath():
        return os.path.join(projectConfig.getDataRootPath('dataset'), projectConfig.PATH_PULL_REQUEST_DISTANCE)

    @staticmethod
    def getFPSDataPath():
        return os.path.join(projectConfig.getDataRootPath('FPS'), projectConfig.projectName)

    @staticmethod
    def getMLDataPath():
        return os.path.join(projectConfig.getDataRootPath('ML'), projectConfig.projectName)

    @staticmethod
    def getCDRDataPath():
        return os.path.join(projectConfig.getDataRootPath('CDR'), projectConfig.projectName)

    @staticmethod
    def getIRDataPath():
        return os.path.join(projectConfig.getDataRootPath('IR'), projectConfig.projectName)

    @staticmethod
    def getACDataPath():
        return os.path.join(projectConfig.getDataRootPath('AC'), projectConfig.projectName)

    @staticmethod
    def getWorkloadInfoDataPath():
        return os.path.join(projectConfig.getDataRootPath('WorkloadRec'), projectConfig.workloadInfo)

    @staticmethod
    def getRF_ADataPath():
        return os.path.join(projectConfig.getDataRootPath('RF'), projectConfig.projectName)

    @staticmethod
    def getPBDataPath():
        return os.path.join(projectConfig.getDataRootPath('PB'), projectConfig.projectName)

    @staticmethod
    def getGADataPath():
        return os.path.join(projectConfig.getDataRootPath('GA'), projectConfig.projectName)

    @staticmethod
    def getTCDataPath():
        return os.path.join(projectConfig.getDataRootPath('TC'), projectConfig.projectName)

    @staticmethod
    def getCHREVDataPath():
        return os.path.join(projectConfig.getDataRootPath('CHREV'), projectConfig.projectName)

    @staticmethod
    def getCADataPath():
        return os.path.join(projectConfig.getDataRootPath('CA'), projectConfig.projectName)

    @staticmethod
    def getSVM_CDataPath():
        return os.path.join(projectConfig.getDataRootPath('SVM_C'), projectConfig.projectName)

    @staticmethod
    def getFPS_ACDataPath():
        return os.path.join(projectConfig.getDataRootPath('FPS_AC'), projectConfig.projectName)

    @staticmethod
    def getIR_ACDataPath():
        return os.path.join(projectConfig.getDataRootPath('IR_AC'), projectConfig.projectName)

    @staticmethod
    def getCN_ACDataPath():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_FPS_AC_DATA)

    @staticmethod
    def getCNDataPath():
        return os.path.join(projectConfig.getDataRootPath('CN'), projectConfig.projectName)

    @staticmethod
    def getCN_IRDataPath():
        return os.path.join(projectConfig.getDataRootPath('CN_IR'), projectConfig.projectName)

    @staticmethod
    def getCFDataPath():
        return os.path.join(projectConfig.getDataRootPath('CF'), projectConfig.projectName)

    @staticmethod
    def getHGDataPath():
        return os.path.join(projectConfig.getDataRootPath('HGRec'), projectConfig.projectName)

    @staticmethod
    def getHGRecFilterDataPath():
        return os.path.join(projectConfig.getDataRootPath('HGRecFilter'), projectConfig.projectName)

    @staticmethod
    def getHGResultPath():
        return os.path.join(projectConfig.getDataRootPath('HGRec'), projectConfig.resultName)

    @staticmethod
    def getXFDataPath():
        return os.path.join(projectConfig.getDataRootPath('XF'), projectConfig.projectName)

    @staticmethod
    def getSEAADataPath():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_SEAA)

    @staticmethod
    def getMyRecDataPath():
        return os.path.join(projectConfig.getDataRootPath('MyRec'), projectConfig.projectName)

    @staticmethod
    def getIR_MyRecDataPath():
        return os.path.join(projectConfig.getDataRootPath('IR_MyRec'), projectConfig.projectName)

    @staticmethod
    def getPRChangeFilePath():
        return os.path.join(projectConfig.getDataRootPath('dataset'), projectConfig.PATH_PR_CHANGE_FILE)

    @staticmethod
    def getReviewDataPath():
        return os.path.join(projectConfig.getDataRootPath('dataset'), projectConfig.PATH_REVIEW)

    @staticmethod
    def getPRTimeLineDataPath():
        return os.path.join(projectConfig.getDataRootPath('dataset'), projectConfig.PATH_TIMELINE)

    @staticmethod
    def getReviewCommentDataPath():
        return os.path.join(projectConfig.getDataRootPath('dataset'), projectConfig.PATH_REVIEW_COMMENT)

    @staticmethod
    def getLogPath():
        return projectConfig.getRootPath() + os.sep + 'log'

    @staticmethod
    def getUserFollowRelation():
        return os.path.join(projectConfig.getDataRootPath('dataset'), projectConfig.PATH_USER_FOLLOW_RELATION)

    @staticmethod
    def getUserWatchRepoRelation():
        return os.path.join(projectConfig.getDataRootPath('dataset'), projectConfig.PATH_USER_WATCH_REPO_RELATION)

    @staticmethod
    def getEARECDataPath():
        return os.path.join(projectConfig.getDataRootPath('EAREC'), projectConfig.projectName)

    @staticmethod
    def getMyRecEditDataPath():
        return os.path.join(projectConfig.getDataRootPath('MyRecEdit'), projectConfig.projectName)

    @staticmethod
    def getAlgorithmPath():
        return projectConfig.getDataRootPath('source code')


if __name__ == "__main__":
    print(projectConfig.getRootPath())
    print(projectConfig.getAlgorithmPath())
    print(projectConfig.getPullRequestDistancePath())
