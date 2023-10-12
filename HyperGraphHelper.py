import pickle
from queue import Queue

from HyperGraph import HyperGraph
from MyRecEdit.Edge import Edge
from MyRecEdit.WeightCalculate import WeightCalculate
from Node import Node
from source.utils.Gexf import Gexf
import os
import numpy as np


class HyperGraphHelper:

    @staticmethod
    def createTrainDataGraph(train_data, train_data_commit, train_data_issue_comment,
                             train_data_review_comment, train_data_y, train_data_committer,
                             train_data_issue_commenter, train_data_review_commenter,
                             trainPrDis, prToRevMat, authToPrMat, prToIssueCommentMat, prToReviewCommentMat,
                             commitToPrMat, c):

        graph = HyperGraph()

        reviewerList = list(set(train_data['review_user_login']))
        for reviewer in reviewerList:
            graph.add_node(nodeType=Node.STR_NODE_TYPE_REVIEWER, contentKey=reviewer,
                           description=f"reviewer:{reviewer}")

        # yao丢掉none
        issueCommenterList = list(set(train_data_issue_comment['comment_user_login']))
        for issue_commenter in issueCommenterList:
            graph.add_node(nodeType=Node.STR_NODE_TYPE_ISSUE_COMMENTER, contentKey=issue_commenter,
                           description=f"issue_commenter:{issue_commenter}")

        reviewCommenterList = list(set(train_data_review_comment['review_comment_user_login']))
        for review_commenter in reviewCommenterList:
            graph.add_node(nodeType=Node.STR_NODE_TYPE_REVIEW_COMMENTER, contentKey=review_commenter,
                           description=f"review_commenter:{review_commenter}")

        committerList = list(set(train_data_commit['commit_user_login']))
        for committer in committerList:
            graph.add_node(nodeType=Node.STR_NODE_TYPE_COMMITTER, contentKey=committer,
                           description=f"committer:{committer}")

        prList = list(set(train_data['pr_number']))
        prList.sort()
        prList = tuple(prList)
        for pr in prList:
            graph.add_node(nodeType=Node.STR_NODE_TYPE_PR, contentKey=pr, description=f"pr:{pr}")

        authorList = list(set(train_data['author_user_login']))
        for author in authorList:
            graph.add_node(nodeType=Node.STR_NODE_TYPE_AUTHOR, contentKey=author, description=f"author:{author}")

        for (p1, p2), weight in trainPrDis.items():
            node_1 = graph.get_node_by_content(Node.STR_NODE_TYPE_PR, p1)
            node_2 = graph.get_node_by_content(Node.STR_NODE_TYPE_PR, p2)
            graph.add_edge(nodes=[node_1.id, node_2.id], edgeType=Edge.STR_EDGE_TYPE_PR_DIS,
                           weight=weight, description=f"pr distance between {p1} and {p2}",
                           queryBeforeAdd=True)

        for pr in prList:
            pr_node = graph.get_node_by_content(Node.STR_NODE_TYPE_PR, pr)
            reviewers = train_data_y[pr]
            reviewer_nodesList = [graph.get_node_by_content(Node.STR_NODE_TYPE_REVIEWER, reviewer).id for reviewer in
                                  reviewers]
            reviewer_nodesList.append(pr_node.id)

            weight = 0
            for reviewer in reviewers:
                weight += prToRevMat[pr][reviewer]
            graph.add_edge(nodes=reviewer_nodesList, edgeType=Edge.STR_EDGE_TYPE_PR_REVIEW_RELATION,
                           weight=c * weight,
                           description=f" pr review relation between pr {pr} and reviewers")

            if pr in train_data_committer:
                committers = train_data_committer[pr]
                committer_nodesList = [graph.get_node_by_content(Node.STR_NODE_TYPE_COMMITTER, committer).id for
                                       committer
                                       in
                                       committers]
                committer_nodesList.append(pr_node.id)

                weight = 0
                for committer in committers:
                    weight += commitToPrMat[pr][committer]

                graph.add_edge(nodes=committer_nodesList, edgeType=Edge.STR_EDGE_TYPE_PR_COMMITTER_RELATION,
                               weight=c * weight,
                               description=f" pr commit relation between pr {pr} and committers")

            if pr in train_data_issue_commenter:
                issue_commenters = train_data_issue_commenter[pr]
                issue_commenter_nodesList = [graph.get_node_by_content(Node.STR_NODE_TYPE_ISSUE_COMMENTER, comment).id
                                             for
                                             comment in
                                             issue_commenters]
                issue_commenter_nodesList.append(pr_node.id)

                weight = 0
                for issue_commenter in issue_commenters:
                    weight += prToIssueCommentMat[pr][issue_commenter]

                graph.add_edge(nodes=issue_commenter_nodesList, edgeType=Edge.STR_EDGE_TYPE_PR_ISSUE_COMMENTER_RELATION,
                               weight=c * weight,
                               description=f" pr comment relation between pr {pr} and issue commenters")

            if pr in train_data_review_commenter:
                review_commenters = train_data_review_commenter[pr]
                review_commenter_nodesList = [graph.get_node_by_content(Node.STR_NODE_TYPE_REVIEW_COMMENTER, comment).id
                                              for
                                              comment in
                                              review_commenters]
                review_commenter_nodesList.append(pr_node.id)

                weight = 0
                for review_commenter in review_commenters:
                    weight += prToReviewCommentMat[pr][review_commenter]

                graph.add_edge(nodes=review_commenter_nodesList,
                               edgeType=Edge.STR_EDGE_TYPE_PR_REVIEW_COMMENTER_RELATION,
                               weight=c * weight,
                               description=f" pr comment relation between pr {pr} and review commenters")

        for pr in prList:
            author = list(set(train_data.loc[train_data['pr_number'] == pr]['author_user_login']))[0]
            pr_node = graph.get_node_by_content(Node.STR_NODE_TYPE_PR, pr)
            author_node = graph.get_node_by_content(Node.STR_NODE_TYPE_AUTHOR, author)
            graph.add_edge(nodes=[pr_node.id, author_node.id], edgeType=Edge.STR_EDGE_TYPE_PR_AUTHOR_RELATION,
                           weight=authToPrMat[author][pr],
                           description=f" pr author relation between pr {pr} and author {author}",
                           nodeObjects=[pr_node, author_node])

        return graph




    @staticmethod
    def getSubGraphByNodes(graph, nodes, project, date, convertDict, key, reviewerFreqDict=None, alpha=0.98, level=3):

        subGraph = HyperGraph()

        subNodes = []
        for node in nodes:

            queue = Queue()

            nodeSet = set()

            node_r = subGraph.add_node(nodeType=node.type, contentKey=node.contentKey,
                                       description=node.description)
            node_r.level = 1
            nodeSet.add(node_r.id)
            queue.put(node_r)
            subNodes.append(node_r)
            while queue.qsize() > 0:
                tempNode = queue.get()

                if tempNode.level >= level:
                    break
                tempNode_origin = graph.get_node_by_content(nodeType=tempNode.type,
                                                            contentKey=tempNode.contentKey)
                for edge_id in tempNode_origin.connectedTo:

                    edge_node_list = []
                    edge_origin = graph.get_edge_by_key(edge_id)
                    for node_it_t in edge_origin.connectedTo:

                        node_origin = graph.get_node_by_key(node_it_t)
                        node_new = subGraph.add_node(nodeType=node_origin.type, contentKey=node_origin.contentKey,
                                                     description=node_origin.description)
                        if node_new.id not in nodeSet and hasattr(node_new, "level") and node_new.id != tempNode.id:
                            if node_new.id == 21:
                                print(21)

                            if node_new.level > tempNode.level + 1:
                                node_new.level = tempNode.level + 1
                        nodeSet.add(node_new.id)
                        if not hasattr(node_new, "level"):
                            if node_new.id == 21:
                                print(21)
                            node_new.level = tempNode.level + 1
                        edge_node_list.append(node_new)

                    subGraph.add_edge(nodes=[node_t.id for node_t in edge_node_list], edgeType=edge_origin.type,
                                      weight=edge_origin.weight, description=edge_origin.description,
                                      queryBeforeAdd=True)
                    for node_t in edge_node_list:
                        if tempNode.level < node_t.level <= level:
                            queue.put(node_t)

        subGraph.updateMatrix()

        y = np.zeros((subGraph.num_nodes, 1))

        nodeInverseMap = {v: k for k, v in subGraph.node_id_map.items()}
        y[nodeInverseMap[subNodes[0].id]][0] = 1
        y[nodeInverseMap[subNodes[1].id]][0] = 1

        I = np.identity(subGraph.num_nodes)
        f = np.dot(np.linalg.inv(I - alpha * subGraph.A), y)

        inverseNodeMap = {k: v for v, k in subGraph.node_id_map.items()}

        reviewer_node_list = []
        for i in range(0, subGraph.num_nodes):
            node_id = subGraph.node_id_map[i]
            node = subGraph.get_node_by_key(node_id)
            if node.type == Node.STR_NODE_TYPE_REVIEWER and reviewerFreqDict[node.contentKey] >= 2:
                reviewer_node_list.append((node, inverseNodeMap[node.id]))

        reverseConvertDict = {v: k for k, v in convertDict.items()}
        scores = {}
        for node, matrix_node_id in reviewer_node_list:
            scores[node.contentKey] = f[matrix_node_id][0]
        print("simple graph:")
        print([(reverseConvertDict[x[0]], x[1]) for x in sorted(scores.items(), key=lambda d: d[1], reverse=True)])

        for i in range(0, subGraph.num_nodes):
            node_id = subGraph.node_id_map[i]
            node = subGraph.get_node_by_key(node_id)
            node.pagerank = f[i][0]

        HyperGraphHelper.toGephiData(project, date, convertDict, subGraph, key=key)

    @staticmethod
    def SaveAndGetGraph(K, c, project, date, train_data, train_data_commit, train_data_issue_comment,
                        train_data_review_comment, train_data_y, train_data_committer,
                        train_data_issue_commenter, train_data_review_commenter, pathDict, prCreatedTimeMap,
                        HyperGraphIsCreated, TrainPRDisIsComputed, AddSelfChargeData):
        if AddSelfChargeData:
            projectName = project + '_self'
        else:
            projectName = project
        if not os.path.exists(
                f'./graph/graph_{K}_{c}/{project}/{projectName}_{date[0]}_{date[1]}_{date[2]}_{date[3]}_graph.pkl'):
            HyperGraphIsCreated = False

        if not HyperGraphIsCreated:
            if not os.path.exists(
                    f'./trainPrDis/trainPrDis_{K}/{project}/{projectName}_{date[0]}_{date[1]}_{date[2]}_{date[3]}_trainPrDis.pkl'):
                TrainPRDisIsComputed = False
            if not TrainPRDisIsComputed:
                trainPrDis = WeightCalculate.getTrainDataPrDistance(train_data, K, pathDict, date, prCreatedTimeMap)
                if not os.path.exists(f'./trainPrDis/trainPrDis_{K}/{project}/'):
                    os.makedirs(f'./trainPrDis/trainPrDis_{K}/{project}/')

            else:
                with open(
                        f'./trainPrDis/trainPrDis_{K}/{project}/{projectName}_{date[0]}_{date[1]}_{date[2]}_{date[3]}_trainPrDis.pkl',
                        'rb') as file:
                    trainPrDis = pickle.loads(file.read())

            prToRevMat = WeightCalculate.buildPrToRevRelation(train_data)
            authToPrMat = WeightCalculate.buildAuthToPrRelation(train_data, date)
            commitToPrMat = WeightCalculate.buildCommitToPrRelation(train_data_commit, date)
            prToIssueCommentMat = WeightCalculate.buildPrToIssueCommentRelation(train_data_issue_comment)
            prToReviewCommentMat = WeightCalculate.buildPrToReviewCommentRelation(train_data_review_comment)
            graph = HyperGraphHelper.createTrainDataGraph(train_data, train_data_commit, train_data_issue_comment,
                                                          train_data_review_comment, train_data_y, train_data_committer,
                                                          train_data_issue_commenter, train_data_review_commenter,
                                                          trainPrDis, prToRevMat, authToPrMat, prToIssueCommentMat,
                                                          prToReviewCommentMat, commitToPrMat, c)

        else:
            with open(
                    f'./graph/graph_{K}_{c}/{project}/{projectName}_{date[0]}_{date[1]}_{date[2]}_{date[3]}_graph.pkl',
                    'rb') as file:
                graph = pickle.loads(file.read())
        # HyperGraphHelper.toGephiData(project, date, convertDict, graph, key='ALL')

        return graph
