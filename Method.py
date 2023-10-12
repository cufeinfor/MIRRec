import math
from datetime import datetime
from source.scikit.FPS.FPSAlgorithm import FPSAlgorithm
from Edge import Edge
from Node import Node
import numpy as np


class Method:
    @staticmethod
    def MultiRelationMethod(graph, pr_num, prList, pathDict, paths2, prCreatedTimeMap, start_time, end_time, author,
                            reviewer_node_list, issue_commenter_node_list,
                            review_commenter_node_list, committer_node_list, re, ct, ic, rc, K, alpha):
        node_1 = graph.add_node(nodeType=Node.STR_NODE_TYPE_PR, contentKey=pr_num, description=f"pr:{pr_num}")
        scores = {}
        for p1 in prList:
            paths1 = list(pathDict[p1]['filename'])
            score = 0
            for filename1 in paths1:
                for filename2 in paths2:
                    score += FPSAlgorithm.LCP_2(filename1, filename2)
            score /= paths1.__len__() * paths2.__len__()

            t2 = prCreatedTimeMap[p1]
            t = (t2 - start_time) / (end_time - start_time)

            scores[p1] = score * math.exp(t - 1)

        KNN = [x[0] for x in sorted(scores.items(), key=lambda d: d[1], reverse=True)[0:K]]
        for p2 in KNN:
            node_2 = graph.get_node_by_content(Node.STR_NODE_TYPE_PR, p2)
            graph.add_edge(nodes=[node_1.id, node_2.id], edgeType=Edge.STR_EDGE_TYPE_PR_DIS,
                           weight=scores[p2], description=f"pr distance between {pr_num} and {p2}",
                           nodeObjects=[node_1, node_2])
        authorNode = graph.get_node_by_content(Node.STR_NODE_TYPE_AUTHOR, author)
        needAddAuthorNode = False
        if authorNode is None:
            needAddAuthorNode = True
            authorNode = graph.add_node(nodeType=Node.STR_NODE_TYPE_AUTHOR, contentKey=author,
                                        description=f"author:{author}")
        graph.add_edge(nodes=[node_1.id, authorNode.id], edgeType=Edge.STR_EDGE_TYPE_PR_AUTHOR_RELATION,
                       weight=0.00001, description=f" pr author relation between pr {pr_num} and author {author}",
                       nodeObjects=[node_1, authorNode])

        time_before_updateMatrix = datetime.now()
        graph.updateMatrix()
        print(f'update：{datetime.now() - time_before_updateMatrix}')

        y = np.zeros((graph.num_nodes, 1))

        nodeInverseMap = {v: k for k, v in graph.node_id_map.items()}
        y[nodeInverseMap[node_1.id]][0] = 1
        y[nodeInverseMap[authorNode.id]][0] = 1

        I = np.identity(graph.num_nodes)
        f = np.dot(np.linalg.inv(I - alpha * graph.A), y)

        scores = {}

        for node, matrix_node_id in reviewer_node_list:
            if node.contentKey != author:
                scores[node.contentKey] = re * f[matrix_node_id][0]

        for node, matrix_node_id in issue_commenter_node_list:
            if node.contentKey != author:
                scores[node.contentKey] += ic * f[matrix_node_id][0]

        for node, matrix_node_id in review_commenter_node_list:
            if node.contentKey != author:
                scores[node.contentKey] += rc * f[matrix_node_id][0]

        for node, matrix_node_id in committer_node_list:
            if node.contentKey != author:
                scores[node.contentKey] += ct * f[matrix_node_id][0]

        if needAddAuthorNode:
            graph.remove_node_by_key(authorNode.id)
        graph.remove_node_by_key(node_1.id)

        return scores

    @staticmethod
    def TurnRecMethod(probabilityOfStay, effort, specializedKnowledge, _alpha, _beta,
                      reverseConvertDict, reviewer_login_list, reviewerImportance=1):
        scores = {}
        for user in reviewer_login_list:
            name = reverseConvertDict[user]
            if name not in probabilityOfStay or name not in effort or name not in specializedKnowledge:
                scores[user] = 0
            else:
                scores[user] = reviewerImportance * math.pow(probabilityOfStay[name] * effort[name], _alpha) * math.pow(
                    1 - specializedKnowledge[name], _beta)
        return scores

    @staticmethod
    def TurnRecBaseMIRRecMethod(MIRRecScores, probabilityOfStay, effort, specializedKnowledge, _alpha, _beta,
                                reverseConvertDict, reviewerImportance=1):
        scores = {}
        for user in list(MIRRecScores.keys()):
            name = reverseConvertDict[user]
            if name not in probabilityOfStay or name not in effort or name not in specializedKnowledge:
                scores[user] = 0
            else:
                scores[user] = reviewerImportance * math.pow(probabilityOfStay[name] * effort[name], _alpha) * math.pow(
                    1 - MIRRecScores[user], _beta)
        return scores
