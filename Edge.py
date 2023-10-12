class Edge:
    """超图的边，边可以包含多个顶点"""

    STR_EDGE_TYPE_PR_DIS = 'pr relation'
    STR_EDGE_TYPE_PR_REVIEW_RELATION = 'review'
    STR_EDGE_TYPE_PR_AUTHOR_RELATION = 'create'
    STR_EDGE_TYPE_PR_COMMITTER_RELATION = 'commit'
    STR_EDGE_TYPE_PR_ISSUE_COMMENTER_RELATION = 'issue comment'
    STR_EDGE_TYPE_PR_REVIEW_COMMENTER_RELATION = 'review comment'


    def __init__(self, key, edgeType, description, weight=0):
        self.id = key
        self.connectedTo = []
        self.weight = weight
        self.type = edgeType
        self.description = description

    def add_nodes(self, nodes):
        for node_id in nodes:
            if node_id not in self.connectedTo:
                self.connectedTo.append(node_id)

    def __str__(self):
        return "node id:" + str(self.id) + " type:" + self.type + "  description:" + self.description
