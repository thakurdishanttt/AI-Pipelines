"""
Knowledge graph module for the Knowledge Graph RAG system.
This module defines the KnowledgeNode and KnowledgeGraph classes for managing the semantic graph.
"""

import networkx as nx

class KnowledgeNode:
    """
    Represents a node in the knowledge graph.
    """

    def __init__(self, node_id, text, embedding):
        self.node_id = node_id
        self.text = text
        self.embedding = embedding

class KnowledgeGraph:
    """
    Represents the semantic knowledge graph.
    """

    def __init__(self):
        self.graph = nx.Graph()

    def add_node(self, node: KnowledgeNode):
        """
        Add a node to the graph.

        Args:
            node (KnowledgeNode): Node to add.
        """
        self.graph.add_node(node.node_id, text=node.text, embedding=node.embedding)

    def add_edge(self, node1_id, node2_id, relation, weight):
        """
        Add an edge between two nodes in the graph.

        Args:
            node1_id: ID of the first node.
            node2_id: ID of the second node.
            relation: Type of relation.
            weight: Weight of the edge.
        """
        self.graph.add_edge(node1_id, node2_id, relation=relation, weight=weight)

    def get_statistics(self):
        """
        Get statistics about the graph.

        Returns:
            dict: Dictionary containing total nodes and edges.
        """
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges()
        }

    def get_subgraph(self, node_ids, depth=1):
        """
        Get a subgraph around specified nodes.

        Args:
            node_ids: List of node IDs.
            depth: Depth of the subgraph.

        Returns:
            Subgraph object.
        """
        subgraph_nodes = set(node_ids)
        for node_id in node_ids:
            neighbors = nx.single_source_shortest_path_length(self.graph, node_id, cutoff=depth)
            subgraph_nodes.update(neighbors.keys())
        return self.graph.subgraph(subgraph_nodes)

    def get_node_text(self, node_id):
        """
        Get the text content of a node.

        Args:
            node_id: ID of the node.

        Returns:
            str: Text content of the node.
        """
        return self.graph.nodes[node_id]["text"] if node_id in self.graph.nodes else None
