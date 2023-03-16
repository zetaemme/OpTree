from dataclasses import dataclass, field
from string import digits
from typing import Literal, Self

import matplotlib.pyplot as plt
from networkx import bfs_tree, draw, draw_networkx_edge_labels
from networkx.drawing.nx_pydot import graphviz_layout
from networkx.readwrite import json_graph

from src.types import Edges, Nodes


@dataclass(init=False, repr=False)
class Tree:
    """Inspired by: https://brandonrozek.com/blog/networkxtree/"""
    anti_replace_idx = 1

    root: dict[Literal["id", "name"], str] | None = field(default_factory=dict)
    nodes: Nodes = field(default_factory=list)
    leaves: Nodes = field(default_factory=list)
    edges: Edges = field(default_factory=list)

    def __init__(self):
        self.nodes = []
        self.leaves = []
        self.edges = []

        self.root = None

    def add_leaf(self, leaf_id: str, label: str) -> None:
        """Adds a terminal node to the tree

        Args:
            leaf_id (str): The leaf to add
            label (str): The label of the edge that joins self with subtree
        """
        leaf_id += str(self.anti_replace_idx)
        Tree.anti_replace_idx += 1

        self.leaves.append({"id": leaf_id, "name": leaf_id})

        if self.root is None:
            self.root = {"id": leaf_id, "name": leaf_id}
        else:
            self.edges.append({"source": self.nodes[-1]["id"], "target": leaf_id, "label": label})

    def add_node(self, parent_id: str | None, node_id: str, label: str) -> None:
        """Adds a not-terminal node to the tree

        Args:
            parent_id (str): The parent of the given tree
            node_id (str): The non-terminal node to add
            label (str): The label of the edge that joins self with subtree
        """
        if self.root is None:
            self.root = {"id": node_id, "name": node_id}
        elif parent_id is not None:
            self.edges.append({"source": parent_id, "target": node_id, "label": label})

        self.nodes.append({"id": node_id, "name": node_id})

    def add_subtree(self, parent_id: str, subtree: Self, label: str) -> None:
        """Adds a given tree as child of the node with id equal to 'parent_id' and labels the edge with 'label'

        Args:
            parent_id (str): The parent of the given tree
            subtree (Self): The tree to add as a child of parent
            label (str): The label of the edge that joins self with subtree
        """
        if subtree.is_empty:
            return

        self.edges.append({"source": parent_id, "target": subtree.root["id"], "label": label})

        self.nodes += subtree.nodes
        self.leaves += subtree.leaves
        self.edges += subtree.edges

    def print(self, as_tree: bool = True) -> None:
        """Plots the tree"""
        def remove_key(data: list[dict], key: str) -> None:
            # checking if data is a dictionary or list
            if isinstance(data, dict):
                # if key is present in dictionary then remove it
                data.pop(key, None)
                # iterating over all the keys in the dictionary
                for key in data:
                    # calling function recursively for nested dictionaries
                    remove_key(data[key], key)
            elif isinstance(data, list):
                # iterating over all the items of the list
                for item in data:
                    # calling function recursively for all elements of the list
                    remove_key(item, key)

        plt.rcParams['figure.figsize'] = [16, 9]

        node_labels = {
            node["id"]: node["name"]
            for node in self.nodes
        }
        leaf_labels = {
            leaf["id"]: leaf["name"].translate(str.maketrans("", "", digits))
            for leaf in self.leaves
        }
        edge_labels = {(edge["source"], edge["target"]): edge["label"] for edge in self.edges}

        remove_key(self.nodes, "name")
        remove_key(self.leaves, "name")

        tree = json_graph.node_link_graph(
            {
                "nodes": self.nodes + self.leaves,
                "links": self.edges
            },
            directed=True,
            multigraph=False
        )
        pos = graphviz_layout(tree, prog="dot")

        if as_tree:
            draw(bfs_tree(tree.to_directed(), self.root["id"]), pos, labels=node_labels | leaf_labels, with_labels=True)
        else:
            draw(tree.to_directed(), pos, labels=node_labels | leaf_labels, with_labels=True)
        draw_networkx_edge_labels(tree, pos, edge_labels=edge_labels)
        plt.show()

    @property
    def is_empty(self) -> bool:
        return not self.edges and not self.nodes and not self.leaves and self.root is None
