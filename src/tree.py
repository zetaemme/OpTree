from dataclasses import dataclass, field
from typing import Literal, Self

import matplotlib.pyplot as plt
from networkx import draw, draw_networkx_edge_labels
from networkx.drawing.nx_pydot import graphviz_layout
from networkx.readwrite import json_graph

from src.types import Edges, Nodes


@dataclass(init=False, repr=False)
class Tree:
    """Inspired by: https://brandonrozek.com/blog/networkxtree/"""
    root: dict[Literal["id", "name"], str] = field(default_factory=dict)
    nodes: Nodes = field(default_factory=list)
    edges: Edges = field(default_factory=list)

    def __init__(self):
        self.nodes = []
        self.edges = []

    def set_root(self, label: str) -> None:
        assert len(self.nodes) == 0, "This method should be called only on empty trees"
        self.nodes.append({"id": label, "name": label})
        self.root = {"id": label, "name": label}

    def add_child(self, *, parent_id: str | None = None, child_id: str, label: str) -> None:
        if parent_id is None:
            parent_id = self.last_added_node["id"]

        self.nodes.append({"id": child_id, "name": child_id})
        self.edges.append({"source": parent_id, "target": child_id, "label": label})

    def add_subtree(self, subtree: Self, label: str) -> None:
        self.edges.append({"source": self.last_added_node["id"], "target": subtree.root["id"], "label": label})
        self.nodes += subtree.nodes
        self.edges += subtree.edges

    def print(self) -> None:
        node_labels = {node["id"]: node["name"] for node in self.nodes}
        edge_labels = {(edge["source"], edge["target"]): edge["label"] for edge in self.edges}

        for node in self.nodes:
            del node["name"]

        tree = json_graph.node_link_graph({"nodes": self.nodes, "links": self.edges}, directed=True, multigraph=False)
        pos = graphviz_layout(tree, prog="dot")

        draw(tree.to_directed(), pos, labels=node_labels, with_labels=True)
        draw_networkx_edge_labels(tree, pos, edge_labels=edge_labels)
        plt.show()

    @property
    def last_added_node(self) -> dict[Literal["id", "name"], str]:
        return self.nodes[-1]
