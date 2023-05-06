import string
from dataclasses import dataclass
from random import choice
from typing import Iterable, Optional, Self
from uuid import UUID, uuid4

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout


@dataclass(init=False, repr=False)
class Tree:
    structure: nx.DiGraph

    def __init__(self):
        self.structure = nx.DiGraph()

    def add_node(
            self,
            objects: list[int],
            pairs_covered: Optional[int],
            node_label: str,
            parent_node: Optional[UUID] = None,
            edge_label: Optional[str] = None
    ) -> UUID:
        node_unique_id = uuid4()
        self.structure.add_node(node_unique_id, label=node_label, objects=objects, pairs=pairs_covered)

        if not self.is_empty and len(self.structure.nodes()) != 1:
            self.structure.add_edge(parent_node, node_unique_id, label=edge_label)

        return node_unique_id

    def add_subtree(self, last_added_node: UUID, subtree: Self, label: str) -> None:
        self.structure.add_nodes_from(subtree.structure.nodes(data=True))
        self.structure.add_edges_from(subtree.structure.edges(data=True))

        self.structure.add_edge(last_added_node, subtree.root, label=label)

    def check_leaves_objects(self, classes: dict[int, str]) -> bool:
        leaves = {
            self.get_label_of_node(leaf) + choice(string.ascii_lowercase): sorted(self.structure.nodes[leaf]["objects"])
            for leaf in self.leaves
        }

        for class_, objects in leaves.items():
            for obj in objects:
                if class_[:-1] != classes[obj]:
                    return False

        return True

    def get_label_of_node(self, node: UUID) -> str:
        return self.structure.nodes[node]["label"]

    def get_root_label(self, ) -> str:
        return self.structure.nodes[self.root]["label"]

    def print(self) -> None:
        plt.rcParams["figure.figsize"] = (20.48, 11.52)
        plt.set_loglevel("info")

        pos = graphviz_layout(self.structure, prog="dot")
        nx.draw_networkx_edges(self.structure, pos)
        nx.draw_networkx_labels(self.structure, pos, labels=nx.get_node_attributes(self.structure, "label"),
                                font_size=8)
        nx.draw_networkx_edge_labels(self.structure, pos, nx.get_edge_attributes(self.structure, "label"))

        plt.tight_layout()
        plt.show()

    @property
    def is_empty(self) -> bool:
        return self.structure.number_of_nodes() == 0

    @property
    def leaves(self) -> Iterable:
        return [
            node
            for node in self.structure.nodes()
            if self.structure.out_degree(node) == 0
        ]

    @property
    def root(self) -> UUID:
        if not self.is_empty:
            root = [node for node, in_degree in self.structure.in_degree if in_degree == 0]  # type: ignore

            if len(root) == 0:
                return list(self.structure.nodes)[0]
            return root[0]
