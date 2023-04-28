from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Optional, Self

import matplotlib.pyplot as plt
import networkx as nx


@dataclass(init=False, repr=False)
class Tree:
    tree: nx.DiGraph
    anti_replace_idx: int = 1
    edge_labels: dict[tuple[str, str], str] = field(default_factory=dict)

    def __init__(self):
        self.tree = nx.DiGraph()
        self.edge_labels = {}

    def add_node(
            self,
            node_label: str,
            parent_node_label: Optional[str] = None,
            edge_label: Optional[str] = None
    ) -> None:
        label = node_label + "_" + str(self.anti_replace_idx)
        self.tree.add_node(label)

        if self.has_root and len(self.tree.nodes()) != 1:
            self.tree.add_edge(parent_node_label, label)
            self.edge_labels[(parent_node_label, label)] = edge_label

        self.anti_replace_idx += 1

    def add_subtree(self, last_added_node: str, subtree: Self, label: str) -> None:
        self.tree.add_nodes_from(subtree.nodes(data=True))
        self.tree.add_edges_from(subtree.tree.edges(data=True))
        self.edge_labels.update(subtree.edge_labels)

        self.tree.add_edge(last_added_node, subtree.root)
        self.edge_labels[(last_added_node, subtree.root)] = label

    def nodes(self, data: bool = False) -> Iterable[str]:
        return self.tree.nodes(data)

    def print(self) -> None:
        pos = nx.spring_layout(self.tree)
        nx.draw_networkx_nodes(self.tree, pos)
        nx.draw_networkx_edges(self.tree, pos)
        nx.draw_networkx_edge_labels(self.tree, pos, self.edge_labels)
        nx.draw_networkx_labels(self.tree, pos)

        plt.show()

    @property
    def has_root(self) -> bool:
        return len(list(nx.topological_sort(self.tree))) != 0

    @property
    def is_empty(self) -> bool:
        return self.tree.number_of_nodes() == 0

    @property
    def root(self) -> str:
        if self.has_root:
            roots = [node for node, in_degree in self.tree.in_degree if in_degree == 0]
            return roots[0]
