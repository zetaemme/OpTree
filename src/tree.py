from dataclasses import dataclass
from typing import Optional, Self
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
            node_label: str,
            parent_node: Optional[UUID] = None,
            edge_label: Optional[str] = None
    ) -> UUID:
        node_unique_id = uuid4()
        self.structure.add_node(node_unique_id, label=node_label)

        if not self.is_empty and len(self.structure.nodes()) != 1:
            self.structure.add_edge(parent_node, node_unique_id, label=edge_label)

        return node_unique_id

    def add_subtree(self, last_added_node: str | UUID, subtree: Self, label: str) -> None:
        last_added_id = last_added_node
        if isinstance(last_added_id, str):
            last_added_id = next((n for n, d in self.structure.nodes(data=True) if d.get('label') == last_added_node),
                                 None)

        # NOTE: Avoids the insertion of self-loops
        if last_added_id == subtree.root:
            return

        self.structure.add_nodes_from(subtree.structure.nodes(data=True))
        self.structure.add_edges_from(subtree.structure.edges(data=True))

        self.structure.add_edge(last_added_id, subtree.root, label=label)

    def get_label_of_node(self, node: UUID) -> str:
        return self.structure.nodes[node]["label"]

    def get_root_label(self, ) -> str:
        return self.structure.nodes[self.root]["label"]

    def print(self) -> None:
        plt.rcParams["figure.figsize"] = (19.20, 10.80)

        pos = graphviz_layout(self.structure, prog="dot")
        nx.draw_networkx_edges(self.structure, pos)
        nx.draw_networkx_labels(self.structure, pos, labels=nx.get_node_attributes(self.structure, "label"))
        nx.draw_networkx_edge_labels(self.structure, pos, nx.get_edge_attributes(self.structure, "label"))

        plt.tight_layout()
        plt.show()

    @property
    def is_empty(self) -> bool:
        return self.structure.number_of_nodes() == 0

    @property
    def root(self) -> UUID:
        if not self.is_empty:
            root = [node for node, in_degree in self.structure.in_degree if in_degree == 0]

            if len(root) == 0:
                return list(self.structure.nodes)[0]
            return root[0]
