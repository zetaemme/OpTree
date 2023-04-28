import string
from collections.abc import Iterable
from dataclasses import dataclass
from random import choices
from typing import Optional, Self

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout


@dataclass(init=False, repr=False)
class Tree:
    tree: nx.DiGraph

    def __init__(self):
        self.tree = nx.DiGraph()

    def add_node(
            self,
            node_label: str,
            parent_node_label: Optional[str] = None,
            edge_label: Optional[str] = None
    ) -> None:
        self.tree.add_node(node_label, label=node_label)

        if not self.is_empty and len(self.tree.nodes()) != 1:
            self.tree.add_edge(parent_node_label, node_label, label=edge_label)

    def add_subtree(self, last_added_node: str, subtree: Self, label: str) -> None:
        # NOTE: Accrocchio fatto per evitare la casistica in cui si ofrmano autoanelli
        if last_added_node == subtree.root:
            return

        self.tree.add_nodes_from(subtree.nodes(data=True))
        self.tree.add_edges_from(subtree.tree.edges(data=True))

        self.tree.add_edge(last_added_node, subtree.root, label=label)

    def nodes(self, data: bool = False) -> Iterable[str]:
        return self.tree.nodes(data)

    def print(self) -> None:
        plt.rcParams["figure.figsize"] = (19.20, 10.80)

        self.separate_nodes_with_multiple_inputs()

        pos = graphviz_layout(self.tree, prog="dot")
        nx.draw_networkx_edges(self.tree, pos)
        nx.draw_networkx_labels(self.tree, pos, labels=nx.get_node_attributes(self.tree, "label"))
        nx.draw_networkx_edge_labels(self.tree, pos, nx.get_edge_attributes(self.tree, "label"))

        plt.tight_layout()
        plt.show()

    def separate_nodes_with_multiple_inputs(self):
        nodes = list(self.tree.nodes())

        for node in nodes:
            in_degree = self.tree.in_degree(node)
            if in_degree > 1:
                # Otteniamo i predecessori del nodo
                predecessors = list(self.tree.predecessors(node))
                # Creiamo i due nuovi nodi
                new_node1 = node + "_".join(choices(string.ascii_lowercase + string.digits, k=5))
                new_node2 = node + "_".join(choices(string.ascii_lowercase + string.digits, k=5))
                # Aggiungiamo i nuovi nodi al grafo
                self.tree.add_node(new_node1, label=node)
                self.tree.add_node(new_node2, label=node)
                # Aggiungiamo gli archi in uscita del nodo originale ai due nuovi nodi
                for neighbor in self.tree.successors(node):
                    if neighbor != node:
                        label = self.tree.get_edge_data(node, neighbor)['label']
                        self.tree.add_edge(new_node1, neighbor, label=label)
                        self.tree.add_edge(new_node2, neighbor, label=label)
                # Aggiungiamo gli archi entranti dei predecessori al primo nuovo nodo
                for predecessor in predecessors[:-1]:
                    label = self.tree.get_edge_data(predecessor, node)['label']
                    self.tree.add_edge(predecessor, new_node1, label=label)
                # Aggiungiamo gli archi entranti dell'ultimo predecessore al secondo nuovo nodo
                label = self.tree.get_edge_data(predecessors[-1], node)['label']
                self.tree.add_edge(predecessors[-1], new_node2, label=label)
                # Rimuoviamo il nodo originale dal grafo
                self.tree.remove_node(node)

    @property
    def is_empty(self) -> bool:
        return self.tree.number_of_nodes() == 0

    @property
    def root(self) -> str:
        if not self.is_empty:
            root = [node for node, in_degree in self.tree.in_degree if in_degree == 0]

            if len(root) == 0:
                return list(self.nodes())[0]
            return root[0]
