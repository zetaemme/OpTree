from collections.abc import Iterable
from dataclasses import dataclass
from random import choice
from typing import Optional, Self

import matplotlib.pyplot as plt
import networkx as nx


@dataclass(init=False, repr=False)
class Tree:
    tree: nx.DiGraph
    anti_replace_idx: int = 1

    def __init__(self):
        self.tree = nx.DiGraph()

    def add_node(
            self,
            node_label: str,
            parent_node_label: Optional[str] = None,
            edge_label: Optional[str] = None
    ) -> None:
        label = node_label + "_" + str(self.anti_replace_idx)
        self.tree.add_node(label)

        if self.has_root and len(self.tree.nodes()) != 1:
            self.tree.add_edge(parent_node_label, label, label=edge_label)

        self.anti_replace_idx += 1

    def add_subtree(self, last_added_node: str, subtree: Self, label: str) -> None:
        self.tree.add_nodes_from(subtree.nodes(data=True))
        self.tree.add_edges_from(subtree.tree.edges(data=True))

        self.tree.add_edge(last_added_node, subtree.root, label=label)

    def nodes(self, data: bool = False) -> Iterable[str]:
        return self.tree.nodes(data)

    def print(self) -> None:
        # NOTE: This function is used in order to plot the graph as a tree
        def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):

            '''
            From Joel's answer at https://stackoverflow.com/a/29597209/2966723
            Licensed under Creative Commons Attribution-Share Alike

            If the graph is a tree this will return the positions to plot this in a
            hierarchical layout.

            G: the graph (must be a tree)

            root: the root node of current branch
            - if the tree is directed and this is not given,
              the root will be found and used
            - if the tree is directed and this is given, then
              the positions will be just for the descendants of this node.
            - if the tree is undirected and not given,
              then a random choice will be used.

            width: horizontal space allocated for this branch - avoids overlap with other branches

            vert_gap: gap between levels of hierarchy

            vert_loc: vertical location of root

            xcenter: horizontal location of root
            '''
            if not nx.is_tree(G):
                raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

            if root is None:
                if isinstance(G, nx.DiGraph):
                    root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
                else:
                    root = choice(list(G.nodes))

            def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
                '''
                see hierarchy_pos docstring for most arguments

                pos: a dict saying where all nodes go if they have been assigned
                parent: parent of this branch. - only affects it if non-directed

                '''

                if pos is None:
                    pos = {root: (xcenter, vert_loc)}
                else:
                    pos[root] = (xcenter, vert_loc)
                children = list(G.neighbors(root))
                if not isinstance(G, nx.DiGraph) and parent is not None:
                    children.remove(parent)
                if len(children) != 0:
                    dx = width / len(children)
                    nextx = xcenter - width / 2 - dx / 2
                    for child in children:
                        nextx += dx
                        pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                             vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                             pos=pos, parent=root)
                return pos

            return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

        pos = hierarchy_pos(self.tree, self.root)
        nx.draw_networkx_nodes(self.tree, pos)
        nx.draw_networkx_edges(self.tree, pos)
        nx.draw_networkx_edge_labels(self.tree, pos, nx.get_edge_attributes(self.tree, "label"))
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
