import abc
from typing import List
import networkx as nx

import matplotlib as mpl

mpl.use("pdf")
import matplotlib.pyplot as plt
import seaborn as sns;

sns.set()

import numpy as np
import tensorflow as tf

import inspect

from boa.core import InputSpec


class Grid(abc.ABC):
    """
    Note: the way categoricals are represented is: 
    
    INTERNALLY they are the index of the string value. This allows a fully numeric internal grid.
    TOWARDS THE SIMULATOR the indicies are converted back to the appropriate strings on-the-fly
    TOWARDS THE OPTIMIZER the indicies are converted into one-hot vectors
    """

    def __init__(self,
                 dim_specs: List[InputSpec],
                 save_dependency_graph_path: str = None,
                 verbose: bool = True,
                 ):

        if not isinstance(dim_specs, List) or \
                not all(map(lambda x: isinstance(x, InputSpec), dim_specs)):
            raise TypeError("dim_spec must be a list of InputSpecs!")

        # Convert all domains to numpy arrays and sort them in ascending order
        # Note: the sorting is crucial, further code assumes the domains are sorted.
        # Note: make sure every `formula` field is occupied
        self.dim_spec = [InputSpec(name,
                                   np.sort(np.array(domain).astype(np.float64)),
                                   formula,
                                   input_type,
                                   ) if input_type in [int, float] else InputSpec(
                                       name, domain, formula, input_type)
                         for name, domain, formula, input_type in dim_specs]

        self.dim_spec_dict = {spec.name: spec for spec in self.dim_spec}

        # Build dependency graph
        self.dep_graph = nx.DiGraph()

        # Add nodes
        for spec in self.dim_spec:
            if spec.name in self.dep_graph:
                raise ValueError(f"Found multiple specifications for '{spec.name}'!")
            self.dep_graph.add_node(spec.name, spec=spec)

        # Add directed edges
        for spec in self.dim_spec:

            sig = inspect.signature(spec.formula)
            formula_deps = tuple(sig.parameters)[1:]

            for dep in formula_deps:
                if dep not in self.dep_graph:
                    raise ValueError(f"Dependency for node '{spec.name} does not exist: {dep}")

                self.dep_graph.add_edge(dep, spec.name, type="formula")

        if save_dependency_graph_path is not None:
            self.save_dependency_graph(save_path=save_dependency_graph_path,
                                       subplots_args={'figsize': (14, 8)})

        # Check if we have a valid dependency graph
        if not nx.is_directed_acyclic_graph(self.dep_graph):
            raise ValueError("The specifications do not specify a valid dependency structure! "
                             "Maybe plot the graph?")

        # Topological sort on the DAG nodes, so that point generation is easy
        self._dim_generation_order = list(nx.topological_sort(self.dep_graph))

        self._max_grid_points = 1

        for spec in dim_specs:
            self._max_grid_points *= len(spec.domain)

        if verbose:
            print(f"Maximum number of grid points: {self.max_grid_points}")

    def save_dependency_graph(self, save_path, subplots_args={'figsize': (14, 8)}):

        sns.set_style("whitegrid")
        fig, ax = plt.subplots(**subplots_args)
        pos = nx.planar_layout(self.dep_graph)

        nx.draw_networkx_nodes(self.dep_graph, pos, ax=ax, node_color='b', alpha=0.6)

        formula_edges = [(u, v) for u, v, e in self.dep_graph.edges(data=True) if e["type"] == "formula"]

        nx.draw_networkx_edges(self.dep_graph,
                               pos,
                               edgelist=formula_edges,
                               ax=ax,
                               arrowsize=15,
                               edge_color='r')

        nx.draw_networkx_labels(self.dep_graph, pos,
                                labels={spec.name: spec.name for spec in self.dim_spec},
                                font_size=18)

        ax.grid(False)

        fig.tight_layout()

        fig.savefig(save_path)

        plt.close(fig)

    @property
    def max_grid_points(self) -> int:
        return self._max_grid_points

    @property
    def dimension(self) -> int:
        return len(self.dim_spec)

    def sample(self, num_grid_points, seed=None):
        pass

    def one_hot_to_categorical(self, points):

        if tf.rank(points) == 1:
            points = points[None, :]

        if tf.rank(points) != 2:
            raise ValueError(f"Points either has to be rank 1 or 2, but was rank {tf.rank(points)}")


        new_points = []
        i = 0

        for spec in self.dim_spec:

            # Handle categorical dimensions
            if spec.input_type == str:
                
                domain_size = len(spec.domain)

                # If we are dealing with a "boolean" categorical
                if domain_size == 2:
                    new_points.append(points[:, i:i + 1])
                    i += 1

                elif domain_size > 2:
                    one_hot = points[:, i:i + domain_size]
                    new_points.append(tf.cast(tf.argmax(one_hot, axis=1)[:, None], tf.float64))

                    i += domain_size

                else:
                    raise ValueError("There shouldn't be a constant dimension!")

            else:
                new_points.append(points[:, i:i + 1])
                i += 1

        return tf.concat(new_points, axis=1)

    def index_to_one_hot(self, points):

        new_points = []

        for i, spec in enumerate(self.dim_spec):

            # Handle categorical dimensions
            if spec.input_type == str:
                
                domain_size = len(spec.domain)

                # If we are dealing with a "boolean" categorical
                if domain_size == 2:
                    new_points.append(points[:, i:i + 1])

                elif domain_size > 2:
                    one_hot = tf.one_hot(indices=tf.cast(points[:, i], tf.int32),
                                         depth=domain_size,
                                         dtype=tf.float64)
                    new_points.append(one_hot)

                else:
                    raise ValueError("There shouldn't be a constant dimension!")

            else:
                new_points.append(points[:, i:i + 1])

        return tf.concat(new_points, axis=1)


    def input_settings_from_points(self, points):

        if tf.rank(points) == 1:
            points = points[None, :]

        if tf.rank(points) != 2:
            raise ValueError(f"Points either has to be rank 1 or 2, but was rank {tf.rank(points)}")

        input_settings_list = []

        # Dictionary of input specifications
        input_specs = nx.get_node_attributes(self.dep_graph, "spec")

        points = self.one_hot_to_categorical(points)

        # Generate setting for every point
        for point in points:

            # dictionary of all grid dimensions with name - value pairs
            base_values = {spec.name: dim for spec, dim in zip(self.dim_spec, point)}

            input_settings = {}
            for spec_name in self._dim_generation_order:

                spec = input_specs[spec_name]
                
                # Convert the input to its type
                if spec.input_type == str:
                    domain_index = tf.cast(base_values[spec_name], tf.int32).numpy()
                    input_settings[spec_name] = f'"{spec.domain[domain_index]}"'

                else:
                    sig = inspect.signature(spec.formula)
                    formula_deps = tuple(sig.parameters)[1:]

                    value = spec.formula(base_values[spec_name],
                                     *[base_values[name] for name in formula_deps])

                    input_settings[spec_name] = spec.input_type(value.numpy())

            input_settings_list.append(input_settings)

        return input_settings_list
