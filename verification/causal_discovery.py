"""
Causal Discovery Algorithms

Implements PC and GES algorithms to discover causal structure
from learned representations.

Based on theory in: theory/causal_formalization.md Section 5
"""

import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional
from itertools import combinations
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


class PCAlgorithm:
    """
    PC (Peter-Clark) Algorithm for causal discovery.

    Discovers causal graph structure using conditional independence tests.
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize PC algorithm.

        Args:
            alpha: Significance level for independence tests
        """
        self.alpha = alpha

    def test_independence(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: Optional[np.ndarray] = None
    ) -> Tuple[bool, float]:
        """
        Test conditional independence X ⊥⊥ Y | Z using partial correlation.

        Args:
            X: First variable [n]
            Y: Second variable [n]
            Z: Conditioning set [n, d] (optional)

        Returns:
            (is_independent, p_value)
        """
        n = len(X)

        if Z is None or Z.shape[1] == 0:
            # Marginal correlation
            corr, p_value = pearsonr(X, Y)
        else:
            # Partial correlation
            # Residuals after regressing out Z
            from sklearn.linear_model import LinearRegression

            lr_X = LinearRegression()
            lr_Y = LinearRegression()

            lr_X.fit(Z, X)
            lr_Y.fit(Z, Y)

            resid_X = X - lr_X.predict(Z)
            resid_Y = Y - lr_Y.predict(Z)

            corr, p_value = pearsonr(resid_X, resid_Y)

        is_independent = p_value > self.alpha
        return is_independent, p_value

    def learn_skeleton(
        self,
        data: np.ndarray,
        var_names: List[str]
    ) -> nx.Graph:
        """
        Learn skeleton (undirected graph) using conditional independence tests.

        Args:
            data: Data matrix [n, p] where p is number of variables
            var_names: Names of variables

        Returns:
            Undirected graph (skeleton)
        """
        n, p = data.shape
        G = nx.complete_graph(p)

        # Rename nodes to variable names
        mapping = {i: var_names[i] for i in range(p)}
        G = nx.relabel_nodes(G, mapping)

        # Store separation sets
        sep_sets = {}

        # Remove edges based on conditional independence
        for order in range(p - 1):
            edges_to_remove = []

            for edge in list(G.edges()):
                i, j = edge
                i_idx = var_names.index(i)
                j_idx = var_names.index(j)

                # Get neighbors of i (excluding j)
                neighbors = list(G.neighbors(i))
                if j in neighbors:
                    neighbors.remove(j)

                # Test all conditioning sets of size 'order'
                if len(neighbors) >= order:
                    for cond_set in combinations(neighbors, order):
                        cond_idx = [var_names.index(v) for v in cond_set]

                        if len(cond_idx) > 0:
                            Z = data[:, cond_idx]
                        else:
                            Z = None

                        is_indep, p_val = self.test_independence(
                            data[:, i_idx],
                            data[:, j_idx],
                            Z
                        )

                        if is_indep:
                            edges_to_remove.append(edge)
                            sep_sets[(i, j)] = cond_set
                            break

            # Remove edges
            for edge in edges_to_remove:
                if G.has_edge(*edge):
                    G.remove_edge(*edge)

        self.sep_sets = sep_sets
        return G

    def orient_edges(self, skeleton: nx.Graph) -> nx.DiGraph:
        """
        Orient edges in skeleton to obtain CPDAG (Completed Partially DAG).

        Uses v-structures and propagation rules.

        Args:
            skeleton: Undirected skeleton graph

        Returns:
            Partially directed graph (CPDAG)
        """
        # Convert to directed graph (initially all edges bidirected)
        G = nx.DiGraph()
        for node in skeleton.nodes():
            G.add_node(node)

        for edge in skeleton.edges():
            G.add_edge(edge[0], edge[1])
            G.add_edge(edge[1], edge[0])

        # Rule 1: Orient v-structures (colliders)
        # If i -> k <- j and i, j not adjacent, orient as collider
        for k in skeleton.nodes():
            neighbors = list(skeleton.neighbors(k))
            for i, j in combinations(neighbors, 2):
                if not skeleton.has_edge(i, j):
                    # Check if k is in the separating set
                    if (i, j) in self.sep_sets:
                        if k not in self.sep_sets[(i, j)]:
                            # Orient i -> k <- j
                            if G.has_edge(k, i):
                                G.remove_edge(k, i)
                            if G.has_edge(k, j):
                                G.remove_edge(k, j)

        # Rule 2: If i -> k and k - j, orient k -> j
        changed = True
        while changed:
            changed = False
            for k in list(G.nodes()):
                in_neighbors = list(G.predecessors(k))
                out_neighbors = list(G.successors(k))

                for i in in_neighbors:
                    if not G.has_edge(k, i):  # i -> k (directed)
                        for j in out_neighbors:
                            if G.has_edge(j, k):  # k - j (undirected)
                                # Orient k -> j
                                G.remove_edge(j, k)
                                changed = True

        return G

    def fit(self, data: np.ndarray, var_names: List[str]) -> nx.DiGraph:
        """
        Run PC algorithm to learn causal graph.

        Args:
            data: Data matrix [n, p]
            var_names: Variable names

        Returns:
            Learned causal graph (CPDAG)
        """
        print(f"Learning skeleton with {data.shape[1]} variables...")
        skeleton = self.learn_skeleton(data, var_names)

        print(f"Orienting edges...")
        cpdag = self.orient_edges(skeleton)

        return cpdag


def extract_model_representations(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str = "cuda"
) -> Dict[str, np.ndarray]:
    """
    Extract representations from trained model for causal discovery.

    Extracts:
    - S: System instruction embeddings
    - U: User input embeddings
    - R: Learned representations
    - O: Output logits

    Args:
        model: Trained causal model
        data_loader: Data loader
        device: Device

    Returns:
        Dictionary with extracted representations
    """
    model.eval()

    all_S = []
    all_U = []
    all_R = []
    all_O = []

    with torch.no_grad():
        for batch in data_loader:
            # System instruction embedding (use mean of tokens)
            sys_instr_ids = batch["system_instruction_ids"].to(device)
            sys_instr_mask = batch["system_instruction_mask"].to(device)

            # User input embedding
            user_input_ids = batch["user_input_ids"].to(device)
            user_input_mask = batch["user_input_mask"].to(device)

            # Forward pass
            outputs = model(
                input_ids=user_input_ids,
                attention_mask=user_input_mask,
                return_representation=True
            )

            # Extract representations
            R = outputs["representation"]  # [batch, hidden]
            O = outputs["logits"].mean(dim=1)  # [batch, vocab] -> [batch, hidden]

            # Simple embeddings for S and U (use first token embedding)
            base_model = model.base_model
            S_embed = base_model.model.embed_tokens(sys_instr_ids).mean(dim=1)
            U_embed = base_model.model.embed_tokens(user_input_ids).mean(dim=1)

            all_S.append(S_embed.cpu().numpy())
            all_U.append(U_embed.cpu().numpy())
            all_R.append(R.cpu().numpy())
            all_O.append(O.cpu().numpy())

    return {
        "S": np.concatenate(all_S, axis=0),
        "U": np.concatenate(all_U, axis=0),
        "R": np.concatenate(all_R, axis=0),
        "O": np.concatenate(all_O, axis=0)
    }


def run_causal_discovery(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str = "cuda",
    save_path: Optional[str] = None
) -> Dict[str, any]:
    """
    Run causal discovery on learned representations.

    Expected causal graph: S -> R <- U, R -> O

    Args:
        model: Trained model
        data_loader: Data loader
        device: Device
        save_path: Path to save graph visualization

    Returns:
        Dictionary with discovered graph and analysis
    """
    print("Extracting representations...")
    representations = extract_model_representations(model, data_loader, device)

    # Subsample for efficiency (use PCA for dimension reduction)
    from sklearn.decomposition import PCA

    n_samples = min(500, representations["S"].shape[0])
    indices = np.random.choice(representations["S"].shape[0], n_samples, replace=False)

    # Reduce dimensionality
    pca = PCA(n_components=5)
    S_reduced = pca.fit_transform(representations["S"][indices])
    U_reduced = pca.fit_transform(representations["U"][indices])
    R_reduced = pca.fit_transform(representations["R"][indices])
    O_reduced = pca.fit_transform(representations["O"][indices])

    # Use first principal component for each variable
    data = np.column_stack([
        S_reduced[:, 0],
        U_reduced[:, 0],
        R_reduced[:, 0],
        O_reduced[:, 0]
    ])
    var_names = ["S", "U", "R", "O"]

    # Run PC algorithm
    print("Running PC algorithm...")
    pc = PCAlgorithm(alpha=0.05)
    learned_graph = pc.fit(data, var_names)

    # Check if learned graph matches expected: S -> R <- U, R -> O
    expected_edges = {("S", "R"), ("U", "R"), ("R", "O")}
    learned_edges = set(learned_graph.edges())

    match_score = len(expected_edges & learned_edges) / len(expected_edges)

    # Visualize
    if save_path:
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(learned_graph)
        nx.draw(learned_graph, pos, with_labels=True, node_color='lightblue',
                node_size=3000, font_size=16, font_weight='bold',
                arrowsize=20, edge_color='gray', width=2)
        plt.title("Discovered Causal Graph")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    return {
        "learned_graph": learned_graph,
        "expected_edges": expected_edges,
        "learned_edges": learned_edges,
        "match_score": match_score,
        "edges_correct": match_score >= 0.66,  # At least 2/3 edges correct
        "summary": {
            "n_nodes": learned_graph.number_of_nodes(),
            "n_edges": learned_graph.number_of_edges(),
            "expected_structure": "S -> R <- U, R -> O",
            "match_score": match_score,
            "structure_correct": match_score >= 0.66
        }
    }


if __name__ == "__main__":
    # Test PC algorithm with synthetic data
    print("Testing PC algorithm with synthetic causal data...")

    # Generate data from known causal graph: X -> Y -> Z
    n = 500
    X = np.random.randn(n)
    Y = 0.8 * X + np.random.randn(n) * 0.3
    Z = 0.7 * Y + np.random.randn(n) * 0.3

    data = np.column_stack([X, Y, Z])
    var_names = ["X", "Y", "Z"]

    pc = PCAlgorithm(alpha=0.05)
    learned_graph = pc.fit(data, var_names)

    print(f"\nLearned edges: {list(learned_graph.edges())}")
    print(f"Expected: X -> Y, Y -> Z")
    print("\nPC algorithm test complete!")
