import csv
import networkx as nx
import numpy as np

def process(adj):
    degrees = np.sum(adj, axis=0)
    print("max degree:", max(degrees))
    print("average degree:", np.mean(degrees))
    adj = np.triu(adj, k=1)
    print("number of edges:", sum(sum(adj)))
    return adj

def save(adj, filename):
    import os
    # Ensure we save to the graphs directory regardless of where script is run from
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(adj)
    print(f"Saved to: {filepath}")

def barabasi_albert():
    # Generate progressively larger Barabási-Albert graphs
    Ns = [100, 500, 1000, 2000, 3000, 5000]
    m = 2  # Each new node connects to 2 existing nodes
    
    for N in Ns:
        print(f"Generating barabasi_albert graph with {N} nodes...")
        G = nx.barabasi_albert_graph(N, m)
        adj = process(nx.adjacency_matrix(G).todense())
        save(adj, f"barabasi_albert_{N}.txt")
        print(f"Saved: barabasi_albert_{N}.txt\n")

def barabasi_albert_custom(sizes, m=2):
    """Generate Barabási-Albert graphs with custom sizes and connectivity"""
    for N in sizes:
        print(f"Generating custom barabasi_albert graph with {N} nodes, m={m}...")
        G = nx.barabasi_albert_graph(N, m)
        adj = process(nx.adjacency_matrix(G).todense())
        save(adj, f"barabasi_albert_{N}.txt")
        print(f"Saved: barabasi_albert_{N}.txt\n")

def dorogovtsev_goltsev_mendes():
    Ns = [1, 2, 3, 4]#[5, 6, 7, 8, 9]
    for N in Ns:
        print("dorogovtsev_goltsev_mendes", N)
        G = nx.dorogovtsev_goltsev_mendes_graph(N)
        adj = process(nx.adjacency_matrix(G).todense())
        save(adj, f"dorogovtsev_goltsev_mendes_{N}.txt")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "barabasi":
            barabasi_albert()
        elif sys.argv[1] == "dorogovtsev":
            dorogovtsev_goltsev_mendes()
        elif sys.argv[1] == "all":
            barabasi_albert()
            dorogovtsev_goltsev_mendes()
        else:
            print("Usage: python generate.py [big|barabasi|dorogovtsev|all]")
            print("  barabasi: Generate Barabási-Albert graphs")
            print("  dorogovtsev: Generate Dorogovtsev-Goltsev-Mendes graphs")
            print("  all: Generate both graph types")
    else:
        # Default behavior - generate standard sets
        print("Generating all standard graphs...")
        barabasi_albert()
        dorogovtsev_goltsev_mendes()
