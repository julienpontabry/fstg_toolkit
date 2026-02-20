import argparse
import json
import os
import pickle
import random
import time
from collections import defaultdict

import networkx as nx
import torch
import torch_geometric.utils as pyg_utils
from common import combined_syn
from common import data
from common import models
from common import utils
from deepsnap.batch import Batch
from subgraph_matching.config import parse_encoder
from subgraph_mining.config import parse_decoder
from subgraph_mining.search_agents import GreedySearchAgent, MCTSSearchAgent
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset, PPI
from tqdm import tqdm


def pattern_growth(dataset, task, args):
    if args.method_type == "end2end":
        model = models.End2EndOrder(1, args.hidden_dim, args)
    elif args.method_type == "mlp":
        model = models.BaselineMLP(1, args.hidden_dim, args)
    else:
        model = models.OrderEmbedder(1, args.hidden_dim, args)
    model.to(utils.get_device())
    model.eval()
    model.load_state_dict(torch.load(args.model_path,
        map_location=utils.get_device()))

    mode = args.graph_mode
    
    original_nodes = {}
    original_edges = {}
    spatial_node_mapping = {}
    
    if args.dataset == 'temporal':
        try:
            original_json_path = args.temporal_json_path
            with open(original_json_path, 'r') as f:
                original_graph_data = json.load(f)
            
            if mode == 's':
                with open(args.temporal_pkl_path, 'rb') as f:
                    loaded_graph = pickle.load(f)
                
                for node_id in loaded_graph.nodes():
                    spatial_node_mapping[node_id] = loaded_graph.nodes[node_id].get('region', 'Unknown')
                
                print(f"Loaded spatial node mapping: {len(spatial_node_mapping)} nodes")
            else:
                for node in original_graph_data['nodes']:
                    node_id = node['id']
                    original_nodes[node_id] = node.get('region', 'Unknown')
            
            if mode == 't':
                for edge in original_graph_data['edges']:
                    if edge.get('type') == 'temporal':
                        source = edge['source']
                        target = edge['target']
                        transition = edge.get('transition', 'Unknown')
                        original_edges[(source, target)] = transition
                print(f"Loaded enrichment data for mode '{mode}': {len(original_graph_data['nodes'])} nodes, {len(original_edges)} directed edges")
            elif mode == 's':
                for edge in original_graph_data['edges']:
                    if edge.get('type') == 'spatial':
                        source = edge['source']
                        target = edge['target']
                        original_edges[(source, target)] = 'spatial'
                        original_edges[(target, source)] = 'spatial'
                print(f"Loaded enrichment data for mode '{mode}': {len(original_graph_data['nodes'])} nodes, {len(original_edges)//2} undirected edges")
            elif mode == 'st':
                for edge in original_graph_data['edges']:
                    source = edge['source']
                    target = edge['target']
                    if edge.get('type') == 'temporal':
                        transition = edge.get('transition', 'Unknown')
                        original_edges[(source, target)] = transition
                    elif edge.get('type') == 'spatial':
                        original_edges[(source, target)] = 'spatial'
                        original_edges[(target, source)] = 'spatial'
                print(f"Loaded enrichment data for mode '{mode}': {len(original_graph_data['nodes'])} nodes, {len(original_edges)} edges (mixed)")
        except Exception as e:
            print(f"Warning: Could not load enrichment data: {e}")

    if task == "graph-labeled":
        dataset, labels = dataset

    neighs = []
    print(len(dataset), "graphs")
    print("search strategy:", args.search_strategy)
    if task == "graph-labeled": 
        print("using label 0")
    
    graphs = []
    for i, graph in enumerate(dataset):
        if task == "graph-labeled" and labels[i] != 0: 
            continue
        if task == "graph-truncate" and i >= 1000: 
            break
        if not type(graph) == nx.Graph:
            graph = pyg_utils.to_networkx(graph).to_undirected()
        graphs.append(graph)
    
    if args.use_whole_graphs:
        neighs = graphs
    else:
        anchors = []
        if args.sample_method == "tree":
            start_time = time.time()
            for j in tqdm(range(args.n_neighborhoods)):
                graph, neigh = utils.sample_neigh(graphs,
                    random.randint(args.min_neighborhood_size,
                        args.max_neighborhood_size))
                neigh = graph.subgraph(neigh)
                neigh = nx.convert_node_labels_to_integers(neigh)
                neigh.add_edge(0, 0)
                neighs.append(neigh)
                if args.node_anchored:
                    anchors.append(0)

    embs = []
    if len(neighs) % args.batch_size != 0:
        print("WARNING: number of graphs not multiple of batch size")
    for i in range(len(neighs) // args.batch_size):
        top = (i+1)*args.batch_size
        with torch.no_grad():
            batch = utils.batch_nx_graphs(neighs[i*args.batch_size:top],
                anchors=anchors if args.node_anchored else None)
            emb = model.emb_model(batch)
            emb = emb.to(torch.device("cpu"))
        embs.append(emb)

    if args.search_strategy == "mcts":
        assert args.method_type == "order"
        agent = MCTSSearchAgent(args.min_pattern_size, args.max_pattern_size,
            model, graphs, embs, node_anchored=args.node_anchored,
            analyze=args.analyze, out_batch_size=args.out_batch_size)
    elif args.search_strategy == "greedy":
        agent = GreedySearchAgent(args.min_pattern_size, args.max_pattern_size,
            model, graphs, embs, node_anchored=args.node_anchored,
            analyze=args.analyze, model_type=args.method_type,
            out_batch_size=args.out_batch_size)
    
    out_graphs = agent.run_search(args.n_trials)
    print(time.time() - start_time, "TOTAL TIME")
    x = int(time.time() - start_time)
    print(x // 60, "mins", x % 60, "secs")

    count_by_size = defaultdict(int)
    motifs_data = {}
    
    TEMPORAL_REVERSE = {0: 'PP', 1: 'PO', 2: 'EQ', 3: 'DC', 4: 'PPi'}
    
    for motif_idx, pattern in enumerate(out_graphs, 1):
        motif_name = f"motif_{motif_idx}"
        
        if mode == 's':
            original_node_ids = list(pattern.nodes())
            node_mapping = {old_id: new_id for new_id, old_id in enumerate(original_node_ids, 1)}
            
            enriched_nodes = []
            for old_id in original_node_ids:
                new_id = node_mapping[old_id]
                region = spatial_node_mapping.get(old_id, 'Unknown')
                node_info = {
                    'id': new_id,
                    'region': region
                }
                enriched_nodes.append(node_info)
            
            enriched_edges = []
            for u, v in pattern.edges():
                new_u = node_mapping[u]
                new_v = node_mapping[v]
                edge_info = {
                    'source': new_u,
                    'target': new_v,
                    'type': 'spatial'
                }
                enriched_edges.append(edge_info)
        
        else:
            original_node_ids = list(pattern.nodes())
            node_mapping = {old_id: new_id for new_id, old_id in enumerate(original_node_ids, 1)}
            
            enriched_nodes = []
            for old_id in original_node_ids:
                new_id = node_mapping[old_id]
                node_info = {
                    'id': new_id,
                    'region': original_nodes.get(old_id, 'Unknown')
                }
                enriched_nodes.append(node_info)
            
            enriched_edges = []
            for u, v in pattern.edges():
                edge_type = pattern[u][v].get('edge_type', 0)
                
                if mode == 't':
                    if (u, v) in original_edges:
                        source_old = u
                        target_old = v
                        transition = original_edges[(u, v)]
                    elif (v, u) in original_edges:
                        source_old = v
                        target_old = u
                        transition = original_edges[(v, u)]
                    else:
                        source_old = u
                        target_old = v
                        transition = TEMPORAL_REVERSE.get(edge_type, 'Unknown')
                    
                    edge_info = {
                        'source': node_mapping[source_old],
                        'target': node_mapping[target_old],
                        'transition': transition
                    }
                
                elif mode == 'st':
                    if edge_type == 5:
                        edge_info = {
                            'source': node_mapping[u],
                            'target': node_mapping[v],
                            'type': 'spatial'
                        }
                    else:
                        if (u, v) in original_edges and original_edges[(u, v)] != 'spatial':
                            source_old = u
                            target_old = v
                            transition = original_edges[(u, v)]
                        elif (v, u) in original_edges and original_edges[(v, u)] != 'spatial':
                            source_old = v
                            target_old = u
                            transition = original_edges[(v, u)]
                        else:
                            source_old = u
                            target_old = v
                            transition = TEMPORAL_REVERSE.get(edge_type, 'Unknown')
                        
                        edge_info = {
                            'source': node_mapping[source_old],
                            'target': node_mapping[target_old],
                            'transition': transition
                        }
                
                enriched_edges.append(edge_info)
        
        motif_info = {
            'nodes': enriched_nodes,
            'edges': enriched_edges,
            'num_nodes': len(pattern),
            'num_edges': pattern.number_of_edges()
        }
        
        motifs_data[motif_name] = motif_info
        count_by_size[len(pattern)] += 1
    
    output_dir = args.temporal_output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    motifs_json_path = os.path.join(output_dir, f"motifs_enriched_{mode}.json")
    with open(motifs_json_path, 'w') as f:
        json.dump(motifs_data, f, indent=2)
    
    print(f"✓ Saved {len(motifs_data)} motifs to {motifs_json_path}")

def main():
    parser = argparse.ArgumentParser(description='Decoder arguments')
    parse_encoder(parser)
    parse_decoder(parser)
    
    parser.add_argument('--temporal_pkl_path', type=str, 
                       default='data_preprocessed/mygraph_temporal.pkl',
                       help='Path to preprocessed temporal graph pickle')
    parser.add_argument('--temporal_json_path', type=str, 
                       default='data/mygraph.json',
                       help='Path to original JSON for enrichment')
    parser.add_argument('--temporal_output_dir', type=str, 
                       default='results',
                       help='Output directory for results')
    parser.add_argument('--graph_mode', type=str, 
                       default='t',
                       choices=['s', 't', 'st'],
                       help='Graph mode: s (spatial), t (temporal), st (spatiotemporal)')
    
    args = parser.parse_args()

    print("Using dataset {}".format(args.dataset))
    print("Graph mode: {}".format(args.graph_mode))
    
    if args.dataset == 'enzymes':
        dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
        task = 'graph'
    elif args.dataset == 'temporal':
        with open(args.temporal_pkl_path, 'rb') as f:
            graph = pickle.load(f)
        print(f"Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        dataset = [graph]
        task = 'graph'
    elif args.dataset == "ppi":
        dataset = PPI(root="/tmp/PPI")
        task = 'graph'

    pattern_growth(dataset, task, args) 

if __name__ == '__main__':
    main()