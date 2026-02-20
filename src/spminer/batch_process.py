"""
Complete Batch Pipeline: Preprocess + Run SPMiner on Multiple Graphs
Usage: python batch_process.py t
       python batch_process.py st

"""
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import networkx as nx

DATA_FOLDER = 'data'
OUTPUT_FOLDER = 'results_batch'
PREPROCESSED_FOLDER = 'data_preprocessed_batch'
NODE_ANCHORED = True

REGION_ENCODING = {
    'Prefrontal cortex': 1,
    'Retrosplenial': 2,
    'Insula': 3,
    'Thalamus': 4,
    'Hippocampe': 5,
    'hippocampal formation': 6,
    'Cortical subplate': 7,
    'Sensory cortex': 8,
    'Olfactory area': 9,
    'Striatum': 10,
    'Pallidum': 11,
    'Hypothalamus': 12,
    'Midbrain': 13,
    'Unknown': 14
}

TEMPORAL_EDGE_ENCODING = {
    'PP': 0, 'PO': 1, 'EQ': 2, 'DC': 3, 'PPi': 4
}

SPATIAL_EDGE_TYPE = 5


def preprocess_graph_temporal(input_json, output_pkl):
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    G = nx.Graph()
    
    for node in data['nodes']:
        node_id = node['id']
        region = node.get('region', 'Unknown')
        region_code = REGION_ENCODING.get(region, 14)
        G.add_node(node_id, node_feature=region_code)
    
    temporal_count = 0
    for edge in data['edges']:
        if edge.get('type') == 'temporal':
            source = edge['source']
            target = edge['target']
            transition = edge.get('transition', 'Unknown')
            
            if transition in TEMPORAL_EDGE_ENCODING and source in G.nodes and target in G.nodes:
                G.add_edge(source, target, edge_type=int(TEMPORAL_EDGE_ENCODING[transition]))
                temporal_count += 1
    
    with open(output_pkl, 'wb') as f:
        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
    
    return G, temporal_count

def preprocess_graph_spatial(input_json, output_pkl):
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    G = nx.Graph()
    
    original_ids = [node['id'] for node in data['nodes']]
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(original_ids, 1)}
    
    for node in data['nodes']:
        old_id = node['id']
        new_id = id_mapping[old_id]
        region = node.get('region', 'Unknown')
        region_code = REGION_ENCODING.get(region, 14)
        G.add_node(new_id, node_feature=region_code, original_id=old_id, region=region)
    
    spatial_count = 0
    for edge in data['edges']:
        if edge.get('type') == 'spatial':
            old_source = edge['source']
            old_target = edge['target']
            
            if old_source in id_mapping and old_target in id_mapping:
                new_source = id_mapping[old_source]
                new_target = id_mapping[old_target]
                G.add_edge(new_source, new_target, edge_type=0)
                spatial_count += 1
    
    with open(output_pkl, 'wb') as f:
        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
    
    return G, spatial_count

def preprocess_graph_spatiotemporal(input_json, output_pkl):
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    G = nx.Graph()
    
    for node in data['nodes']:
        node_id = node['id']
        region = node.get('region', 'Unknown')
        region_code = REGION_ENCODING.get(region, 14)
        G.add_node(node_id, node_feature=region_code)
    
    temporal_count = 0
    for edge in data['edges']:
        if edge.get('type') == 'temporal':
            source = edge['source']
            target = edge['target']
            transition = edge.get('transition', 'Unknown')
            
            if transition in TEMPORAL_EDGE_ENCODING and source in G.nodes and target in G.nodes:
                G.add_edge(source, target, edge_type=int(TEMPORAL_EDGE_ENCODING[transition]))
                temporal_count += 1
    
    spatial_count = 0
    for edge in data['edges']:
        if edge.get('type') == 'spatial':
            source = edge['source']
            target = edge['target']
            
            if source in G.nodes and target in G.nodes:
                if not G.has_edge(source, target):
                    G.add_edge(source, target, edge_type=SPATIAL_EDGE_TYPE)
                    spatial_count += 1
    
    with open(output_pkl, 'wb') as f:
        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
    
    return G, temporal_count, spatial_count

def preprocess_graph(input_json, output_pkl, mode):
    print(f'  [1/3] Preprocessing {input_json.name} (mode: {mode})...')
    
    if mode == 't':
        G, edge_count = preprocess_graph_temporal(input_json, output_pkl)
        print(f'         Saved: {G.number_of_nodes()} nodes, {edge_count} temporal edges')
    elif mode == 's':
        G, edge_count = preprocess_graph_spatial(input_json, output_pkl)
        print(f'         Saved: {G.number_of_nodes()} nodes, {edge_count} spatial edges')
    elif mode == 'st':
        G, temporal_count, spatial_count = preprocess_graph_spatiotemporal(input_json, output_pkl)
        print(f'         Saved: {G.number_of_nodes()} nodes, {temporal_count} temporal + {spatial_count} spatial edges')
    
    return G

def run_spminer(pkl_path, json_path, output_dir, mode, node_anchored=True):
    print(f'  [2/3] Running SPMiner...')
    
    python_exe = sys.executable
    
    cmd = [
        python_exe, '-m', 'subgraph_mining.decoder',
        '--dataset=temporal',
        f'--temporal_pkl_path={pkl_path}',
        f'--temporal_json_path={json_path}',
        f'--temporal_output_dir={output_dir}',
        f'--graph_mode={mode}'
    ]
    
    if node_anchored:
        cmd.append('--node_anchored')
    
    print('        ' + '-'*60)
    result = subprocess.run(cmd)
    print('        ' + '-'*60)
    
    if result.returncode != 0:
        print(f'\n        SPMiner failed with return code {result.returncode}')
        return False
    
    print(f'\n        SPMiner completed successfully')
    return True

def process_single_graph(json_file, graph_name, mode):
    print(f'\n{"="*70}')
    print(f'Processing: {graph_name} (mode: {mode})')
    print(f'{"="*70}')
    
    try:
        pkl_path = os.path.join(PREPROCESSED_FOLDER, f'{graph_name}_{mode}.pkl')
        G = preprocess_graph(json_file, pkl_path, mode)
        
        graph_output_dir = os.path.join(OUTPUT_FOLDER, graph_name)
        os.makedirs(graph_output_dir, exist_ok=True)
        
        success = run_spminer(pkl_path, str(json_file), graph_output_dir, mode, NODE_ANCHORED)
        
        if not success:
            return {'status': 'spminer_failed'}
        
        print(f'  [3/3] Checking results...')
        motifs_file = os.path.join(graph_output_dir, f'motifs_enriched_{mode}.json')
        
        if os.path.exists(motifs_file):
            with open(motifs_file, 'r') as f:
                motifs = json.load(f)
            
            print(f'         Found {len(motifs)} motifs')
            print(f'         Saved to: {motifs_file}')
            
            return {
                'status': 'success',
                'num_motifs': len(motifs),
                'output_file': motifs_file
            }
        else:
            print(f'         No motifs file generated')
            return {'status': 'no_results'}
    
    except Exception as e:
        print(f'         Error: {e}')
        return {'status': f'error: {str(e)}'}

def main():
    if len(sys.argv) < 2:
        print("  Example: python batch_process.py t")
        print("  Example: python batch_process.py st")
        return
    
    modes = []
    valid_modes = {'s', 't', 'st'}
    for arg in sys.argv[1:]:
        if arg in valid_modes:
            modes.append(arg)
        else:
            print(f" Invalid mode: {arg}")
            print(f"  Valid modes: {', '.join(valid_modes)}")
            return
    
    if not modes:
        print(" No valid modes specified")
        return
    
    print("="*70)
    print("BATCH SPMINER PIPELINE")
    print("="*70)
    print(f"Configuration:")
    print(f"  Input folder: {DATA_FOLDER}/")
    print(f"  Output folder: {OUTPUT_FOLDER}/")
    print(f"  Modes: {', '.join(modes)}")
    print(f"  Node anchored: {NODE_ANCHORED}")
    print("="*70)
    
    json_files = sorted(Path(DATA_FOLDER).glob('*.json'))
    
    if not json_files:
        print(f"\n No JSON files found in {DATA_FOLDER}/")
        print("  Please add your graph JSON files to this folder")
        return
    
    print(f'\n Found {len(json_files)} JSON files:')
    for f in json_files:
        print(f'  - {f.name}')
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(PREPROCESSED_FOLDER, exist_ok=True)
    
    total_tasks = len(json_files) * len(modes)
    task_num = 0
    
    for json_file in json_files:
        graph_name = json_file.stem
        
        for mode in modes:
            task_num += 1
            print(f'\n[{task_num}/{total_tasks}] ', end='')
            process_single_graph(json_file, graph_name, mode)
    
    print(f'\n{"="*70}')
    print(' ALL GRAPHS PROCESSED')
    print(f'{"="*70}')
    print(f'\nResults saved in: {OUTPUT_FOLDER}/')
    print(f'  Output files: motifs_enriched_s.json, motifs_enriched_t.json, motifs_enriched_st.json')
    print(f'{"="*70}\n')

if __name__ == '__main__':
    main()