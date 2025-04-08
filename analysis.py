import os
import json
import numpy as np

def analyze_hidden_states(directory='hidden_states'):
    """
    Analyzes hidden states JSON files and provides statistics.
    
    Args:
        directory (str): Path to directory containing hidden states JSON files
    
    Returns:
        dict: Analysis report containing statistics and metadata
    """
    analysis_report = {}
    
    # Iterate through all JSON files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.json') and not filename.startswith('all_hidden_states_'):
            filepath = os.path.join(directory, filename)
            print(f"\nAnalyzing {filename}...")
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Basic file information
            file_info = {
                'model_name': filename.split('_')[0],
                'dataset_name': '_'.join(filename.split('_')[1:-1]),
                'total_samples': len(data),
                'layers': {},
                'dimensional_analysis': {},
                'value_stats': {},
                'anomalies': []
            }

            if not data:
                print("Empty file detected, skipping...")
                continue

            # Get layer structure from first sample
            first_sample = data[0]
            layers = list(first_sample.keys())
            num_layers = len(layers)
            hidden_dim = len(first_sample[layers[0]]) if num_layers > 0 else 0
            
            # Check consistency across samples
            layer_consistency = True
            dimension_consistency = True
            
            for sample_idx, sample in enumerate(data):
                # Check layer count consistency
                if len(sample) != num_layers:
                    file_info['anomalies'].append(
                        f"Sample {sample_idx} has {len(sample)} layers (expected {num_layers})"
                    )
                    layer_consistency = False
                
                # Check dimension consistency
                for layer, values in sample.items():
                    if len(values) != hidden_dim:
                        file_info['anomalies'].append(
                            f"Sample {sample_idx} layer {layer} has dim {len(values)} (expected {hidden_dim})"
                        )
                        dimension_consistency = False

            # Collect layer-wise statistics
            for layer in layers:
                layer_values = []
                for sample in data:
                    layer_values.extend(sample[layer])
                
                file_info['layers'][layer] = {
                    'dimension': hidden_dim,
                    'value_mean': np.mean(layer_values),
                    'value_std': np.std(layer_values),
                    'value_min': np.min(layer_values),
                    'value_max': np.max(layer_values)
                }

            # Add dimensional analysis
            file_info['dimensional_analysis'] = {
                'consistent_layers': layer_consistency,
                'consistent_dimensions': dimension_consistency,
                'num_layers': num_layers,
                'hidden_dimension': hidden_dim
            }

            # Add to report
            analysis_report[filename] = file_info

            # Print summary
            print(f"Model: {file_info['model_name']}")
            print(f"Dataset: {file_info['dataset_name']}")
            print(f"Total samples: {file_info['total_samples']}")
            print(f"Number of layers: {num_layers}")
            print(f"Hidden dimension size: {hidden_dim}")
            print(f"Consistent layers across samples: {layer_consistency}")
            print(f"Consistent dimensions: {dimension_consistency}")
            if file_info['anomalies']:
                print(f"Found {len(file_info['anomalies'])} anomalies")
            
    return analysis_report


def describe_hidden_states(filepath, num_examples=2, num_values=5):
    """
    Describes the exact structure of hidden states JSON files with examples.
    
    Args:
        filepath (str): Path to the JSON file
        num_examples (int): Number of sample examples to show
        num_values (int): Number of values to display per layer
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if not data:
        print("The file is empty!")
        return
    
    first_sample = data[0]
    
    print("\n=== Hidden States Data Structure ===")
    print(f"File: {os.path.basename(filepath)}")
    print(f"Total samples: {len(data)}")
    print(f"Number of layers: {len(first_sample)}")
    print(f"Hidden dimension size: {len(first_sample['layer_0'])}")
    
    print("\n=== Structure Details ===")
    print("1. Top level: List of samples (order matches input DataFrame)")
    print("2. Each sample: Dictionary with layer-wise CLS token embeddings")
    print("3. Layer keys: 'layer_0' to 'layer_N' where N = num_layers-1")
    print("4. Each layer: List of floats (length = hidden_dimension)")
    
    print("\n=== Example Samples ===")
    for i in range(min(num_examples, len(data))):
        print(f"\nSample {i + 1}:")
        sample = data[i]
        for layer in sorted(sample.keys()):
            values = sample[layer]
            print(f"  {layer}: [{', '.join(map(str, values[:num_values]))}, ...] "
                  f"(total {len(values)} values)")
    
    print("\n=== Value Statistics ===")
    all_values = [val for sample in data for layer in sample.values() for val in layer]
    print(f"Global mean: {np.mean(all_values):.4f}")
    print(f"Global std: {np.std(all_values):.4f}")
    print(f"Min value: {np.min(all_values):.4f}")
    print(f"Max value: {np.max(all_values):.4f}")



def describe_all_hidden_states(directory='hidden_states', num_examples=2, num_values=5):
    """
    Describes the structure of all hidden states JSON files in a directory.
    
    Args:
        directory (str): Path to directory containing JSON files
        num_examples (int): Number of sample examples to show per file
        num_values (int): Number of values to display per layer
    """
    # Get all JSON files in directory (excluding combined files)
    json_files = [f for f in os.listdir(directory) 
                if f.endswith('.json') and not f.startswith('all_hidden_states_')]
    
    if not json_files:
        print(f"No JSON files found in {directory}")
        return
    
    print(f"\nFound {len(json_files)} hidden state files to analyze:")
    
    for i, filename in enumerate(json_files, 1):
        filepath = os.path.join(directory, filename)
        print(f"\n{'='*50}")
        print(f"File {i}/{len(json_files)}: {filename}")
        print(f"{'='*50}")
        
        try:
            describe_hidden_states(filepath, num_examples, num_values)
        except Exception as e:
            print(f"Error analyzing {filename}: {str(e)}")
            continue

    print("\nAnalysis complete for all files!")