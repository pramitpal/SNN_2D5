"""
Mesh Topology and Trace File Generator

This script generates BookSim configuration files for mesh network topology
and packet trace schedules from a DataFrame of communication patterns.
"""

import math
import pandas as pd


def create_label_mapping(labels):
    """Map labels to sequential IDs: ['T0', 'T5', 'ACC0'] -> {'T0': 0, 'T5': 1, 'ACC0': 2}"""
    return {label: idx for idx, label in enumerate(labels)}


def find_best_mesh_dimensions(num_chiplets):
    """Calculate optimal mesh dimensions for given number of chiplets."""
    rows = int(math.sqrt(num_chiplets))
    cols = math.ceil(num_chiplets / rows)
    return rows, cols


def generate_router_node_mapping(num_routers):
    """Generate router to node mapping configuration."""
    output_lines = []
    for router_id in range(num_routers):
        line = f"router {router_id} node {router_id}"
        output_lines.append(line)
    return output_lines


def generate_mesh_booksim_config(num_routers, latency=1):
    """Generate mesh topology configuration for BookSim."""
    rows, cols = find_best_mesh_dimensions(num_routers)
    output_lines = []
    
    for router_id in range(num_routers):
        # Calculate 2D position
        row = router_id // cols
        col = router_id % cols
        
        # Build the connection line for this router
        line = f"router {router_id}"
        neighbors = []
        
        # Check all 4 directions and collect neighbors
        # East neighbor (col + 1)
        if col < cols - 1:
            east = router_id + 1
            if east < num_routers:
                neighbors.append((east, "East"))
        
        # West neighbor (col - 1)
        if col > 0:
            west = router_id - 1
            neighbors.append((west, "West"))
        
        # South neighbor (row + 1)
        if row < rows - 1:
            south = router_id + cols
            if south < num_routers:
                neighbors.append((south, "South"))
        
        # North neighbor (row - 1)
        if row > 0:
            north = router_id - cols
            neighbors.append((north, "North"))
        
        # Add all neighbors to the line
        for neighbor_id, direction in neighbors:
            line += f" router {neighbor_id} {latency}"
        
        output_lines.append(line)
    
    return output_lines


def calculate_manhattan_distance(num_chiplets, source, destination):
    """Calculate Manhattan distance between two chiplets in the mesh."""
    rows, cols = find_best_mesh_dimensions(num_chiplets)

    def id_to_position(chiplet_id):
        idx = chiplet_id - 1
        row = idx // cols
        col = idx % cols
        return (row, col)

    src_row, src_col = id_to_position(source)
    dst_row, dst_col = id_to_position(destination)

    # Pure Manhattan distance calculation
    manhattan_hops = abs(dst_row - src_row) + abs(dst_col - src_col)

    return {
        'num_hops': manhattan_hops,
        'grid_size': (rows, cols),
        'source_pos': (src_row, src_col),
        'dest_pos': (dst_row, dst_col)
    }


def generate_packet_schedule(df, calculate_hops, time=0):
    """Generate packet schedule from DataFrame of communication patterns."""
    output_lines = []
    
    # Create mapping from labels to sequential IDs
    label_to_id = create_label_mapping(df.index)
    num_chiplets = len(df.index)

    for row_label in df.index:
        for col_label in df.columns:
            packets = df.loc[row_label, col_label]
            source = label_to_id[row_label]  # Sequential ID (0, 1, 2, ...)
            destination = label_to_id[col_label]  # Sequential ID (0, 1, 2, ...)
            no_hops = calculate_hops(num_chiplets, source+1, destination+1)['num_hops']

            for i in range(packets):
                line = f"{source} {destination} {time}"
                time += 1
                output_lines.append(line)

    return output_lines


def save_to_file(filename, content, ext=''):
    """Save content to file."""
    with open(filename + ext, 'w') as f:
        f.write('\n'.join(content))
        f.write('\n')


def generate_booksim_files(df, topology_filename='anynet_file', trace_filename='trace_file'):
    """
    Main function to generate mesh topology and trace files from DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Communication matrix where df.loc['Ci', 'Cj'] represents number of packets
        from chiplet i to chiplet j. Index and columns can have any labels 
        (e.g., 'C0', 'T0', 'ACC0', 'LIF0', etc.) - they will be mapped to 
        sequential IDs 0, 1, 2, ...
    topology_filename : str, optional
        Filename for the topology configuration (default: 'anynet_file')
    trace_filename : str, optional
        Filename for the trace file (default: 'trace_file')
    
    Returns:
    --------
    tuple : (topology_lines, trace_lines)
        Lists of configuration lines generated
    """
    # Calculate number of chiplets (simply count the rows/columns)
    num_chiplets = len(df.index)
    
    # Generate topology configuration
    total_topology = (generate_router_node_mapping(num_chiplets) + 
                     generate_mesh_booksim_config(num_chiplets))
    
    # Save topology file
    save_to_file(topology_filename, total_topology)
    # print(f"Topology file saved to: {topology_filename}")
    
    # Generate trace file
    trace_file = generate_packet_schedule(df, calculate_manhattan_distance)
    
    # Save trace file
    save_to_file(trace_filename, trace_file, '.txt')
    # print(f"Trace file saved to: {trace_filename}.txt")
    
    return total_topology, trace_file


# if __name__ == "__main__":
#     # Example usage
#     print("Mesh Topology and Trace File Generator")
#     print("=" * 50)
    
#     # Create example DataFrame with mixed labels
#     matrix = [
#         [5, 74, 0, 1, 3],
#         [7, 7, 0, 1, 200],
#         [2, 70, 0, 1, 1],
#         [2, 7, 0, 1, 9],
#         [1, 7, 0, 1, 13]
#     ]
    
#     df = pd.DataFrame(
#         matrix,
#         index=['T0', 'T1', 'T2', 'ACC0', 'LIF0'],
#         columns=['T0', 'T1', 'T2', 'ACC0', 'LIF0']
#     )
    
#     print("\nInput DataFrame:")
#     print(df)
#     print()
    
#     # Generate files
#     topology, trace = generate_booksim_files(
#         df,
#         topology_filename='anynet_file',
#         trace_filename='trace_file'
#     )
    
#     print("\nGeneration complete!")
#     print(f"Total topology lines: {len(topology)}")
#     print(f"Total trace lines: {len(trace)}")
