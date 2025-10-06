"""
SNN 2.5D Mapping Tool
Maps neural network layers onto chiplets and calculates traffic matrices
"""

import math
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional


class SNNMapper:
    """
    Maps Spiking Neural Network layers to 2.5D chiplet architecture
    and generates traffic matrices at system and tile levels.
    """
    
    def __init__(
        self,
        weights: List[Tuple],
        layer_groups: List[List],
        NPE: int = 19,
        NT: int = 16,
        X: int = 128,
        P: int = 100,
        Vmem_res: int = 4,
        Timestep: int = 5,
        NoC_buswidth: int = 32,
        NoI_buswidth: int = 32,
        allow_break_columns: bool = True,
        include_chiplets: bool = True
    ):
        """
        Initialize the SNN Mapper.
        
        Parameters:
        -----------
        weights : List[Tuple]
            List of layer configurations: (IFM_H, IFM_W, IFM_C, K_H, K_W, K_N, Pool, Stride)
        layer_groups : List[List]
            Layer grouping with LIF placement: [[layer_ids], lif_chiplet_id]
        NPE : int
            Number of processing elements per tile
        NT : int
            Number of tiles per chiplet
        X : int
            Crossbar dimension
        P : int
            Fill percentage (0-100)
        Vmem_res : int
            Voltage memory resolution (bits)
        Timestep : int
            Number of timesteps
        NoC_buswidth : int
            Network-on-Chip bus width
        NoI_buswidth : int
            Network-on-Interface bus width
        allow_break_columns : bool
            Allow column breaking across chiplets
        include_chiplets : bool
            Include other chiplet labels (C0, C1, etc.) in tile-level matrices
        """
        self.weights = weights
        self.layer_groups = layer_groups
        self.NPE = NPE
        self.NT = NT
        self.X = X
        self.P = P
        self.Vmem_res = Vmem_res
        self.Timestep = Timestep
        self.NoC_buswidth = NoC_buswidth
        self.NoI_buswidth = NoI_buswidth
        self.allow_break_columns = allow_break_columns
        self.include_chiplets = include_chiplets
        
        # Will be computed during run()
        self.tunable_params = None
        self.xbars = None
        self.IFMS = None
        self.OFMS = None
        self.layer_output_sizes = None
        self.chiplet_data = None
        
    def _calc_tunable_params(self):
        """Calculate tunable parameters for each layer."""
        xbars = []
        params = []
        IFMS = []
        OFMS = []
        
        for each in self.weights:
            IFM = each[0] * each[1] * each[2]
            param = each[2] * each[3] * each[4] * each[5]
            xbar = math.ceil(each[3] * each[4] * each[2] / self.X) * math.ceil(each[5] / self.X)
            OFM = each[0] * each[1] * each[5]
            
            params.append(param)
            xbars.append(xbar)
            IFMS.append(IFM)
            OFMS.append(OFM)
            
        return params, xbars, IFMS, OFMS
    
    def _generate_chiplet_mapping(self):
        """Generate chiplet mapping for all layers."""
        chip_capacity = self.NT * self.NPE
        usable_capacity = max(0, min(chip_capacity, math.floor(chip_capacity * (self.P / 100.0))))
        
        # Build column/row info for each layer
        XX = []
        for i, each in enumerate(self.weights):
            cols = each[5]
            rows = math.ceil(self.tunable_params[i] / max(cols, 1)) if cols > 0 else 0
            XX.append([cols, rows, int(self.xbars[i])])
        
        Chiplet = []
        remaining_usable = usable_capacity
        
        def new_chip():
            return {
                "Layers_filled": [],
                "Crossbars_filled_respective_layer": [],
                "Crossbars_remaining_respective_layer": [],
                "Layer_tile_distribution": {},
                "Empty_crossbars": chip_capacity
            }
        
        def chip_used(blk):
            return sum(blk["Crossbars_filled_respective_layer"])
        
        def finalize_chip(blk):
            blk["Empty_crossbars"] = chip_capacity - chip_used(blk)
        
        chip = new_chip()
        current_tile = 0
        current_tile_used = 0
        
        def add_layer_allocation(layer_num, crossbars_alloc, crossbars_remaining):
            nonlocal current_tile, current_tile_used
            
            # Check tile limit
            crossbars_to_place = crossbars_alloc
            temp_tile = current_tile
            temp_used = current_tile_used
            remaining_crossbars = crossbars_to_place
            max_tile_needed = temp_tile
            
            while remaining_crossbars > 0:
                space_in_current_tile = self.NPE - temp_used
                if remaining_crossbars <= space_in_current_tile:
                    break
                else:
                    remaining_crossbars -= space_in_current_tile
                    max_tile_needed += 1
                    temp_used = 0
            
            if max_tile_needed >= self.NT:
                return False
            
            # Check if layer exists
            if layer_num in chip["Layers_filled"]:
                layer_idx = chip["Layers_filled"].index(layer_num)
                chip["Crossbars_filled_respective_layer"][layer_idx] += crossbars_alloc
                chip["Crossbars_remaining_respective_layer"][layer_idx] = crossbars_remaining
            else:
                chip["Layers_filled"].append(layer_num)
                chip["Crossbars_filled_respective_layer"].append(crossbars_alloc)
                chip["Crossbars_remaining_respective_layer"].append(crossbars_remaining)
                chip["Layer_tile_distribution"][layer_num] = {}
            
            # Place crossbars
            remaining_crossbars = crossbars_to_place
            while remaining_crossbars > 0:
                space_in_current_tile = self.NPE - current_tile_used
                
                if remaining_crossbars <= space_in_current_tile:
                    if current_tile in chip["Layer_tile_distribution"][layer_num]:
                        chip["Layer_tile_distribution"][layer_num][current_tile] += remaining_crossbars
                    else:
                        chip["Layer_tile_distribution"][layer_num][current_tile] = remaining_crossbars
                    current_tile_used += remaining_crossbars
                    remaining_crossbars = 0
                else:
                    if current_tile in chip["Layer_tile_distribution"][layer_num]:
                        chip["Layer_tile_distribution"][layer_num][current_tile] += space_in_current_tile
                    else:
                        chip["Layer_tile_distribution"][layer_num][current_tile] = space_in_current_tile
                    remaining_crossbars -= space_in_current_tile
                    current_tile += 1
                    current_tile_used = 0
            
            if current_tile_used == self.NPE:
                current_tile += 1
                current_tile_used = 0
            
            return True
        
        def reset_tile_tracking():
            nonlocal current_tile, current_tile_used
            current_tile = 0
            current_tile_used = 0
        
        i = 0
        layers_to_place = min(len(self.weights), len(XX))
        
        while i < layers_to_place:
            cols, rows, total_need = XX[i]
            remaining_need = total_need
            
            atomic_chunk = math.ceil(rows / self.X) if cols > self.X and rows > 0 else total_need
            
            if cols <= self.X:
                if remaining_need > remaining_usable:
                    finalize_chip(chip)
                    Chiplet.append(chip)
                    chip = new_chip()
                    remaining_usable = usable_capacity
                    reset_tile_tracking()
                
                if remaining_need > remaining_usable:
                    finalize_chip(chip)
                    Chiplet.append(chip)
                    chip = new_chip()
                    remaining_usable = usable_capacity
                    reset_tile_tracking()
                    i += 1
                    continue
                
                if not add_layer_allocation(i + 1, remaining_need, 0):
                    finalize_chip(chip)
                    Chiplet.append(chip)
                    chip = new_chip()
                    remaining_usable = usable_capacity
                    reset_tile_tracking()
                    add_layer_allocation(i + 1, remaining_need, 0)
                
                remaining_usable -= remaining_need
                i += 1
                
                if remaining_usable == 0:
                    finalize_chip(chip)
                    Chiplet.append(chip)
                    chip = new_chip()
                    remaining_usable = usable_capacity
                    reset_tile_tracking()
                
                continue
            
            # cols > X: splitting allowed
            while remaining_need > 0:
                if remaining_usable == 0:
                    finalize_chip(chip)
                    Chiplet.append(chip)
                    chip = new_chip()
                    remaining_usable = usable_capacity
                    reset_tile_tracking()
                
                alloc = 0
                if remaining_usable >= atomic_chunk:
                    k = min(remaining_usable // atomic_chunk, math.ceil(remaining_need / atomic_chunk))
                    k = max(k, 1)
                    alloc = min(k * atomic_chunk, remaining_need)
                else:
                    if self.allow_break_columns and remaining_usable > 0:
                        alloc = min(remaining_usable, remaining_need)
                    else:
                        finalize_chip(chip)
                        Chiplet.append(chip)
                        chip = new_chip()
                        remaining_usable = usable_capacity
                        reset_tile_tracking()
                        continue
                
                remaining_need -= alloc
                remaining_usable -= alloc
                
                if not add_layer_allocation(i + 1, alloc, max(remaining_need, 0)):
                    finalize_chip(chip)
                    Chiplet.append(chip)
                    chip = new_chip()
                    remaining_usable = usable_capacity
                    reset_tile_tracking()
                    add_layer_allocation(i + 1, alloc, max(remaining_need, 0))
                    remaining_need -= alloc
                    remaining_usable -= alloc
                
                if remaining_need == 0:
                    i += 1
                    break
        
        if chip["Layers_filled"] or chip_used(chip) > 0 or usable_capacity < chip_capacity:
            finalize_chip(chip)
            Chiplet.append(chip)
        
        return Chiplet
    
    def _get_tile_to_lif_traffic(self, layer_id, use_tile_accumulators=False, use_TA_names=False):
        """Get traffic from tiles to LIF."""
        lif_chiplet = None
        group_index = None
        
        for i, group_info in enumerate(self.layer_groups):
            layers = group_info[0]
            if layer_id in layers:
                lif_chiplet = group_info[1]
                group_index = i
                break
        
        if lif_chiplet is None:
            return [[0, "LIF_NOT_FOUND", 0, "NOT_FOUND", "NOT_FOUND"]]
        
        lif_name = f"LIF{group_index}"
        acc_name = f"ACC{group_index}"
        
        layer_idx = layer_id - 1
        output_channels = self.weights[layer_idx][5]
        IFM_H = self.weights[layer_idx][0]
        IFM_W = self.weights[layer_idx][1]
        
        total_ofm = IFM_H * IFM_W * output_channels
        
        crossbars_per_column = int(math.ceil(self.tunable_params[layer_idx] / (self.X * output_channels)))
        total_crossbars = int(self.xbars[layer_idx])
        num_columns = int(total_crossbars / crossbars_per_column)
        
        ofm_per_column = total_ofm / num_columns
        traffic_per_column = math.ceil(ofm_per_column * self.Vmem_res * self.Timestep / self.NoC_buswidth)
        
        if total_crossbars == 0:
            return [[0, acc_name, 0, "NOT_FOUND", lif_chiplet],
                    [acc_name, lif_name, 0, lif_chiplet, lif_chiplet]]
        
        results = []
        global_crossbar_index = 0
        
        for chiplet_id, chiplet in enumerate(self.chiplet_data):
            if layer_id in chiplet.get('Layer_tile_distribution', {}):
                tile_distribution = chiplet['Layer_tile_distribution'][layer_id]
                
                for tile_id, tile_count in tile_distribution.items():
                    tile_start = global_crossbar_index
                    tile_end = global_crossbar_index + tile_count
                    
                    columns_in_tile = set()
                    for cb_idx in range(tile_start, tile_end):
                        col_id = cb_idx // crossbars_per_column
                        columns_in_tile.add(col_id)
                    
                    num_columns_in_tile = len(columns_in_tile)
                    crossbars_to_ta_traffic = tile_count * traffic_per_column
                    ta_to_acc_traffic = crossbars_to_ta_traffic // tile_count
                    
                    if use_tile_accumulators and tile_count > 1:
                        ta_identifier = f"TA{tile_id}" if use_TA_names else tile_id
                        results.append([tile_id, ta_identifier, crossbars_to_ta_traffic,
                                      chiplet_id, chiplet_id])
                        results.append([ta_identifier, acc_name, ta_to_acc_traffic,
                                      chiplet_id, lif_chiplet])
                    else:
                        results.append([tile_id, acc_name, traffic_per_column,
                                      chiplet_id, lif_chiplet])
                    
                    global_crossbar_index += tile_count
        
        for col_id in range(num_columns):
            results.append([acc_name, lif_name, traffic_per_column, lif_chiplet, lif_chiplet])
        
        return results
    
    def _get_lif_to_tile_traffic(self, layer_id):
        """Get traffic from LIF to tiles."""
        target_group = None
        lif_chiplet = None
        group_index = None
        previous_layer = None
        previous_layer_lif_chiplet = None
        previous_group_index = None
        
        for i, group_info in enumerate(self.layer_groups):
            layers = group_info[0]
            if layer_id in layers:
                target_group = layers
                lif_chiplet = group_info[1]
                group_index = i
                layer_index = layers.index(layer_id)
                
                if layer_index > 0:
                    previous_layer = layers[layer_index - 1]
                    previous_layer_lif_chiplet = lif_chiplet
                    previous_group_index = group_index
                else:
                    previous_layer = layer_id - 1
                    if previous_layer > 0:
                        for j, prev_group_info in enumerate(self.layer_groups):
                            prev_layers = prev_group_info[0]
                            if previous_layer in prev_layers:
                                previous_layer_lif_chiplet = prev_group_info[1]
                                previous_group_index = j
                                break
                break
        
        if target_group is None:
            return [[f"LIF{group_index}", 0, 0, "NOT_FOUND", "NOT_FOUND"]]
        
        if previous_layer is None or previous_layer <= 0:
            return [[f"LIF{group_index}", 0, 0, "NO_PREVIOUS", lif_chiplet]]
        
        if previous_layer_lif_chiplet is None:
            return [[f"LIF{group_index}", 0, 0, "PREV_LIF_NOT_FOUND", lif_chiplet]]
        
        source_lif_name = f"LIF{previous_group_index}"
        previous_layer_ofm = self.layer_output_sizes.get(previous_layer, 0)
        
        layer_idx = layer_id - 1
        current_output_channels = self.weights[layer_idx][5]
        
        current_crossbars_per_column = int(math.ceil(self.tunable_params[layer_idx] / (self.X * current_output_channels)))
        current_total_crossbars = int(self.xbars[layer_idx])
        current_num_columns = int(current_total_crossbars / current_crossbars_per_column)
        
        input_per_column = math.ceil(previous_layer_ofm / current_num_columns * self.Timestep / self.NoC_buswidth)
        
        results = []
        global_crossbar_index = 0
        
        for chiplet_id, chiplet in enumerate(self.chiplet_data):
            if layer_id in chiplet.get('Layer_tile_distribution', {}):
                tile_distribution = chiplet['Layer_tile_distribution'][layer_id]
                
                for tile_id, tile_count in tile_distribution.items():
                    tile_start = global_crossbar_index
                    tile_end = global_crossbar_index + tile_count
                    
                    columns_in_tile = set()
                    for cb_idx in range(tile_start, tile_end):
                        col_id = cb_idx // current_crossbars_per_column
                        columns_in_tile.add(col_id)
                    
                    num_columns_in_tile = len(columns_in_tile)
                    tile_traffic = num_columns_in_tile * input_per_column
                    
                    results.append([source_lif_name, tile_id, tile_traffic,
                                  previous_layer_lif_chiplet, chiplet_id])
                    
                    global_crossbar_index += tile_count
        
        return results
    
    def _get_layer_output_traffic_chiplet(self, layer_id):
        """Get output traffic from chiplets to LIF."""
        layer_chiplet_distribution = {}
        for chiplet_id, chiplet in enumerate(self.chiplet_data):
            if layer_id in chiplet.get('Layer_tile_distribution', {}):
                tile_distribution = chiplet['Layer_tile_distribution'][layer_id]
                num_tiles = len(tile_distribution)
                layer_chiplet_distribution[chiplet_id] = num_tiles
        
        lif_chiplet = None
        group_index = None
        for i, group_info in enumerate(self.layer_groups):
            layers = group_info[0]
            if layer_id in layers:
                lif_chiplet = group_info[1]
                group_index = i
                break
        
        if lif_chiplet is None:
            return [[0, "NOT_FOUND", "NOT_FOUND"]]
        
        layer_idx = layer_id - 1
        output_channels = self.weights[layer_idx][5]
        IFM_H = self.weights[layer_idx][0]
        IFM_W = self.weights[layer_idx][1]
        
        total_ofm = IFM_H * IFM_W * output_channels
        
        crossbars_per_column = int(math.ceil(self.tunable_params[layer_idx] / (self.X * output_channels)))
        total_crossbars = int(self.xbars[layer_idx])
        num_columns = int(total_crossbars / crossbars_per_column)
        
        ofm_per_column = total_ofm / num_columns
        traffic_per_column = math.ceil(ofm_per_column * self.Vmem_res * self.Timestep / self.NoI_buswidth)
        
        if len(layer_chiplet_distribution) == 0:
            return [[0, "NOT_FOUND", lif_chiplet]]
        
        results = []
        
        for chiplet_id, num_tiles in layer_chiplet_distribution.items():
            chiplet_traffic = num_tiles * traffic_per_column
            results.append([chiplet_traffic, chiplet_id, lif_chiplet])
        
        for col_id in range(num_columns):
            results.append([traffic_per_column, lif_chiplet, lif_chiplet])
        
        return results
    
    def _get_layer_input_traffic_chiplet(self, layer_id):
        """Get input traffic from previous LIF to chiplets."""
        target_group = None
        lif_chiplet = None
        previous_layer = None
        previous_layer_lif_chiplet = None
        
        for group_info in self.layer_groups:
            layers = group_info[0]
            if layer_id in layers:
                target_group = layers
                lif_chiplet = group_info[1]
                layer_index = layers.index(layer_id)
                
                if layer_index > 0:
                    previous_layer = layers[layer_index - 1]
                    previous_layer_lif_chiplet = lif_chiplet
                else:
                    previous_layer = layer_id - 1
                    if previous_layer > 0:
                        for prev_group_info in self.layer_groups:
                            prev_layers = prev_group_info[0]
                            if previous_layer in prev_layers:
                                previous_layer_lif_chiplet = prev_group_info[1]
                                break
                break
        
        if target_group is None:
            return [[0, "NOT_FOUND", "NOT_FOUND"]]
        
        if previous_layer is None or previous_layer <= 0:
            return [[0, "NO_PREVIOUS", lif_chiplet]]
        
        if previous_layer_lif_chiplet is None:
            return [[0, "PREV_LIF_NOT_FOUND", lif_chiplet]]
        
        previous_layer_ofm = self.layer_output_sizes.get(previous_layer, 0)
        
        layer_idx = layer_id - 1
        current_output_channels = self.weights[layer_idx][5]
        
        current_crossbars_per_column = int(math.ceil(self.tunable_params[layer_idx] / (self.X * current_output_channels)))
        current_total_crossbars = int(self.xbars[layer_idx])
        current_num_columns = int(current_total_crossbars / current_crossbars_per_column)
        
        input_per_column = math.ceil(previous_layer_ofm / current_num_columns * self.Timestep / self.NoI_buswidth)
        
        chiplet_columns = {}
        global_crossbar_index = 0
        
        for chiplet_id, chiplet in enumerate(self.chiplet_data):
            if layer_id in chiplet.get('Layer_tile_distribution', {}):
                tile_distribution = chiplet['Layer_tile_distribution'][layer_id]
                
                for tile_id, tile_count in tile_distribution.items():
                    tile_start = global_crossbar_index
                    tile_end = global_crossbar_index + tile_count
                    
                    for cb_idx in range(tile_start, tile_end):
                        col_id = cb_idx // current_crossbars_per_column
                        if chiplet_id not in chiplet_columns:
                            chiplet_columns[chiplet_id] = set()
                        chiplet_columns[chiplet_id].add(col_id)
                    
                    global_crossbar_index += tile_count
        
        if len(chiplet_columns) == 0:
            return [[0, previous_layer_lif_chiplet, "NOT_FOUND"]]
        
        results = []
        
        for chiplet_id, columns in chiplet_columns.items():
            num_columns_in_chiplet = len(columns)
            chiplet_traffic = num_columns_in_chiplet * input_per_column
            results.append([chiplet_traffic, previous_layer_lif_chiplet, chiplet_id])
        
        return results
    
    def _create_chiplet_matrix(self, results_list):
        """Create system-level chiplet traffic matrix."""
        N = max(max(tuple[1] for tuple in sublist) for sublist in results_list) + 1
        matrix = np.zeros((N, N), dtype=int)
        
        for result_layer in results_list:
            for data_flow, from_chiplet, to_chiplet in result_layer:
                if from_chiplet < N and to_chiplet < N:
                    matrix[from_chiplet][to_chiplet] += data_flow
        
        chiplet_names = [f'C{i}' for i in range(N)]
        df = pd.DataFrame(matrix, index=chiplet_names, columns=chiplet_names)
        
        return matrix, df
    
    def _create_tile_matrix(self, results_list, target_chiplet, use_tile_accumulators=False, use_TA_names=False, include_chiplets=True):
        """Create tile-level traffic matrix for a specific chiplet."""
        tile_labels = [f"T{i}" for i in range(self.NT)]
        
        target_lifs = []
        target_accs = []
        target_tas = []
        
        if use_tile_accumulators:
            for i in range(self.NT):
                if use_TA_names:
                    target_tas.append(f"TA{i}")
        
        for i, (layers, lif_chiplet) in enumerate(self.layer_groups):
            if lif_chiplet == target_chiplet:
                target_accs.append(f"ACC{i}")
                target_lifs.append(f"LIF{i}")
        
        # Find max chiplet ID from traffic entries, filtering for numeric chiplet IDs only
        max_chiplet = 0
        for sublist in results_list:
            if sublist:
                for entry in sublist:
                    if len(entry) >= 5:
                        source_chiplet = entry[3]
                        dest_chiplet = entry[4]
                        if isinstance(source_chiplet, int):
                            max_chiplet = max(max_chiplet, source_chiplet)
                        if isinstance(dest_chiplet, int):
                            max_chiplet = max(max_chiplet, dest_chiplet)
        
        N = max_chiplet + 1
        
        # Only include other chiplet labels if include_chiplets is True
        if include_chiplets:
            other_chiplet_labels = [f"C{i}" for i in range(N) if i != target_chiplet]
        else:
            other_chiplet_labels = []
        
        if use_tile_accumulators and use_TA_names:
            all_labels = tile_labels + target_tas + target_accs + target_lifs + other_chiplet_labels
        else:
            all_labels = tile_labels + target_accs + target_lifs + other_chiplet_labels
        
        matrix_size = len(all_labels)
        matrix = np.zeros((matrix_size, matrix_size), dtype=int)
        label_to_idx = {label: i for i, label in enumerate(all_labels)}
        
        for result_layer in results_list:
            for traffic_entry in result_layer:
                if len(traffic_entry) >= 5:
                    source, dest, traffic, source_chiplet, dest_chiplet = traffic_entry[:5]
                    
                    if source_chiplet == target_chiplet or dest_chiplet == target_chiplet:
                        source_label = None
                        dest_label = None
                        
                        # Handle source
                        if isinstance(source, int):
                            if source_chiplet == target_chiplet:
                                source_label = f"T{source}"
                            else:
                                source_label = f"C{source_chiplet}"
                        elif isinstance(source, str):
                            if source.startswith(("TA", "ACC", "LIF")):
                                if source_chiplet == target_chiplet:
                                    source_label = source
                                else:
                                    source_label = f"C{source_chiplet}"
                        
                        # Handle destination
                        if isinstance(dest, int):
                            if dest_chiplet == target_chiplet:
                                dest_label = f"T{dest}"
                            else:
                                dest_label = f"C{dest_chiplet}"
                        elif isinstance(dest, str):
                            if dest.startswith(("TA", "ACC", "LIF")):
                                if dest_chiplet == target_chiplet:
                                    dest_label = dest
                                else:
                                    dest_label = f"C{dest_chiplet}"
                        
                        if (source_label and dest_label and
                            source_label in label_to_idx and dest_label in label_to_idx):
                            src_idx = label_to_idx[source_label]
                            dest_idx = label_to_idx[dest_label]
                            matrix[src_idx][dest_idx] += traffic
        
        df = pd.DataFrame(matrix, index=all_labels, columns=all_labels)
        
        return matrix, df
    
    def run(self) -> Dict:
        """
        Run the mapping process and generate all traffic matrices.
        
        Returns:
        --------
        Dict containing:
            - 'system_matrix': pandas.DataFrame - System-level chiplet traffic matrix
            - 'tile_matrices': Dict[int, pandas.DataFrame] - Tile matrices for each chiplet
            - 'total_noi_traffic': int - Total inter-chiplet traffic (NoI)
            - 'noc_traffic': Dict[int, int] - NoC traffic per chiplet (intra-chiplet, excluding diagonal)
            - 'num_chiplets': int - Total number of chiplets used
            - 'chiplet_mapping': List - Detailed chiplet allocation information
        """
        # Calculate parameters
        self.tunable_params, self.xbars, self.IFMS, self.OFMS = self._calc_tunable_params()
        self.layer_output_sizes = dict(zip(range(1, len(self.OFMS) + 1), self.OFMS))
        
        # Generate chiplet mapping
        self.chiplet_data = self._generate_chiplet_mapping()
        
        # Collect all traffic data
        all_chiplet_result = []
        all_tile_result = []
        
        for layer in self.layer_output_sizes.keys():
            # Chiplet-level traffic
            o_chiplet = self._get_layer_output_traffic_chiplet(layer)
            o_tile = self._get_tile_to_lif_traffic(layer, use_tile_accumulators=True, use_TA_names=False)
            
            if layer > 1:
                i_chiplet = self._get_layer_input_traffic_chiplet(layer)
                i_tile = self._get_lif_to_tile_traffic(layer)
                all_chiplet_result.append(i_chiplet)
                all_tile_result.append(i_tile)
            
            all_chiplet_result.append(o_chiplet)
            all_tile_result.append(o_tile)
        
        # Create system matrix
        matrix_system, df_system = self._create_chiplet_matrix(all_chiplet_result)
        
        # Create tile matrices for each chiplet (based on user preference)
        N = max(max(tuple[1] for tuple in sublist) for sublist in all_chiplet_result) + 1
        tile_matrices = {}
        noc_traffic = {}
        
        for chiplet_id in range(N):
            # Tile matrix for user (respects include_chiplets setting)
            _, df_tile = self._create_tile_matrix(
                all_tile_result,
                target_chiplet=chiplet_id,
                use_tile_accumulators=False,
                use_TA_names=False,
                include_chiplets=self.include_chiplets
            )
            tile_matrices[chiplet_id] = df_tile
            
            # NoC traffic calculation (always use include_chiplets=False)
            _, df_tile_noc = self._create_tile_matrix(
                all_tile_result,
                target_chiplet=chiplet_id,
                use_tile_accumulators=False,
                use_TA_names=False,
                include_chiplets=False  # Always False for NoC calculation
            )
            tile_matrix_np = df_tile_noc.to_numpy()
            noc_traffic[chiplet_id] = int(tile_matrix_np.sum() - np.trace(tile_matrix_np))
        
        # Calculate total NoI traffic (excluding diagonal)
        total_noi_traffic = int(matrix_system.sum() - np.trace(matrix_system))
        
        return {
            'system_matrix': df_system,
            'tile_matrices': tile_matrices,
            'total_noi_traffic': total_noi_traffic,
            'noc_traffic': noc_traffic,
            'num_chiplets': N,
            'chiplet_mapping': self.chiplet_data
        }


# Convenience function for quick usage
def map_snn_to_chiplets(
    weights: List[Tuple],
    layer_groups: List[List],
    **kwargs
) -> Dict:
    """
    Convenience function to map SNN layers to chiplets.
    
    Parameters match SNNMapper.__init__()
    Returns the same dict as SNNMapper.run()
    """
    mapper = SNNMapper(weights=weights, layer_groups=layer_groups, **kwargs)
    return mapper.run()
