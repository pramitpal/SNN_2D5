import numpy as np

#@title count no of xbars & tiles required function
def pe_tile_count(in_ch_list, out_ch_list, out_dim_list, k, xbar, pe_per_tile):
    # ----------------------------------------------------------------------
    # 1. CALCULATE THE NUMBER OF PROCESSING ENGINES (PEs) PER LAYER 
    # ----------------------------------------------------------------------
    # A Processing Engine (PE) is the basic calculator for our network.
    # We need to figure out how many PEs are required for each layer.

    num_layer = len(out_ch_list)  # Get the total number of layers in the network.
    pe_list = []                  # Create an empty list to store the PE count for each layer.

    # Loop through each layer to calculate its PE requirement.
    for i in range(num_layer):
        # Calculate the number of PEs needed for the current layer.
        # This is determined by how the input and output channels of the layer
        # map onto the crossbar arrays ('xbar'), which are the core of a PE.
        # We use np.ceil to round up, because you can't have a fraction of a hardware unit.
        num_pe = np.ceil(in_ch_list[i] / xbar) * np.ceil(out_ch_list[i] / xbar)
        pe_list.append(num_pe)
    
    # ----------------------------------------------------------------------
    # 2. SANITY CHECK: ENSURE LAYERS ARE IN AN EFFICIENT ORDER 
    # ----------------------------------------------------------------------
    # For hardware efficiency, it's often better to process layers that
    # require fewer PEs first. This part checks if the layers are sorted by PE count.

    flag = 0
    if (pe_list == sorted(pe_list)):
        flag = 1  # If the list is already sorted, set the flag to 1.

    if (flag):
        print("Layers in ascending order based on xbars required")
    else:
        print("Check layer order") # If not sorted, it prints a warning.
        return # And exits the function.

    # ----------------------------------------------------------------------
    # 3. CALCULATE TOTAL PEs AND THE NUMBER OF TILES NEEDED 
    # ----------------------------------------------------------------------
    # Now we sum up all the PEs and figure out how many "Tiles" we need.
    # A Tile is a physical container on the chip that holds a group of PEs.

    # Sum the PEs from all layers to get the total number of PEs.
    num_pe = sum(pe_list)
    print(f'No. of PEs {num_pe}')

    # Check if the total PEs can be evenly divided into tiles.
    if (num_pe % pe_per_tile != 0):
        # If there's a remainder, we need one extra tile for the leftover PEs.
        num_tiles = (num_pe // pe_per_tile) + 1
    else:
        # If it divides evenly, the number of tiles is a simple division.
        num_tiles = num_pe / pe_per_tile

    print(f'No. of Tiles {num_tiles}')

    # ----------------------------------------------------------------------
    # 4. RETURN THE FINAL TILE COUNT 
    # ----------------------------------------------------------------------
    # The function returns the total number of tiles required to build the hardware.
    return num_tiles



#@title Compute total area needed function
# All areas are in square micrometers (µm^2)
def compute_area(in_ch_list, in_dim_list, out_ch_list, out_dim_list, xbar_size, k, pe_per_tile, n_tiles, device):
    # ----------------------------------------------------------------------
    # 1. DEFINE THE AREA OF BASIC HARDWARE COMPONENTS 
    # ----------------------------------------------------------------------
    # These are the physical sizes of the fundamental building blocks of the chip.

    # Set the area for a single crossbar array based on the memory technology.
    # RRAM is generally denser (smaller area) than SRAM.
    if device == 'rram':
        xbar_ar = 26.2144
    elif device == 'sram':
        xbar_ar = 671.089

    # --- Areas for other fixed components ---
    Tile_buff = 0.7391 * 64 * 128  # Area of the buffer within a tile
    Temp_Buff = 484.643999         # Area of a temporary buffer
    Sub = 13411.41498              # Area of a subtractor circuit
    ADC = 693.633                  # Area of an Analog-to-Digital Converter
    Htree = 216830 * 2             # Area of the H-Tree network (for data distribution)
    MUX = 45.9                     # Area of a multiplexer

    # ----------------------------------------------------------------------
    # 2. CALCULATE THE AREA OF A SINGLE PROCESSING ENGINE (PE) 
    # ----------------------------------------------------------------------
    # A PE is the core computational unit. Its total area is the sum of its internal components.

    # The PE area consists of the crossbar arrays plus the associated ADCs and MUXs.
    PE_ar = k * k * xbar_ar + (xbar_size / 8) * (ADC + MUX)

    # ----------------------------------------------------------------------
    # 3. CALCULATE THE AREA OF A SINGLE TILE 
    # ----------------------------------------------------------------------
    # A Tile is a larger block that groups several PEs together with shared resources.

    # The total area of a tile is the sum of all its PEs plus shared components
    # like buffers, subtractors, and the H-Tree network.
    Tile_ar_ov = (xbar_size / 8) * Sub + Temp_Buff + Tile_buff + pe_per_tile * PE_ar + Htree

    # ----------------------------------------------------------------------
    # 4. CALCULATE THE TOTAL CHIP AREA 
    # ----------------------------------------------------------------------
    # Finally, we add up the area of all tiles and memory components to get the final chip size.

    # The total compute area is the area of all tiles plus the memory needed to store input feature maps.
    total_compute_ar_ov = Tile_ar_ov * n_tiles + np.sum(
        np.array(in_ch_list) * np.array(in_dim_list) * np.array(in_dim_list)) * 0.7391 * 22

    # Calculate the area required for the memory that stores neuron states (e.g., membrane potentials).
    neuron_mem = np.sum(
        np.array(out_ch_list[0:5]) * np.array(out_dim_list[0:5]) * np.array(out_dim_list[0:5])) * 0.7391 * 22

    # The final total area is the sum of the compute area and the neuron memory area.
    total_ar = total_compute_ar_ov + neuron_mem

    # This calculates the area of just the "digital" parts of the chip for analysis.
    digital_area = (Temp_Buff + Tile_buff + pe_per_tile * PE_ar) * n_tiles + np.sum(
        np.array(in_ch_list) * np.array(in_dim_list) * np.array(in_dim_list)) * 0.7391 * 22

    # ----------------------------------------------------------------------
    # 5. FINAL OUTPUT 
    # ----------------------------------------------------------------------
    # print(f'total_compute_area {total_ar} µm^2')

    # ----------------------------------------------------------------------
    # 5. RETURN A DETAILED BREAKDOWN OF THE AREA 
    # ----------------------------------------------------------------------
    # The function returns a tuple containing four specific area values. This is
    # useful for more detailed analysis of how the chip area is distributed.

    # The tuple contains the following four values, in order:
    #   1. neuron_mem: The total area dedicated to storing neuron states (e.g., membrane potentials).
    #   2. Core Compute Area: The combined area of all PEs and subtractors across all tiles. This represents the core processing hardware.
    #   3. digital_area: The total area of all digital components, including buffers, PEs, and input feature map memory.
    #   4. total_ar: The final, all-inclusive area of the entire chip. This is the same value that was just printed.
    return (neuron_mem, (pe_per_tile * PE_ar + (xbar_size / 8) * Sub) * n_tiles, digital_area, total_ar)

#@title Compute total Energy function
def compute_energy(in_ch_list, in_dim_list, out_ch_list, out_dim_list, xbar_size, k, n_tiles, device, time_steps):
    # All energies are in picoJoules (pJ)

    # ----------------------------------------------------------------------
    # 1. DEFINE ENERGY COST OF BASIC HARDWARE COMPONENTS 
    # ----------------------------------------------------------------------
    # These are the fundamental energy costs for different parts of the hardware.
    # They are like the price tags on individual LEGO bricks.

    # Set the energy cost for a crossbar operation based on the memory technology used.
    # RRAM (Resistive RAM) is more energy-efficient than SRAM (Static RAM).
    if device == 'rram':
        xbar_ar = 1.76423  # Energy for one operation in an RRAM crossbar
    elif device == 'sram':
        xbar_ar = 671.089  # Energy for one operation in an SRAM crossbar

    # --- Energy costs for other components ---
    Tile_buff = 397          # Energy for the buffer within a tile
    Temp_Buff = 0.2          # Energy for a temporary buffer
    Sub = 1.15E-6            # Energy for a subtractor circuit
    ADC = 2.03084            # Energy for an Analog-to-Digital Converter

    Htree = 19.64 * 8        # Energy for the H-Tree network (distributes data)
    MUX = 0.094245           # Energy for a multiplexer
    mem_fetch = 4.64         # Energy to fetch data from memory
    neuron = 1.274 * 4.0     # Energy to update a single neuron's state

    # ----------------------------------------------------------------------
    # 2. CALCULATE ENERGY FOR A SINGLE PROCESSING ENGINE (PE) CYCLE 
    # ----------------------------------------------------------------------
    # A PE is the core computational unit. We sum the costs of its parts
    # to find the energy it consumes in one operational cycle.

    # First, calculate the energy of the crossbar array within the PE.
    # Note: 'PE_energy_component'. It's based on area parameters but used for energy.
    PE_energy_component = k * k * xbar_ar + (xbar_size / 8) * (ADC + MUX)

    # Now, sum up all component energies to get the total energy for one PE cycle.
    PE_cycle_energy = Htree + mem_fetch + neuron + xbar_size / 8 * PE_energy_component + (xbar_size / 8) * 16 * Sub + (
                xbar_size / 8) * Temp_Buff + Tile_buff

    # ----------------------------------------------------------------------
    # 3. COMPUTE TOTAL ENERGY LAYER BY LAYER  LAYER-BY-LAYER 
    # ----------------------------------------------------------------------
    # We loop through each layer of the neural network to calculate its
    # specific energy consumption and add it to the grand total.

    energy_layerwise = []  # A list to store the energy of each layer
    tot_energy = 0         # Initialize total energy to zero
    tot_pe_cycle = 0       # Initialize total PE cycles to zero

    # Loop through each layer of the SNN
    for i in range(len(out_ch_list)):
        # Calculate the total number of cycles the PEs need to run for this layer.
        # This depends on the number of input/output channels and the size of the feature maps.
        Total_PE_cycle = np.ceil(out_ch_list[i] / xbar_size) * np.ceil(in_ch_list[i] / xbar_size) * (
                    out_dim_list[i] * out_dim_list[i])

        # Calculate the energy for this layer.
        # It's the number of cycles * energy per cycle * number of time steps (for SNNs).
        layer_energy = Total_PE_cycle * PE_cycle_energy * time_steps

        # Add this layer's energy to the total energy.
        tot_energy += layer_energy

        # Keep track of the total PE cycles.
        tot_pe_cycle += Total_PE_cycle

        # Store the layer-specific energy.
        energy_layerwise.append(layer_energy)

    # ----------------------------------------------------------------------
    # 4. FINAL OUTPUT 
    # ----------------------------------------------------------------------
    # Print the final calculated total energy.
    print(f'total_energy {tot_energy} pJ')

    # Return the list containing the energy consumption of each individual layer.
    return energy_layerwise

#@title Compute Latency function
# All latencies are in nanoseconds (ns)
def latency_gen(in_ch_list, out_ch_list, out_dim_list, k, xbar, pe_per_tile, PE_cycle, time_steps, pipeline_overlap_percent=25):
    # ----------------------------------------------------------------------
    # 1. CALCULATE PE & TILE REQUIREMENTS (Similar to other functions) 
    # ----------------------------------------------------------------------
    # First, it repeats the initial steps of calculating how many PEs are needed
    # for each layer and checking if they are in an efficient order.

    num_layer = len(out_ch_list)
    pe_list = []

    for i in range(num_layer):
        # NOTE: Unlike pe_tile_count, this does not use np.ceil, which implies
        # this calculation might be a simplification for the latency model.
        num_pe = (in_ch_list[i] / xbar) * (out_ch_list[i] / xbar)
        pe_list.append(num_pe)

    # Sanity check for layer order
    flag = 0
    if (pe_list == sorted(pe_list)):
        flag = 1
    if (flag):
        print("Layers in order")
    else:
        print("Check layer order")
        return

    # Calculate total PEs and Tiles
    num_pe = sum(pe_list)
    if (num_pe % pe_per_tile != 0):
        num_tiles = (num_pe // pe_per_tile) + 1
    else:
        num_tiles = num_pe / pe_per_tile

    # ----------------------------------------------------------------------
    # 2. MAP PEs to PHYSICAL TILES 
    # ----------------------------------------------------------------------
    # This section models how the PEs for different layers are physically placed onto the hardware tiles.
    # This is important for understanding data movement, but this information isn't
    # directly used in the final latency calculation in this specific version of the code.

    tile_mat = np.zeros(int(num_tiles * pe_per_tile)) # A flat array representing all PE slots on all tiles.

    # Fill the array, assigning each layer's PEs to the available slots.
    i = 0
    while (i < num_tiles):
        for j in range(len(pe_list)):
            tile_mat[i:i + int(pe_list[j])] = j + 1 # Assign layer number (j+1)
            i = i + int(pe_list[j])
    
    
    # Reshape the flat array into a 2D matrix where rows are tiles and columns are PEs.
    tile_mat = tile_mat.reshape((int(num_tiles), pe_per_tile))
    # display(tile_mat)

    # The following code (tile_dist, mult_tile) analyzes this distribution
    # but the results are not used later in the latency calculation itself.
    tile_dist = np.zeros((int(num_tiles), num_layer))

    
    for i in range(int(num_tiles)):
        for j in range(num_layer):
            tile_dist[i, j] = np.sum(tile_mat[i] == (j + 1))

    mult_tile = []
    for i in range(num_layer):
        if (np.sum(tile_dist[:, i] != 0) == 1):
            mult_tile.append(0)
        else:
            mult_tile.append(1)

    # ----------------------------------------------------------------------
    # 3. MODEL THE PIPELINED EXECUTION SCHEDULE 
    # ----------------------------------------------------------------------
    # This is the core of the latency calculation. It determines the start and
    # stop "timestamps" (in terms of clock cycles) for each layer.

    # 'cp' defines the "checkpoint" for pipelining. A value of 0.75 means
    # the next layer can start once the current layer has completed 75% of its work.
    cp = [1-pipeline_overlap_percent/100.0] * (num_layer+1) # A list of checkpoint percentages for each layer.

    # Calculate the number of cycles until the checkpoint is reached for each layer.
    checkpoints_1 = []
    for i in range(num_layer):
        cycles_to_checkpoint = int(cp[i] * out_dim_list[i] * out_dim_list[i] * out_ch_list[i]) * time_steps
        checkpoints_1.append(cycles_to_checkpoint)

    # Calculate the cumulative sum of these checkpoint cycles. This tells us
    # when each subsequent layer is allowed to start.
    temp = np.cumsum(checkpoints_1)

    # 'starts' is an array of the absolute start time (in cycles) for each layer.
    starts = 1 + temp
    starts = np.insert(starts, 0, 1) # Layer 0 starts at cycle 1.
    starts = starts[:-1]

    # Now, calculate the *total* number of cycles required to complete each layer.
    checkpoints_2 = []
    for i in range(num_layer):
        total_layer_cycles = int(out_dim_list[i] * out_dim_list[i] * out_ch_list[i]) * time_steps
        checkpoints_2.append(total_layer_cycles)

    # 'halts' is an array of the absolute HALT time (in cycles) for each layer.
    # It's calculated by adding the total cycles for a layer ('checkpoints_2')
    # to the cumulative time of the previous layer's checkpoints.
    temp = np.cumsum(checkpoints_1)
    temp = temp[:-1]
    temp = np.insert(temp, 0, 0)
    halts = checkpoints_2 + temp

    # ----------------------------------------------------------------------
    # 4. CALCULATE AND PRINT FINAL LATENCY 
    # ----------------------------------------------------------------------
    # The final latency is determined by the halt time of the very last layer.

    # Get the final halt time (in cycles) from the last element of the 'halts' array.
    final_halt_cycle = np.array(halts)[-1]
    # Convert the total cycles into nanoseconds by multiplying by the time per cycle.
    final_latency_ns = final_halt_cycle * PE_cycle

    print(f'Final Latency {final_latency_ns} ns')
