#!/usr/bin/env python
# coding: utf-8

# 

# 

# In[1]:


import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import hist
from hist import Hist
import pandas as pd
import matplotlib

def load_dataset(file_path, detector_names=("VirtualDetector", "Sample", "NTuple")):
    """
    Load datasets from a ROOT file for multiple detector groups.

    Args:
        file_path (str): Path to the ROOT file.
        detector_names (tuple or list): Names of detector groups inside the file.
    
    Returns:
        dict: Dictionary of detector_name -> uproot.ReadOnlyDirectory.
    """
    loaded = {}
    try:
        file = uproot.open(file_path)
        for name in detector_names:
            if name in file:
                loaded[name] = file[name]
                print(f"Loaded detector group '{name}' from: {file_path}")
            else:
                print(f"Detector group '{name}' not found in: {file_path}")
        return loaded
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return {}

def list_detector_keys(detector_group):
    """
    List all available virtual detectors in the detector group.

    Args:
        detector_group (uproot.ReadOnlyDirectory): A specific detector group object.
    
    Returns:
        list: List of keys (detector positions) in the group.
    """
    return detector_group.keys() if detector_group else []

def get_virtual_detector_data(group, detector_key, variable_key):
    """
    Retrieve data from a specific detector and variable.

    Args:
        group (uproot.ReadOnlyDirectory): The detector group (e.g., VirtualDetector or Sample).
        detector_key (str): The detector position key.
        variable_key (str): The variable to extract (e.g., 'PDGid', 'x', 'y').

    Returns:
        awkward.Array or None: The extracted data or None if not found.
    """
    if group is None:
        print("Detector group is not loaded properly.")
        return None
    if detector_key not in group:
        print(f"Detector key '{detector_key}' not found. Available: {group.keys()}")
        return None
    if variable_key not in group[detector_key].keys():
        print(f"Variable '{variable_key}' not in '{detector_key}'. Available: {group[detector_key].keys()}")
        return None
    try:
        return group[detector_key][variable_key].array()
    except Exception as e:
        print(f"Error extracting data: {e}")
        return None

def plot_1D_histogram(data, xlabel, title, bins=None, xlim=None, discrete=False, save_path=None):
    """
    Plot a 1D histogram with automatic bin width of 1 if bins are not specified.

    Args:
        data (array-like): The data to plot.
        xlabel (str): Label for the x-axis.
        title (str): Title of the histogram.
        bins (int or None): Number of bins (if None, auto-bin width = 1).
        xlim (tuple or None): (min, max) limits for x-axis.
        discrete (bool): If True, treat data as discrete and use a bar plot.
    """
    # Convert awkward arrays to numpy if needed
    data = ak.to_numpy(ak.flatten(data, axis=None)) if isinstance(data, ak.Array) else np.array(data)

    # Auto-detect binning if bins is not provided
    if bins is None:
        data_min, data_max = data.min(), data.max()
        if xlim is not None:
            data_min, data_max = xlim  # Override with xlim if given
        bins = np.arange(data_min, data_max + 1, 1)  # Ensure bin width = 1

    plt.figure(figsize=(8, 6))

    if discrete:
        # Discrete case: Use bar plot
        unique_values, counts = np.unique(data, return_counts=True)
        plt.bar(unique_values, counts, width=10.0, edgecolor='black', alpha=0.7)
        plt.xticks(unique_values)
    else:
        # Continuous case: Use histogram
        plt.hist(data, bins=bins, alpha=0.7, edgecolor='black')

    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True)

    # Set x-axis limits if provided
    if xlim is not None:
        plt.xlim(xlim)

    # Save plot if save_path is specified
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # High-resolution PNG
        print(f"Plot saved to {save_path}")

    plt.show()

def plot_1D_histogram_overlay(datasets, labels, xlabel, title, bins=None, alpha=0.7, colors=None, xmin=None, xmax=None, discrete=False, save_path=None):
    """
    Plot overlaid 1D histograms for multiple datasets with auto binning and discrete support.

    Args:
        datasets (list of array-like): List of datasets to plot.
        labels (list of str): List of labels corresponding to each dataset.
        xlabel (str): Label for the x-axis.
        title (str): Title of the histogram.
        bins (int or None): Number of bins (if None, auto bin width = 1).
        alpha (float): Transparency for better visibility.
        colors (list of str): List of colors for each dataset.
        xmin (float): Minimum x-axis value (optional).
        xmax (float): Maximum x-axis value (optional).
        discrete (bool): If True, use bar plots based on unique values.
        save_path (str): Optional path to save the plot as a PNG file.
    """
    # Convert awkward arrays to numpy if needed
    datasets = [ak.to_numpy(ak.flatten(data, axis=None)) if isinstance(data, ak.Array) else np.array(data) for data in datasets]

    # Determine plot range
    data_min = min(min(d) for d in datasets)
    data_max = max(max(d) for d in datasets)

    if xmin is not None:
        data_min = xmin
    if xmax is not None:
        data_max = xmax

    # Define colors if not provided
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, len(datasets)))

    plt.figure(figsize=(8, 6))

    if discrete:
        # For discrete values, use bar plots
        unique_values = np.unique(np.concatenate(datasets))
        for data, label, color in zip(datasets, labels, colors):
            counts, _ = np.histogram(data, bins=np.append(unique_values, unique_values[-1] + 1))
            plt.bar(unique_values, counts, width=10.0, alpha=alpha, edgecolor='black', label=label, color=color)
        plt.xticks(unique_values)
    else:
        # For continuous data, use overlaid histograms
        if bins is None:
            bins = np.arange(data_min, data_max + 1, 1)  # Auto bin width = 1
        for data, label, color in zip(datasets, labels, colors):
            plt.hist(data, bins=bins, alpha=alpha, edgecolor='black', label=label, color=color)

    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if xmin is not None and xmax is not None:
        plt.xlim(xmin, xmax)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_1D_histogram_stacked(datasets, labels, xlabel, title, bins=None, alpha=0.7, colors=None, xmin=None, xmax=None, discrete=False, save_path=None):
    """
    Plot a stacked 1D histogram for multiple datasets with auto binning based on data range.

    Args:
        datasets (list of array-like): List of datasets to plot.
        labels (list of str): List of labels corresponding to each dataset.
        xlabel (str): Label for the x-axis.
        title (str): Title of the histogram.
        bins (int or None): Number of bins (if None, auto-bin width = 1).
        alpha (float): Transparency for better visibility.
        colors (list of str): List of colors for each dataset.
        xmin (float): Minimum x-axis value (if None, auto-detect from data).
        xmax (float): Maximum x-axis value (if None, auto-detect from data).
        discrete (bool): If True, treat data as discrete and use a bar plot.
    """
    # Convert awkward arrays to numpy if needed
    datasets = [ak.to_numpy(ak.flatten(data, axis=None)) if isinstance(data, ak.Array) else np.array(data) for data in datasets]

    # Auto-detect bin range
    if bins is None:
        data_min = min(min(d) for d in datasets)
        data_max = max(max(d) for d in datasets)
        if xmin is not None and xmax is not None:
            data_min, data_max = xmin, xmax  # Override with given range
        bins = np.arange(data_min, data_max + 1, 1)  # Ensure bin width = 1

    plt.figure(figsize=(8, 6))

    if discrete:
        # Discrete case: Use bar plot
        unique_values = np.unique(np.concatenate(datasets))
        counts_list = [np.histogram(d, bins=np.append(unique_values, unique_values[-1] + 1))[0] for d in datasets]

        for counts, label, color in zip(counts_list, labels, colors or plt.cm.viridis(np.linspace(0, 1, len(datasets)))):
            plt.bar(unique_values, counts, width=10.0, alpha=alpha, edgecolor='black', label=label, color=color)
        plt.xticks(unique_values)
    else:
        # Continuous case: Use histogram
        plt.hist(datasets, bins=bins, stacked=True, alpha=alpha, label=labels, color=colors or plt.cm.viridis(np.linspace(0, 1, len(datasets))), edgecolor='black')

    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Set x-axis limits if provided
    if xmin is not None and xmax is not None:
        plt.xlim(xmin, xmax)

    # Save plot if save_path is specified
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # High-resolution PNG
        print(f"Plot saved to {save_path}")

    plt.show()

def plot_2D_histogram(data_x, data_y, xlabel, ylabel, title, xlim=None, ylim=None, vmin=None, vmax=None, bins=50, save_path=None):
    """
    Plot a 2D histogram with optional axis and color range control.

    Args:
        data_x (array-like): X-axis data.
        data_y (array-like): Y-axis data.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the histogram.
        xlim (tuple): (min, max) for x-axis.
        ylim (tuple): (min, max) for y-axis.
        vmin (float): Min value for color scale.
        vmax (float): Max value for color scale.
        bins (int): Number of bins (default: 50).
    """
    plt.figure(figsize=(8, 6))

    # Ensure proper conversion from awkward array to numpy
    data_x = ak.to_numpy(ak.flatten(data_x, axis=None)) if isinstance(data_x, ak.Array) else np.array(data_x)
    data_y = ak.to_numpy(ak.flatten(data_y, axis=None)) if isinstance(data_y, ak.Array) else np.array(data_y)

    # Remove any NaN values that may cause issues
    valid_mask = ~np.isnan(data_x) & ~np.isnan(data_y)
    data_x = data_x[valid_mask]
    data_y = data_y[valid_mask]

    # Create histogram
    h = plt.hist2d(data_x, data_y, bins=bins, cmap="viridis", vmin=vmin, vmax=vmax)

    plt.colorbar(label="Count")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Apply axis limits only if specified
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.grid(True)

    # Save plot if save_path is specified
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # High-resolution PNG
        print(f"Plot saved to {save_path}")
        
    plt.show()

def plot_scatter(data_x, data_y, xlabel, ylabel, title, xlim=None, ylim=None, alpha=1.0, s=10, save_path=None):
    """
    Plot a scatter plot for x vs y.
    
    Args:
        data_x (array-like): X-axis data.
        data_y (array-like): Y-axis data.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
        xlim (tuple): (min, max) for x-axis.
        ylim (tuple): (min, max) for y-axis.
        alpha (float): Transparency of points.
        s (int): Marker size.
    """
    if len(data_x) != 0 and len(data_y) != 0:
        # Convert Awkward arrays to NumPy only if necessary
        if isinstance(data_x, ak.Array):
            if data_x.ndim > 1:  # Check if it's nested
                data_x = ak.to_numpy(ak.flatten(data_x))
            else:
                data_x = ak.to_numpy(data_x)
    
        if isinstance(data_y, ak.Array):
            if data_y.ndim > 1:
                data_y = ak.to_numpy(ak.flatten(data_y))
            else:
                data_y = ak.to_numpy(data_y)
    
        # Default limits
        if xlim is None:
            xlim = (np.min(data_x), np.max(data_x))
        if ylim is None:
            ylim = (np.min(data_y), np.max(data_y))
    
        # Scatter plot
        plt.figure(figsize=(8, 8))
        label = f"Count = {len(data_x)}"
        plt.scatter(data_x, data_y, alpha=alpha, s=s, color="blue", label=label)
         
        plt.legend(loc="upper right", fontsize="small")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.grid(True)
    
        # Save plot if save_path is specified
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')  # High-resolution PNG
            print(f"Plot saved to {save_path}")
    
        plt.show()


def filter_by_pdgid(data_pdgid, pdg_values, save_csv_path=None, print_counts=False):
    """
    Filter data based on specific PDG ID values, optionally print and save counts.

    Args:
        data_pdgid (awkward.Array): PDG ID data array.
        pdg_values (int or list): Single PDG ID or a list of PDG IDs to filter.
        save_csv_path (str or None): Path to save PDG ID counts as a CSV file.
        print_counts (bool): Whether to print the counts of each matched PDG ID.

    Returns:
        awkward.Array: Boolean mask selecting only the desired PDG IDs.
    """
    # Ensure pdg_values is a list
    if not isinstance(pdg_values, (list, tuple)):
        pdg_values = [pdg_values]

    # Create boolean mask
    mask = ak.any(data_pdgid[:, None] == ak.Array(pdg_values), axis=1)

    # Print and/or save counts if requested
    if print_counts or save_csv_path:
        # Extract matching PDG values
        filtered_pdg = ak.to_numpy(data_pdgid[mask])
        unique_vals, counts = np.unique(filtered_pdg, return_counts=True)

        if print_counts:
            print("PDG ID counts:")
            for pdg, count in zip(unique_vals, counts):
                print(f"  PDG {int(pdg)}: {count}")

        if save_csv_path:
            df = pd.DataFrame({
                "PDG_ID": unique_vals,
                "Count": counts
            })
            df.to_csv(save_csv_path, index=False)
            print(f"PDG ID counts saved to: {save_csv_path}")

    return mask


def filter_by_pdgid_and_initpos(
    data_pdgid,
    pdg_values,
    init_x=None,
    init_y=None,
    init_z=None,
    data_initx=None,
    data_inity=None,
    data_initz=None,
    save_csv_path=None,
    print_counts=False ):
    """
    Filter events by PDG ID and optional initial position ranges (InitX/Y/Z).

    Args:
        data_pdgid (ak.Array): PDG ID array.
        pdg_values (int or list): Single PDG ID or list of IDs.
        init_x (tuple or None): (min_x, max_x) or None to skip.
        init_y (tuple or None): (min_y, max_y) or None to skip.
        init_z (tuple or None): (min_z, max_z) or None to skip.
        data_initx (ak.Array or None): InitX array (required if init_x is used).
        data_inity (ak.Array or None): InitY array (required if init_y is used).
        data_initz (ak.Array or None): InitZ array (required if init_z is used).
        save_csv_path (str or None): Path to save count summary as CSV.
        print_counts (bool): Print matching PDG counts if True.

    Returns:
        ak.Array: Boolean mask satisfying all specified filters.
    """
    # Ensure PDG values is a list
    if not isinstance(pdg_values, (list, tuple)):
        pdg_values = [pdg_values]
    
    # PDG ID mask
    mask = ak.any(data_pdgid[:, None] == ak.Array(pdg_values), axis=1)

    # InitX filter
    if init_x is not None:
        if data_initx is None:
            raise ValueError("data_initx is required when filtering on init_x.")
        mask = mask & (data_initx >= init_x[0]) & (data_initx <= init_x[1])
    
    # InitY filter
    if init_y is not None:
        if data_inity is None:
            raise ValueError("data_inity is required when filtering on init_y.")
        mask = mask & (data_inity >= init_y[0]) & (data_inity <= init_y[1])
    
    # InitZ filter
    if init_z is not None:
        if data_initz is None:
            raise ValueError("data_initz is required when filtering on init_z.")
        mask = mask & (data_initz >= init_z[0]) & (data_initz <= init_z[1])

    # Print and/or save counts
    if print_counts or save_csv_path:
        filtered_pdg = ak.to_numpy(data_pdgid[mask])
        unique_vals, counts = np.unique(filtered_pdg, return_counts=True)

        if print_counts:
            print("Filtered PDG ID counts:")
            for pdg, count in zip(unique_vals, counts):
                print(f"  PDG {int(pdg)}: {count}")

        if save_csv_path:
            df = pd.DataFrame({
                "PDG_ID": unique_vals,
                "Count": counts
            })
            df.to_csv(save_csv_path, index=False)
            print(f"Saved PDG ID counts to: {save_csv_path}")

    return mask


# In[2]:


# Load both VirtualDetector and Sample groups
detector_groups = load_dataset("./g4beamline.root")

# Access each group separately
virtual_group = detector_groups.get("VirtualDetector")
sample_group = detector_groups.get("Sample")
trace_group = detector_groups.get("Trace")

# List detectors
print("VirtualDetector keys:", list_detector_keys(virtual_group))
print("Sample keys:", list_detector_keys(sample_group))
print("Trace keys", list_detector_keys(trace_group))

# List of variables 
variable_keys = ["PDGid", "x", "y", "z", "t", "Ex", "Ey", "Ez", "Px", "Py", "Pz", "ParentID", "InitX", "InitY", "InitZ", "InitKE", "TrackID", "EventID"]  

# Dictionary to store extracted data
detector_data = {}

# Loop through each detector group
for group_name, group in detector_groups.items():
    if group is None:
        continue
    for detector_key in group.keys():
        # Clean the key to make it a valid Python variable name
        clean_key = detector_key.replace(";1", "").replace("-", "_").replace(" ", "_")
        for var in variable_keys:
            # Construct variable name: e.g., PDGid_Det_Solenoid_1
            var_name = f"{var}_{clean_key}"
            # Load the variable data
            data = get_virtual_detector_data(group, detector_key, var)
            # Store it in the dictionary
            detector_data[var_name] = data
            print(f"Loaded {var_name}")



# In[63]:


# ====================================================================================================================
# ====================================================================================================================
# Det_Solenoid_1
# ====================================================================================================================
# ====================================================================================================================

# Check if PDG ID data was successfully extracted
if detector_data["PDGid_Det_Solenoid_1"] is not None:
    
    # =======================================================================================
    # Select PDGID
    # =======================================================================================
    
    # Convert to numpy array (for compatibility)
    PDGid_Det_Solenoid_1 = ak.to_numpy(detector_data["PDGid_Det_Solenoid_1"]).astype(int)

    # PDG ID for muons, pions, and kaons 
    PDGID_muons = [-13, 13]
    PDGID_muplus = [13]
    PDGID_muminus = [-13]
    PDGID_pions = [-211, 211]
    PDGID_piplus = [211]
    PDGID_piminus = [-211]
    PDGID_muons_pions_kaons = [13, -13, 211, -211, 321, -321]
    
    # Apply PDG ID filtering 
    select_muons_Det_Solenoid_1 = filter_by_pdgid(PDGid_Det_Solenoid_1, PDGID_muons)
    select_muplus_Det_Solenoid_1 = filter_by_pdgid(PDGid_Det_Solenoid_1, PDGID_muplus)
    select_muminus_Det_Solenoid_1 = filter_by_pdgid(PDGid_Det_Solenoid_1, PDGID_muminus)
    
    select_pions_Det_Solenoid_1 = filter_by_pdgid(PDGid_Det_Solenoid_1, PDGID_pions)
    select_piplus_Det_Solenoid_1 = filter_by_pdgid(PDGid_Det_Solenoid_1, PDGID_piplus)
    select_piminus_Det_Solenoid_1 = filter_by_pdgid(PDGid_Det_Solenoid_1, PDGID_piminus)
    
    select_muons_pions_kaons_Det_Solenoid_1 = filter_by_pdgid(PDGid_Det_Solenoid_1, PDGID_muons_pions_kaons, "./Target_Rotate_theta_0_Beam_alpha_0_beta_0_End_of_Solenoid_1.csv", True)
    select_protons_Det_Solenoid_1 = filter_by_pdgid(PDGid_Det_Solenoid_1, 2212)
    
    # =======================================================================================
    # Extract positions x, y for muons at Det_Solenoid_1
    # =======================================================================================
    
    x_Muons_Det_Solenoid_1 = detector_data['x_Det_Solenoid_1'][select_muons_Det_Solenoid_1]
    y_Muons_Det_Solenoid_1 = detector_data['y_Det_Solenoid_1'][select_muons_Det_Solenoid_1]
    
    # =======================================================================================
    # Extract positions x, y for pions at Det_Solenoid_1
    # =======================================================================================

    x_Pions_Det_Solenoid_1 = detector_data['x_Det_Solenoid_1'][select_pions_Det_Solenoid_1]
    y_Pions_Det_Solenoid_1 = detector_data['y_Det_Solenoid_1'][select_pions_Det_Solenoid_1]

    # =======================================================================================
    # Calculate Muons Energies at Det_Solenoid_1
    # =======================================================================================

    # Muons
    Px_Muons_Det_Solenoid_1 = detector_data['Px_Det_Solenoid_1'][select_muons_Det_Solenoid_1]
    Py_Muons_Det_Solenoid_1 = detector_data['Py_Det_Solenoid_1'][select_muons_Det_Solenoid_1]
    Pz_Muons_Det_Solenoid_1 = detector_data['Pz_Det_Solenoid_1'][select_muons_Det_Solenoid_1]

    P_Muons_Det_Solenoid_1 = np.sqrt(Px_Muons_Det_Solenoid_1**2 + Py_Muons_Det_Solenoid_1**2 + Pz_Muons_Det_Solenoid_1**2)
    
    # Mu plus
    Px_MuPlus_Det_Solenoid_1 = detector_data['Px_Det_Solenoid_1'][select_muplus_Det_Solenoid_1]
    Py_MuPlus_Det_Solenoid_1 = detector_data['Py_Det_Solenoid_1'][select_muplus_Det_Solenoid_1]
    Pz_MuPlus_Det_Solenoid_1 = detector_data['Pz_Det_Solenoid_1'][select_muplus_Det_Solenoid_1]

    P_MuPlus_Det_Solenoid_1 = np.sqrt(Px_MuPlus_Det_Solenoid_1**2 + Py_MuPlus_Det_Solenoid_1**2 + Pz_MuPlus_Det_Solenoid_1**2)

    # Mu minus
    Px_MuMinus_Det_Solenoid_1 = detector_data['Px_Det_Solenoid_1'][select_muminus_Det_Solenoid_1]
    Py_MuMinus_Det_Solenoid_1 = detector_data['Py_Det_Solenoid_1'][select_muminus_Det_Solenoid_1]
    Pz_MuMinus_Det_Solenoid_1 = detector_data['Pz_Det_Solenoid_1'][select_muminus_Det_Solenoid_1]

    P_MuMinus_Det_Solenoid_1 = np.sqrt(Px_MuMinus_Det_Solenoid_1**2 + Py_MuMinus_Det_Solenoid_1**2 + Pz_MuMinus_Det_Solenoid_1**2)

    Muon_Mass = 105.67 # MeV
    # Muons
    E_Muons_Det_Solenoid_1 = np.sqrt(Muon_Mass**2 + Px_Muons_Det_Solenoid_1**2 + Py_Muons_Det_Solenoid_1**2 + Pz_Muons_Det_Solenoid_1**2)
    KE_Muons_Det_Solenoid_1 = E_Muons_Det_Solenoid_1 - Muon_Mass
    # Mu plus
    E_MuPlus_Det_Solenoid_1 = np.sqrt(Muon_Mass**2 + Px_MuPlus_Det_Solenoid_1**2 + Py_MuPlus_Det_Solenoid_1**2 + Pz_MuPlus_Det_Solenoid_1**2)
    KE_MuPlus_Det_Solenoid_1 = E_MuPlus_Det_Solenoid_1 - Muon_Mass
    # Mu minus
    E_MuMinus_Det_Solenoid_1 = np.sqrt(Muon_Mass**2 + Px_MuMinus_Det_Solenoid_1**2 + Py_MuMinus_Det_Solenoid_1**2 + Pz_MuMinus_Det_Solenoid_1**2)
    KE_MuMinus_Det_Solenoid_1 = E_MuMinus_Det_Solenoid_1 - Muon_Mass
    
    # =======================================================================================
    # Calculate Pions Energies at Det_Solenoid_1
    # =======================================================================================

    # Pions
    Px_Pions_Det_Solenoid_1 = detector_data['Px_Det_Solenoid_1'][select_pions_Det_Solenoid_1]
    Py_Pions_Det_Solenoid_1 = detector_data['Py_Det_Solenoid_1'][select_pions_Det_Solenoid_1]
    Pz_Pions_Det_Solenoid_1 = detector_data['Pz_Det_Solenoid_1'][select_pions_Det_Solenoid_1]

    P_Pions_Det_Solenoid_1 = np.sqrt(Px_Pions_Det_Solenoid_1**2 + Py_Pions_Det_Solenoid_1**2 + Pz_Pions_Det_Solenoid_1**2)

    # Pi plus
    Px_PiPlus_Det_Solenoid_1 = detector_data['Px_Det_Solenoid_1'][select_piplus_Det_Solenoid_1]
    Py_PiPlus_Det_Solenoid_1 = detector_data['Py_Det_Solenoid_1'][select_piplus_Det_Solenoid_1]
    Pz_PiPlus_Det_Solenoid_1 = detector_data['Pz_Det_Solenoid_1'][select_piplus_Det_Solenoid_1]

    P_PiPlus_Det_Solenoid_1 = np.sqrt(Px_PiPlus_Det_Solenoid_1**2 + Py_PiPlus_Det_Solenoid_1**2 + Pz_PiPlus_Det_Solenoid_1**2)

    # Pi minus
    Px_PiMinus_Det_Solenoid_1 = detector_data['Px_Det_Solenoid_1'][select_piminus_Det_Solenoid_1]
    Py_PiMinus_Det_Solenoid_1 = detector_data['Py_Det_Solenoid_1'][select_piminus_Det_Solenoid_1]
    Pz_PiMinus_Det_Solenoid_1 = detector_data['Pz_Det_Solenoid_1'][select_piminus_Det_Solenoid_1]

    P_PiMinus_Det_Solenoid_1 = np.sqrt(Px_PiMinus_Det_Solenoid_1**2 + Py_PiMinus_Det_Solenoid_1**2 + Pz_PiMinus_Det_Solenoid_1**2)
    
    Pion_Mass = 139.57 # MeV
    # Pions
    E_Pions_Det_Solenoid_1 = np.sqrt(Pion_Mass**2 + Px_Pions_Det_Solenoid_1**2 + Py_Pions_Det_Solenoid_1**2 + Pz_Pions_Det_Solenoid_1**2)
    KE_Pions_Det_Solenoid_1 = E_Pions_Det_Solenoid_1 - Pion_Mass
    # Pi plus
    E_PiPlus_Det_Solenoid_1 = np.sqrt(Pion_Mass**2 + Px_PiPlus_Det_Solenoid_1**2 + Py_PiPlus_Det_Solenoid_1**2 + Pz_PiPlus_Det_Solenoid_1**2)
    KE_PiPlus_Det_Solenoid_1 = E_PiPlus_Det_Solenoid_1 - Pion_Mass
    # Pions
    E_PiMinus_Det_Solenoid_1 = np.sqrt(Pion_Mass**2 + Px_PiMinus_Det_Solenoid_1**2 + Py_PiMinus_Det_Solenoid_1**2 + Pz_PiMinus_Det_Solenoid_1**2)
    KE_PiMinus_Det_Solenoid_1 = E_PiMinus_Det_Solenoid_1 - Pion_Mass

    # =======================================================================================
    # Calculate Protons Energies at Det_Solenoid_1
    # =======================================================================================

    Px_Protons_Det_Solenoid_1 = detector_data['Px_Det_Solenoid_1'][select_protons_Det_Solenoid_1]
    Py_Protons_Det_Solenoid_1 = detector_data['Py_Det_Solenoid_1'][select_protons_Det_Solenoid_1]
    Pz_Protons_Det_Solenoid_1 = detector_data['Pz_Det_Solenoid_1'][select_protons_Det_Solenoid_1]
    
    P_Protons_Det_Solenoid_1 = np.sqrt(Px_Protons_Det_Solenoid_1**2 + Py_Protons_Det_Solenoid_1**2 + Pz_Protons_Det_Solenoid_1**2)
    
    Proton_Mass = 938.27 # MeV
    
    E_Protons_Det_Solenoid_1 = np.sqrt(Proton_Mass**2 + Px_Protons_Det_Solenoid_1**2 + Py_Protons_Det_Solenoid_1**2 + Pz_Protons_Det_Solenoid_1**2)
    KE_Protons_Det_Solenoid_1 = E_Protons_Det_Solenoid_1 - Proton_Mass
    
    # =======================================================================================
    # Plotting
    # =======================================================================================
    
    # Plot 1D histogram of PDG IDs 
    plot_1D_histogram(
    detector_data['PDGid_Det_Solenoid_1'][select_muons_pions_kaons_Det_Solenoid_1], "PDG ID",
    r"PDG ID Distribution at the End of the Solenoid 1 (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
    discrete=True, save_path="PDGID_muons_pions_kaons_Det_Solenoid_1.png"
    ) 

    plot_1D_histogram(
    detector_data['PDGid_Det_Solenoid_1'][select_protons_Det_Solenoid_1], "PDG ID",
    r"PDG ID Distribution at the End of Solenoid 1 (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
    discrete=True, save_path="PDGID_protons_Det_Solenoid_1.png"
    )

    # Plot 1D histogram of Energies 
    plot_1D_histogram(
    E_Protons_Det_Solenoid_1, "Energy (MeV)",
    r"Energy Distribution of the Protons at the End of Solenoid 1 (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
    save_path="E_Protons_Det_Solenoid_1.png", bins=100
    )
    
    plot_1D_histogram(
    E_Muons_Det_Solenoid_1, "Energy (MeV)",
    r"Energy Distribution of the Muons at the End of Solenoid 1 (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
    save_path="E_Muons_Det_Solenoid_1.png", bins=100
    )

    plot_1D_histogram(
    E_Pions_Det_Solenoid_1, "Energy (MeV)",
    r"Energy Distribution of the Pions at the End of Solenoid 1 (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
    save_path="E_Pions_Det_Solenoid_1.png", bins=100
    )
    

    # # Plot stacked 1D histogram of PDG IDs with custom x-axis limits
    # plot_1D_histogram_overlay(
    # datasets=[PDGid_1000[select_muons_pions_kaons], PDGid_Det_Target[select_muons_pions_kaons_Det_Target], PDGid_200[select_muons_pions_kaons_200]], 
    # labels=["#theta = 0, #alpha = 0, #beta = 0", "600mm Tungsten", "200mm Tungsten"], 
    # xlabel="PDG ID", 
    # title="PDG ID Distribution at the End of Solenoid",
    # discrete=True,
    # save_path="Compare_PDGID_TungstenSize.png"
    # )

    # plot_1D_histogram_overlay(
    # datasets=[E_Muons_1000, E_Muons_Det_Target, E_Muons_200], 
    # labels=["#theta = 0, #alpha = 0, #beta = 0", "600mm Tungsten", "200mm Tungsten"], 
    # xlabel="Energy (Mev)", 
    # title="Energy Distribution of the Muons at the End of Solenoid",
    # bins=100,
    # save_path="Compare_EMuons_TungstenSize.png"
    # )

    # plot_1D_histogram_overlay(
    # datasets=[E_Protons_1000, E_Protons_Det_Target, E_Protons_200], 
    # labels=["#theta = 0, #alpha = 0, #beta = 0", "600mm Tungsten", "200mm Tungsten"], 
    # xlabel="Energy (Mev)", 
    # title="Energy Distribution of the Protons at the End of Solenoid",
    # bins=100,
    # save_path="Compare_EProtons_TungstenSize.png"
    # )

    # Plot 2D scatter (x vs y) for filtered muons
    plot_scatter(x_Muons_Det_Solenoid_1, y_Muons_Det_Solenoid_1, "x (mm)", "y (mm)", r"Muon Distribution at the End of Solenoid 1 (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",(-710, 710), (-710, 710), \
                s = 1, save_path = "Pos_Muons_Det_Solenoid_1.png")
    plot_scatter(x_Pions_Det_Solenoid_1, y_Pions_Det_Solenoid_1, "x (mm)", "y (mm)", r"Pions Distribution at the End of Solenoid 1 (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",(-710, 710), (-710, 710), 
                s = 1, save_path = "Pos_Pions_Det_Solenoid_1.png")
    # plot_scatter(x_Muons_Det_Target, y_Muons_Det_Target, "x (mm)", "y (mm)", "Muon Distribution at the End of Solenoid (600mm Tungsten)",(-5000, 5000), (-5000, 5000))
    # plot_scatter(x_Muons_200, y_Muons_200, "x (mm)", "y (mm)", "Muon Distribution at the End of Solenoid (200mm Tungsten)",(-5000, 5000), (-5000, 5000))
else:
    print("Failed to retrieve PDG ID data.")


# In[64]:


# ====================================================================================================================
# ====================================================================================================================
# Det_Solenoid_2
# ====================================================================================================================
# ====================================================================================================================

# Check if PDG ID data was successfully extracted
if detector_data["PDGid_Det_Solenoid_2"] is not None:
    
    # =======================================================================================
    # Select PDGID
    # =======================================================================================
    
    # Convert to numpy array (for compatibility)
    PDGid_Det_Solenoid_2 = ak.to_numpy(detector_data["PDGid_Det_Solenoid_2"]).astype(int)

    # PDG ID for muons, pions, and kaons 
    PDGID_muons = [-13, 13]
    PDGID_muplus = [13]
    PDGID_muminus = [-13]
    PDGID_pions = [-211, 211]
    PDGID_piplus = [211]
    PDGID_piminus = [-211]
    PDGID_muons_pions_kaons = [13, -13, 211, -211, 321, -321]
    
    # Apply PDG ID filtering 
    select_muons_Det_Solenoid_2 = filter_by_pdgid(PDGid_Det_Solenoid_2, PDGID_muons)
    select_muplus_Det_Solenoid_2 = filter_by_pdgid(PDGid_Det_Solenoid_2, PDGID_muplus)
    select_muminus_Det_Solenoid_2 = filter_by_pdgid(PDGid_Det_Solenoid_2, PDGID_muminus)
    
    select_pions_Det_Solenoid_2 = filter_by_pdgid(PDGid_Det_Solenoid_2, PDGID_pions)
    select_piplus_Det_Solenoid_2 = filter_by_pdgid(PDGid_Det_Solenoid_2, PDGID_piplus)
    select_piminus_Det_Solenoid_2 = filter_by_pdgid(PDGid_Det_Solenoid_2, PDGID_piminus)
    
    select_muons_pions_kaons_Det_Solenoid_2 = filter_by_pdgid(PDGid_Det_Solenoid_2, PDGID_muons_pions_kaons, "./Target_Rotate_theta_0_Beam_alpha_0_beta_0_End_of_Solenoid_2.csv", True)
    select_protons_Det_Solenoid_2 = filter_by_pdgid(PDGid_Det_Solenoid_2, 2212)
    
    # =======================================================================================
    # Extract positions x, y for muons at Det_Solenoid_2
    # =======================================================================================
    
    x_Muons_Det_Solenoid_2 = detector_data['x_Det_Solenoid_2'][select_muons_Det_Solenoid_2]
    y_Muons_Det_Solenoid_2 = detector_data['y_Det_Solenoid_2'][select_muons_Det_Solenoid_2]
    
    # =======================================================================================
    # Extract positions x, y for pions at Det_Solenoid_2
    # =======================================================================================

    x_Pions_Det_Solenoid_2 = detector_data['x_Det_Solenoid_2'][select_pions_Det_Solenoid_2]
    y_Pions_Det_Solenoid_2 = detector_data['y_Det_Solenoid_2'][select_pions_Det_Solenoid_2]

    # =======================================================================================
    # Calculate Muons Energies at Det_Solenoid_2
    # =======================================================================================

    # Muons
    Px_Muons_Det_Solenoid_2 = detector_data['Px_Det_Solenoid_2'][select_muons_Det_Solenoid_2]
    Py_Muons_Det_Solenoid_2 = detector_data['Py_Det_Solenoid_2'][select_muons_Det_Solenoid_2]
    Pz_Muons_Det_Solenoid_2 = detector_data['Pz_Det_Solenoid_2'][select_muons_Det_Solenoid_2]

    P_Muons_Det_Solenoid_2 = np.sqrt(Px_Muons_Det_Solenoid_2**2 + Py_Muons_Det_Solenoid_2**2 + Pz_Muons_Det_Solenoid_2**2)
    
    # Mu plus
    Px_MuPlus_Det_Solenoid_2 = detector_data['Px_Det_Solenoid_2'][select_muplus_Det_Solenoid_2]
    Py_MuPlus_Det_Solenoid_2 = detector_data['Py_Det_Solenoid_2'][select_muplus_Det_Solenoid_2]
    Pz_MuPlus_Det_Solenoid_2 = detector_data['Pz_Det_Solenoid_2'][select_muplus_Det_Solenoid_2]

    P_MuPlus_Det_Solenoid_2 = np.sqrt(Px_MuPlus_Det_Solenoid_2**2 + Py_MuPlus_Det_Solenoid_2**2 + Pz_MuPlus_Det_Solenoid_2**2)

    # Mu minus
    Px_MuMinus_Det_Solenoid_2 = detector_data['Px_Det_Solenoid_2'][select_muminus_Det_Solenoid_2]
    Py_MuMinus_Det_Solenoid_2 = detector_data['Py_Det_Solenoid_2'][select_muminus_Det_Solenoid_2]
    Pz_MuMinus_Det_Solenoid_2 = detector_data['Pz_Det_Solenoid_2'][select_muminus_Det_Solenoid_2]

    P_MuMinus_Det_Solenoid_2 = np.sqrt(Px_MuMinus_Det_Solenoid_2**2 + Py_MuMinus_Det_Solenoid_2**2 + Pz_MuMinus_Det_Solenoid_2**2)

    Muon_Mass = 105.67 # MeV
    # Muons
    E_Muons_Det_Solenoid_2 = np.sqrt(Muon_Mass**2 + Px_Muons_Det_Solenoid_2**2 + Py_Muons_Det_Solenoid_2**2 + Pz_Muons_Det_Solenoid_2**2)
    KE_Muons_Det_Solenoid_2 = E_Muons_Det_Solenoid_2 - Muon_Mass
    # Mu plus
    E_MuPlus_Det_Solenoid_2 = np.sqrt(Muon_Mass**2 + Px_MuPlus_Det_Solenoid_2**2 + Py_MuPlus_Det_Solenoid_2**2 + Pz_MuPlus_Det_Solenoid_2**2)
    KE_MuPlus_Det_Solenoid_2 = E_MuPlus_Det_Solenoid_2 - Muon_Mass
    # Mu minus
    E_MuMinus_Det_Solenoid_2 = np.sqrt(Muon_Mass**2 + Px_MuMinus_Det_Solenoid_2**2 + Py_MuMinus_Det_Solenoid_2**2 + Pz_MuMinus_Det_Solenoid_2**2)
    KE_MuMinus_Det_Solenoid_2 = E_MuMinus_Det_Solenoid_2 - Muon_Mass
    
    # =======================================================================================
    # Calculate Pions Energies at Det_Solenoid_2
    # =======================================================================================

    # Pions
    Px_Pions_Det_Solenoid_2 = detector_data['Px_Det_Solenoid_2'][select_pions_Det_Solenoid_2]
    Py_Pions_Det_Solenoid_2 = detector_data['Py_Det_Solenoid_2'][select_pions_Det_Solenoid_2]
    Pz_Pions_Det_Solenoid_2 = detector_data['Pz_Det_Solenoid_2'][select_pions_Det_Solenoid_2]

    P_Pions_Det_Solenoid_2 = np.sqrt(Px_Pions_Det_Solenoid_2**2 + Py_Pions_Det_Solenoid_2**2 + Pz_Pions_Det_Solenoid_2**2)

    # Pi plus
    Px_PiPlus_Det_Solenoid_2 = detector_data['Px_Det_Solenoid_2'][select_piplus_Det_Solenoid_2]
    Py_PiPlus_Det_Solenoid_2 = detector_data['Py_Det_Solenoid_2'][select_piplus_Det_Solenoid_2]
    Pz_PiPlus_Det_Solenoid_2 = detector_data['Pz_Det_Solenoid_2'][select_piplus_Det_Solenoid_2]

    P_PiPlus_Det_Solenoid_2 = np.sqrt(Px_PiPlus_Det_Solenoid_2**2 + Py_PiPlus_Det_Solenoid_2**2 + Pz_PiPlus_Det_Solenoid_2**2)

    # Pi minus
    Px_PiMinus_Det_Solenoid_2 = detector_data['Px_Det_Solenoid_2'][select_piminus_Det_Solenoid_2]
    Py_PiMinus_Det_Solenoid_2 = detector_data['Py_Det_Solenoid_2'][select_piminus_Det_Solenoid_2]
    Pz_PiMinus_Det_Solenoid_2 = detector_data['Pz_Det_Solenoid_2'][select_piminus_Det_Solenoid_2]

    P_PiMinus_Det_Solenoid_2 = np.sqrt(Px_PiMinus_Det_Solenoid_2**2 + Py_PiMinus_Det_Solenoid_2**2 + Pz_PiMinus_Det_Solenoid_2**2)
    
    Pion_Mass = 139.57 # MeV
    # Pions
    E_Pions_Det_Solenoid_2 = np.sqrt(Pion_Mass**2 + Px_Pions_Det_Solenoid_2**2 + Py_Pions_Det_Solenoid_2**2 + Pz_Pions_Det_Solenoid_2**2)
    KE_Pions_Det_Solenoid_2 = E_Pions_Det_Solenoid_2 - Pion_Mass
    # Pi plus
    E_PiPlus_Det_Solenoid_2 = np.sqrt(Pion_Mass**2 + Px_PiPlus_Det_Solenoid_2**2 + Py_PiPlus_Det_Solenoid_2**2 + Pz_PiPlus_Det_Solenoid_2**2)
    KE_PiPlus_Det_Solenoid_2 = E_PiPlus_Det_Solenoid_2 - Pion_Mass
    # Pions
    E_PiMinus_Det_Solenoid_2 = np.sqrt(Pion_Mass**2 + Px_PiMinus_Det_Solenoid_2**2 + Py_PiMinus_Det_Solenoid_2**2 + Pz_PiMinus_Det_Solenoid_2**2)
    KE_PiMinus_Det_Solenoid_2 = E_PiMinus_Det_Solenoid_2 - Pion_Mass

    # =======================================================================================
    # Calculate Protons Energies at Det_Solenoid_2
    # =======================================================================================

    Px_Protons_Det_Solenoid_2 = detector_data['Px_Det_Solenoid_2'][select_protons_Det_Solenoid_2]
    Py_Protons_Det_Solenoid_2 = detector_data['Py_Det_Solenoid_2'][select_protons_Det_Solenoid_2]
    Pz_Protons_Det_Solenoid_2 = detector_data['Pz_Det_Solenoid_2'][select_protons_Det_Solenoid_2]
    
    P_Protons_Det_Solenoid_2 = np.sqrt(Px_Protons_Det_Solenoid_2**2 + Py_Protons_Det_Solenoid_2**2 + Pz_Protons_Det_Solenoid_2**2)
    
    Proton_Mass = 938.27 # MeV
    
    E_Protons_Det_Solenoid_2 = np.sqrt(Proton_Mass**2 + Px_Protons_Det_Solenoid_2**2 + Py_Protons_Det_Solenoid_2**2 + Pz_Protons_Det_Solenoid_2**2)
    KE_Protons_Det_Solenoid_2 = E_Protons_Det_Solenoid_2 - Proton_Mass
    
    # =======================================================================================
    # Plotting
    # =======================================================================================
    
    # Plot 1D histogram of PDG IDs 
    plot_1D_histogram(
    detector_data['PDGid_Det_Solenoid_2'][select_muons_pions_kaons_Det_Solenoid_2], "PDG ID",
    r"PDG ID Distribution at the End of the Solenoid 2 (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
    discrete=True, save_path="PDGID_muons_pions_kaons_Det_Solenoid_2.png"
    ) 

    plot_1D_histogram(
    detector_data['PDGid_Det_Solenoid_2'][select_protons_Det_Solenoid_2], "PDG ID",
    r"PDG ID Distribution at the End of Solenoid 2 (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
    discrete=True, save_path="PDGID_protons_Det_Solenoid_2.png"
    )

    # Plot 1D histogram of Energies 
    plot_1D_histogram(
    E_Protons_Det_Solenoid_2, "Energy (MeV)",
    r"Energy Distribution of the Protons at the End of Solenoid 2 (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
    save_path="E_Protons_Det_Solenoid_2.png", bins=100
    )
    
    plot_1D_histogram(
    E_Muons_Det_Solenoid_2, "Energy (MeV)",
    r"Energy Distribution of the Muons at the End of Solenoid 2 (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
    save_path="E_Muons_Det_Solenoid_2.png", bins=100
    )

    plot_1D_histogram(
    E_Pions_Det_Solenoid_2, "Energy (MeV)",
    r"Energy Distribution of the Pions at the End of Solenoid 2 (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
    save_path="E_Pions_Det_Solenoid_2.png", bins=100
    )
    

    # # Plot stacked 1D histogram of PDG IDs with custom x-axis limits
    # plot_1D_histogram_overlay(
    # datasets=[PDGid_1000[select_muons_pions_kaons], PDGid_Det_Target[select_muons_pions_kaons_Det_Target], PDGid_200[select_muons_pions_kaons_200]], 
    # labels=["#theta = 0, #alpha = 0, #beta = 0", "600mm Tungsten", "200mm Tungsten"], 
    # xlabel="PDG ID", 
    # title="PDG ID Distribution at the End of Solenoid",
    # discrete=True,
    # save_path="Compare_PDGID_TungstenSize.png"
    # )

    # plot_1D_histogram_overlay(
    # datasets=[E_Muons_1000, E_Muons_Det_Target, E_Muons_200], 
    # labels=["#theta = 0, #alpha = 0, #beta = 0", "600mm Tungsten", "200mm Tungsten"], 
    # xlabel="Energy (Mev)", 
    # title="Energy Distribution of the Muons at the End of Solenoid",
    # bins=100,
    # save_path="Compare_EMuons_TungstenSize.png"
    # )

    # plot_1D_histogram_overlay(
    # datasets=[E_Protons_1000, E_Protons_Det_Target, E_Protons_200], 
    # labels=["#theta = 0, #alpha = 0, #beta = 0", "600mm Tungsten", "200mm Tungsten"], 
    # xlabel="Energy (Mev)", 
    # title="Energy Distribution of the Protons at the End of Solenoid",
    # bins=100,
    # save_path="Compare_EProtons_TungstenSize.png"
    # )

    # Plot 2D scatter (x vs y) for filtered muons
    plot_scatter(x_Muons_Det_Solenoid_2, y_Muons_Det_Solenoid_2, "x (mm)", "y (mm)", r"Muon Distribution at the End of Solenoid 2 (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",(-710, 710), (-710, 710), \
                s = 1, save_path = "Pos_Muons_Det_Solenoid_2.png")
    plot_scatter(x_Pions_Det_Solenoid_2, y_Pions_Det_Solenoid_2, "x (mm)", "y (mm)", r"Pions Distribution at the End of Solenoid 2 (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",(-710, 710), (-710, 710), 
                s = 1, save_path = "Pos_Pions_Det_Solenoid_2.png")
    # plot_scatter(x_Muons_Det_Target, y_Muons_Det_Target, "x (mm)", "y (mm)", "Muon Distribution at the End of Solenoid (600mm Tungsten)",(-5000, 5000), (-5000, 5000))
    # plot_scatter(x_Muons_200, y_Muons_200, "x (mm)", "y (mm)", "Muon Distribution at the End of Solenoid (200mm Tungsten)",(-5000, 5000), (-5000, 5000))
else:
    print("Failed to retrieve PDG ID data.")


# In[4]:


# ====================================================================================================================
# ====================================================================================================================
# Det__2000mm_Solenoid
# ====================================================================================================================
# ====================================================================================================================

# Check if PDG ID data was successfully extracted
if detector_data["PDGid_Det__2000mm_Solenoid"] is not None:
    
    # =======================================================================================
    # Select PDGID
    # =======================================================================================
    
    # Convert to numpy array (for compatibility)
    PDGid_Det_minus2000mm_Solenoid = ak.to_numpy(detector_data["PDGid_Det__2000mm_Solenoid"]).astype(int)

    # PDG ID for muons, pions, and kaons 
    PDGID_muons = [-13, 13]
    PDGID_pions = [-211, 211]
    PDGID_muons_pions_kaons = [13, -13, 211, -211, 321, -321]
    
    # Apply PDG ID filtering 
    select_muons_Det_minus2000mm_Solenoid = filter_by_pdgid(PDGid_Det_minus2000mm_Solenoid, PDGID_muons)
    select_pions_Det_minus2000mm_Solenoid = filter_by_pdgid(PDGid_Det_minus2000mm_Solenoid, PDGID_pions)
    select_muons_pions_kaons_Det_minus2000mm_Solenoid = filter_by_pdgid(PDGid_Det_minus2000mm_Solenoid, PDGID_muons_pions_kaons, "./Target_Rotate_theta_0_Beam_alpha_0_beta_0_End_of_Solenoid_Back.csv", True)
    select_protons_Det_minus2000mm_Solenoid = filter_by_pdgid(PDGid_Det_minus2000mm_Solenoid, 2212)

    
    
    # =======================================================================================
    # Extract positions x, y for muons at Det__2000mm_Solenoid
    # =======================================================================================
    
    x_Muons_Det_minus2000mm_Solenoid = detector_data['x_Det__2000mm_Solenoid'][select_muons_Det_minus2000mm_Solenoid]
    y_Muons_Det_minus2000mm_Solenoid = detector_data['y_Det__2000mm_Solenoid'][select_muons_Det_minus2000mm_Solenoid]
    
    # =======================================================================================
    # Extract positions x, y for pions at Det__2000mm_Solenoid
    # =======================================================================================

    x_Pions_Det_minus2000mm_Solenoid = detector_data['x_Det__2000mm_Solenoid'][select_pions_Det_minus2000mm_Solenoid]
    y_Pions_Det_minus2000mm_Solenoid = detector_data['y_Det__2000mm_Solenoid'][select_pions_Det_minus2000mm_Solenoid]

    # =======================================================================================
    # Calculate Muons Energies at Det__2000mm_Solenoid
    # =======================================================================================

    Px_Muons_Det_minus2000mm_Solenoid = detector_data['Px_Det__2000mm_Solenoid'][select_muons_Det_minus2000mm_Solenoid]
    Py_Muons_Det_minus2000mm_Solenoid = detector_data['Py_Det__2000mm_Solenoid'][select_muons_Det_minus2000mm_Solenoid]
    Pz_Muons_Det_minus2000mm_Solenoid = detector_data['Pz_Det__2000mm_Solenoid'][select_muons_Det_minus2000mm_Solenoid]

    Muon_Mass = 105.67 # MeV

    E_Muons_Det_minus2000mm_Solenoid = np.sqrt(Muon_Mass**2 + Px_Muons_Det_minus2000mm_Solenoid**2 + Py_Muons_Det_minus2000mm_Solenoid**2 + Pz_Muons_Det_minus2000mm_Solenoid**2)
    
    # =======================================================================================
    # Calculate Pions Energies at Det__2000mm_Solenoid
    # =======================================================================================

    Px_Pions_Det_minus2000mm_Solenoid = detector_data['Px_Det__2000mm_Solenoid'][select_pions_Det_minus2000mm_Solenoid]
    Py_Pions_Det_minus2000mm_Solenoid = detector_data['Py_Det__2000mm_Solenoid'][select_pions_Det_minus2000mm_Solenoid]
    Pz_Pions_Det_minus2000mm_Solenoid = detector_data['Pz_Det__2000mm_Solenoid'][select_pions_Det_minus2000mm_Solenoid]

    Pion_Mass = 139.57 # MeV

    E_Pions_Det_minus2000mm_Solenoid = np.sqrt(Pion_Mass**2 + Px_Pions_Det_minus2000mm_Solenoid**2 + Py_Pions_Det_minus2000mm_Solenoid**2 + Pz_Pions_Det_minus2000mm_Solenoid**2)

    # =======================================================================================
    # Calculate Protons Energies at Det__2000mm_Solenoid
    # =======================================================================================

    # New mask: Exclude initial beam protons (TrackID == 1)
    exclude_initial_beam = (detector_data['TrackID_Det__2000mm_Solenoid'] != 1)
    
    # Combine both masks
    final_proton_mask = select_protons_Det_minus2000mm_Solenoid & exclude_initial_beam
    
    # Apply mask to extract momentum components
    Px_Protons_Det_minus2000mm_Solenoid = detector_data['Px_Det__2000mm_Solenoid'][final_proton_mask]
    Py_Protons_Det_minus2000mm_Solenoid = detector_data['Py_Det__2000mm_Solenoid'][final_proton_mask]
    Pz_Protons_Det_minus2000mm_Solenoid = detector_data['Pz_Det__2000mm_Solenoid'][final_proton_mask]


    Proton_Mass = 938.27 # MeV
    
    E_Protons_Det_minus2000mm_Solenoid = np.sqrt(Proton_Mass**2 + Px_Protons_Det_minus2000mm_Solenoid**2 + Py_Protons_Det_minus2000mm_Solenoid**2 + Pz_Protons_Det_minus2000mm_Solenoid**2)
    
    # =======================================================================================
    # Plotting
    # =======================================================================================
    
    # Plot 1D histogram of PDG IDs 
    plot_1D_histogram(
    detector_data['PDGid_Det__2000mm_Solenoid'][select_muons_pions_kaons_Det_minus2000mm_Solenoid], "PDG ID",
    r"PDG ID Distribution at the End of the Solenoid Back (KE= 8 GeV, r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
    discrete=True, save_path="PDGID_muons_pions_kaons_Det__2000mm_Solenoid.png"
    ) 

    plot_1D_histogram(
    detector_data['PDGid_Det__2000mm_Solenoid'][select_muons_pions_kaons_Det_minus2000mm_Solenoid], "PDG ID",
    r"PDG ID Distribution at the End of Solenoid Back (KE= 8 GeV, r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
    discrete=True, save_path="PDGID_protons_Det__2000mm_Solenoid.png"
    )

    # Plot 1D histogram of Energies 
    plot_1D_histogram(
    E_Protons_Det_minus2000mm_Solenoid, "Energy (MeV)",
    r"Energy Distribution of the Protons at the End of Solenoid Back (KE= 8 GeV, r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
    save_path="E_Protons_Det__2000mm_Solenoid.png", bins=100
    )
    
    plot_1D_histogram(
    E_Muons_Det_minus2000mm_Solenoid, "Energy (MeV)",
    r"Energy Distribution of the Muons at the End of Solenoid Back (KE= 8 GeV, r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
    save_path="E_Muons_Det__2000mm_Solenoid.png", bins=100
    )

    plot_1D_histogram(
    E_Pions_Det_minus2000mm_Solenoid, "Energy (MeV)",
    r"Energy Distribution of the Pions at the End of Solenoid Back (KE= 8 GeV, r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
    save_path="E_Pions_Det__2000mm_Solenoid.png", bins=100
    )
    

    # # Plot stacked 1D histogram of PDG IDs with custom x-axis limits
    # plot_1D_histogram_overlay(
    # datasets=[PDGid_1000[select_muons_pions_kaons], PDGid_Det_Target[select_muons_pions_kaons_Det_Target], PDGid_200[select_muons_pions_kaons_200]], 
    # labels=["#theta = 0, #alpha = 0, #beta = 0", "600mm Tungsten", "200mm Tungsten"], 
    # xlabel="PDG ID", 
    # title="PDG ID Distribution at the End of Solenoid",
    # discrete=True,
    # save_path="Compare_PDGID_TungstenSize.png"
    # )

    # plot_1D_histogram_overlay(
    # datasets=[E_Muons_1000, E_Muons_Det_Target, E_Muons_200], 
    # labels=["#theta = 0, #alpha = 0, #beta = 0", "600mm Tungsten", "200mm Tungsten"], 
    # xlabel="Energy (Mev)", 
    # title="Energy Distribution of the Muons at the End of Solenoid",
    # bins=100,
    # save_path="Compare_EMuons_TungstenSize.png"
    # )

    # plot_1D_histogram_overlay(
    # datasets=[E_Protons_1000, E_Protons_Det_Target, E_Protons_200], 
    # labels=["#theta = 0, #alpha = 0, #beta = 0", "600mm Tungsten", "200mm Tungsten"], 
    # xlabel="Energy (Mev)", 
    # title="Energy Distribution of the Protons at the End of Solenoid",
    # bins=100,
    # save_path="Compare_EProtons_TungstenSize.png"
    # )

    # Plot 2D scatter (x vs y) for filtered muons
    plot_scatter(x_Muons_Det_minus2000mm_Solenoid, y_Muons_Det_minus2000mm_Solenoid, "x (mm)", "y (mm)", r"Muon Distribution at the End of Solenoid Back (KE= 8 GeV, r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",(-710, 710), (-710, 710), \
                s = 1, save_path = "Pos_Muons_Det__2000mm_Solenoid.png")
    plot_scatter(x_Pions_Det_minus2000mm_Solenoid, y_Pions_Det_minus2000mm_Solenoid, "x (mm)", "y (mm)", r"Pions Distribution at the End of Solenoid Back (KE= 8 GeV, r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",(-710, 710), (-710, 710), 
                s = 1, save_path = "Pos_Pions_Det__2000mm_Solenoid.png")
    # plot_scatter(x_Muons_Det_Target, y_Muons_Det_Target, "x (mm)", "y (mm)", "Muon Distribution at the End of Solenoid (600mm Tungsten)",(-5000, 5000), (-5000, 5000))
    # plot_scatter(x_Muons_200, y_Muons_200, "x (mm)", "y (mm)", "Muon Distribution at the End of Solenoid (200mm Tungsten)",(-5000, 5000), (-5000, 5000))
else:
    print("Failed to retrieve PDG ID data.")


# In[5]:


# ====================================================================================================================
# ====================================================================================================================
# Det_Target_Side
# ====================================================================================================================
# ====================================================================================================================

# Check if PDG ID data was successfully extracted
if detector_data["PDGid_Det_Target_Side"] is not None:
    
    # =======================================================================================
    # Select PDGID
    # =======================================================================================
    
    # Convert to numpy array (for compatibility)
    PDGid_Det_Target_Side = ak.to_numpy(detector_data["PDGid_Det_Target_Side"]).astype(int)

    # PDG ID for muons, pions, and kaons 
    PDGID_muons = [-13, 13]
    PDGID_muplus = [13]
    PDGID_muminus = [-13]
    PDGID_pions = [-211, 211]
    PDGID_piplus = [211]
    PDGID_piminus = [-211]
    PDGID_muons_pions_kaons = [13, -13, 211, -211, 321, -321]
    
    # Apply PDG ID filtering 
    select_muons_Det_Target_Side = filter_by_pdgid(PDGid_Det_Target_Side, PDGID_muons)
    select_muplus_Det_Target_Side = filter_by_pdgid(PDGid_Det_Target_Side, PDGID_muplus)
    select_muminus_Det_Target_Side = filter_by_pdgid(PDGid_Det_Target_Side, PDGID_muminus)
    
    select_pions_Det_Target_Side = filter_by_pdgid(PDGid_Det_Target_Side, PDGID_pions)
    select_piplus_Det_Target_Side = filter_by_pdgid(PDGid_Det_Target_Side, PDGID_piplus)
    select_piminus_Det_Target_Side = filter_by_pdgid(PDGid_Det_Target_Side, PDGID_piminus)
    
    select_muons_pions_kaons_Det_Target_Side = filter_by_pdgid(PDGid_Det_Target_Side, PDGID_muons_pions_kaons, "./Target_Rotate_theta_0_Beam_alpha_0_beta_0_Target_Side.csv", True)
    select_protons_Det_Target_Side = filter_by_pdgid(PDGid_Det_Target_Side, 2212)
    
    # =======================================================================================
    # Extract positions x, y for muons at Det_Target_Side
    # =======================================================================================
    
    x_Muons_Det_Target_Side = detector_data['x_Det_Target_Side'][select_muons_Det_Target_Side]
    y_Muons_Det_Target_Side = detector_data['y_Det_Target_Side'][select_muons_Det_Target_Side]
    
    # =======================================================================================
    # Extract positions x, y for pions at Det_Target_Side
    # =======================================================================================

    x_Pions_Det_Target_Side = detector_data['x_Det_Target_Side'][select_pions_Det_Target_Side]
    y_Pions_Det_Target_Side = detector_data['y_Det_Target_Side'][select_pions_Det_Target_Side]

    # =======================================================================================
    # Calculate Muons Energies at Det_Target_Side
    # =======================================================================================

    # Muons
    Px_Muons_Det_Target_Side = detector_data['Px_Det_Target_Side'][select_muons_Det_Target_Side]
    Py_Muons_Det_Target_Side = detector_data['Py_Det_Target_Side'][select_muons_Det_Target_Side]
    Pz_Muons_Det_Target_Side = detector_data['Pz_Det_Target_Side'][select_muons_Det_Target_Side]

    P_Muons_Det_Target_Side = np.sqrt(Px_Muons_Det_Target_Side**2 + Py_Muons_Det_Target_Side**2 + Pz_Muons_Det_Target_Side**2)
    
    # Mu plus
    Px_MuPlus_Det_Target_Side = detector_data['Px_Det_Target_Side'][select_muplus_Det_Target_Side]
    Py_MuPlus_Det_Target_Side = detector_data['Py_Det_Target_Side'][select_muplus_Det_Target_Side]
    Pz_MuPlus_Det_Target_Side = detector_data['Pz_Det_Target_Side'][select_muplus_Det_Target_Side]

    P_MuPlus_Det_Target_Side = np.sqrt(Px_MuPlus_Det_Target_Side**2 + Py_MuPlus_Det_Target_Side**2 + Pz_MuPlus_Det_Target_Side**2)

    # Mu minus
    Px_MuMinus_Det_Target_Side = detector_data['Px_Det_Target_Side'][select_muminus_Det_Target_Side]
    Py_MuMinus_Det_Target_Side = detector_data['Py_Det_Target_Side'][select_muminus_Det_Target_Side]
    Pz_MuMinus_Det_Target_Side = detector_data['Pz_Det_Target_Side'][select_muminus_Det_Target_Side]

    P_MuMinus_Det_Target_Side = np.sqrt(Px_MuMinus_Det_Target_Side**2 + Py_MuMinus_Det_Target_Side**2 + Pz_MuMinus_Det_Target_Side**2)

    Muon_Mass = 105.67 # MeV
    # Muons
    E_Muons_Det_Target_Side = np.sqrt(Muon_Mass**2 + Px_Muons_Det_Target_Side**2 + Py_Muons_Det_Target_Side**2 + Pz_Muons_Det_Target_Side**2)
    KE_Muons_Det_Target_Side = E_Muons_Det_Target_Side - Muon_Mass
    # Mu plus
    E_MuPlus_Det_Target_Side = np.sqrt(Muon_Mass**2 + Px_MuPlus_Det_Target_Side**2 + Py_MuPlus_Det_Target_Side**2 + Pz_MuPlus_Det_Target_Side**2)
    KE_MuPlus_Det_Target_Side = E_MuPlus_Det_Target_Side - Muon_Mass
    # Mu minus
    E_MuMinus_Det_Target_Side = np.sqrt(Muon_Mass**2 + Px_MuMinus_Det_Target_Side**2 + Py_MuMinus_Det_Target_Side**2 + Pz_MuMinus_Det_Target_Side**2)
    KE_MuMinus_Det_Target_Side = E_MuMinus_Det_Target_Side - Muon_Mass
    
    # =======================================================================================
    # Calculate Pions Energies at Det_Target_Side
    # =======================================================================================

    # Pions
    Px_Pions_Det_Target_Side = detector_data['Px_Det_Target_Side'][select_pions_Det_Target_Side]
    Py_Pions_Det_Target_Side = detector_data['Py_Det_Target_Side'][select_pions_Det_Target_Side]
    Pz_Pions_Det_Target_Side = detector_data['Pz_Det_Target_Side'][select_pions_Det_Target_Side]

    P_Pions_Det_Target_Side = np.sqrt(Px_Pions_Det_Target_Side**2 + Py_Pions_Det_Target_Side**2 + Pz_Pions_Det_Target_Side**2)

    # Pi plus
    Px_PiPlus_Det_Target_Side = detector_data['Px_Det_Target_Side'][select_piplus_Det_Target_Side]
    Py_PiPlus_Det_Target_Side = detector_data['Py_Det_Target_Side'][select_piplus_Det_Target_Side]
    Pz_PiPlus_Det_Target_Side = detector_data['Pz_Det_Target_Side'][select_piplus_Det_Target_Side]

    P_PiPlus_Det_Target_Side = np.sqrt(Px_PiPlus_Det_Target_Side**2 + Py_PiPlus_Det_Target_Side**2 + Pz_PiPlus_Det_Target_Side**2)

    # Pi minus
    Px_PiMinus_Det_Target_Side = detector_data['Px_Det_Target_Side'][select_piminus_Det_Target_Side]
    Py_PiMinus_Det_Target_Side = detector_data['Py_Det_Target_Side'][select_piminus_Det_Target_Side]
    Pz_PiMinus_Det_Target_Side = detector_data['Pz_Det_Target_Side'][select_piminus_Det_Target_Side]

    P_PiMinus_Det_Target_Side = np.sqrt(Px_PiMinus_Det_Target_Side**2 + Py_PiMinus_Det_Target_Side**2 + Pz_PiMinus_Det_Target_Side**2)
    
    Pion_Mass = 139.57 # MeV
    # Pions
    E_Pions_Det_Target_Side = np.sqrt(Pion_Mass**2 + Px_Pions_Det_Target_Side**2 + Py_Pions_Det_Target_Side**2 + Pz_Pions_Det_Target_Side**2)
    KE_Pions_Det_Target_Side = E_Pions_Det_Target_Side - Pion_Mass
    # Pi plus
    E_PiPlus_Det_Target_Side = np.sqrt(Pion_Mass**2 + Px_PiPlus_Det_Target_Side**2 + Py_PiPlus_Det_Target_Side**2 + Pz_PiPlus_Det_Target_Side**2)
    KE_PiPlus_Det_Target_Side = E_PiPlus_Det_Target_Side - Pion_Mass
    # Pions
    E_PiMinus_Det_Target_Side = np.sqrt(Pion_Mass**2 + Px_PiMinus_Det_Target_Side**2 + Py_PiMinus_Det_Target_Side**2 + Pz_PiMinus_Det_Target_Side**2)
    KE_PiMinus_Det_Target_Side = E_PiMinus_Det_Target_Side - Pion_Mass

    # =======================================================================================
    # Calculate Protons Energies at Det_Target_Side
    # =======================================================================================

    Px_Protons_Det_Target_Side = detector_data['Px_Det_Target_Side'][select_protons_Det_Target_Side]
    Py_Protons_Det_Target_Side = detector_data['Py_Det_Target_Side'][select_protons_Det_Target_Side]
    Pz_Protons_Det_Target_Side = detector_data['Pz_Det_Target_Side'][select_protons_Det_Target_Side]
    
    P_Protons_Det_Target_Side = np.sqrt(Px_Protons_Det_Target_Side**2 + Py_Protons_Det_Target_Side**2 + Pz_Protons_Det_Target_Side**2)
    
    Proton_Mass = 938.27 # MeV
    
    E_Protons_Det_Target_Side = np.sqrt(Proton_Mass**2 + Px_Protons_Det_Target_Side**2 + Py_Protons_Det_Target_Side**2 + Pz_Protons_Det_Target_Side**2)
    KE_Protons_Det_Target_Side = E_Protons_Det_Target_Side - Proton_Mass

    # =======================================================================================
    # Select InitZ
    # =======================================================================================
    
    # Convert to numpy array (for compatibility)
    InitZ_Det_Target_Side = ak.to_numpy(detector_data["InitZ_Det_Target_Side"]).astype(float)
    
    Reference_Z_Start_of_Target = 1800  # baseline Z of target front face
    initz_offsets = np.arange(50, 351, 100)  # from 0 to 400 in steps of 50
    tolerance = 50 # +- 0.1 mm tolerance

    # Color cycle for offsets (you can use your own color map if desired)
    colors = matplotlib.colormaps.get_cmap("tab10").resampled(len(initz_offsets))
    
    for offset in initz_offsets:
        initz_center = Reference_Z_Start_of_Target + offset
        z_range = (initz_center - tolerance, initz_center + tolerance)
    
        # Filter muons
        muon_mask = filter_by_pdgid_and_initpos(
            data_pdgid=detector_data['PDGid_Det_Target_Side'],
            pdg_values=[13, -13],
            init_z=z_range,
            data_initz=detector_data['InitZ_Det_Target_Side'],
            print_counts=False
        )
    
        # Filter pions
        pion_mask = filter_by_pdgid_and_initpos(
            data_pdgid=detector_data['PDGid_Det_Target_Side'],
            pdg_values=[211, -211],
            init_z=z_range,
            data_initz=detector_data['InitZ_Det_Target_Side'],
            print_counts=False
        )
    
        # Extract positions for muons
        x_mu = detector_data['x_Det_Target_Side'][muon_mask]
        y_mu = detector_data['y_Det_Target_Side'][muon_mask]
        z_mu = detector_data['z_Det_Target_Side'][muon_mask] - Reference_Z_Start_of_Target

        # Extract momentum for muons
        px_mu = detector_data['Px_Det_Target_Side'][muon_mask]
        py_mu = detector_data['Py_Det_Target_Side'][muon_mask]
        pz_mu = detector_data['Pz_Det_Target_Side'][muon_mask]
    
        # Extract positions for pions
        x_pi = detector_data['x_Det_Target_Side'][pion_mask]
        y_pi = detector_data['y_Det_Target_Side'][pion_mask]
        z_pi = detector_data['z_Det_Target_Side'][pion_mask] - Reference_Z_Start_of_Target

        # Extract momentum for pions
        px_pi = detector_data['Px_Det_Target_Side'][pion_mask]
        py_pi = detector_data['Py_Det_Target_Side'][pion_mask]
        pz_pi = detector_data['Pz_Det_Target_Side'][pion_mask]
    
        # Generate plots
        pos_muon_plot_path = f"Pos_Initz_{offset}_Muons_Det_Target_Side.png"
        pos_pion_plot_path = f"Pos_Initz_{offset}_Pions_Det_Target_Side.png"

        plot_scatter(
            x_mu, y_mu, "x (mm)", "y (mm)",
            fr"Muon Distribution (InitZ = {offset} mm) at Target Side (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)", (-35, 35), (-35, 35), s=1, save_path=pos_muon_plot_path
        )
    
        plot_scatter(
            x_pi, y_pi, "x (mm)", "y (mm)",
            fr"Pion Distribution (InitZ = {offset} mm) at Target Side (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)", (-35, 35), (-35, 35), s=1, save_path=pos_pion_plot_path
        )

        # ================================
        # Phase Space Projections (Muons)
        # ================================
        
        plot_scatter(
            x_mu, px_mu, "x (mm)", r"p_x (MeV)",
            fr"Muon Phase Space: x vs p_x (InitZ = {offset} mm, r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
            s=1, save_path=f"PhaseSpace_x_px_Initz_{offset}_Muons.png"
        )
        
        plot_scatter(
            y_mu, py_mu, "y (mm)", r"p_y (MeV)",
            fr"Muon Phase Space: y vs p_y (InitZ = {offset} mm, r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
            s=1, save_path=f"PhaseSpace_y_py_Initz_{offset}_Muons.png"
        )
        
        plot_scatter(
            z_mu, pz_mu, "z (mm)", r"p_z (MeV)",
            fr"Muon Phase Space: z vs p_z (InitZ = {offset} mm, r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
            s=1, save_path=f"PhaseSpace_z_pz_Initz_{offset}_Muons.png"
        )
        
        # ================================
        # Phase Space Projections (Pions)
        # ================================
        
        plot_scatter(
            x_pi, px_pi, "x (mm)", r"p_x (MeV)",
            fr"Pion Phase Space: x vs p_x (InitZ = {offset} mm, r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
            s=1, save_path=f"PhaseSpace_x_px_Initz_{offset}_Pions.png"
        )
        
        plot_scatter(
            y_pi, py_pi, "y (mm)", r"p_y (MeV)",
            fr"Pion Phase Space: y vs p_y (InitZ = {offset} mm, r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
            s=1, save_path=f"PhaseSpace_y_py_Initz_{offset}_Pions.png"
        )
        
        plot_scatter(
            z_pi, pz_pi, "z (mm)", r"p_z (MeV)",
            fr"Pion Phase Space: z vs p_z (InitZ = {offset} mm, r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
            s=1, save_path=f"PhaseSpace_z_pz_Initz_{offset}_Pions.png"
        )
    
        print(f"Saved plots for InitZ = {offset} mm")


        
    
    # --------------------
    # Muon Plot
    # --------------------
    import matplotlib
    import numpy as np
    
    Reference_Z_Start_of_Target = 1800
    initz_offsets = np.arange(50, 351, 100)
    tolerance = 50
    colors = matplotlib.colormaps.get_cmap("tab20").resampled(len(initz_offsets))
    
    # ==========================
    # Combined Muon XY Position
    # ==========================
    plt.figure(figsize=(8, 8))
    for i, offset in enumerate(initz_offsets):
        initz_center = Reference_Z_Start_of_Target + offset
        z_range = (initz_center - tolerance, initz_center + tolerance)
    
        muon_mask = filter_by_pdgid_and_initpos(
            data_pdgid=detector_data['PDGid_Det_Target_Side'],
            pdg_values=[13, -13],
            init_z=z_range,
            data_initz=detector_data['InitZ_Det_Target_Side']
        )
        x_mu = detector_data['x_Det_Target_Side'][muon_mask]
        y_mu = detector_data['y_Det_Target_Side'][muon_mask]
    
        if len(x_mu) > 0:
            plt.scatter(x_mu, y_mu, s=5, color=colors(i), alpha=0.8,
                        label=f"InitZ={offset}mm (n={len(x_mu)})")
    
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title(r"Muon XY Distribution at Target Side (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
    plt.xlim(-35, 35)
    plt.ylim(-35, 35)
    plt.legend(markerscale=4, fontsize="small", loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Combined_Pos_InitZ_0_to_400_Muons_Det_Target_Side.png")
    plt.close()
    print("Saved combined scatter plot for muons across InitZ offsets.")
    
    # ==========================
    # Combined Pion XY Position
    # ==========================
    plt.figure(figsize=(8, 8))
    for i, offset in enumerate(initz_offsets):
        initz_center = Reference_Z_Start_of_Target + offset
        z_range = (initz_center - tolerance, initz_center + tolerance)
    
        pion_mask = filter_by_pdgid_and_initpos(
            data_pdgid=detector_data['PDGid_Det_Target_Side'],
            pdg_values=[211, -211],
            init_z=z_range,
            data_initz=detector_data['InitZ_Det_Target_Side']
        )
        x_pi = detector_data['x_Det_Target_Side'][pion_mask]
        y_pi = detector_data['y_Det_Target_Side'][pion_mask]
    
        if len(x_pi) > 0:
            plt.scatter(x_pi, y_pi, s=5, color=colors(i), alpha=0.8,
                        label=f"InitZ={offset}mm (n={len(x_pi)})")
    
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title(r"Pion XY Distribution at Target Side (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
    plt.xlim(-35, 35)
    plt.ylim(-35, 35)
    plt.legend(markerscale=4, fontsize="small", loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Combined_Pos_InitZ_0_to_400_Pions_Det_Target_Side.png")
    plt.close()
    print("Saved combined scatter plot for pions across InitZ offsets.") 
    
    # =======================================================================================
    # 
    # =======================================================================================
    
    
    # =======================================================================================
    # Plotting
    # =======================================================================================
    
    # Plot 1D histogram of PDG IDs 
    plot_1D_histogram(
    detector_data['PDGid_Det_Target_Side'][select_muons_pions_kaons_Det_Target_Side], "PDG ID",
    r"PDG ID Distribution at the Side of the Target (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
    discrete=True, save_path="PDGID_muons_pions_kaons_Det_Target_Side.png"
    ) 

    plot_1D_histogram(
    detector_data['PDGid_Det_Target_Side'][select_protons_Det_Target_Side], "PDG ID",
    r"PDG ID Distribution at the Side of the Target (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
    discrete=True, save_path="PDGID_protons_Det_Target_Side.png"
    )

    # Plot 1D histogram of Energies 
    plot_1D_histogram(
    E_Protons_Det_Target_Side, "Energy (MeV)",
    r"Energy Distribution of the Protons at the Side of the Target (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
    save_path="E_Protons_Det_Target_Side.png", bins=100
    )
    
    plot_1D_histogram(
    E_Muons_Det_Target_Side, "Energy (MeV)",
    r"Energy Distribution of the Muons at the Side of the Target (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
    save_path="E_Muons_Det_Target_Side.png", bins=100
    )

    plot_1D_histogram(
    E_Pions_Det_Target_Side, "Energy (MeV)",
    r"Energy Distribution of the Pions at the Side of the Target (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
    save_path="E_Pions_Det_Target_Side.png", bins=100
    )
    

    # # Plot stacked 1D histogram of PDG IDs with custom x-axis limits
    # plot_1D_histogram_overlay(
    # datasets=[PDGid_1000[select_muons_pions_kaons], PDGid_Det_Target[select_muons_pions_kaons_Det_Target], PDGid_200[select_muons_pions_kaons_200]], 
    # labels=["#theta = 0, #alpha = 0, #beta = 0", "600mm Tungsten", "200mm Tungsten"], 
    # xlabel="PDG ID", 
    # title="PDG ID Distribution at the Side of the Target",
    # discrete=True,
    # save_path="Compare_PDGID_TungstenSize.png"
    # )

    # plot_1D_histogram_overlay(
    # datasets=[E_Muons_1000, E_Muons_Det_Target, E_Muons_200], 
    # labels=["#theta = 0, #alpha = 0, #beta = 0", "600mm Tungsten", "200mm Tungsten"], 
    # xlabel="Energy (Mev)", 
    # title="Energy Distribution of the Muons at the Side of the Target",
    # bins=100,
    # save_path="Compare_EMuons_TungstenSize.png"
    # )

    # plot_1D_histogram_overlay(
    # datasets=[E_Protons_1000, E_Protons_Det_Target, E_Protons_200], 
    # labels=["#theta = 0, #alpha = 0, #beta = 0", "600mm Tungsten", "200mm Tungsten"], 
    # xlabel="Energy (Mev)", 
    # title="Energy Distribution of the Protons at the Side of the Target",
    # bins=100,
    # save_path="Compare_EProtons_TungstenSize.png"
    # )

    # Plot 2D scatter (x vs y) for filtered muons and pions
    plot_scatter(x_Muons_Det_Target_Side, y_Muons_Det_Target_Side, "x (mm)", "y (mm)", r"Muon Distribution at the Side of the Target (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",(-35, 35), (-35, 35), \
                s = 1, save_path = "Pos_Muons_Det_Target_Side.png")
    plot_scatter(x_Pions_Det_Target_Side, y_Pions_Det_Target_Side, "x (mm)", "y (mm)", r"Pions Distribution at the Side of the Target (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",(-35, 35), (-35, 35), 
                s = 1, save_path = "Pos_Pions_Det_Target_Side.png")
   
else:
    print("Failed to retrieve PDG ID data.")


# In[ ]:





# In[6]:


from coffea import hist
from coffea.hist.plot import plot1d
from cycler import cycle
import pandas as pd

initz_offsets = np.arange(50, 351, 100)  # from 0 to 400 in steps of 50

# Reference in the Target
Reference_Z_Start_of_Target = 1800
tolerance = 50

# Check if PDG ID data was successfully extracted
if detector_data["PDGid_Det_Target_Side"] is not None:
    
    # ==========================
    # Combined Phase Space Plots (Muons)
    # ==========================
    def combined_phase_plot(x_data_all, p_data_all, xlabel, ylabel, title, filename, offsets, colors):
        plt.figure(figsize=(8, 8))
        for i, (x_vals, p_vals) in enumerate(zip(x_data_all, p_data_all)):
            if len(x_vals) > 0 and len(p_vals) > 0:
                plt.scatter(x_vals, p_vals, s=5, alpha=0.8, color=colors[i],
                            label=f"InitZ={offsets[i]}mm (n={len(x_vals)})")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.legend(markerscale=4, fontsize="small", loc="upper right")
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
        print(f"Saved {filename}")

    # def select_earliest_hits(detector_data, mask):
    #     """Selects the first hit (smallest t) per (EventID, TrackID) pair."""
    
    #     # Extract relevant fields with mask applied
    #     event_ids = detector_data['EventID_Det_Target_Side'][mask]
    #     track_ids = detector_data['TrackID_Det_Target_Side'][mask]
    #     times = detector_data['t_Det_Target_Side'][mask]
    
    #     # Build table of all fields
    #     table = ak.zip({
    #         "event": event_ids,
    #         "track": track_ids,
    #         "t": times,
    #         "x": detector_data['x_Det_Target_Side'][mask],
    #         "y": detector_data['y_Det_Target_Side'][mask],
    #         "z": detector_data['z_Det_Target_Side'][mask],
    #         "px": detector_data['Px_Det_Target_Side'][mask],
    #         "py": detector_data['Py_Det_Target_Side'][mask],
    #         "pz": detector_data['Pz_Det_Target_Side'][mask],
    #     })
    
    #     # Step 1: Sort first by `event_id` to ensure events are grouped together
    #     i_event = ak.argsort(event_ids, stable=True)  # Stable sort preserves relative order
    #     event_sorted = event_ids[i_event]             # Apply the sort to event_ids
        
    #     # Step 2: Now sort by `track_id` *within* each sorted event group
    #     # This is a nested sort workaround since ak.argsort doesn't support RecordArrays
    #     # First, map track_ids according to the event sort
    #     track_sorted_within_event = track_ids[i_event]
        
    #     # Then sort these track_ids
    #     i_track = ak.argsort(track_sorted_within_event, stable=True)
        
    #     # Step 3: Compose the final lexicographical sort index
    #     # Apply the track sort to the event sort indices
    #     lex_sorted_index = i_event[i_track]
        
    #     # Step 4: Reorder the data tables according to the lexicographical sort index
    #     table_sorted = table[lex_sorted_index]
    #     event_sorted = event_ids[lex_sorted_index]
    #     track_sorted = track_ids[lex_sorted_index]
        
    #     # Step 5: Identify the *start* of each new (event, track) pair (i.e., group boundaries)
    #     # The first entry is always a new group
    #     first_is_new = ak.Array([True])
        
    #     # For all other entries, mark `True` if event or track differs from the previous row
    #     rest_is_new = (
    #         event_sorted[1:] != event_sorted[:-1]
    #     ) | (
    #         track_sorted[1:] != track_sorted[:-1]
    #     )
        
    #     # Combine the first and rest to form the full group-start boolean array
    #     is_new = ak.concatenate([first_is_new, rest_is_new], axis=0)

    #     # Compute run lengths and group
    #     run_lengths = ak.run_lengths(is_new)
    #     grouped = ak.unflatten(table_sorted, run_lengths)
    
    #     # Select first (earliest) hit in each group by `t`
    #     min_indices = ak.argmin(grouped["t"], axis=1)
    #     earliest_hits = grouped[min_indices]
    
    #     return earliest_hits

    def select_earliest_hits(detector_data, mask):
        """
        Selects the earliest detector hit (smallest t) for each (EventID, TrackID) pair using pandas.
        
        Args:
            detector_data: dict of awkward arrays
            mask: Boolean awkward array mask to filter relevant hits
            
        Returns:
            Pandas DataFrame of selected hits with one row per (EventID, TrackID) pair.
        """
    
        # Convert masked data to flat numpy arrays
        df = pd.DataFrame({
            "event": ak.to_numpy(detector_data["EventID_Det_Target_Side"][mask]),
            "track": ak.to_numpy(detector_data["TrackID_Det_Target_Side"][mask]),
            "t":     ak.to_numpy(detector_data["t_Det_Target_Side"][mask]),
            "x":     ak.to_numpy(detector_data["x_Det_Target_Side"][mask]),
            "y":     ak.to_numpy(detector_data["y_Det_Target_Side"][mask]),
            "z":     ak.to_numpy(detector_data["z_Det_Target_Side"][mask]),
            "px":    ak.to_numpy(detector_data["Px_Det_Target_Side"][mask]),
            "py":    ak.to_numpy(detector_data["Py_Det_Target_Side"][mask]),
            "pz":    ak.to_numpy(detector_data["Pz_Det_Target_Side"][mask]),
        })
    
        # Sort by event, track, then time to ensure earliest hits come first
        df_sorted = df.sort_values(by=["event", "track", "t"], ascending=[True, True, True])
    
        # Drop duplicates by keeping the first hit per (event, track) pair
        df_unique = df_sorted.drop_duplicates(subset=["event", "track"], keep="first").reset_index(drop=True)
    
        return df_unique


    # Collect muon & pion data for all offsets
    x_all_mu, px_all_mu = [], []
    y_all_mu, py_all_mu = [], []
    z_all_mu, pz_all_mu = [], []
    
    x_all_pi, px_all_pi = [], []
    y_all_pi, py_all_pi = [], []
    z_all_pi, pz_all_pi = [], []

    for offset in initz_offsets:
        initz_center = Reference_Z_Start_of_Target + offset
        z_range = (initz_center - tolerance, initz_center + tolerance)

    
        # === Muons ===
        muon_mask = filter_by_pdgid_and_initpos(
            data_pdgid=detector_data['PDGid_Det_Target_Side'],
            pdg_values=[13, -13],
            init_z=z_range,
            data_initz=detector_data['InitZ_Det_Target_Side']
        )
        
        mu_hits_df = select_earliest_hits(detector_data, muon_mask)
        x_all_mu.append(mu_hits_df["x"].to_numpy())
        y_all_mu.append(mu_hits_df["y"].to_numpy())
        z_all_mu.append(mu_hits_df["z"].to_numpy())
        px_all_mu.append(mu_hits_df["px"].to_numpy())
        py_all_mu.append(mu_hits_df["py"].to_numpy())
        pz_all_mu.append(mu_hits_df["pz"].to_numpy())
        
        # === Pions ===
        pion_mask = filter_by_pdgid_and_initpos(
            data_pdgid=detector_data['PDGid_Det_Target_Side'],
            pdg_values=[211, -211],
            init_z=z_range,
            data_initz=detector_data['InitZ_Det_Target_Side']
        )
        
        pi_hits_df = select_earliest_hits(detector_data, pion_mask)
        x_all_pi.append(pi_hits_df["x"].to_numpy())
        y_all_pi.append(pi_hits_df["y"].to_numpy())
        z_all_pi.append(pi_hits_df["z"].to_numpy())
        px_all_pi.append(pi_hits_df["px"].to_numpy())
        py_all_pi.append(pi_hits_df["py"].to_numpy())
        pz_all_pi.append(pi_hits_df["pz"].to_numpy())


    colors = ['blue', 'orange', 'green', 'red']

    # ==========================
    # Muon Combined Phase Space
    # ==========================
    combined_phase_plot(x_all_mu, px_all_mu, "x (mm)", "p_x (MeV/c)",
                        r"Muon Phase Space: x vs $p_x$ (All InitZ, r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
                        "Combined_PhaseSpace_x_px_Muons.png", initz_offsets, colors)
    
    combined_phase_plot(y_all_mu, py_all_mu, "y (mm)", "p_y (MeV/c)",
                        r"Muon Phase Space: y vs $p_y$ (All InitZ, r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
                        "Combined_PhaseSpace_y_py_Muons.png", initz_offsets, colors)
    
    combined_phase_plot(z_all_mu, pz_all_mu, "z (mm)", "p_z (MeV/c)",
                        r"Muon Phase Space: z vs $p_z$ (All InitZ, r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
                        "Combined_PhaseSpace_z_pz_Muons.png", initz_offsets, colors)
    
    # ==========================
    # Pion Combined Phase Space
    # ==========================
    combined_phase_plot(x_all_pi, px_all_pi, "x (mm)", "p_x (MeV/c)",
                        r"Pion Phase Space: x vs $p_x$ (All InitZ, r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
                        "Combined_PhaseSpace_x_px_Pions.png", initz_offsets, colors)
    
    combined_phase_plot(y_all_pi, py_all_pi, "y (mm)", "p_y (MeV/c)",
                        r"Pion Phase Space: y vs $p_y$ (All InitZ, r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
                        "Combined_PhaseSpace_y_py_Pions.png", initz_offsets, colors)
    
    combined_phase_plot(z_all_pi, pz_all_pi, "z (mm)", "p_z (MeV/c)",
                        r"Pion Phase Space: z vs $p_z$ (All InitZ, r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
                        "Combined_PhaseSpace_z_pz_Pions.png", initz_offsets, colors)
    

    ordered_initz_labels = [f"InitZ={z}" for z in [50, 150, 250, 350]]


    def plot_grouped_1d(hist_obj, var, xlabel, title, fname):
        h1d = hist_obj.integrate("var", var) 
        fig, ax = plt.subplots(figsize=(7, 5))
        plot1d(h1d, ax=ax, overlay="initz", order=ordered_initz_labels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Entries")
        ax.set_title(title)
        ax.legend(title=f"InitZ")
        plt.tight_layout()
        plt.savefig(fname)
        plt.show()

    
    # Create histogram object with two categorical axes: var and initz group
    
    hist1d_mu_x_y = hist.Hist("Muon 1D by InitZ",
        hist.Cat("initz", "InitZ Group"),
        hist.Cat("var", "Variable"),
        hist.Bin("value", "Value", 35, -35, 35)
    )

    hist1d_mu_px_py = hist.Hist("Muon 1D by InitZ",
        hist.Cat("initz", "InitZ Group"),
        hist.Cat("var", "Variable"),
        hist.Bin("value", "Value", 50, -410, 410)
    )
    
    hist1d_mu_z = hist.Hist("Muon Longitudinal by InitZ",
        hist.Cat("initz", "InitZ Group"),
        hist.Cat("var", "Variable"),
        hist.Bin("value", "Value", 100, 1800, 2200)
    )

    hist1d_mu_pz = hist.Hist("Muon Longitudinal by InitZ",
        hist.Cat("initz", "InitZ Group"),
        hist.Cat("var", "Variable"),
        hist.Bin("value", "Value", 50, -1000, 1000)
    )

    hist1d_mu_E = hist.Hist("Muon Energy by InitZ",
        hist.Cat("initz", "InitZ Group"),
        hist.Cat("var", "Variable"),
        hist.Bin("value", "Value", 400, 100, 900)
    )

    hist1d_pi_x_y = hist.Hist("Pion 1D by InitZ",
        hist.Cat("initz", "InitZ Group"),
        hist.Cat("var", "Variable"),
        hist.Bin("value", "Value", 35, -35, 35)
    )

    hist1d_pi_px_py = hist.Hist("Pion 1D by InitZ",
        hist.Cat("initz", "InitZ Group"),
        hist.Cat("var", "Variable"),
        hist.Bin("value", "Value", 50, -410, 410)
    )
    
    hist1d_pi_z = hist.Hist("Pion Longitudinal by InitZ",
        hist.Cat("initz", "InitZ Group"),
        hist.Cat("var", "Variable"),
        hist.Bin("value", "Value", 100, 1800, 2200)
    )

    hist1d_pi_pz = hist.Hist("Pion Longitudinal by InitZ",
        hist.Cat("initz", "InitZ Group"),
        hist.Cat("var", "Variable"),
        hist.Bin("value", "Value", 50, -1000, 1000)
    )

    hist1d_pi_E = hist.Hist("Pion Energy by InitZ",
        hist.Cat("initz", "InitZ Group"),
        hist.Cat("var", "Variable"),
        hist.Bin("value", "Value", 50, 0, 2000)
    )

    # Separate 2D histograms for muons and pions by InitZ and XY plane
    hist2d_mu_xy = hist.Hist(
        "Muon XY Position",                        
        hist.Cat("initz", "Initial Z Position"),
        hist.Bin("x", "X (mm)", 62, -31, 31),
        hist.Bin("y", "Y (mm)", 62, -31, 31),
    )
    
    hist2d_pi_xy = hist.Hist(
        "Pion XY Position",                        
        hist.Cat("initz", "Initial Z Position"),
        hist.Bin("x", "X (mm)", 62, -31, 31),
        hist.Bin("y", "Y (mm)", 62, -31, 31),
    )

    Muon_Mass = 105.67 # MeV
    Pion_Mass = 139.57 # MeV

    for i, offset in enumerate(initz_offsets):
        label = f"InitZ={offset}"
        
        # Filter valid entries
        def safe_concat(arr_list):
            return np.concatenate([ak.to_numpy(a) for a in arr_list if len(a) > 0]) if arr_list else np.array([])
    
        x_mu_vals = safe_concat([x_all_mu[i]])
        y_mu_vals = safe_concat([y_all_mu[i]])
        z_mu_vals = safe_concat([z_all_mu[i]])
        px_mu_vals = safe_concat([px_all_mu[i]])
        py_mu_vals = safe_concat([py_all_mu[i]])
        pz_mu_vals = safe_concat([pz_all_mu[i]])
        E_mu_vals = np.sqrt(Muon_Mass**2 + px_mu_vals**2 + py_mu_vals**2 + pz_mu_vals**2)
        
        # Fill transverse vars
        hist1d_mu_x_y.fill(initz=label, var="x", value=x_mu_vals)
        hist1d_mu_x_y.fill(initz=label, var="y", value=y_mu_vals)
        hist1d_mu_px_py.fill(initz=label, var="px", value=px_mu_vals)
        hist1d_mu_px_py.fill(initz=label, var="py", value=py_mu_vals)
        hist1d_mu_E.fill(initz=label, var="E", value=E_mu_vals)
        
        hist2d_mu_xy.fill(initz=label, x=x_mu_vals, y=y_mu_vals)
        h2d_mu_xy = hist2d_mu_xy.integrate("initz", label)
        ax = hist.plot2d(h2d_mu_xy, xaxis="x")
        ax.set_title(fr"Muons {label} x vs y (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.figure.tight_layout()
        ax.figure.savefig(f"Pos_Muons_x_vs_y_{label}.png")
        plt.show()
    
        # Fill longitudinal vars
        hist1d_mu_z.fill(initz=label, var="z", value=z_mu_vals)
        hist1d_mu_pz.fill(initz=label, var="pz", value=pz_mu_vals)

        x_pi_vals = safe_concat([x_all_pi[i]])
        y_pi_vals = safe_concat([y_all_pi[i]])
        z_pi_vals = safe_concat([z_all_pi[i]])
        px_pi_vals = safe_concat([px_all_pi[i]])
        py_pi_vals = safe_concat([py_all_pi[i]])
        pz_pi_vals = safe_concat([pz_all_pi[i]])
        E_pi_vals = np.sqrt(Pion_Mass**2 + px_pi_vals**2 + py_pi_vals**2 + pz_pi_vals**2)
    
        # Fill transverse vars
        hist1d_pi_x_y.fill(initz=label, var="x", value=x_pi_vals)
        hist1d_pi_x_y.fill(initz=label, var="y", value=y_pi_vals)
        hist1d_pi_px_py.fill(initz=label, var="px", value=px_pi_vals)
        hist1d_pi_px_py.fill(initz=label, var="py", value=py_pi_vals)
        hist1d_pi_E.fill(initz=label, var="E", value=E_pi_vals)
        
        hist2d_pi_xy.fill(initz=label, x=x_pi_vals, y=y_pi_vals)
        h2d_pi_xy = hist2d_pi_xy.integrate("initz", label)
        ax = hist.plot2d(h2d_pi_xy, xaxis="x")
        ax.set_title(fr"Pions {label} x vs y ( r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.figure.tight_layout()
        ax.figure.savefig(f"Pos_Pions_x_vs_y_{label}.png")
        plt.show()
        
        # Fill longitudinal vars
        hist1d_pi_z.fill(initz=label, var="z", value=z_pi_vals)
        hist1d_pi_pz.fill(initz=label, var="pz", value=pz_pi_vals)

    # Transverse / spatial variables
    for var in ["x", "y"]:
        plot_grouped_1d(hist1d_mu_x_y, var,
                        xlabel=f"{var} [mm]",
                        title=fr"Muon {var} Distribution by InitZ (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
                        fname=f"Muon_{var}_by_InitZ.png")
    
    for var in ["px", "py"]:
        plot_grouped_1d(hist1d_mu_px_py, var,
                        xlabel=f"{var} [MeV]",
                        title=fr"Muon {var} Distribution by InitZ (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
                        fname=f"Muon_{var}_by_InitZ.png")
        
    # Longitudinal variables
    for var in ["z"]:
        plot_grouped_1d(hist1d_mu_z, var,
                        xlabel=f"{var} [mm]",
                        title=fr"Muon {var} Distribution by InitZ (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
                        fname=f"Muon_{var}_by_InitZ.png")
    for var in ["pz"]:
        plot_grouped_1d(hist1d_mu_pz, var,
                        xlabel=f"{var} [MeV]",
                        title=fr"Muon {var} Distribution by InitZ (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
                        fname=f"Muon_{var}_by_InitZ.png")
    for var in ["E"]:
        plot_grouped_1d(hist1d_mu_E, var,
                        xlabel=f"{var} [MeV]",
                        title=fr"Muon {var} Distribution by InitZ (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
                        fname=f"Muon_{var}_by_InitZ.png")

     # Transverse / spatial variables
    for var in ["x", "y"]:
        plot_grouped_1d(hist1d_pi_x_y, var,
                        xlabel=f"{var} [mm]",
                        title=fr"Pion {var} Distribution by InitZ (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
                        fname=f"Pion_{var}_by_InitZ.png")
    
    for var in ["px", "py"]:
        plot_grouped_1d(hist1d_pi_px_py, var,
                        xlabel=f"{var} [MeV]",
                        title=fr"Pion {var} Distribution by InitZ (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
                        fname=f"Pion_{var}_by_InitZ.png")
        
    # Longitudinal variables
    for var in ["z"]:
        plot_grouped_1d(hist1d_pi_z, var,
                        xlabel=f"{var} [mm]",
                        title=fr"Pion {var} Distribution by InitZ (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
                        fname=f"Pion_{var}_by_InitZ.png")
    for var in ["pz"]:
        plot_grouped_1d(hist1d_pi_pz, var,
                        xlabel=f"{var} [MeV]",
                        title=fr"Pion {var} Distribution by InitZ (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
                        fname=f"Pion_{var}_by_InitZ.png")

    for var in ["E"]:
        plot_grouped_1d(hist1d_pi_E, var,
                        xlabel=f"{var} [MeV]",
                        title=fr"Pion {var} Distribution by InitZ (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
                        fname=f"Pion_{var}_by_InitZ.png")
  




# In[ ]:





# In[ ]:





# In[7]:


from coffea import hist


# PDG ID for protons
PROTON_PDGID = 2212

# Reference in the Target
Reference_Z_Start_of_Target = 1800

# Extract sample detector keys (cleaned)
sample_keys = list_detector_keys(sample_group)
sample_clean_keys = [key.replace(";1", "").replace("-", "_").replace(" ", "_") for key in sample_keys]

# Initialize arrays to hold proton beam positions
x_protons, y_protons, z_protons = [], [], []

# Loop over all Sample detectors
for raw_key, clean_key in zip(sample_keys, sample_clean_keys):
    pdg = detector_data.get(f"PDGid_{clean_key}")
    trackid = detector_data.get(f"TrackID_{clean_key}")
    x = detector_data.get(f"x_{clean_key}")
    y = detector_data.get(f"y_{clean_key}")
    z = detector_data.get(f"z_{clean_key}")

    if pdg is None or trackid is None or x is None or y is None or z is None:
        print(f"Skipping {clean_key}: missing data.")
        continue

    # Filter for PDG ID == 2212 and TrackID == 1
    mask = (pdg == PROTON_PDGID) & (trackid == 1)

    # Append filtered proton positions
    x_protons.append(x[mask])
    y_protons.append(y[mask])
    z_protons.append(z[mask])

# Concatenate all proton hits across detectors
x_protons = np.concatenate([ak.to_numpy(a) for a in x_protons if len(a) > 0])
y_protons = np.concatenate([ak.to_numpy(a) for a in y_protons if len(a) > 0])
z_protons = np.concatenate([ak.to_numpy(a) for a in z_protons if len(a) > 0])


# Step 1: Count protons per sample detector
proton_counts = []
detector_labels = []

for raw_key, clean_key in zip(sample_keys, sample_clean_keys):
    pdg = detector_data.get(f"PDGid_{clean_key}")
    trackid = detector_data.get(f"TrackID_{clean_key}")
    
    if pdg is None or trackid is None:
        continue

    # Mask for protons with TrackID == 1
    mask = (pdg == PROTON_PDGID) & (trackid == 1)
    count = ak.sum(mask)
    
    detector_labels.append(clean_key)
    proton_counts.append(count)

# Step 2: Save to CSV
df_counts = pd.DataFrame({
    "Detector": detector_labels,
    "Proton_Count": proton_counts
})
df_counts.to_csv("Proton_Counts_Per_Sample.csv", index=False)
print("Saved proton counts to Proton_Counts_Per_Sample.csv")

# Step 3: Plot histogram
x_labels = [f"{i:.0f} mm" for i in np.linspace(0, 401, len(proton_counts))]
plt.figure(figsize=(6, 6))
plt.bar(x_labels, proton_counts, color="steelblue")
plt.xticks(rotation=90)
plt.xlabel("Sample Detector")
plt.ylabel("Number of Protons (PDGID=2212, TrackID=1)")
plt.title(fr"Proton Counts per Sample Detector (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
plt.tight_layout()
plt.savefig("Proton_Counts_Per_Sample.png")
plt.show()

# ================
# Plot x vs z
# ================
plt.figure(figsize=(6, 5))
plt.scatter(z_protons - Reference_Z_Start_of_Target, x_protons, s=1, alpha=0.7)
plt.xlim(-10, 410)
plt.ylim(-31, 31)
plt.xlabel("z (mm)")
plt.ylabel("x (mm)")
plt.title(fr"Proton Beam: x vs z (TrackID = 1) (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
plt.grid(True)
plt.tight_layout()
plt.savefig("Proton_x_vs_z_Sample_TrackID1.png")
plt.show()

# ================
# Plot y vs z
# ================
plt.figure(figsize=(6, 5))
plt.scatter(z_protons - Reference_Z_Start_of_Target, y_protons, s=1, alpha=0.7)
plt.xlim(-10, 410)
plt.ylim(-31, 31)
plt.xlabel("z (mm)")
plt.ylabel("y (mm)")
plt.title(fr"Proton Beam: y vs z (TrackID = 1) (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
plt.grid(True)
plt.tight_layout()
plt.savefig("Proton_y_vs_z_Sample_TrackID1.png")
plt.show()

# ================
# Plot x vs y
# ================
plt.figure(figsize=(6, 6))
plt.scatter(x_protons, y_protons, s=1, alpha=0.7)
plt.xlim(-31, 31)
plt.ylim(-31, 31)
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.title(fr"Proton Beam: x vs y (TrackID = 1) (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
plt.grid(True)
plt.tight_layout()
plt.savefig("Proton_x_vs_y_Sample_TrackID1.png")
plt.show()

# Proton 2D histograms with "var" tag
hist2d_xz_yz = hist.Hist("Proton Tracks",
    hist.Cat("plane", "Projection Plane"),  # xz or yz
    hist.Bin("x", "X (mm)", 20, -5, 5),   # general x axis
    hist.Bin("y", "Y (mm)", 205, -5.0, 405)  # general y axis (for z, shifted)
)

hist2d_xy = hist.Hist("Proton Tracks",
    hist.Cat("plane", "Projection Plane"),  #  xy
    hist.Bin("x", "X (mm)", 20, -5, 5),   # general x axis
    hist.Bin("y", "Y (mm)", 20, -5, 5)  # general y axis (for z, shifted)
)

# Fill histograms with the correct plane label
hist2d_xz_yz.fill(plane="xz", x=x_protons, y=z_protons - Reference_Z_Start_of_Target)
hist2d_xz_yz.fill(plane="yz", x=y_protons, y=z_protons - Reference_Z_Start_of_Target)
hist2d_xy.fill(plane="xy", x=x_protons, y=y_protons)

# Plotting function for one 2D projection
def plot_hist2d_tagged(h, plane_label, xaxis, xlabel, ylabel, title, fname):
    h2d = h.integrate("plane", plane_label)  # reduce to 2D
    ax = hist.plot2d(h2d, xaxis=xaxis)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.figure.tight_layout()
    ax.figure.savefig(fname)
    plt.show()


# Render 2D histogram plots
plot_hist2d_tagged(hist2d_xz_yz, "xz", "y", "z (mm)", "x (mm)", fr"Proton Beam: x vs z (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)", "Proton_x_vs_z_hist2d_tagged.png")
plot_hist2d_tagged(hist2d_xz_yz, "yz", "y", "z (mm)", "y (mm)", fr"Proton Beam: y vs z (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)", "Proton_y_vs_z_hist2d_tagged.png")
plot_hist2d_tagged(hist2d_xy, "xy", "x", "x (mm)", "y (mm)", fr"Proton Beam: x vs y (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)", "Proton_x_vs_y_hist2d_tagged.png")

print("Saved filtered proton beam scatter plots (TrackID == 1).")


# In[3]:


# Load both VirtualDetector and Sample groups
detector_groups = load_dataset("./g4beamline.root", detector_names={'NTuple'})

# Access each group separately
ntuple_group = detector_groups.get("NTuple")

# List detectors
print("ntuple keys", list_detector_keys(ntuple_group))

# List of variables 
variable_keys = ["PDGid", "x", "y", "z", "Ex", "Ey", "Ez", "Px", "Py", "Pz", "ParentID", "InitX", "InitY", "InitZ", "InitKE", "TrackID", "EventID", "t"]  

# Dictionary to store extracted data
detector_data_NTuple = {}

# Loop through each detector group
for group_name, group in detector_groups.items():
    if group is None:
        continue
    for detector_key in group.keys():
        # Clean the key to make it a valid Python variable name
        clean_key = detector_key.replace(";1", "").replace("-", "_").replace(" ", "_")
        for var in variable_keys:
            # Construct variable name: e.g., PDGid_Det_Solenoid_1
            var_name = f"{var}_{clean_key}"
            # Load the variable data
            data = get_virtual_detector_data(group, detector_key, var)
            # Store it in the dictionary
            detector_data_NTuple[var_name] = data
            print(f"Loaded {var_name}")






# In[ ]:





# In[6]:


import matplotlib.pyplot as plt
from coffea import hist
import numpy as np

# Concatenate data from all z-planes
x_all_proton = []
y_all_proton = []
z_all_proton = []

x_all_piplus = []
y_all_piplus = []
z_all_piplus = []

x_all_piminus = []
y_all_piminus = []
z_all_piminus = []

import awkward as ak
import pandas as pd
import numpy as np

# Temporary lists to gather all hits across all z-planes
piplus_hits = []
piminus_hits = []

# Loop through all detector planes and gather pion hits
for z in range(1800, 2201, 10):
    key = f"Z{z}"

    # Mask for pi+ and pi−
    piplus_mask = (detector_data_NTuple[f"PDGid_{key}"] == 211)
    piminus_mask = (detector_data_NTuple[f"PDGid_{key}"] == -211)
    Pion_Mass = 139.57 # MeV

    # Extract info for pi+
    df_pi_plus = pd.DataFrame({
        "event": ak.to_numpy(detector_data_NTuple[f"EventID_{key}"][piplus_mask]),
        "track": ak.to_numpy(detector_data_NTuple[f"TrackID_{key}"][piplus_mask]),
        "time": ak.to_numpy(detector_data_NTuple[f"t_{key}"][piplus_mask]),
        "InitX": ak.to_numpy(detector_data_NTuple[f"InitX_{key}"][piplus_mask]),
        "InitY": ak.to_numpy(detector_data_NTuple[f"InitY_{key}"][piplus_mask]),
        "InitZ": ak.to_numpy(detector_data_NTuple[f"InitZ_{key}"][piplus_mask]),
        "InitKE": ak.to_numpy(detector_data_NTuple[f"InitKE_{key}"][piplus_mask]),
        "InitP": np.sqrt(detector_data_NTuple[f"InitKE_{key}"][piplus_mask]**2 + 2*Pion_Mass*detector_data_NTuple[f"InitKE_{key}"][piplus_mask]),
        "Px": ak.to_numpy(detector_data_NTuple[f"Px_{key}"][piplus_mask]),
        "Py": ak.to_numpy(detector_data_NTuple[f"Py_{key}"][piplus_mask]),
        "Pz": ak.to_numpy(detector_data_NTuple[f"Pz_{key}"][piplus_mask]),
    })

    # Extract info for pi−
    df_pi_minus = pd.DataFrame({
        "event": ak.to_numpy(detector_data_NTuple[f"EventID_{key}"][piminus_mask]),
        "track": ak.to_numpy(detector_data_NTuple[f"TrackID_{key}"][piminus_mask]),
        "time": ak.to_numpy(detector_data_NTuple[f"t_{key}"][piminus_mask]),
        "InitX": ak.to_numpy(detector_data_NTuple[f"InitX_{key}"][piminus_mask]),
        "InitY": ak.to_numpy(detector_data_NTuple[f"InitY_{key}"][piminus_mask]),
        "InitZ": ak.to_numpy(detector_data_NTuple[f"InitZ_{key}"][piminus_mask]),
        "InitKE": ak.to_numpy(detector_data_NTuple[f"InitKE_{key}"][piminus_mask]),
        "InitP": np.sqrt(detector_data_NTuple[f"InitKE_{key}"][piminus_mask]**2 + 2*Pion_Mass*detector_data_NTuple[f"InitKE_{key}"][piminus_mask]),
        "Px": ak.to_numpy(detector_data_NTuple[f"Px_{key}"][piminus_mask]),
        "Py": ak.to_numpy(detector_data_NTuple[f"Py_{key}"][piminus_mask]),
        "Pz": ak.to_numpy(detector_data_NTuple[f"Pz_{key}"][piminus_mask]),
    })

    piplus_hits.append(df_pi_plus)
    piminus_hits.append(df_pi_minus)

# Combine across all z planes
df_all_piplus = pd.concat(piplus_hits, ignore_index=True)
df_all_piminus = pd.concat(piminus_hits, ignore_index=True)

# Drop duplicates globally across all planes
df_unique_piplus = df_all_piplus.sort_values(by=["event", "track", "time"]).drop_duplicates(subset=["event", "track"], keep="first")
df_unique_piminus = df_all_piminus.sort_values(by=["event", "track", "time"]).drop_duplicates(subset=["event", "track"], keep="first")

# Output arrays
init_x_piplus = df_unique_piplus["InitX"].values
init_y_piplus = df_unique_piplus["InitY"].values
init_z_piplus = df_unique_piplus["InitZ"].values
px_piplus = df_unique_piplus["Px"].values
py_piplus = df_unique_piplus["Py"].values
pz_piplus = df_unique_piplus["Pz"].values

init_x_piminus = df_unique_piminus["InitX"].values
init_y_piminus = df_unique_piminus["InitY"].values
init_z_piminus = df_unique_piminus["InitZ"].values
px_piminus = df_unique_piminus["Px"].values
py_piminus = df_unique_piminus["Py"].values
pz_piminus = df_unique_piminus["Pz"].values


for z in range(1800, 2201, 10):
    key = f"Z{z}"
    proton_mask = ( detector_data_NTuple[f"PDGid_{key}"] == 2212 )
    piplus_mask = ( detector_data_NTuple[f"PDGid_{key}"] == 211 )
    piminus_mask = ( detector_data_NTuple[f"PDGid_{key}"] == -211 )
    
    x_all_proton.append(detector_data_NTuple[f"x_{key}"][proton_mask])
    y_all_proton.append(detector_data_NTuple[f"y_{key}"][proton_mask])
    z_all_proton.append(np.full_like(detector_data_NTuple[f"x_{key}"][proton_mask], z))  # z position as constant per plane

    x_all_piplus.append(detector_data_NTuple[f"x_{key}"][piplus_mask])
    y_all_piplus.append(detector_data_NTuple[f"y_{key}"][piplus_mask])
    z_all_piplus.append(np.full_like(detector_data_NTuple[f"x_{key}"][piplus_mask], z))  # z position as constant per plane

    x_all_piminus.append(detector_data_NTuple[f"x_{key}"][piminus_mask])
    y_all_piminus.append(detector_data_NTuple[f"y_{key}"][piminus_mask])
    z_all_piminus.append(np.full_like(detector_data_NTuple[f"x_{key}"][piminus_mask], z))  # z position as constant per plane

# Convert lists to flat numpy arrays
x_all_proton = np.concatenate(x_all_proton)
y_all_proton = np.concatenate(y_all_proton)
z_all_proton = np.concatenate(z_all_proton)

x_all_piplus = np.concatenate(x_all_piplus)
y_all_piplus = np.concatenate(y_all_piplus)
z_all_piplus = np.concatenate(z_all_piplus)

x_all_piminus = np.concatenate(x_all_piminus)
y_all_piminus = np.concatenate(y_all_piminus)
z_all_piminus = np.concatenate(z_all_piminus)

# Shift z so it maps into the histogram y-axis range
z_shifted_proton = z_all_proton - 1800 
z_shifted_piplus = z_all_piplus - 1800 
z_shifted_piminus = z_all_piminus - 1800

# Create histogram with a categorical "plane" axis
hist2d_xz_yz = hist.Hist("Tracks",
    hist.Cat("plane", "Projection Plane"),    # xz or yz or xy
    hist.Bin("x", "X (mm)", 50, -5, 5),     # x axis
    hist.Bin("y", "Y (mm)", 410, -5, 405)     # covers full z range or y range
)

hist2d_xy = hist.Hist("Tracks",
    hist.Cat("plane", "Projection Plane"),    # xz or yz or xy
    hist.Bin("x", "X (mm)", 50, -5, 5),    # x axis
    hist.Bin("y", "Y (mm)", 50, -5, 5)     # covers full z range or y range
)

hist1d_pi_x_y = hist.Hist("Pion 1D Distribution",
        hist.Cat("particle", "Particles Type"),
        hist.Cat("var", "Variable"),
        hist.Bin("value", "Value", 50, -5, 5)
    )

hist1d_pi_px_py = hist.Hist("Pion 1D by InitZ",
        hist.Cat("particle", "Particles Type"),
        hist.Cat("var", "Variable"),
        hist.Bin("value", "Value", 50, -410, 410)
    )

hist1d_pi_z = hist.Hist("Pion 1D Distribution",
        hist.Cat("particle", "Particles Type"),
        hist.Cat("var", "Variable"),
        hist.Bin("value", "Value", 50, -8, 400)
    )

hist1d_pi_pz = hist.Hist("Pion Longitudinal by InitZ",
        hist.Cat("particle", "Particles Type"),
        hist.Cat("var", "Variable"),
        hist.Bin("value", "Value", 50, -500, 1500)
    )

# Fill the histogram with tagged planes
hist2d_xz_yz.fill(plane="Proton xz", x=x_all_proton, y=z_shifted_proton)
hist2d_xz_yz.fill(plane="Proton yz", x=y_all_proton, y=z_shifted_proton)
hist2d_xy.fill(plane="Proton xy", x=x_all_proton, y=y_all_proton)

hist2d_xz_yz.fill(plane="PiPlus xz", x=init_x_piplus, y=(init_z_piplus - 1800.0))
hist2d_xz_yz.fill(plane="PiPlus yz", x=init_y_piplus, y=(init_z_piplus - 1800.0))
hist2d_xy.fill(plane="PiPlus xy", x=init_x_piplus, y=init_y_piplus)

hist2d_xz_yz.fill(plane="PiMinus xz", x=init_x_piminus, y=(init_z_piminus - 1800.0))
hist2d_xz_yz.fill(plane="PiMinus yz", x=init_y_piminus, y=(init_z_piminus - 1800.0))
hist2d_xy.fill(plane="PiMinus xy", x=init_x_piminus, y=init_y_piminus)

hist1d_pi_x_y.fill(particle="PiPlus", var="InitX", value=init_x_piplus)
hist1d_pi_x_y.fill(particle="PiPlus", var="InitY", value=init_y_piplus)
hist1d_pi_z.fill(particle="PiPlus", var="InitZ", value=(init_z_piplus - 1800))
hist1d_pi_px_py.fill(particle="PiPlus", var="Px", value=px_piplus)
hist1d_pi_px_py.fill(particle="PiPlus", var="Py", value=py_piplus)
hist1d_pi_pz.fill(particle="PiPlus", var="Pz", value=pz_piplus)

hist1d_pi_x_y.fill(particle="PiMinus", var="InitX", value=init_x_piminus)
hist1d_pi_x_y.fill(particle="PiMinus", var="InitY", value=init_y_piminus)
hist1d_pi_z.fill(particle="PiMinus", var="InitZ", value=(init_z_piminus - 1800))
hist1d_pi_px_py.fill(particle="PiMinus", var="Px", value=px_piminus)
hist1d_pi_px_py.fill(particle="PiMinus", var="Py", value=py_piminus)
hist1d_pi_pz.fill(particle="PiMinus", var="Pz", value=pz_piminus)

######## Proton ########
# Plot x vs z
# Reduce to 2D by selecting one value for the categorical axis
Proton_h2d_xz = hist2d_xz_yz.integrate("plane", "Proton xz")
ax = hist.plot2d(Proton_h2d_xz, xaxis="y")
ax.set_title(fr"Protons zNtuple x vs z (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
ax.set_xlabel("z (mm)")
ax.set_ylabel("x (mm)")
ax.figure.tight_layout()
ax.figure.savefig("Proton_x_vs_z_NTuple.png")
plt.show()

# Plot y vs z
# Reduce to 2D by selecting one value for the categorical axis
Proton_h2d_yz = hist2d_xz_yz.integrate("plane", "Proton yz")
ax = hist.plot2d(Proton_h2d_yz, xaxis="y")
ax.set_title(fr"Protons zNtuple y vs z (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
ax.set_xlabel("z (mm)")
ax.set_ylabel("y (mm)")
ax.figure.tight_layout()
ax.figure.savefig("Proton_y_vs_z_NTuple.png")
plt.show()

# Plot y vs x
# Reduce to 2D by selecting one value for the categorical axis
Proton_h2d_xy = hist2d_xy.integrate("plane", "Proton xy")
ax = hist.plot2d(Proton_h2d_xy, xaxis="x")
ax.set_title(fr"Protons zNtuple x vs y (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.figure.tight_layout()
ax.figure.savefig("Proton_x_vs_y_NTuple.png")
plt.show()

######## PiPlus ########
# Plot InitX vs InitZ
PiPlus_h2d_xz = hist2d_xz_yz.integrate("plane", "PiPlus xz")
ax = hist.plot2d(PiPlus_h2d_xz, xaxis="y")
ax.set_title(fr"PiPlus zNtuple InitX vs InitZ (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
ax.set_xlabel("z (mm)")
ax.set_ylabel("x (mm)")
ax.figure.tight_layout()
ax.figure.savefig("PiPlus_InitX_vs_InitZ_NTuple.png")
plt.show()

# Plot y vs z
# Reduce to 2D by selecting one value for the categorical axis
PiPlus_h2d_yz = hist2d_xz_yz.integrate("plane", "PiPlus yz")
ax = hist.plot2d(PiPlus_h2d_yz, xaxis="y")
ax.set_title(fr"PiPlus zNtuple InitY vs InitZ (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
ax.set_xlabel("z (mm)")
ax.set_ylabel("y (mm)")
ax.figure.tight_layout()
ax.figure.savefig("PiPlus_InitY_vs_InitZ_NTuple.png")
plt.show()

# Plot y vs x
# Reduce to 2D by selecting one value for the categorical axis
PiPlus_h2d_xy = hist2d_xy.integrate("plane", "PiPlus xy")
ax = hist.plot2d(PiPlus_h2d_xy, xaxis="x")
ax.set_title(fr"PiPlus zNtuple InitX vs InitY (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.figure.tight_layout()
ax.figure.savefig("PiPlus_InitX_vs_InitY_NTuple.png")
plt.show()

######## PiMinus ########
# Plot InitX vs InitZ
PiMinus_h2d_xz = hist2d_xz_yz.integrate("plane", "PiMinus xz")
ax = hist.plot2d(PiMinus_h2d_xz, xaxis="y")
ax.set_title(fr"PiMinus zNtuple InitX vs InitZ (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
ax.set_xlabel("z (mm)")
ax.set_ylabel("x (mm)")
ax.figure.tight_layout()
ax.figure.savefig("PiMinus_InitX_vs_InitZ_NTuple.png")
plt.show()

# Plot y vs z
# Reduce to 2D by selecting one value for the categorical axis
PiMinus_h2d_yz = hist2d_xz_yz.integrate("plane", "PiMinus yz")
ax = hist.plot2d(PiMinus_h2d_yz, xaxis="y")
ax.set_title(fr"PiMinus zNtuple InitY vs InitZ (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
ax.set_xlabel("z (mm)")
ax.set_ylabel("y (mm)")
ax.figure.tight_layout()
ax.figure.savefig("PiMinus_InitY_vs_InitZ_NTuple.png")
plt.show()

# Plot y vs x
# Reduce to 2D by selecting one value for the categorical axis
PiMinus_h2d_xy = hist2d_xy.integrate("plane", "PiMinus xy")
ax = hist.plot2d(PiMinus_h2d_xy, xaxis="x")
ax.set_title(fr"PiMinus zNtuple InitX vs InitY (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.figure.tight_layout()
ax.figure.savefig("PiMinus_InitX_vs_InitY_NTuple.png")
plt.show()


def plot_pions_Init_1d(hist_obj, var, xlabel, title, fname):
        h1d = hist_obj.integrate("var", var)
        n_entries_PiPlus = h1d.integrate("particle", "PiPlus").values(overflow='all')[()].sum()
        n_entries_PiMinus = h1d.integrate("particle", "PiMinus").values(overflow='all')[()].sum()
        h1d.axis("particle").index("PiPlus").label = fr"$\pi^+$ (n={(n_entries_PiPlus)})"
        h1d.axis("particle").index("PiMinus").label = fr"$\pi^-$ (n={(n_entries_PiMinus)})"
        fig, ax = plt.subplots(figsize=(6, 4))
        hist.plot1d(h1d, ax=ax, overlay="particle")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Entries")
        ax.set_title(title)
        ax.grid(True)
        ax.legend(title=f"Particle")
        plt.tight_layout()
        plt.savefig(fname)
        plt.show()

def plot_pions_momentum_1d(hist_obj, var, xlabel, title, fname):
        h1d = hist_obj.integrate("var", var) 
        n_entries_PiPlus = h1d.integrate("particle", "PiPlus").values(overflow='all')[()].sum()
        n_entries_PiMinus = h1d.integrate("particle", "PiMinus").values(overflow='all')[()].sum()
        h1d.axis("particle").index("PiPlus").label = fr"$\pi^+$ (n={(n_entries_PiPlus)})"
        h1d.axis("particle").index("PiMinus").label = fr"$\pi^-$ (n={(n_entries_PiMinus)})"
        fig, ax = plt.subplots(figsize=(6, 4))
        hist.plot1d(h1d, ax=ax, overlay="particle", stack=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Entries")
        ax.set_title(title)
        ax.grid(True)
        ax.legend(title=f"Particle")
        plt.tight_layout()
        plt.savefig(fname)
        plt.show()

for var in ["InitX", "InitY"]:
    plot_pions_Init_1d( hist1d_pi_x_y, var,
                        xlabel=f"{var} [mm]",
                        title=fr"Pions {var} Distribution Inside Target (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
                        fname=f"Pions_{var}_Inside_Target.png")
for var in ["InitZ"]:
    plot_pions_Init_1d( hist1d_pi_z, var,
                        xlabel=f"{var} [mm]",
                        title=fr"Pions {var} Distribution Inside Target (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
                        fname=f"Pions_{var}_Inside_Target.png")

for var in ["Pz"]:
    plot_pions_momentum_1d( hist1d_pi_pz, var,
                        xlabel=f"{var} [MeV]",
                        title=fr"Pions {var} Distribution Inside Target (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
                        fname=f"Pions_{var}_Inside_Target.png")

for var in ["Px", "Py"]:
    plot_pions_momentum_1d( hist1d_pi_px_py, var,
                        xlabel=f"{var} [MeV]",
                        title=fr"Pions {var} Distribution Inside Target (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)",
                        fname=f"Pions_{var}_Inside_Target.png")
# print(total[i] for i in range(1, 52))
# total_first_sec = 0
# total_first_sec += sum(total[i] for i in range(1, 52))   
# print(total_first_sec)


# In[7]:


def select_earliest_hits(detector_data, mask):
        """
        Selects the earliest detector hit (smallest t) for each (EventID, TrackID) pair using pandas.
        
        Args:
            detector_data: dict of awkward arrays
            mask: Boolean awkward array mask to filter relevant hits
            
        Returns:
            Pandas DataFrame of selected hits with one row per (EventID, TrackID) pair.
        """
    
        # Convert masked data to flat numpy arrays
        df = pd.DataFrame({
            "event": ak.to_numpy(detector_data["EventID_Det_Target_Side"][mask]),
            "track": ak.to_numpy(detector_data["TrackID_Det_Target_Side"][mask]),
            "t":     ak.to_numpy(detector_data["t_Det_Target_Side"][mask]),
            "x":     ak.to_numpy(detector_data["x_Det_Target_Side"][mask]),
            "y":     ak.to_numpy(detector_data["y_Det_Target_Side"][mask]),
            "z":     ak.to_numpy(detector_data["z_Det_Target_Side"][mask]),
            "px":    ak.to_numpy(detector_data["Px_Det_Target_Side"][mask]),
            "py":    ak.to_numpy(detector_data["Py_Det_Target_Side"][mask]),
            "pz":    ak.to_numpy(detector_data["Pz_Det_Target_Side"][mask]),
        })
    
        # Sort by event, track, then time to ensure earliest hits come first
        df_sorted = df.sort_values(by=["event", "track", "t"], ascending=[True, True, True])
    
        # Drop duplicates by keeping the first hit per (event, track) pair
        df_unique = df_sorted.drop_duplicates(subset=["event", "track"], keep="first").reset_index(drop=True)
    
        return df_unique

def plot_pions_momentum_1d(hist_obj, var, xlabel, title, fname):
        h1d = hist_obj.integrate("var", var) 
        fig, ax = plt.subplots(figsize=(6, 4))
        hist.plot1d(h1d, ax=ax, overlay="particle", stack=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Entries")
        ax.set_title(title)
        ax.grid(True)
        ax.legend(title=f"Particle")
        plt.tight_layout()
        plt.savefig(fname)
        plt.show()

def filter_by_pdgid(data_pdgid, pdg_values, save_csv_path=None, print_counts=False):
    """
    Filter data based on specific PDG ID values, optionally print and save counts.

    Args:
        data_pdgid (awkward.Array): PDG ID data array.
        pdg_values (int or list): Single PDG ID or a list of PDG IDs to filter.
        save_csv_path (str or None): Path to save PDG ID counts as a CSV file.
        print_counts (bool): Whether to print the counts of each matched PDG ID.

    Returns:
        awkward.Array: Boolean mask selecting only the desired PDG IDs.
    """
    # Ensure pdg_values is a list
    if not isinstance(pdg_values, (list, tuple)):
        pdg_values = [pdg_values]

    # Create boolean mask
    mask = ak.any(data_pdgid[:, None] == ak.Array(pdg_values), axis=1)

    # Print and/or save counts if requested
    if print_counts or save_csv_path:
        # Extract matching PDG values
        filtered_pdg = ak.to_numpy(data_pdgid[mask])
        unique_vals, counts = np.unique(filtered_pdg, return_counts=True)

        if print_counts:
            print("PDG ID counts:")
            for pdg, count in zip(unique_vals, counts):
                print(f"  PDG {int(pdg)}: {count}")

        if save_csv_path:
            df = pd.DataFrame({
                "PDG_ID": unique_vals,
                "Count": counts
            })
            df.to_csv(save_csv_path, index=False)
            print(f"PDG ID counts saved to: {save_csv_path}")

    return mask

px_all_pi = []
py_all_pi = []
pz_all_pi = []
E_all_pi = []
Pion_Mass = 139.57 # (MeV)

hist1d_pi_px_py_all = hist.Hist("Pion 1D by InitZ",
        hist.Cat("particle", "Particles Type"),
        hist.Cat("var", "Variable"),
        hist.Bin("value", "Value", 50, -410, 410)
    )

hist1d_pi_pz_all = hist.Hist("Pion Longitudinal by InitZ",
        hist.Cat("particle", "Particles Type"),
        hist.Cat("var", "Variable"),
        hist.Bin("value", "Value", 50, -500, 1500)
    )

hist1d_pi_KE_all = hist.Hist("Pion Longitudinal by InitZ",
        hist.Cat("particle", "Particles Type"),
        hist.Cat("var", "Variable"),
        hist.Bin("value", "Value", 50, -40, 2000)
    )

hist1d_pi_px_py_all_Eff = hist.Hist("Pion 1D by InitZ",
        hist.Cat("particle", "Particles Type"),
        hist.Cat("var", "Variable"),
        hist.Bin("value", "Value", 50, -410, 410)
    )

hist1d_pi_pz_all_Eff = hist.Hist("Pion Longitudinal by InitZ",
        hist.Cat("particle", "Particles Type"),
        hist.Cat("var", "Variable"),
        hist.Bin("value", "Value", 50, -500, 1500)
    )

hist1d_pi_KE_all_Eff = hist.Hist("Pion Longitudinal by InitZ",
        hist.Cat("particle", "Particles Type"),
        hist.Cat("var", "Variable"),
        hist.Bin("value", "Value", 50, -40, 2000)
    )


if detector_data["PDGid_Det_Target_Side"] is not None:
    
    pion_mask = filter_by_pdgid(
            data_pdgid=detector_data['PDGid_Det_Target_Side'],
            pdg_values=[211, -211]
        )
    
    pi_hits_df = select_earliest_hits(detector_data, pion_mask)
    px_all_pi.append(pi_hits_df["px"].to_numpy())
    py_all_pi.append(pi_hits_df["py"].to_numpy())
    pz_all_pi.append(pi_hits_df["pz"].to_numpy())
    
    px_all_pi = np.concatenate(px_all_pi)
    py_all_pi = np.concatenate(py_all_pi)
    pz_all_pi = np.concatenate(pz_all_pi)
    KE_all_pi = np.sqrt(px_all_pi**2 + py_all_pi**2 + pz_all_pi**2 + Pion_Mass**2) - Pion_Mass

hist1d_pi_px_py_all.fill(particle="Pions Target Side", var = "Px", value=px_all_pi)
hist1d_pi_px_py_all.fill(particle="Pions Inside Target", var = "Px", value=px_piminus)
hist1d_pi_px_py_all.fill(particle="Pions Inside Target", var = "Px", value=px_piplus)
hist1d_pi_px_py_all.axis("particle").index("Pions Target Side").label = fr"Around Target (n={len(px_all_pi)})"
hist1d_pi_px_py_all.axis("particle").index("Pions Inside Target").label = fr"Inside Target (n={len(px_piminus) + len(px_piplus)})"

#Eff_pi_px = hist1d_pi_px_py_all.integrate("particle", "Pions Target Side").integrate("var", "Px").values()[()] / hist1d_pi_px_py_all.integrate("particle", "Pions Inside Target").integrate("var", "Px").values()[()]

hist1d_pi_KE_all.fill(particle="Pions Target Side", var = "KE", value=KE_all_pi) 
hist1d_pi_KE_all.fill(particle="Pions Inside Target", var = "KE", value=(np.sqrt(px_piminus**2 + py_piminus**2 + pz_piminus**2 + Pion_Mass**2) - Pion_Mass))
hist1d_pi_KE_all.fill(particle="Pions Inside Target", var = "KE", value=(np.sqrt(px_piplus**2 + py_piplus**2 + pz_piplus**2 + Pion_Mass**2) - Pion_Mass))
hist1d_pi_KE_all.axis("particle").index("Pions Target Side").label = fr"Around Target (n={len(KE_all_pi)})"
hist1d_pi_KE_all.axis("particle").index("Pions Inside Target").label = fr"Inside Target (n={len(np.sqrt(px_piminus**2 + py_piminus**2 + pz_piminus**2 + Pion_Mass**2) - Pion_Mass) + len(np.sqrt(px_piplus**2 + py_piplus**2 + pz_piplus**2 + Pion_Mass**2) - Pion_Mass)})"

hist1d_pi_px_py_all.fill(particle="Pions Target Side", var = "Py", value=py_all_pi)
hist1d_pi_px_py_all.fill(particle="Pions Inside Target", var = "Py", value=py_piminus)
hist1d_pi_px_py_all.fill(particle="Pions Inside Target", var = "Py", value=py_piplus)
hist1d_pi_px_py_all.axis("particle").index("Pions Target Side").label = fr"Around Target (n={len(py_all_pi)})"
hist1d_pi_px_py_all.axis("particle").index("Pions Inside Target").label = fr"Inside Target (n={len(py_piminus) + len(py_piplus)})"

hist1d_pi_pz_all.fill(particle="Pions Target Side", var = "Pz", value=pz_all_pi)
hist1d_pi_pz_all.fill(particle="Pions Inside Target", var = "Pz", value=pz_piminus)
hist1d_pi_pz_all.fill(particle="Pions Inside Target", var = "Pz", value=pz_piplus)
hist1d_pi_pz_all.axis("particle").index("Pions Target Side").label = fr"Around Target (n={len(pz_all_pi)})"
hist1d_pi_pz_all.axis("particle").index("Pions Inside Target").label = fr"Inside Target (n={len(pz_piminus) + len(pz_piplus)})"
        
colors=['red', 'dodgerblue']

overlay_Px_h1d = hist1d_pi_px_py_all.integrate("var", "Px") 
fig, ax = plt.subplots(figsize=(6, 4))
hist.plot1d(overlay_Px_h1d, ax=ax, overlay="particle", line_opts=dict(color=colors))
ax.set_xlabel("Px (MeV)")
ax.set_ylabel("Entries")
ax.set_title(fr"Comparison of Pions Px Distribution (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
ax.grid(True)
ax.legend(title=f"Detectors")
plt.tight_layout()
plt.savefig("Compare_Target_Inside_Target_Side_Px.png")
plt.show()

fig, ax = plt.subplots(figsize=(6, 4))
hist.plotratio(num=hist1d_pi_px_py_all.integrate("particle", "Pions Target Side").integrate("var", "Px"), 
               denom=hist1d_pi_px_py_all.integrate("particle", "Pions Inside Target").integrate("var", "Px"), 
               ax=ax,
               error_opts={'color': 'darkblue', 'marker': '.', },
               denom_fill_opts={},
               unc='num')
ax.set_xlabel("Px (MeV)")
ax.set_ylabel("Efficiency")
ax.set_title(fr"Efficiency of Pions Px Distribution (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
ax.grid(True)
plt.tight_layout()
plt.savefig("Eff_Target_Inside_Target_Side_Px.png")
plt.show()

overlay_Py_h1d = hist1d_pi_px_py_all.integrate("var", "Py") 
fig, ax = plt.subplots(figsize=(6, 4))
hist.plot1d(overlay_Py_h1d, ax=ax, overlay="particle", line_opts=dict(color=colors))
ax.set_xlabel("Py (MeV)")
ax.set_ylabel("Entries")
ax.set_title(fr"Comparison of Pions Py Distribution (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
ax.grid(True)
ax.legend(title=f"Detectors")
plt.tight_layout()
plt.savefig("Compare_Target_Inside_Target_Side_Py.png")
plt.show()

# fig, ax = plt.subplots(figsize=(6, 4))
# hist.plotratio(num=hist1d_pi_px_py_all.integrate("particle", "Pions Target Side").integrate("var", "Py"), 
#                denom=hist1d_pi_px_py_all.integrate("particle", "Pions Inside Target").integrate("var", "Py"), 
#                ax=ax,
#                error_opts={'color': 'darkblue', 'marker': '.', },
#                denom_fill_opts={},
#                unc='num')
# ax.set_xlabel("Py (MeV)")
# ax.set_ylabel("Efficiency")
# ax.set_title(fr"Efficiency of Pions Py Distribution (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
# ax.grid(True)
# plt.tight_layout()
# plt.savefig("Eff_Target_Inside_Target_Side_Py.png")
# plt.show()

overlay_Pz_h1d = hist1d_pi_pz_all.integrate("var", "Pz") 
fig, ax = plt.subplots(figsize=(6, 4))
hist.plot1d(overlay_Pz_h1d, ax=ax, overlay="particle", line_opts=dict(color=colors))
ax.set_xlabel("Pz (MeV)")
ax.set_ylabel("Entries")
ax.set_title(fr"Comparison of Pions Pz Distribution (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
ax.grid(True)
ax.legend(title=f"Detectors")
plt.tight_layout()
plt.savefig("Compare_Target_Inside_Target_Side_Pz.png")
plt.show()

# fig, ax = plt.subplots(figsize=(6, 4))
# hist.plotratio(num=hist1d_pi_pz_all.integrate("particle", "Pions Target Side").integrate("var", "Pz"), 
#                denom=hist1d_pi_pz_all.integrate("particle", "Pions Inside Target").integrate("var", "Pz"), 
#                ax=ax,
#                error_opts={'color': 'darkblue', 'marker': '.', },
#                denom_fill_opts={},
#                unc='num')
# ax.set_xlabel("Pz (MeV)")
# ax.set_ylabel("Efficiency")
# ax.set_xlim(left=0)
# ax.set_title(fr"Efficiency of Pions Pz Distribution (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
# ax.grid(True)
# plt.tight_layout()
# plt.savefig("Eff_Target_Inside_Target_Side_Pz.png")
# plt.show()

overlay_KE_h1d = hist1d_pi_KE_all.integrate("var", "KE") 
fig, ax = plt.subplots(figsize=(6, 4))
hist.plot1d(overlay_KE_h1d, ax=ax, overlay="particle", line_opts=dict(color=colors))
ax.set_xlabel("KE (MeV)")
ax.set_ylabel("Entries")
ax.set_title(fr"Comparison of Pions KE Distribution (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
ax.grid(True)
ax.legend(title=f"Detectors")
plt.tight_layout()
plt.savefig("Compare_Target_Inside_Target_Side_KE.png")
plt.show()

# fig, ax = plt.subplots(figsize=(6, 4))
# hist.plotratio(num=hist1d_pi_KE_all.integrate("particle", "Pions Target Side").integrate("var", "KE"), 
#                denom=hist1d_pi_KE_all.integrate("particle", "Pions Inside Target").integrate("var", "KE"), 
#                ax=ax,
#                error_opts={'color': 'darkblue', 'marker': '.', },
#                denom_fill_opts={},
#                unc='num')
# ax.set_xlabel("KE (MeV)")
# ax.set_ylabel("Efficiency")
# ax.set_xlim(left=0)
# ax.set_title(fr"Efficiency of Pions KE Distribution (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
# ax.grid(True)
# plt.tight_layout()
# plt.savefig("Eff_Target_Inside_Target_Side_KE.png")
# plt.show()



# In[17]:


def plot_pions_momentum_1d(hist_obj, var, xlabel, title, fname):
        h1d = hist_obj.integrate("var", var) 
        fig, ax = plt.subplots(figsize=(6, 4))
        hist.plot1d(h1d, ax=ax, overlay="particle", stack=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Entries")
        ax.set_title(title)
        ax.grid(True)
        ax.legend(title=f"Particle")
        plt.tight_layout()
        plt.savefig(fname)
        plt.show()

def filter_by_pdgid(data_pdgid, pdg_values, save_csv_path=None, print_counts=False):
    """
    Filter data based on specific PDG ID values, optionally print and save counts.

    Args:
        data_pdgid (awkward.Array): PDG ID data array.
        pdg_values (int or list): Single PDG ID or a list of PDG IDs to filter.
        save_csv_path (str or None): Path to save PDG ID counts as a CSV file.
        print_counts (bool): Whether to print the counts of each matched PDG ID.

    Returns:
        awkward.Array: Boolean mask selecting only the desired PDG IDs.
    """
    # Ensure pdg_values is a list
    if not isinstance(pdg_values, (list, tuple)):
        pdg_values = [pdg_values]

    # Create boolean mask
    mask = ak.any(data_pdgid[:, None] == ak.Array(pdg_values), axis=1)

    # Print and/or save counts if requested
    if print_counts or save_csv_path:
        # Extract matching PDG values
        filtered_pdg = ak.to_numpy(data_pdgid[mask])
        unique_vals, counts = np.unique(filtered_pdg, return_counts=True)

        if print_counts:
            print("PDG ID counts:")
            for pdg, count in zip(unique_vals, counts):
                print(f"PDG {int(pdg)}: {count}")

        if save_csv_path:
            df = pd.DataFrame({
                "PDG_ID": unique_vals,
                "Count": counts
            })
            df.to_csv(save_csv_path, index=False)
            print(f"PDG ID counts saved to: {save_csv_path}")

    return mask

#============================================================================================#
##############################################################################################
#################################### Inside the Taregt #######################################
##############################################################################################
#============================================================================================#

# Concatenate data from all z-planes
x_all_proton = []
y_all_proton = []
z_all_proton = []

x_all_piplus = []
y_all_piplus = []
z_all_piplus = []

x_all_piminus = []
y_all_piminus = []
z_all_piminus = []

import awkward as ak
import pandas as pd

Pion_Mass = 139.57  # MeV


# Categories of pion interactions
interaction_categories = {
    "primary":   lambda parentid: parentid == 1,
    "secondary": lambda parentid: (parentid >= 1000) & (parentid < 2000),
    "tertiary":  lambda parentid: (parentid >= 2000) & (parentid < 3000),
}

# To store results for each category
results = {}

for category, parentid_condition in interaction_categories.items():
    piplus_hits = []
    piminus_hits = []

    for z in range(1800, 2201, 10):
        key = f"Z{z}"

        parent_ids = detector_data_NTuple[f"ParentID_{key}"]
        pdg_ids = detector_data_NTuple[f"PDGid_{key}"]

        # Create masks for pi+ and pi− with appropriate parent ID condition
        piplus_mask = (pdg_ids == 211) & parentid_condition(parent_ids)
        piminus_mask = (pdg_ids == -211) & parentid_condition(parent_ids)

        df_pi_plus = pd.DataFrame({
            "event": ak.to_numpy(detector_data_NTuple[f"EventID_{key}"][piplus_mask]),
            "track": ak.to_numpy(detector_data_NTuple[f"TrackID_{key}"][piplus_mask]),
            "t": ak.to_numpy(detector_data_NTuple[f"t_{key}"][piplus_mask]),
            "InitX": ak.to_numpy(detector_data_NTuple[f"InitX_{key}"][piplus_mask]),
            "InitY": ak.to_numpy(detector_data_NTuple[f"InitY_{key}"][piplus_mask]),
            "InitZ": ak.to_numpy(detector_data_NTuple[f"InitZ_{key}"][piplus_mask]),
            "InitKE": ak.to_numpy(detector_data_NTuple[f"InitKE_{key}"][piplus_mask]),
            "InitP": np.sqrt(detector_data_NTuple[f"InitKE_{key}"][piplus_mask]**2 + 2*Pion_Mass*detector_data_NTuple[f"InitKE_{key}"][piplus_mask]),
            "Px":    ak.to_numpy(detector_data_NTuple[f"Px_{key}"][piplus_mask]),
            "Py":    ak.to_numpy(detector_data_NTuple[f"Py_{key}"][piplus_mask]),
            "Pz":    ak.to_numpy(detector_data_NTuple[f"Pz_{key}"][piplus_mask]),
            "x":    ak.to_numpy(detector_data_NTuple[f"x_{key}"][piplus_mask]),
            "y":    ak.to_numpy(detector_data_NTuple[f"y_{key}"][piplus_mask]),
            "z":    ak.to_numpy(detector_data_NTuple[f"z_{key}"][piplus_mask]),
        })

        df_pi_minus = pd.DataFrame({
            "event": ak.to_numpy(detector_data_NTuple[f"EventID_{key}"][piminus_mask]),
            "track": ak.to_numpy(detector_data_NTuple[f"TrackID_{key}"][piminus_mask]),
            "t": ak.to_numpy(detector_data_NTuple[f"t_{key}"][piminus_mask]),
            "InitX": ak.to_numpy(detector_data_NTuple[f"InitX_{key}"][piminus_mask]),
            "InitY": ak.to_numpy(detector_data_NTuple[f"InitY_{key}"][piminus_mask]),
            "InitZ": ak.to_numpy(detector_data_NTuple[f"InitZ_{key}"][piminus_mask]),
            "InitKE": ak.to_numpy(detector_data_NTuple[f"InitKE_{key}"][piminus_mask]),
            "InitP": np.sqrt(detector_data_NTuple[f"InitKE_{key}"][piminus_mask]**2 + 2*Pion_Mass*detector_data_NTuple[f"InitKE_{key}"][piminus_mask]),
            "Px":    ak.to_numpy(detector_data_NTuple[f"Px_{key}"][piminus_mask]),
            "Py":    ak.to_numpy(detector_data_NTuple[f"Py_{key}"][piminus_mask]),
            "Pz":    ak.to_numpy(detector_data_NTuple[f"Pz_{key}"][piminus_mask]),
            "x":    ak.to_numpy(detector_data_NTuple[f"x_{key}"][piminus_mask]),
            "y":    ak.to_numpy(detector_data_NTuple[f"y_{key}"][piminus_mask]),
            "z":    ak.to_numpy(detector_data_NTuple[f"z_{key}"][piminus_mask]),
        })

        piplus_hits.append(df_pi_plus)
        piminus_hits.append(df_pi_minus)

    # Combine and select only earliest hits per (event, track)
    df_all_piplus = pd.concat(piplus_hits, ignore_index=True)
    df_all_piminus = pd.concat(piminus_hits, ignore_index=True)

    df_unique_piplus = df_all_piplus.sort_values(by=["event", "track", "t"]).drop_duplicates(subset=["event", "track"], keep="first")
    df_unique_piminus = df_all_piminus.sort_values(by=["event", "track", "t"]).drop_duplicates(subset=["event", "track"], keep="first")

    # Store result
    results[category] = {
        "piplus": df_unique_piplus,
        "piminus": df_unique_piminus,
    }


#============================================================================================#
##############################################################################################
######## Function used for a.Around the Target, b.Det_Solenoid_1, c.Det_Solenoid_2 ###########
##############################################################################################
#============================================================================================#

def select_particles_by_charge_and_category(detector_data, detector_types, mask, pdgid_categories):
    """
    Select earliest detector hit (smallest t) for each (EventID, TrackID),
    combining multiple detector types into one dataset,
    and grouping specified PDGID types by parentID category.

    Args:
        detector_data: dict of awkward arrays
        detector_types: list of detector type strings (e.g. ["Det_Target_Front", "Det_Target_End"])
        mask: Boolean awkward array mask
        pdgid_categories: dict of labels to PDGIDs

    Returns:
        Dict with structure: result[(label, category)] -> pandas DataFrame
    """
    dfs = []

    for i, detector_type in enumerate(detector_types):
        if f"EventID_{detector_type}" not in detector_data:
            continue

        current_mask = mask[i]

        df_part = pd.DataFrame({
            "event":    ak.to_numpy(detector_data[f"EventID_{detector_type}"][current_mask]),
            "track":    ak.to_numpy(detector_data[f"TrackID_{detector_type}"][current_mask]),
            "t":        ak.to_numpy(detector_data[f"t_{detector_type}"][current_mask]),
            "x":        ak.to_numpy(detector_data[f"x_{detector_type}"][current_mask]),
            "y":        ak.to_numpy(detector_data[f"y_{detector_type}"][current_mask]),
            "z":        ak.to_numpy(detector_data[f"z_{detector_type}"][current_mask]),
            "InitX":    ak.to_numpy(detector_data[f"InitX_{detector_type}"][current_mask]),
            "InitY":    ak.to_numpy(detector_data[f"InitY_{detector_type}"][current_mask]),
            "InitZ":    ak.to_numpy(detector_data[f"InitZ_{detector_type}"][current_mask]),
            "InitKE":   ak.to_numpy(detector_data[f"InitKE_{detector_type}"][current_mask]),
            "InitP":    np.sqrt(detector_data[f"InitKE_{detector_type}"][current_mask]**2 + 2*Pion_Mass*detector_data[f"InitKE_{detector_type}"][current_mask]),
            "Px":       ak.to_numpy(detector_data[f"Px_{detector_type}"][current_mask]),
            "Py":       ak.to_numpy(detector_data[f"Py_{detector_type}"][current_mask]),
            "Pz":       ak.to_numpy(detector_data[f"Pz_{detector_type}"][current_mask]),
            "pdgid":    ak.to_numpy(detector_data[f"PDGid_{detector_type}"][current_mask]),
            "parentid": ak.to_numpy(detector_data[f"ParentID_{detector_type}"][current_mask]),
        })

        dfs.append(df_part)

    if not dfs:
        return {} # return nothing

    # Combine all into one DataFrame
    df = pd.concat(dfs, ignore_index=True)

    # Sort by event, track, and t to get earliest hit
    df_sorted = df.sort_values(by=["event", "track", "t"])
    df_unique = df_sorted.drop_duplicates(subset=["event", "track"], keep="first").reset_index(drop=True)

    categories = {
        "primary":   lambda pid: pid == 1,
        "secondary": lambda pid: (pid >= 1000) & (pid < 2000),
        "tertiary":  lambda pid: (pid >= 2000) & (pid < 3000),
    }

    result = {}
    for label, pdgid in pdgid_categories.items():
        df_selected = df_unique[df_unique["pdgid"] == pdgid]
        for cat_name, cat_fn in categories.items():
            cat_mask = cat_fn(df_selected["parentid"])
            result[(label, cat_name)] = df_selected[cat_mask].reset_index(drop=True)

    return result


#============================================================================================#
##############################################################################################
#################################### Around the Taregt #######################################
##############################################################################################
#============================================================================================#

Pion_Mass = 139.57  # MeV

detector_types_target_side = ["Det_Target_Side", "Det_Target_Front", "Det_Target_End"]

# Make sure both detectors exist
available_types = [
    dt for dt in detector_types_target_side
    if detector_data.get(f"PDGid_{dt}") is not None
]

if available_types:
    pion_masks = []
    for dt in available_types:
        pion_masks.append(
            filter_by_pdgid(
                data_pdgid=detector_data[f"PDGid_{dt}"],
                pdg_values=[211, -211]
            )
        )

        pdgid_map = {
        "piplus": 211,
        "piminus": -211,
        # Add more particles if desired, e.g.
        # "mu+": -13,
        # "mu-": 13,
    }

    # Call the new function with multiple detector types and multiple masks
    categorized_hits_Det_Target_Around = select_particles_by_charge_and_category(
        detector_data=detector_data,
        detector_types=available_types,
        mask=pion_masks,
        pdgid_categories=pdgid_map
    )


    # print(categorized_hits_Det_Target_Around)
    # # Example: extract KE for secondary pi+
    # df_pi_secondary = categorized_hits_Det_Target_Around[("piplus", "secondary")]
    # px = df_pi_secondary["Px"].to_numpy()
    # py = df_pi_secondary["Py"].to_numpy()
    # pz = df_pi_secondary["Pz"].to_numpy()
    
    # KE = np.sqrt(px**2 + py**2 + pz**2 + Pion_Mass**2) - Pion_Mass

#=============================================================================================#
###############################################################################################
###################################### Det_Solenoid_1 #########################################
###############################################################################################
#=============================================================================================#

Pion_Mass = 139.57  # MeV

if detector_data["PDGid_Det_Solenoid_1"] is not None:
    pion_mask = filter_by_pdgid(
    data_pdgid=detector_data["PDGid_Det_Solenoid_1"],
    pdg_values=[211, -211]
    )
    
    pdgid_map = {
        "piplus": 211,
        "piminus": -211,
        # Add more particles if desired, e.g.
        # "mu+": -13,
        # "mu-": 13,
    }
    
    categorized_hits_Det_Solenoid_1 = select_particles_by_charge_and_category(detector_data, ["Det_Solenoid_1"], [pion_mask], pdgid_map)

#=============================================================================================#
###############################################################################################
###################################### Det_Solenoid_2 #########################################
###############################################################################################
#=============================================================================================#

Pion_Mass = 139.57  # MeV

if detector_data["PDGid_Det_Solenoid_2"] is not None:
    pion_mask = filter_by_pdgid(
    data_pdgid=detector_data["PDGid_Det_Solenoid_2"],
    pdg_values=[211, -211]
    )
    
    pdgid_map = {
        "piplus": 211,
        "piminus": -211,
        # Add more particles if desired, e.g.
        # "mu+": -13,
        # "mu-": 13,
    }
    
    categorized_hits_Det_Solenoid_2 = select_particles_by_charge_and_category(detector_data, ["Det_Solenoid_2"], [pion_mask], pdgid_map)

#============================================================================================#
##############################################################################################
######################################### Plotting ###########################################
##############################################################################################
#============================================================================================#

import numpy as np
from coffea import hist
import matplotlib.pyplot as plt

# Define mass
Pion_Mass = 139.57  # MeV

# Create histograms
histograms_momentum = {
    "Px": hist.Hist("Pion Px", hist.Cat("region", "Region"), hist.Cat("category", "Interaction"), hist.Cat("particle", "Particle Type"), hist.Bin("value", "Px (MeV)", 50, -410, 410)),
    "Py": hist.Hist("Pion Py", hist.Cat("region", "Region"), hist.Cat("category", "Interaction"), hist.Cat("particle", "Particle Type"), hist.Bin("value", "Py (MeV)", 50, -410, 410)),
    "Pz": hist.Hist("Pion Pz", hist.Cat("region", "Region"), hist.Cat("category", "Interaction"), hist.Cat("particle", "Particle Type"), hist.Bin("value", "Pz (MeV)", 50, -200, 1800)),
    "KE": hist.Hist("Pion KE", hist.Cat("region", "Region"), hist.Cat("category", "Interaction"), hist.Cat("particle", "Particle Type"), hist.Bin("value", "KE (MeV)", 50, -40, 2000)),
    "P": hist.Hist("Pion P", hist.Cat("region", "Region"), hist.Cat("category", "Interaction"), hist.Cat("particle", "Particle Type"), hist.Bin("value", "P (MeV)", 50, -40, 2000)),
}

histograms_position = {
    "x": hist.Hist("Pion x", hist.Cat("region", "Region"), hist.Cat("category", "Interaction"), hist.Cat("particle", "Particle Type"), hist.Bin("value", "x (mm)", 32, -32, 32)),
    "y": hist.Hist("Pion y", hist.Cat("region", "Region"), hist.Cat("category", "Interaction"), hist.Cat("particle", "Particle Type"), hist.Bin("value", "y (mm)", 32, -32, 32)),
    "z": hist.Hist("Pion z", hist.Cat("region", "Region"), hist.Cat("category", "Interaction"), hist.Cat("particle", "Particle Type"), hist.Bin("value", "z (mm)", 100, 1800.0, 2200)),
}

histograms_combined_label_momentum = {
    "Px": hist.Hist("Pion Px", hist.Cat("region", "Region"), hist.Cat("label", "Interaction + Particle"), hist.Bin("value", "Px (MeV)", 50, -410, 410)),
    "Py": hist.Hist("Pion Py", hist.Cat("region", "Region"),hist.Cat("label", "Interaction + Particle"), hist.Bin("value", "Py (MeV)", 50, -410, 410)),
    "Pz": hist.Hist("Pion Pz", hist.Cat("region", "Region"),hist.Cat("label", "Interaction + Particle"), hist.Bin("value", "Pz (MeV)", 50, -200, 1800)),
    "KE": hist.Hist("Pion KE", hist.Cat("region", "Region"),hist.Cat("label", "Interaction + Particle"), hist.Bin("value", "KE (MeV)", 50, -40, 2000)),
    "P": hist.Hist("Pion P", hist.Cat("region", "Region"),hist.Cat("label", "Interaction + Particle"), hist.Bin("value", "P (MeV)", 50, -40, 2000)),
}

histograms_combined_label_position = {
    "x": hist.Hist("Pion x", hist.Cat("region", "Region"), hist.Cat("label", "Interaction + Particle"), hist.Bin("value", "x (mm)", 32, -32, 32)),
    "y": hist.Hist("Pion y", hist.Cat("region", "Region"),hist.Cat("label", "Interaction + Particle"), hist.Bin("value", "y (mm)", 32, -32, 32)),
    "z": hist.Hist("Pion z", hist.Cat("region", "Region"),hist.Cat("label", "Interaction + Particle"), hist.Bin("value", "z (mm)", 100, 1800.0, 2200.0)),
}

histograms_phase_space = {
    "x_px": hist.Hist(
        "x_px_phase_space",
        hist.Cat("label", "Interaction + Particle"),
        hist.Cat("region", "Region"),
        hist.Bin("x", "x (mm)", 50, -400, 400),
        hist.Bin("px", "Px (MeV)", 50, -410, 410)
    ),
    "y_py": hist.Hist(
        "y_py_phase_space",
        hist.Cat("label", "Interaction + Particle"),
        hist.Cat("region", "Region"),
        hist.Bin("y", "y (mm)", 50, -400, 400),
        hist.Bin("py", "Py (MeV)", 50, -410, 410)
    )
}

# ======================================
# Fill histograms from inside the target
# ======================================
for category, charge_map in results.items():
    for charge, df in charge_map.items():
        label_Interaction = f"{category}"
        label_Particle_Type = f"{charge}"
        x, y, z = df["InitX"].to_numpy(), df["InitY"].to_numpy(), df["InitZ"].to_numpy()
        px, py, pz = df["Px"].to_numpy(), df["Py"].to_numpy(), df["Pz"].to_numpy()
        KE = df['InitKE'].to_numpy()
        P = df['InitP'].to_numpy()

        histograms_position["x"].fill(region="Inside", category=label_Interaction, particle=label_Particle_Type, value=x)
        histograms_position["y"].fill(region="Inside", category=label_Interaction, particle=label_Particle_Type, value=y)
        histograms_position["z"].fill(region="Inside", category=label_Interaction, particle=label_Particle_Type, value=z)
        histograms_momentum["Px"].fill(region="Inside", category=label_Interaction, particle=label_Particle_Type, value=px)
        histograms_momentum["Py"].fill(region="Inside", category=label_Interaction, particle=label_Particle_Type, value=py)
        histograms_momentum["Pz"].fill(region="Inside", category=label_Interaction, particle=label_Particle_Type, value=pz)
        histograms_momentum["KE"].fill(region="Inside", category=label_Interaction, particle=label_Particle_Type, value=KE)
        histograms_momentum["P"].fill(region="Inside", category=label_Interaction, particle=label_Particle_Type, value=P)

        if charge == 'piplus':
            label = f"{category} ($\pi^+$)" #(n={len(df['Px'])})"
        elif charge == 'piminus':
            label = f"{category} ($\pi^-$)" #(n={len(df['Px'])})"
        histograms_combined_label_position["x"].fill(region="Inside", label=label, value=x)
        histograms_combined_label_position["y"].fill(region="Inside",label=label, value=y)
        histograms_combined_label_position["z"].fill(region="Inside",label=label, value=z)
        histograms_combined_label_momentum["Px"].fill(region="Inside", label=label, value=px)
        histograms_combined_label_momentum["Py"].fill(region="Inside",label=label, value=py)
        histograms_combined_label_momentum["Pz"].fill(region="Inside",label=label, value=pz)
        histograms_combined_label_momentum["KE"].fill(region="Inside",label=label, value=KE)
        histograms_combined_label_momentum["P"].fill(region="Inside",label=label, value=P)

# ======================================
# Fill histograms from around the target
# ======================================
for (charge, category), df in categorized_hits_Det_Target_Around.items():
    label_Interaction = f"{category}"
    label_Particle_Type = f"{charge}"
    x, y, z = df["x"].to_numpy(), df["y"].to_numpy(), df["z"].to_numpy()
    px, py, pz = df["Px"].to_numpy(), df["Py"].to_numpy(), df["Pz"].to_numpy()
    KE = np.sqrt(px**2 + py**2 + pz**2 + Pion_Mass**2) - Pion_Mass
    P = np.sqrt(KE**2 + 2*Pion_Mass*KE)

    histograms_position["x"].fill(region="Around", category=label_Interaction, particle=label_Particle_Type, value=x)
    histograms_position["y"].fill(region="Around", category=label_Interaction, particle=label_Particle_Type, value=y)
    histograms_position["z"].fill(region="Around", category=label_Interaction, particle=label_Particle_Type, value=z)
    histograms_momentum["Px"].fill(region="Around", category=label_Interaction, particle=label_Particle_Type, value=px)
    histograms_momentum["Py"].fill(region="Around", category=label_Interaction, particle=label_Particle_Type, value=py)
    histograms_momentum["Pz"].fill(region="Around", category=label_Interaction, particle=label_Particle_Type, value=pz)
    histograms_momentum["KE"].fill(region="Around", category=label_Interaction, particle=label_Particle_Type, value=KE)
    histograms_momentum["P"].fill(region="Around", category=label_Interaction, particle=label_Particle_Type, value=P)

    if charge == 'piplus':
        label = f"{category} ($\pi^+$)" #(n={len(df['Px'])})"
    elif charge == 'piminus':
        label = f"{category} ($\pi^-$)" #(n={len(df['Px'])})"
    histograms_combined_label_position["x"].fill(region="Around", label=label, value=x)
    histograms_combined_label_position["y"].fill(region="Around", label=label, value=y)
    histograms_combined_label_position["z"].fill(region="Around", label=label, value=z)
    histograms_combined_label_momentum["Px"].fill(region="Around", label=label, value=px)
    histograms_combined_label_momentum["Py"].fill(region="Around", label=label, value=py)
    histograms_combined_label_momentum["Pz"].fill(region="Around", label=label, value=pz)
    histograms_combined_label_momentum["KE"].fill(region="Around", label=label, value=KE)
    histograms_combined_label_momentum["P"].fill(region="Around", label=label, value=P)

# ===================================
# Fill histograms from Det_Solenoid_1
# ===================================
for (charge, category), df in categorized_hits_Det_Solenoid_1.items():
    label_Interaction = f"{category}"
    label_Particle_Type = f"{charge}"
    x, y, z = df["x"].to_numpy(), df["y"].to_numpy(), df["z"].to_numpy()
    px, py, pz = df["Px"].to_numpy(), df["Py"].to_numpy(), df["Pz"].to_numpy()
    KE = np.sqrt(px**2 + py**2 + pz**2 + Pion_Mass**2) - Pion_Mass
    P = np.sqrt(KE**2 + 2*Pion_Mass*KE)

    histograms_position["x"].fill(region="Det_Solenoid_1", category=label_Interaction, particle=label_Particle_Type, value=x)
    histograms_position["y"].fill(region="Det_Solenoid_1", category=label_Interaction, particle=label_Particle_Type, value=y)
    histograms_position["z"].fill(region="Det_Solenoid_1", category=label_Interaction, particle=label_Particle_Type, value=z)
    histograms_momentum["Px"].fill(region="Det_Solenoid_1", category=label_Interaction, particle=label_Particle_Type, value=px)
    histograms_momentum["Py"].fill(region="Det_Solenoid_1", category=label_Interaction, particle=label_Particle_Type, value=py)
    histograms_momentum["Pz"].fill(region="Det_Solenoid_1", category=label_Interaction, particle=label_Particle_Type, value=pz)
    histograms_momentum["KE"].fill(region="Det_Solenoid_1", category=label_Interaction, particle=label_Particle_Type, value=KE)
    histograms_momentum["P"].fill(region="Det_Solenoid_1", category=label_Interaction, particle=label_Particle_Type, value=P)

    if charge == 'piplus':
        label = f"{category} ($\pi^+$)" #(n={len(df['Px'])})"
    elif charge == 'piminus':
        label = f"{category} ($\pi^-$)" #(n={len(df['Px'])})"
    histograms_combined_label_position["x"].fill(region="Det_Solenoid_1", label=label, value=x)
    histograms_combined_label_position["y"].fill(region="Det_Solenoid_1", label=label, value=y)
    histograms_combined_label_position["z"].fill(region="Det_Solenoid_1", label=label, value=z)
    histograms_combined_label_momentum["Px"].fill(region="Det_Solenoid_1", label=label, value=px)
    histograms_combined_label_momentum["Py"].fill(region="Det_Solenoid_1", label=label, value=py)
    histograms_combined_label_momentum["Pz"].fill(region="Det_Solenoid_1", label=label, value=pz)
    histograms_combined_label_momentum["KE"].fill(region="Det_Solenoid_1", label=label, value=KE)
    histograms_combined_label_momentum["P"].fill(region="Det_Solenoid_1", label=label, value=P)

    histograms_phase_space['x_px'].fill(region="Det_Solenoid_1", label=label, x=x, px=px)
    histograms_phase_space['y_py'].fill(region="Det_Solenoid_1", label=label, y=y, py=py)

# ===================================
# Fill histograms from Det_Solenoid_2
# ===================================
for (charge, category), df in categorized_hits_Det_Solenoid_2.items():
    label_Interaction = f"{category}"
    label_Particle_Type = f"{charge}"
    x, y, z = df["x"].to_numpy(), df["y"].to_numpy(), df["z"].to_numpy()
    px, py, pz = df["Px"].to_numpy(), df["Py"].to_numpy(), df["Pz"].to_numpy()
    KE = np.sqrt(px**2 + py**2 + pz**2 + Pion_Mass**2) - Pion_Mass
    P = np.sqrt(KE**2 + 2*Pion_Mass*KE)

    histograms_position["x"].fill(region="Det_Solenoid_2", category=label_Interaction, particle=label_Particle_Type, value=x)
    histograms_position["y"].fill(region="Det_Solenoid_2", category=label_Interaction, particle=label_Particle_Type, value=y)
    histograms_position["z"].fill(region="Det_Solenoid_2", category=label_Interaction, particle=label_Particle_Type, value=z)
    histograms_momentum["Px"].fill(region="Det_Solenoid_2", category=label_Interaction, particle=label_Particle_Type, value=px)
    histograms_momentum["Py"].fill(region="Det_Solenoid_2", category=label_Interaction, particle=label_Particle_Type, value=py)
    histograms_momentum["Pz"].fill(region="Det_Solenoid_2", category=label_Interaction, particle=label_Particle_Type, value=pz)
    histograms_momentum["KE"].fill(region="Det_Solenoid_2", category=label_Interaction, particle=label_Particle_Type, value=KE)
    histograms_momentum["P"].fill(region="Det_Solenoid_2", category=label_Interaction, particle=label_Particle_Type, value=P)

    if charge == 'piplus':
        label = f"{category} ($\pi^+$)" #(n={len(df['Px'])})"
    elif charge == 'piminus':
        label = f"{category} ($\pi^-$)" #(n={len(df['Px'])})"
    histograms_combined_label_position["x"].fill(region="Det_Solenoid_2", label=label, value=x)
    histograms_combined_label_position["y"].fill(region="Det_Solenoid_2", label=label, value=y)
    histograms_combined_label_position["z"].fill(region="Det_Solenoid_2", label=label, value=z)
    histograms_combined_label_momentum["Px"].fill(region="Det_Solenoid_2", label=label, value=px)
    histograms_combined_label_momentum["Py"].fill(region="Det_Solenoid_2", label=label, value=py)
    histograms_combined_label_momentum["Pz"].fill(region="Det_Solenoid_2", label=label, value=pz)
    histograms_combined_label_momentum["KE"].fill(region="Det_Solenoid_2", label=label, value=KE)
    histograms_combined_label_momentum["P"].fill(region="Det_Solenoid_2", label=label, value=P)

    histograms_phase_space['x_px'].fill(region="Det_Solenoid_2", label=label, x=x, px=px)
    histograms_phase_space['y_py'].fill(region="Det_Solenoid_2", label=label, y=y, py=py)


# ===========================================
# Plot all stack histograms for inside target
# ===========================================
for var, h in histograms_momentum.items():

    stack_fill_opts = {
    'alpha': 1.0,
    'color':('mediumseagreen', 'deepskyblue', 'gold')
    }
    
    h = h.integrate("region", "Inside").integrate("particle")
    # Get all interaction types from the 'category' axis
    categories = h.axis("category").identifiers()
    # Generate labels with entry counts for each category
    labels = [
        f"{cat.name} (n={int(h.integrate('category', cat).values(overflow='all')[()].sum())})"
        for cat in categories
    ]
    fig, ax = plt.subplots(figsize=(8, 6))
    hist.plot1d(h, ax=ax, overlay="category", stack=True, fill_opts=stack_fill_opts,clear=False)
    ax.set_xlabel(f"{var} (MeV)")
    ax.set_ylabel("Entries")
    ax.set_title(fr"Stacked Distribution of {var} for Pions Inside Target (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
    ax.grid(True)
    ax.legend(title="Interaction Type", labels=labels)
    plt.tight_layout()
    plt.savefig(f"Stacked_Pions_{var.upper()}_Interactions_Inside_Target.png")
    plt.show()

for var, h in histograms_combined_label_momentum.items():

    stack_fill_opts = {
    'alpha': 1.0,
    'color':('springgreen', 'olivedrab', 'powderblue', 'darkturquoise', 'khaki', 'goldenrod')
    }
    
    h = h.integrate("region", "Inside")
    # Get all interaction types from the 'category' axis
    labels_particle_categories = h.axis("label").identifiers()
    # Generate labels with entry counts for each category
    labels = [
        f"{lbl.name} (n={int(h.integrate('label', lbl).values(overflow='all')[()].sum())})"
        for lbl in labels_particle_categories
    ]
    fig, ax = plt.subplots(figsize=(8, 6))
    hist.plot1d(h, ax=ax, overlay="label", stack=True, fill_opts=stack_fill_opts)
    ax.set_xlabel(f"{var} (MeV)")
    ax.set_ylabel("Entries")
    ax.set_title(fr"Stacked distribution of {var} by interaction and Charge Inside Target (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
    ax.grid(True)
    ax.legend(title="Interaction + Particle", labels=labels)
    plt.tight_layout()
    plt.savefig(f"Stacked_Pions_{var.upper()}_Interactions_and_Charge_Inside_Target.png")
    plt.show()

# ===============================================
# Plot all stack histograms for Around the Target
# ===============================================
for var, h in histograms_momentum.items():

    stack_fill_opts = {
    'alpha': 1.0,
    'color':('mediumorchid', 'darkorange', 'mediumblue')
    }
    
    h = h.integrate("region", "Around").integrate("particle")
    # Get all interaction types from the 'category' axis
    categories = h.axis("category").identifiers()
    # Generate labels with entry counts for each category
    labels = [
        f"{cat.name} (n={int(h.integrate('category', cat).values(overflow='all')[()].sum())})"
        for cat in categories
    ]
    fig, ax = plt.subplots(figsize=(8, 6))
    hist.plot1d(h, ax=ax, overlay="category", stack=True, fill_opts=stack_fill_opts, clear=False)
    ax.set_xlabel(f"{var} (MeV)")
    ax.set_ylabel("Entries")
    ax.set_title(fr"Stacked Distribution of {var} for Pions Around Target (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
    ax.grid(True)
    ax.legend(title="Interaction Type", labels=labels)
    plt.tight_layout()
    plt.savefig(f"Stacked_Pions_{var.upper()}_Interactions_Around_Target.png")
    plt.show()

for var, h in histograms_combined_label_momentum.items():

    stack_fill_opts = {
    'alpha': 1.0,
    'color':('violet', 'mediumpurple', 'navajowhite', 'sandybrown', 'cornflowerblue', 'darkblue')
    }
    
    h = h.integrate("region", "Around")
    # Get all interaction types from the 'category' axis
    labels_particle_categories = h.axis("label").identifiers()
    # Generate labels with entry counts for each category
    labels = [
        f"{lbl.name} (n={int(h.integrate('label', lbl).values(overflow='all')[()].sum())})"
        for lbl in labels_particle_categories
    ]
    fig, ax = plt.subplots(figsize=(8, 6))
    hist.plot1d(h, ax=ax, overlay="label", stack=True, fill_opts=stack_fill_opts)
    ax.set_xlabel(f"{var} (MeV)")
    ax.set_ylabel("Entries")
    ax.set_title(fr"Stacked distribution of {var} by interaction and Charge Around Target (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
    ax.grid(True)
    ax.legend(title="Interaction + Particle", labels=labels)
    plt.tight_layout()
    plt.savefig(f"Stacked_Pions_{var.upper()}_Interactions_and_Charge_Around_Target.png")
    plt.show()

# ==============================================================================
# Plot Comparison of histograms between Inside the Target and  Around the Target 
# ==============================================================================

# Define consistent colors for each label
line_colors_target = [
    'red', 'dodgerblue', 'limegreen', 'magenta', 'yellow', 'orange'
]

line_colors_Det = [
    'darkred', 'steelblue', 'green', 'darkviolet', 'gold', 'navajowhite'
]

for var in ["x", "y", "z"]:
    fig, ax = plt.subplots(figsize=(10, 8))
    h_all = histograms_combined_label_position[var]
    labels = list(h_all.axis("label").identifiers())

    legend_handles = []
    legend_labels = []

    for color_target, lbl in zip(line_colors_target, labels):
        # Plot Inside (solid)
        h_inside = h_all.integrate("region", "Inside").integrate("label", lbl)
        if h_inside.values().get(()) is not None:
            hist.plot1d(
                h_inside,
                ax=ax,
                line_opts={"color": color_target, "linestyle": "-"},
                clear=False
            )
            count_inside = int(h_inside.values(overflow="all")[()].sum())
            #handle_inside = ax.get_lines()[-1]
            #legend_handles.append(handle_inside)
            legend_labels.append(f"{lbl.name} [Inside] (n={count_inside})")

        # Plot Around (dashed)
        h_around = h_all.integrate("region", "Around").integrate("label", lbl)
        if h_around.values().get(()) is not None:
            hist.plot1d(
                h_around,
                ax=ax,
                line_opts={"color": color_target, "linestyle": "--"},
                clear=False
            )
            count_around = int(h_around.values(overflow="all")[()].sum())
            #handle_around = ax.get_lines()[-1]
            #legend_handles.append(handle_around)
            legend_labels.append(f"{lbl.name} [Around] (n={count_around})")

    ax.set_xlabel(f"{var} (mm)")
    ax.set_ylabel("Entries")
    ax.set_title(fr"{var} Position by Category and Region (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
    ax.grid(True)
    ax.legend(legend_labels, fontsize=10, loc="best", title="Interaction + Region")
    plt.tight_layout()
    plt.savefig(f"Overlay_Pions_{var}_Position_By_Label_And_Region.png")
    plt.show()

for var in ["z"]:
    h_all = histograms_combined_label_position[var]
    labels = list(h_all.axis("label").identifiers())

    for color_target, lbl in zip(line_colors_target, labels):
        fig, ax = plt.subplots(figsize=(7, 5))
        legend_handles = []
        legend_labels = []
        
        # Plot Inside (solid)
        h_inside = h_all.integrate("region", "Inside").integrate("label", lbl)
        if h_inside.values().get(()) is not None:
            hist.plot1d(
                h_inside,
                ax=ax,
                line_opts={"color": color_target, "linestyle": "-"},
                clear=False
            )
            count_inside = int(h_inside.values(overflow="all")[()].sum())
            #handle_inside = ax.get_lines()[-1]
            #legend_handles.append(handle_inside)
            legend_labels.append(f"{lbl.name} [Inside] (n={count_inside})")

        # Plot Around (dashed)
        h_around = h_all.integrate("region", "Around").integrate("label", lbl)
        if h_around.values().get(()) is not None:
            hist.plot1d(
                h_around,
                ax=ax,
                line_opts={"color": color_target, "linestyle": "--"},
                clear=False
            )
            count_around = int(h_around.values(overflow="all")[()].sum())
            #handle_around = ax.get_lines()[-1]
            #legend_handles.append(handle_around)
            legend_labels.append(f"{lbl.name} [Around] (n={count_around})")
        
        ax.set_xlabel(f"{var} (mm)")
        ax.set_ylabel("Entries")
        ax.set_title(fr"{var} Position of {lbl.name} (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
        ax.grid(True)
        ax.legend(legend_labels, fontsize=9, loc="best", title="Interaction + Region")
        plt.tight_layout()
        if '$\pi^-$' in lbl.name:
            save_png_lbl_name = lbl.name.replace(' ($\pi^-$)', '_piminus')
        elif '$\pi^+$' in lbl.name:
            save_png_lbl_name = lbl.name.replace(' ($\pi^+$)', '_piplus')
        plt.savefig(fr"Pions_{var}_Position_of_{save_png_lbl_name}.png")
        plt.show()

# ==============================================================================================================
# Plot Comparison of histograms between Inside the Target, Around the Target, Det Solenoid 1, and Det Solenoid 2
# ==============================================================================================================

for var in ["P"]:
    h_all = histograms_combined_label_momentum[var]
    labels = list(h_all.axis("label").identifiers())

    for color_target, color_det, lbl in zip(line_colors_target, line_colors_Det, labels):
        fig, ax = plt.subplots(figsize=(7, 5))
        legend_handles = []
        legend_labels = []
        
        # Plot Inside (solid)
        h_inside = h_all.integrate("region", "Inside").integrate("label", lbl)
        if h_inside.values().get(()) is not None:
            hist.plot1d(
                h_inside,
                ax=ax,
                line_opts={"color": color_target, "linestyle": "-"},
                clear=False
            )
            count_inside = int(h_inside.values(overflow="all")[()].sum())
            #handle_inside = ax.get_lines()[-1]
            #legend_handles.append(handle_inside)
            legend_labels.append(f"{lbl.name} [Inside] (n={count_inside})")

        # Plot Around (dashed)
        h_around = h_all.integrate("region", "Around").integrate("label", lbl)
        if h_around.values().get(()) is not None:
            hist.plot1d(
                h_around,
                ax=ax,
                line_opts={"color": color_target, "linestyle": "--"},
                clear=False
            )
            count_around = int(h_around.values(overflow="all")[()].sum())
            #handle_around = ax.get_lines()[-1]
            #legend_handles.append(handle_around)
            legend_labels.append(f"{lbl.name} [Around] (n={count_around})")

        # Plot Det_Solenoid_1 (dotted)
        h_det_solenoid_1 = h_all.integrate("region", "Det_Solenoid_1").integrate("label", lbl)
        if h_det_solenoid_1.values().get(()) is not None:
            hist.plot1d(
                h_det_solenoid_1,
                ax=ax,
                line_opts={"color": color_det, "linestyle": "-"},
                clear=False
            )
            count_det_solenoid_1 = int(h_det_solenoid_1.values(overflow="all")[()].sum())
            #handle_around = ax.get_lines()[-1]
            #legend_handles.append(handle_around)
            legend_labels.append(f"{lbl.name} [Det Solenoid 1] (n={count_det_solenoid_1})")

        # Plot Det_Solenoid_2 (loosely dashdotdotted)
        h_det_solenoid_2 = h_all.integrate("region", "Det_Solenoid_2").integrate("label", lbl)
        if h_det_solenoid_2.values().get(()) is not None:
            hist.plot1d(
                h_det_solenoid_2,
                ax=ax,
                line_opts={"color": color_det, "linestyle": '--'},
                clear=False
            )
            count_det_solenoid_2 = int(h_det_solenoid_2.values(overflow="all")[()].sum())
            #handle_around = ax.get_lines()[-1]
            #legend_handles.append(handle_around)
            legend_labels.append(f"{lbl.name} [Det Solenoid 2] (n={count_det_solenoid_2})")
        
        ax.set_xlabel(f"{var} (MeV)")
        ax.set_ylabel("Entries")
        ax.set_title(fr"{var} of {lbl.name} (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
        ax.grid(True)
        ax.legend(legend_labels, fontsize=9, loc="best", title="Interaction + Region")
        plt.tight_layout()
        if '$\pi^-$' in lbl.name:
            save_png_lbl_name = lbl.name.replace(' ($\pi^-$)', '_piminus')
        elif '$\pi^+$' in lbl.name:
            save_png_lbl_name = lbl.name.replace(' ($\pi^+$)', '_piplus')
        plt.savefig(fr"Pions_{var}_Position_of_{save_png_lbl_name}.png")
        plt.show()

for var in ["Px", "Py", "Pz"]:
    h_all = histograms_combined_label_momentum[var]
    labels = list(h_all.axis("label").identifiers())

    for color_target, color_det, lbl in zip(line_colors_target, line_colors_Det, labels):
        fig, ax = plt.subplots(figsize=(7, 5))
        legend_labels = []
        max_bin_value = 0  # <-- Track max y

        # Plot Inside (solid)
        h_inside = h_all.integrate("region", "Inside").integrate("label", lbl)
        if h_inside.values().get(()) is not None:
            hist.plot1d(
                h_inside,
                ax=ax,
                line_opts={"color": color_target, "linestyle": "-"},
                clear=False
            )
            count_inside = int(h_inside.values(overflow="all")[()].sum())
            legend_labels.append(f"{lbl.name} [Inside] (n={count_inside})")
            max_bin_value = max(max_bin_value, h_inside.values()[()].max())

        # Plot Around (dashed)
        h_around = h_all.integrate("region", "Around").integrate("label", lbl)
        if h_around.values().get(()) is not None:
            hist.plot1d(
                h_around,
                ax=ax,
                line_opts={"color": color_target, "linestyle": "--"},
                clear=False
            )
            count_around = int(h_around.values(overflow="all")[()].sum())
            legend_labels.append(f"{lbl.name} [Around] (n={count_around})")
            max_bin_value = max(max_bin_value, h_around.values()[()].max())

        # Plot Det_Solenoid_1 (solid)
        h_det_solenoid_1 = h_all.integrate("region", "Det_Solenoid_1").integrate("label", lbl)
        if h_det_solenoid_1.values().get(()) is not None:
            hist.plot1d(
                h_det_solenoid_1,
                ax=ax,
                line_opts={"color": color_det, "linestyle": "-"},
                clear=False
            )
            count_det_solenoid_1 = int(h_det_solenoid_1.values(overflow="all")[()].sum())
            legend_labels.append(f"{lbl.name} [Det Solenoid 1] (n={count_det_solenoid_1})")
            max_bin_value = max(max_bin_value, h_det_solenoid_1.values()[()].max())

        # Plot Det_Solenoid_2 (dashed)
        h_det_solenoid_2 = h_all.integrate("region", "Det_Solenoid_2").integrate("label", lbl)
        if h_det_solenoid_2.values().get(()) is not None:
            hist.plot1d(
                h_det_solenoid_2,
                ax=ax,
                line_opts={"color": color_det, "linestyle": '--'},
                clear=False
            )
            count_det_solenoid_2 = int(h_det_solenoid_2.values(overflow="all")[()].sum())
            legend_labels.append(f"{lbl.name} [Det Solenoid 2] (n={count_det_solenoid_2})")
            max_bin_value = max(max_bin_value, h_det_solenoid_2.values()[()].max())

        # Ensure Y-limits include everything with margin
        if max_bin_value > 0:
            ax.set_ylim(0, 1.02 * max_bin_value)

        ax.set_xlabel(f"{var} (MeV)")
        ax.set_ylabel("Entries")
        ax.set_title(fr"{var} of {lbl.name} (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
        ax.grid(True)
        ax.legend(legend_labels, fontsize=9, loc="best", title="Interaction + Region")
        plt.tight_layout()

        if '$\pi^-$' in lbl.name:
            save_png_lbl_name = lbl.name.replace(' ($\pi^-$)', '_piminus')
        elif '$\pi^+$' in lbl.name:
            save_png_lbl_name = lbl.name.replace(' ($\pi^+$)', '_piplus')
        else:
            save_png_lbl_name = lbl.name.replace(' ', '_')

        plt.savefig(fr"Pions_{var}_Position_of_{save_png_lbl_name}.png")
        plt.show()

import re


for phase_var, axes in [("x_px", ("x", "px")), ("y_py", ("y", "py"))]:
    h_all = histograms_phase_space[phase_var]
    labels = list(h_all.axis("label").identifiers())

    for lbl in labels:
        # Plot for Det_Solenoid_1
        h_det_solenoid_1 = h_all.integrate("region", "Det_Solenoid_1").integrate("label", lbl)
        if h_det_solenoid_1.values().get(()) is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            hist.plot2d(
                h_det_solenoid_1,
                ax=ax,
                xaxis=axes[0],
                patch_opts={"cmap": "viridis"}
            )
            ax.set_title(fr"{phase_var.replace('_', ' vs ')}: {lbl} (Det_Solenoid_1, r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
            ax.set_xlabel(f"{axes[0]} (mm)")
            ax.set_ylabel(f"{axes[1]} (MeV)")
            ax.grid(False)
            plt.tight_layout()

            # Clean filename
            label_str = lbl.name
            label_str = label_str.replace("+", "plus").replace("-", "minus")
            label_str = label_str.replace(" ", "_")
            label_str = re.sub(r"[()\$\^\\]", "", label_str)
            plt.savefig(f"PhaseSpace_{phase_var}_DetSolenoid1_{label_str}.png")
            plt.show()

        # Plot for Det_Solenoid_2
        h_det_solenoid_2 = h_all.integrate("region", "Det_Solenoid_2").integrate("label", lbl)
        if h_det_solenoid_2.values().get(()) is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            hist.plot2d(
                h_det_solenoid_2,
                ax=ax,
                xaxis=axes[0],
                patch_opts={"cmap": "viridis"}
            )
            ax.set_title(fr"{phase_var.replace('_', ' vs ')}: {lbl} (Det_Solenoid_2, r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
            ax.set_xlabel(f"{axes[0]} (mm)")
            ax.set_ylabel(f"{axes[1]} (MeV)")
            ax.grid(False)
            plt.tight_layout()

            # Clean filename
            label_str = lbl.name
            label_str = label_str.replace("+", "plus").replace("-", "minus")
            label_str = label_str.replace(" ", "_")
            label_str = re.sub(r"[()\$\^\\]", "", label_str)
            plt.savefig(f"PhaseSpace_{phase_var}_DetSolenoid2_{label_str}.png")
            plt.show()


# In[8]:


# Convert both sets to DataFrames
def convert_to_df(df, region_name):
    df = df.copy()
    df = df.rename(columns={
        "Px": f"Px_{region_name}",
        "Py": f"Py_{region_name}",
        "Pz": f"Pz_{region_name}",
        "z": f"z_{region_name}",
        "InitX": f"InitX_{region_name}",
        "InitY": f"InitY_{region_name}",
        "InitZ": f"InitZ_{region_name}",
        "t": f"t_{region_name}"
    })
    if region_name == 'Inside':
        df["KE_" + region_name] = df['InitKE']
    elif region_name == 'Around':    
        df["KE_" + region_name] = np.sqrt(df[f"Px_{region_name}"]**2 + df[f"Py_{region_name}"]**2 + df[f"Pz_{region_name}"]**2 + Pion_Mass**2) - Pion_Mass
        
    return df[["event", "track", f"Px_{region_name}", f"Py_{region_name}", f"Pz_{region_name}", f"z_{region_name}", f"KE_{region_name}", f"InitX_{region_name}", f"InitY_{region_name}", f"InitZ_{region_name}", f"t_{region_name}"]]

df_inside_all = []
df_around_all = []

for charge in ["piplus", "piminus"]:
    for cat in results:
        df_inside = convert_to_df(results[cat][charge], "Inside")
        df_inside["interaction"] = cat
        df_inside["particle"] = charge
        df_inside_all.append(df_inside)

        df_around = convert_to_df(categorized_hits_Det_Target_Around[charge, cat], "Around")
        df_around["interaction"] = cat
        df_around["particle"] = charge
        df_around_all.append(df_around)

# Combine and merge
# (df_merge: only keeps the pions that match inside and around the target
# df_merge_all: keeps all the pions and puts 0 for Px, Py, Pz, z, ... for Pions that doesn't match 
df_inside_all = pd.concat(df_inside_all, ignore_index=True)
df_around_all = pd.concat(df_around_all, ignore_index=True)
df_merged = pd.merge(df_inside_all, df_around_all, on=["event", "track", "interaction", "particle"])
df_merged_all = pd.merge(df_inside_all, df_around_all, on=["event", "track", "interaction", "particle"], how="outer") # keep all rows
cols_to_fill = [
    col for col in df_merged.columns
    if col not in ["event", "track", "interaction", "particle"]
]

df_merged_all[cols_to_fill] = df_merged_all[cols_to_fill].fillna(0)

# Compute: Eff px, py, pz, KE
df_merged["Eff_Px"] = np.abs(df_merged["Px_Around"]) / np.abs(df_merged["Px_Inside"])
df_merged["Eff_Py"] = np.abs(df_merged["Py_Around"]) / np.abs(df_merged["Py_Inside"])
df_merged["Eff_Pz"] = np.abs(df_merged["Pz_Around"]) / np.abs(df_merged["Pz_Inside"])
df_merged["Eff_KE"] = np.abs(df_merged["KE_Around"]) / np.abs(df_merged["KE_Inside"])

# Clean up: avoid inf/nan
df_merged = df_merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["Eff_Px", "Eff_Py", "Eff_Pz", "Eff_KE"])

# # Create histograms
# histograms_EFF = {
#     "Eff_Px": hist.Hist("Pion Px Efficiency", hist.Cat("category", "Interaction"), hist.Cat("particle", "Particle Type"), hist.Bin("value", "Px Efficiency", 50, -410, 410)),
#     "Eff_Py": hist.Hist("Pion Py Efficiency", hist.Cat("category", "Interaction"), hist.Cat("particle", "Particle Type"), hist.Bin("value", "Py Efficiency", 50, -410, 410)),
#     "Eff_Pz": hist.Hist("Pion Pz Efficiency", hist.Cat("category", "Interaction"), hist.Cat("particle", "Particle Type"), hist.Bin("value", "Pz Efficiency", 50, -500, 1500)),
#     "Eff_KE": hist.Hist("Pion KE Efficiency", hist.Cat("category", "Interaction"), hist.Cat("particle", "Particle Type"), hist.Bin("value", "KE Efficiency", 50, -40, 2000)),
# }

# histograms_combined_Eff = {
#     "Eff_Px": hist.Hist("Pion Px Efficiency", hist.Cat("label", "Interaction + Particle"), hist.Bin("value", "Px Efficiency", 50, -410, 410)),
#     "Eff_Py": hist.Hist("Pion Py Efficiency", hist.Cat("label", "Interaction + Particle"), hist.Bin("value", "Py Efficiency", 50, -410, 410)),
#     "Eff_Pz": hist.Hist("Pion Pz Efficiency", hist.Cat("label", "Interaction + Particle"), hist.Bin("value", "Pz Efficiency", 50, -500, 1500)),
#     "Eff_KE": hist.Hist("Pion KE Efficiency", hist.Cat("label", "Interaction + Particle"), hist.Bin("value", "KE Efficiency", 50, -40, 2000)),
# }

# # ==================================
# # Fill histograms for the efficiency
# # ==================================
# for _, row in df_merged.iterrows():
#     interaction = row["interaction"]
#     particle = row["particle"]
#     label = f"{interaction} ($\pi^+$)" if particle == "piplus" else f"{interaction} ($\pi^-$)"

#     histograms_EFF["Eff_Px"].fill(category=interaction, particle=particle, value=row["Eff_Px"])
#     histograms_EFF["Eff_Py"].fill(category=interaction, particle=particle, value=row["Eff_Py"])
#     histograms_EFF["Eff_Pz"].fill(category=interaction, particle=particle, value=row["Eff_Pz"])
#     histograms_EFF["Eff_KE"].fill(category=interaction, particle=particle, value=row["Eff_KE"])

#     histograms_combined_Eff["Eff_Px"].fill(label=label, value=row["Eff_Px"])
#     histograms_combined_Eff["Eff_Py"].fill(label=label, value=row["Eff_Py"])
#     histograms_combined_Eff["Eff_Pz"].fill(label=label, value=row["Eff_Pz"])
#     histograms_combined_Eff["Eff_KE"].fill(label=label, value=row["Eff_KE"])

# # ============================================
# # Plot all combined efficiency histograms
# # ============================================
# plot_settings = {
#     "Eff_Px": ("Px Efficiency", "Px (Around / Inside)", -410.0, 410.0),
#     "Eff_Py": ("Py Efficiency", "Py (Around / Inside)", -410.0, 410),
#     "Eff_Pz": ("Pz Efficiency", "Pz (Around / Inside)", -500.0, 1500.0),
#     "Eff_KE": ("KE Efficiency", "KE (Around / Inside)", -40.0, 2000.0),
# }

# for var, (title, xlabel, xmin, xmax) in plot_settings.items():
#     h = histograms_combined_Eff[var]
    
#     fig, ax = plt.subplots(figsize=(7, 5))
#     hist.plot1d(h, ax=ax, overlay="label", clear=False,
#                 fill_opts={'alpha': 0.8})
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel("Entries")
#     ax.set_title(f"Pion {title} by Interaction + Charge")
#     ax.set_xlim(xmin, xmax)
#     ax.grid(True)
#     ax.legend(title="Interaction + Particle")
#     plt.tight_layout()
#     plt.savefig(f"Pion_{var}_Eff_Overlay_by_Category_and_Charge.png")
#     plt.show()



# Create histograms
histograms = {
    "z": hist.Hist("Pion z", hist.Cat("region", "Region"), hist.Cat("category", "Interaction"), hist.Cat("particle", "Particle Type"), hist.Bin("value", "z (mm)", 100, 1800, 2200)),
    "Px": hist.Hist("Pion Px", hist.Cat("region", "Region"), hist.Cat("category", "Interaction"), hist.Cat("particle", "Particle Type"), hist.Bin("value", "Px (MeV)", 50, -410, 410)),
    "Py": hist.Hist("Pion Py", hist.Cat("region", "Region"), hist.Cat("category", "Interaction"), hist.Cat("particle", "Particle Type"), hist.Bin("value", "Py (MeV)", 50, -410, 410)),
    "Pz": hist.Hist("Pion Pz", hist.Cat("region", "Region"), hist.Cat("category", "Interaction"), hist.Cat("particle", "Particle Type"), hist.Bin("value", "Pz (MeV)", 50, -500, 1500)),
    "KE": hist.Hist("Pion KE", hist.Cat("region", "Region"), hist.Cat("category", "Interaction"), hist.Cat("particle", "Particle Type"), hist.Bin("value", "KE (MeV)", 50, -40, 2000)),
    "Eff_KE": hist.Hist("Pion KE Efficiency", hist.Cat("category", "Interaction"), hist.Cat("particle", "Particle Type"), hist.Bin("value", "KE Efficiency", 50, -40, 2000)),
}

histograms_combined = {
    "z": hist.Hist("Pion z", hist.Cat("region", "Region"),hist.Cat("label", "Interaction + Particle"), hist.Bin("value", "z (mm)", 100, 1800, 2200)),
    "Px": hist.Hist("Pion Px", hist.Cat("region", "Region"), hist.Cat("label", "Interaction + Particle"), hist.Bin("value", "Px (MeV)", 50, -410, 410)),
    "Py": hist.Hist("Pion Py", hist.Cat("region", "Region"),hist.Cat("label", "Interaction + Particle"), hist.Bin("value", "Py (MeV)", 50, -410, 410)),
    "Pz": hist.Hist("Pion Pz", hist.Cat("region", "Region"),hist.Cat("label", "Interaction + Particle"), hist.Bin("value", "Pz (MeV)", 50, -500, 1500)),
    "KE": hist.Hist("Pion KE", hist.Cat("region", "Region"),hist.Cat("label", "Interaction + Particle"), hist.Bin("value", "KE (MeV)", 50, -40, 2000)),
    "Eff_KE": hist.Hist("Pion KE Efficiency", hist.Cat("label", "Interaction + Particle"), hist.Bin("value", "KE Efficiency", 50, -40, 2000)),
}


# Define the 2D histogram
hist_2d_KE_EffKE = hist.Hist(
    "Counts",
    hist.Cat("label", "Interaction + Particle"),
    hist.Bin("KE_Inside", r"KE Inside Target (MeV)", 50, -40, 2000),
    hist.Bin("Eff_KE", r"Efficiency (KE Around Target / KE Inside Target)", 120, 0.0, 1.2),
)

hist_2d_z_KE = hist.Hist(
    "Counts",
    hist.Cat("label", "Interaction + Particle"),
    hist.Bin("z", r"z Inside Target (mm)", 100, 0.0, 400.0),
    hist.Bin("KE", r"KE Inside Target (MeV)", 50, -40.0, 2000.0),
)

hist_2d_z_EffKE = hist.Hist(
    "Counts",
    hist.Cat("label", "Interaction + Particle"),
    hist.Bin("z", r"z Inside Target (mm)", 100, 0.0, 400.0),
    hist.Bin("Eff_KE", r"Efficiency (KE Around Target / KE Inside Target)", 120, 0.0, 1.2),
)

hist_2d_z_z = hist.Hist(
    "Counts",
    hist.Cat("label", "Interaction + Particle"),
    hist.Bin("z_inside", r"z Inside Target (mm)", 100, 0.0, 400.0),
    hist.Bin("z_around", r"z Around Target (mm)", 100, 0.0, 400.0),
)



# Fill histogram (Filling the energy loss of the pions that have been both produce inside and captured outside)
for _, row in df_merged.iterrows():
    interaction = row["interaction"]
    particle = row["particle"]
    label = f"{interaction} ($\pi^+$)" if particle == "piplus" else f"{interaction} ($\pi^-$)"

    hist_2d_KE_EffKE.fill(
        label=label,
        KE_Inside=row["KE_Inside"],
        Eff_KE=row["Eff_KE"]
    )

    hist_2d_z_EffKE.fill(
        label=label,
        z=row["InitZ_Inside"] - 1800,
        Eff_KE=row["Eff_KE"]
    )

    hist_2d_z_KE.fill(
        label=label,
        z=row["InitZ_Inside"] - 1800,
        KE=row["KE_Inside"]
    )
    
    if row["z_Around"] == 0.0:
        hist_2d_z_z.fill(
            label=label,
            z_inside=row["InitZ_Inside"] - 1800,
            z_around=row["z_Around"]
        )
    else:
        hist_2d_z_z.fill(
            label=label,
            z_inside=row["InitZ_Inside"] - 1800,
            z_around=row["z_Around"] - 1800
        )

import re

for lbl in hist_2d_KE_EffKE.axis("label").identifiers():
    h = hist_2d_KE_EffKE.integrate("label", lbl)

    fig, ax = plt.subplots(figsize=(9, 7))
    hist.plot2d(
        h,
        xaxis="KE_Inside",
        ax=ax,
        xoverflow='none',
        yoverflow='none',
        patch_opts={"cmap": "viridis"},
    )
    ax.set_title(fr"Energy Loss Distribution by Initial KE: {lbl} (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
    plt.tight_layout()
    # Start from the label name
    label_str = lbl.name
    
    # Replace special characters
    label_str = label_str.replace("+", "plus").replace("-", "minus")
    label_str = label_str.replace(" ", "_")  # Replace spaces with underscores
    
    # Remove unwanted symbols: (, ), $, ^
    label_str = re.sub(r"[()\$\^\\]", "", label_str)
    plt.savefig(f"Eff_KE_Inside_Distribution_{label_str}.png")
    plt.show()

for lbl in hist_2d_z_EffKE.axis("label").identifiers():
    h = hist_2d_z_EffKE.integrate("label", lbl)

    fig, ax = plt.subplots(figsize=(9, 7))
    hist.plot2d(
        h,
        xaxis="z",
        ax=ax,
        xoverflow='none',
        yoverflow='none',
        patch_opts={"cmap": "viridis"},
    )
    ax.set_title(fr"Energy Loss Distribution by Initial z position: {lbl} (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
    plt.tight_layout()
    # Start from the label name
    label_str = lbl.name
    
    # Replace special characters
    label_str = label_str.replace("+", "plus").replace("-", "minus")
    label_str = label_str.replace(" ", "_")  # Replace spaces with underscores
    
    # Remove unwanted symbols: (, ), $, ^
    label_str = re.sub(r"[()\$\^\\]", "", label_str)
    plt.savefig(f"2D_z_Inside_Distribution_{label_str}.png")
    plt.show()

for lbl in hist_2d_z_KE.axis("label").identifiers():
    h = hist_2d_z_KE.integrate("label", lbl)

    fig, ax = plt.subplots(figsize=(9, 7))
    hist.plot2d(
        h,
        xaxis="z",
        ax=ax,
        xoverflow='none',
        yoverflow='none',
        patch_opts={"cmap": "viridis"},
    )
    ax.set_title(fr"Matched Pions KE Distribution: {lbl} (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
    plt.tight_layout()
    # Start from the label name
    label_str = lbl.name
    
    # Replace special characters
    label_str = label_str.replace("+", "plus").replace("-", "minus")
    label_str = label_str.replace(" ", "_")  # Replace spaces with underscores
    
    # Remove unwanted symbols: (, ), $, ^
    label_str = re.sub(r"[()\$\^\\]", "", label_str)
    plt.savefig(f"Matched_Pions_z_vs_KE_{label_str}.png")
    plt.show()

for lbl in hist_2d_z_z.axis("label").identifiers():
    h = hist_2d_z_z.integrate("label", lbl)

    fig, ax = plt.subplots(figsize=(9, 7))
    hist.plot2d(
        h,
        xaxis="z_inside",
        ax=ax,
        xoverflow='none',
        yoverflow='none',
        patch_opts={"cmap": "viridis"},
    )
    ax.set_title(fr"Initial z position vs Around z position: {lbl} (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
    plt.tight_layout()
    # Start from the label name
    label_str = lbl.name
    
    # Replace special characters
    label_str = label_str.replace("+", "plus").replace("-", "minus")
    label_str = label_str.replace(" ", "_")  # Replace spaces with underscores
    
    # Remove unwanted symbols: (, ), $, ^
    label_str = re.sub(r"[()\$\^\\]", "", label_str)
    plt.savefig(f"2D_z_Inside_z_Around_Distribution_{label_str}.png")
    plt.show()

# ===========================================
# Select unmatched pions
# (they have all "Around" columns == 0)
# ===========================================

# This assumes 'Px_Around' is always present if matched
unmatched_pions = df_merged_all[df_merged_all["Px_Around"] == 0]

# ------------------------------------------------
# Define histograms for unmatched pions (Inside only)
# ------------------------------------------------
histograms_unmatched_inside = {
    "Px": hist.Hist(
        "Counts",
        hist.Cat("label", "Interaction + Particle"),
        hist.Bin("value", "Px (MeV)", 50, -410, 410)
    ),
    "Py": hist.Hist(
        "Counts",
        hist.Cat("label", "Interaction + Particle"),
        hist.Bin("value", "Py (MeV)", 50, -410, 410)
    ),
    "Pz": hist.Hist(
        "Counts",
        hist.Cat("label", "Interaction + Particle"),
        hist.Bin("value", "Pz (MeV)", 50, -500, 1500)
    ),
    "z": hist.Hist(
        "Counts",
        hist.Cat("label", "Interaction + Particle"),
        hist.Bin("value", "z (mm)", 100, 1800, 2200)
    )
}

# ------------------------------------------------
# Fill histograms
# ------------------------------------------------
for _, row in unmatched_pions.iterrows():
    interaction = row["interaction"]
    particle = row["particle"]
    label = f"{interaction} ($\pi^+$)" if particle == "piplus" else f"{interaction} ($\pi^-$)"
    
    histograms_unmatched_inside["Px"].fill(label=label, value=row["Px_Inside"])
    histograms_unmatched_inside["Py"].fill(label=label, value=row["Py_Inside"])
    histograms_unmatched_inside["Pz"].fill(label=label, value=row["Pz_Inside"])
    histograms_unmatched_inside["z"].fill(label=label, value=row["InitZ_Inside"])

# ------------------------------------------------
# Plotting
# ------------------------------------------------

line_colors = [
    'red', 'dodgerblue', 'limegreen', 'magenta', 'yellow', 'orange'
]

for var, h in histograms_unmatched_inside.items():
    labels = h.axis("label").identifiers()
    for color, lbl in zip(line_colors, labels):
        h_proj = h.integrate("label", lbl)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        hist.plot1d(h_proj, ax=ax, clear=False, line_opts={"color": color, "linestyle": "-"})
        ax.set_xlabel(f"{var} (MeV)" if var != "z" else "z (mm)")
        ax.set_ylabel("Entries")
        ax.set_title(fr"Unmatched Pions {var} Distribution: {lbl} (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
        ax.grid(True)
        plt.tight_layout()

        legend_labels = []
        count_unmatched = int(h_proj.values(overflow="all")[()].sum())
        legend_labels.append(f"Unmatched {lbl.name} (n={count_unmatched})")
        ax.legend(legend_labels, loc='best')
        
        # Clean label for filename
        label_str = lbl.name
        label_str = label_str.replace("+", "plus").replace("-", "minus")
        label_str = label_str.replace(" ", "_")
        label_str = re.sub(r"[()\$\^\\]", "", label_str)
        
        plt.savefig(f"Unmatched_Pions_{var}_{label_str}.png")
        plt.show()


# ------------------------------------------------
# Define 2D histogram for unmatched pions
# ------------------------------------------------
hist2d_z_KE_unmatched = hist.Hist(
    "Counts",
    hist.Cat("label", "Interaction + Particle"),
    hist.Bin("z", "z (mm)", 100, 1800, 2200),
    hist.Bin("KE", "KE (MeV)", 50, -40, 2000)
)

# ------------------------------------------------
# Fill 2D histogram
# ------------------------------------------------
for _, row in unmatched_pions.iterrows():
    interaction = row["interaction"]
    particle = row["particle"]
    label = f"{interaction} ($\pi^+$)" if particle == "piplus" else f"{interaction} ($\pi^-$)"
    
    hist2d_z_KE_unmatched.fill(
        label=label,
        z=row["InitZ_Inside"],
        KE=row["KE_Inside"]
    )

# ------------------------------------------------
# Plot 2D histogram
# ------------------------------------------------
for lbl in hist2d_z_KE_unmatched.axis("label").identifiers():
    h_proj = hist2d_z_KE_unmatched.integrate("label", lbl)
    
    fig, ax = plt.subplots(figsize=(9, 7))
    hist.plot2d(
        h_proj,
        xaxis="z",
        ax=ax,
        patch_opts={"cmap": "viridis"},
        xoverflow='none',
        yoverflow='none'
    )
    
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("KE (MeV)")
    ax.set_title(fr"Unmatched Pions: z vs KE Distribution {lbl} (r = 30mm, $\theta = 0, \alpha = 0, \beta = 0$)")
    ax.grid(True)
    plt.tight_layout()
    
    # Clean label for filename
    label_str = lbl.name
    label_str = label_str.replace("+", "plus").replace("-", "minus")
    label_str = label_str.replace(" ", "_")
    label_str = re.sub(r"[()\$\^\\]", "", label_str)
    
    plt.savefig(f"Unmatched_Pions_z_vs_KE_{label_str}.png")
    plt.show()

    
# # ==================================
# # Fill histograms for the efficiency
# # ==================================
# for _, row in df_merged.iterrows():
#     label_interaction = row["interaction"]
#     label_particle = row["particle"]
#     label = f"{label_interaction} ($\pi^+$)" if label_particle == "piplus" else f"{label_interaction} ($\pi^-$)"

#     histograms["Px"].fill(region="Inside", category=label_Interaction, particle=label_Particle_Type, value=row['Px_Inside'])
#     histograms["Py"].fill(region="Inside", category=label_Interaction, particle=label_Particle_Type, value=row['Py_Inside'])
#     histograms["Pz"].fill(region="Inside", category=label_Interaction, particle=label_Particle_Type, value=row['Pz_Inside'])
#     histograms["KE"].fill(region="Inside", category=label_Interaction, particle=label_Particle_Type, value=row['KE_Inside'])
    
#     histograms_combined["Px"].fill(region="Inside", label=label, value=row['Px_Inside'])
#     histograms_combined["Py"].fill(region="Inside",label=label, value=row['Py_Inside'])
#     histograms_combined["Pz"].fill(region="Inside",label=label, value=row['Pz_Inside'])
#     histograms_combined["KE"].fill(region="Inside",label=label, value=row['KE_Inside'])

#     histograms["Px"].fill(region="Around", category=label_Interaction, particle=label_Particle_Type, value=row['Px_Around'])
#     histograms["Py"].fill(region="Around", category=label_Interaction, particle=label_Particle_Type, value=row['Py_Around'])
#     histograms["Pz"].fill(region="Around", category=label_Interaction, particle=label_Particle_Type, value=row['Pz_Around'])
#     histograms["KE"].fill(region="Around", category=label_Interaction, particle=label_Particle_Type, value=row['KE_Around'])
    
#     histograms_combined["Px"].fill(region="Around", label=label, value=row['Px_Around'])
#     histograms_combined["Py"].fill(region="Around", label=label, value=row['Py_Around'])
#     histograms_combined["Pz"].fill(region="Around", label=label, value=row['Pz_Around'])
#     histograms_combined["KE"].fill(region="Around", label=label, value=row['KE_Around'])


# # ===============================
# # Plot Efficiency Ratio Histograms
# # ===============================
# plot_vars = {
#     "Px": ("Px (MeV)", -410, 410),
#     "Py": ("Py (MeV)", -410, 410),
#     "Pz": ("Pz (MeV)", -500, 1500),
#     "KE": ("KE (MeV)", 0, 2000)
# }

# for var, (xlabel, xmin, xmax) in plot_vars.items():
#     fig, ax = plt.subplots(figsize=(6, 4))

#     hist.plotratio(
#         num=histograms_combined[var].integrate("region", "Around").integrate("label", f'primary ($\pi^+$)'),
#         denom=histograms_combined[var].integrate("region", "Inside").integrate("label", f'primary ($\pi^+$)'),
#         ax=ax,
#         error_opts={'color': 'darkblue', 'marker': '.'},
#         denom_fill_opts={},
#         unc='num'
#     )

#     ax.set_xlabel(xlabel)
#     ax.set_ylabel("Efficiency")
#     ax.set_xlim(xmin, xmax)
#     ax.set_ylim(0.6, 1.2)
#     ax.set_title(f"Pion {var} Efficiency Ratio (Around / Inside)")
#     ax.grid(True)
#     plt.tight_layout()
#     plt.savefig(f"Eff_Target_Around_vs_Inside_{var}.png")
#     plt.show()


# In[9]:


# count = 0
# error_Eff_KE = []
# error_KE_Inside = []
# error_KE_Around = []

# for i in range(len(df_merged['Eff_KE'])):
#     if df_merged['Eff_KE'][i] > 1.05:
#         error_Eff_KE.append(df_merged['Eff_KE'][i])
#         error_KE_Inside.append(df_merged['KE_Inside'][i])
#         error_KE_Around.append(df_merged['KE_Around'][i])
#         count += 1
# print(count)
# print(error_Eff_KE)
# print(error_KE_Inside)
# print(error_KE_Around)

print(df_inside_all)

print(df_around_all)


print(df_merged)
# plt.scatter(df_merged['KE_Inside'], df_merged['Eff_KE'])
# plt.xlim(100)
# plt.show()

# print(df_merged[df_merged["Eff_KE"] > 1][["event", "track", "KE_Inside", "KE_Around", "Eff_KE", "InitX_Inside", "InitY_Inside", "InitZ_Inside", "InitX_Around", "InitY_Around", "InitZ_Around", "t_Inside", "t_Around"]])



# In[5]:


import numpy as np
import matplotlib.pyplot as plt

def load_solenoid_field(filepath):
    """
    Load solenoid field data from a G4beamline-style printfield file.
    Returns a NumPy array with columns: r, z, Br, Bz
    """
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip().startswith("data"):
                break  # Found start of data section
        for line in f:
            if line.strip():
                tokens = line.strip().split()
                if len(tokens) == 4:
                    r, z, Br, Bz = map(float, tokens)
                    data.append((r, z, Br, Bz))
    return np.array(data)

def plot_Bz_vs_r(files, labels, title="Axial Magnetic Field $B_z$ vs Radius", save_path=None):
    """
    Plot Bz vs r from one or more files.

    Args:
        files (list of str): List of file paths.
        labels (list of str): Labels for each dataset.
        title (str): Plot title.
        save_path (str or None): Path to save figure, if desired.
    """
    plt.figure(figsize=(8, 6))

    for filepath, label in zip(files, labels):
        data = load_solenoid_field(filepath)
        r, Bz = data[:, 0], data[:, 3]
        plt.plot(r, Bz, label=label)

    plt.xlabel("Radius $r$ (mm)")
    plt.ylabel("Axial Magnetic Field $B_z$ (T)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to {save_path}")

    plt.show()

file1 = "./solenoid_1_field_cylinder_Z0_0mm.txt"
file2 = "./solenoid_1_field_cylinder_Z0_800mm.txt"
file3 = "./solenoid_1_field_cylinder_Z0_-800mm.txt"
file4 = "./solenoid_2_field_cylinder_Z0_0mm.txt"
file5 = "./solenoid_2_field_cylinder_Z0_800mm.txt"
file6 = "./solenoid_2_field_cylinder_Z0_-800mm.txt"

plot_Bz_vs_r(
    files=[file1, file2, file3, file4, file5, file6],
    labels=[
        r"$I = 96.68,\mathrm{A/mm^2},\ z = 2000\,\mathrm{mm}$",
        r"$I = 96.68,\mathrm{A/mm^2},\ z = 2800 \,\mathrm{mm}$",
        r"$I = 96.68,\mathrm{A/mm^2},\ z = 1200\,\mathrm{mm}$",
        r"$I = 59.208,\mathrm{A/mm^2},\ z = 4000\,\mathrm{mm}$",
        r"$I = 59.208,\mathrm{A/mm^2},\ z = 4800 \,\mathrm{mm}$",
        r"$I = 59.208,\mathrm{A/mm^2},\ z = 3200\,\mathrm{mm}$",
    ],
    title=r"Axial Magnetic Field $B_z$ vs Radius at Different $z$ Positions",
    save_path="Compare_Magneticfield.png"
)



# In[6]:


import pandas as pd
import matplotlib.pyplot as plt

# Replace with your filename
filename = "FieldAlongZ.txt"

# Read the file, skipping the header lines that start with #
data = pd.read_csv(
    filename,
    comment="#",
    delim_whitespace=True,
    header=None,
    names=["x", "y", "z", "t", "Bx", "By", "Bz", "Ex", "Ey", "Ez"]
)

# Extract z and Bz columns
z = data["z"]
Bz = data["Bz"]

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(z, Bz, color='blue')
# add shaded regions
plt.axvspan(1000, 3000, color='red', alpha=0.2, label='Solenoid 1')
plt.axvspan(3000, 5000, color='green', alpha=0.2, label='Solenoid 2')
plt.xlabel("Z (mm)")
plt.ylabel("Bz (T)")
plt.title("Bz (peak field) vs Z")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./BzPeak_vs_Z.png')
plt.show()



# In[15]:


import uproot
import numpy as np
from coffea import hist
import matplotlib.pyplot as plt

# Open the ROOT file
file = uproot.open("g4beamline.root")

# Access both TTrees
tree1 = file["Trace/AllTracks;1"]
tree2 = file["Trace/AllTracks;2"]

# Reference in the Target
Reference_Z_Start_of_Target = 1800

# Load branches into dicts
vars_to_load = ["x", "y", "z", "Px", "Py", "Pz", "PDGid"]
data1 = tree1.arrays(vars_to_load, library="np")
data2 = tree2.arrays(vars_to_load, library="np")

# Concatenate all variables
data = {var: np.concatenate([data1[var], data2[var]]) for var in vars_to_load}

# Filter: only keep protons (PDGid == 2212)
is_proton = (data["PDGid"] == 2212)
within_Target = ((data['z'] <= 405 + Reference_Z_Start_of_Target) & (data['z'] >= 0 + Reference_Z_Start_of_Target))

# Combine both selections
selected = is_proton & within_Target

filtered_data = {k: v[selected] for k, v in data.items()}

# === Coffea 1D Histogramming ===
hist1d_x_y = hist.Hist("Proton Tracks", hist.Cat("var", "Variable"), hist.Bin("value", "Value", 80, -40, 40))
hist1d_z = hist.Hist("Proton Tracks", hist.Cat("var", "Variable"), hist.Bin("value", "Value", 400, 1.0, 399))

for var in ["x", "y", "Px", "Py"]:
    hist1d_x_y.fill(var=var, value=filtered_data[var])

for var in ["z", "Pz"]:
    hist1d_z.fill(var=var, value=filtered_data[var] - Reference_Z_Start_of_Target)
    
# Plot 1D histograms
for var in ["x", "y", "Px", "Py"]:
    plt.figure()
    hist.plot1d(hist1d_x_y[var])
    plt.title(f"Histogram of {var}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
for var in ["z", "Pz"]:
    plt.figure()
    hist.plot1d(hist1d_z[var])
    plt.title(f"Histogram of {var}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === 2D Scatter Histograms ===
def plot2d(x, y, xlabel, ylabel):
    plt.figure()
    plt.hist2d(x, y, bins=100, cmap="viridis")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{xlabel} vs {ylabel}")
    plt.colorbar(label="Counts")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot2d(filtered_data["x"], filtered_data["z"], "x", "z")
plot2d(filtered_data["y"], filtered_data["z"], "y", "z")


# In[ ]:


# Load datasets
dataset = load_dataset('./TungstenTarget-50x50x1000_Solenoid-4T-3000x12500/TungstenTarget-50x50x1000_Solenoid-4T-3000x12500.root')
dataset_600 = load_dataset('./TungstenTarget-50x50x600_Solenoid-4T-3000x12500/TungstenTarget-50x50x600_Solenoid-4T-3000x12500.root')
dataset_200 = load_dataset('./TungstenTarget-50x50x200_Solenoid-4T-3000x12500/TungstenTarget-50x50x200_Solenoid-4T-3000x12500.root')

# List available detectors
detectors = list_detector_keys(dataset)
detectors_600 = list_detector_keys(dataset_600)
detectors_200 = list_detector_keys(dataset_200)
print("Available Detectors in Dataset 1000:", detectors)
print("Available Detectors in Dataset 600:", detectors_600)
print("Available Detectors in Dataset 200:", detectors_200)

# Choose a virtual detector position
detector_key = "Det_1150cm_Solenoid"
detector_key_600 = "Det_1190cm_Solenoid"
detector_key_200 = "Det_1230cm_Solenoid"

# Extract PDG ID data from the chosen detector
PDGid = get_virtual_detector_data(dataset, detector_key, "PDGid")
PDGid_600 = get_virtual_detector_data(dataset_600, detector_key_600, "PDGid")
PDGid_200 = get_virtual_detector_data(dataset_200, detector_key_200, "PDGid")


# In[ ]:


# Load Datasets
dataset_4T = load_dataset('./TungstenTarget-50x50x600_Solenoid-4T-3000x12500/TungstenTarget-50x50x600_Solenoid-4T-3000x12500.root')
dataset_6T = load_dataset('./TungstenTarget-50x50x600_Solenoid-6T-3000x12500/TungstenTarget-50x50x600_Solenoid-6T-3000x12500.root')
dataset_8T = load_dataset('./TungstenTarget-50x50x600_Solenoid-8T-3000x12500/TungstenTarget-50x50x600_Solenoid-8T-3000x12500.root')

# Detector key
detector_key = "Det_1190cm_Solenoid"

# Extract PDG ID data
PDGid_4T = get_virtual_detector_data(dataset_4T, detector_key, "PDGid")
PDGid_6T = get_virtual_detector_data(dataset_6T, detector_key, "PDGid")
PDGid_8T = get_virtual_detector_data(dataset_8T, detector_key, "PDGid")

if PDGid_4T is not None and PDGid_6T is not None and PDGid_8T is not None:
    # Convert to NumPy
    PDGid_4T = ak.to_numpy(PDGid_4T).astype(int)
    PDGid_6T = ak.to_numpy(PDGid_6T).astype(int)
    PDGid_8T = ak.to_numpy(PDGid_8T).astype(int)

    # PDG ID filters
    PDGID_muons = [-13, 13]
    PDGID_muons_pions_kaons = [13, -13, 211, -211, 321, -321]

    # Apply PDG ID filtering
    mask_mu_4T = filter_by_pdgid(PDGid_4T, PDGID_muons)
    mask_mu_6T = filter_by_pdgid(PDGid_6T, PDGID_muons)
    mask_mu_8T = filter_by_pdgid(PDGid_8T, PDGID_muons)

    mask_mpk_4T = filter_by_pdgid(PDGid_4T, PDGID_muons_pions_kaons)
    mask_mpk_6T = filter_by_pdgid(PDGid_6T, PDGID_muons_pions_kaons)
    mask_mpk_8T = filter_by_pdgid(PDGid_8T, PDGID_muons_pions_kaons)

    mask_pr_4T = filter_by_pdgid(PDGid_4T, 2212)
    mask_pr_6T = filter_by_pdgid(PDGid_6T, 2212)
    mask_pr_8T = filter_by_pdgid(PDGid_8T, 2212)

    # Get positions
    def get_xy(dataset, mask):
        x = get_virtual_detector_data(dataset, detector_key, "x")[mask]
        y = get_virtual_detector_data(dataset, detector_key, "y")[mask]
        return x, y

    x_mu_4T, y_mu_4T = get_xy(dataset_4T, mask_mu_4T)
    x_mu_6T, y_mu_6T = get_xy(dataset_6T, mask_mu_6T)
    x_mu_8T, y_mu_8T = get_xy(dataset_8T, mask_mu_8T)

    # Get momenta
    def get_momenta(dataset, mask):
        px = get_virtual_detector_data(dataset, detector_key, "Px")[mask]
        py = get_virtual_detector_data(dataset, detector_key, "Py")[mask]
        pz = get_virtual_detector_data(dataset, detector_key, "Pz")[mask]
        return px, py, pz

    Px_mu_4T, Py_mu_4T, Pz_mu_4T = get_momenta(dataset_4T, mask_mu_4T)
    Px_mu_6T, Py_mu_6T, Pz_mu_6T = get_momenta(dataset_6T, mask_mu_6T)
    Px_mu_8T, Py_mu_8T, Pz_mu_8T = get_momenta(dataset_8T, mask_mu_8T)

    Px_pr_4T, Py_pr_4T, Pz_pr_4T = get_momenta(dataset_4T, mask_pr_4T)
    Px_pr_6T, Py_pr_6T, Pz_pr_6T = get_momenta(dataset_6T, mask_pr_6T)
    Px_pr_8T, Py_pr_8T, Pz_pr_8T = get_momenta(dataset_8T, mask_pr_8T)

    # Masses
    mu_mass = 105.67
    pr_mass = 938.27

    # Energies
    E_mu_4T = np.sqrt(mu_mass**2 + Px_mu_4T**2 + Py_mu_4T**2 + Pz_mu_4T**2)
    E_mu_6T = np.sqrt(mu_mass**2 + Px_mu_6T**2 + Py_mu_6T**2 + Pz_mu_6T**2)
    E_mu_8T = np.sqrt(mu_mass**2 + Px_mu_8T**2 + Py_mu_8T**2 + Pz_mu_8T**2)

    E_pr_4T = np.sqrt(pr_mass**2 + Px_pr_4T**2 + Py_pr_4T**2 + Pz_pr_4T**2)
    E_pr_6T = np.sqrt(pr_mass**2 + Px_pr_6T**2 + Py_pr_6T**2 + Pz_pr_6T**2)
    E_pr_8T = np.sqrt(pr_mass**2 + Px_pr_8T**2 + Py_pr_8T**2 + Pz_pr_8T**2)

    # PDG ID Plots
    plot_1D_histogram_overlay(
        datasets=[PDGid_4T[mask_mpk_4T], PDGid_6T[mask_mpk_6T], PDGid_8T[mask_mpk_8T]],
        labels=[r"$4\,\mathrm{T}$", r"$6\,\mathrm{T}$", r"$8\,\mathrm{T}$"],
        xlabel="PDG ID",
        title="Overlayed PDG ID Distribution at End of Solenoid (600mm Tungsten)",
        discrete=True,
        save_path="Compare_PDGID_Overlay_Solenoid600mm_Magneticfield.png"
    )
    
    # Energy plots (muons and protons)
    plot_1D_histogram_overlay(
        datasets=[E_mu_4T, E_mu_6T, E_mu_8T],
        labels=[r"$4\,\mathrm{T}$", r"$6\,\mathrm{T}$", r"$8\,\mathrm{T}$"],
        xlabel="Energy (MeV)",
        title="Compare_Muon Energy Distribution at End of Solenoid (600mm Tungsten)",
        bins=100,
        save_path="Muon_Energy_Solenoid600mm_Magneticfield.png"
    )
    
    plot_1D_histogram_overlay(
        datasets=[E_pr_4T, E_pr_6T, E_pr_8T],
        labels=[r"$4\,\mathrm{T}$", r"$6\,\mathrm{T}$", r"$8\,\mathrm{T}$"],
        xlabel="Energy (MeV)",
        title="Proton Energy Distribution at End of Solenoid (600mm Tungsten)",
        bins=100,
        save_path="Compare_Proton_Energy_Solenoid600mm_Magneticfield.png"
    )
    
    # 2D scatter plots for muons
    plot_scatter(
        x_mu_4T, y_mu_4T, "x (mm)", "y (mm)", "Muon Distribution (600mm, 4T)", 
        (-2000, 2000), (-2000, 2000), save_path="Muon_Distribution_600mm_4T.png"
    )
    
    plot_scatter(
        x_mu_6T, y_mu_6T, "x (mm)", "y (mm)", "Muon Distribution (600mm, 6T)", 
        (-2000, 2000), (-2000, 2000), save_path="Muon_Distribution_600mm_6T.png"
    )
    
    plot_scatter(
        x_mu_8T, y_mu_8T, "x (mm)", "y (mm)", "Muon Distribution (600mm, 8T)", 
        (-2000, 2000), (-2000, 2000), save_path="Muon_Distribution_600mm_8T.png"
    )

else:
    print("Failed to retrieve PDG ID data.")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# # --------------------------------------------
# # Example Usage: Loading and Plotting Data
# # --------------------------------------------

# # Load datasets
# dataset = load_dataset('./TungstenTarget-50x50x1000_Solenoid-4T-3000x12500/TungstenTarget-50x50x1000_Solenoid-4T-3000x12500.root')
# dataset_200 = load_dataset('TungstenTarget-50x50x200_Solenoid-4T-3000x12500/TungstenTarget-50x50x200_Solenoid-4T-3000x12500.root')

# # List available detectors
# detectors = list_detector_keys(dataset)
# detectors_200 = list_detector_keys(dataset_200)
# print("Available Detectors in Dataset 1000:", detectors)
# print("Available Detectors in Dataset 200:", detectors_200)

# # Choose a virtual detector position (example: 'Det_400cm_Solenoid')
# detector_key = "Det_400cm_Solenoid"

# # Extract PDG ID data from the chosen detector
# PDGid = get_virtual_detector_data(dataset, detector_key, "PDGid")
# PDGid_200 = get_virtual_detector_data(dataset_200, detector_key, "PDGid")

# # Check if PDG ID data was successfully extracted
# if PDGid is not None and PDGid_200 is not None:
#     # Convert to numpy array (for compatibility)
#     PDGid = ak.to_numpy(PDGid).astype(int)
#     PDGid_200 = ak.to_numpy(PDGid_200).astype(int)

#     # Apply PDG ID filtering (e.g., only select muons PDG ID = ±13)
#     mask_muons = filter_by_pdgid(PDGid, 13, -13)
#     mask_muons_200 = filter_by_pdgid(PDGid_200, 13, -13)

#     mask_protons_neutrons = filter_by_pdgid(PDGid, 2212, 2112)
#     mask_protons_neutrons_200 = filter_by_pdgid(PDGid_200, 2212, 2112)
    

#     # Extract positions x, y for muons
#     x = get_virtual_detector_data(dataset, detector_key, "x")[mask_muons]
#     y = get_virtual_detector_data(dataset, detector_key, "y")[mask_muons]

#     x_200 = get_virtual_detector_data(dataset_200, detector_key, "x")[mask_muons_200]
#     y_200 = get_virtual_detector_data(dataset_200, detector_key, "y")[mask_muons_200]

#     # Plot 1D histogram of PDG IDs (Full dataset)
#     plot_1D_histogram(PDGid, "PDG ID", "PDG ID Distribution", xlim=(0, 100))
#     plot_1D_histogram(PDGid, "PDG ID", "PDG ID Distribution at 400cm (#theta = 0, #alpha = 0, #beta = 0)")
#     plot_1D_histogram(PDGid_200, "PDG ID", "PDG ID Distribution at 400cm (200mm Tungsten)")
#     plot_1D_histogram(PDGid[mask_protons_neutrons], "PDG ID", "PDG ID Distribution at 400cm (#theta = 0, #alpha = 0, #beta = 0)", bins=np.unique(PDGid[mask_protons_neutrons]))
#     plot_1D_histogram(PDGid_200[mask_protons_neutrons_200], "PDG ID", "PDG ID Distribution at 400cm (200mm Tungsten)", bins=np.unique(PDGid_200[mask_protons_neutrons_200]))

#     # Plot stacked 1D histogram of PDG IDs with custom x-axis limits
#     plot_1D_histogram_stacked(
#     datasets=[PDGid[mask_muons], PDGid_200[mask_muons_200]], 
#     labels=["#theta = 0, #alpha = 0, #beta = 0", "200mm Tungsten"], 
#     xlabel="PDG ID", 
#     title="Stacked PDG ID Distribution at 400cm",
#     discrete=True
#     )


#     # # Plot 2D histogram (x vs y) for filtered muons
#     # plot_2D_histogram(x, y, "x (mm)", "y (mm)", "Muon Distribution at 400cm (#theta = 0, #alpha = 0, #beta = 0)", (-3000, 3000), (-3000, 3000),-1, 100, bins=500)

#     # plot_2D_histogram(x_200, y_200, "x (mm)", "y (mm)", "Muon Distribution at 400cm (200mm Tungsten)")

#     # Plot 2D scatter (x vs y) for filtered muons
#     plot_scatter(x, y, "x (mm)", "y (mm)", "Muon Distribution at 400cm (#theta = 0, #alpha = 0, #beta = 0)",(-5000, 5000), (-5000, 5000))
#     plot_scatter(x_200, y_200, "x (mm)", "y (mm)", "Muon Distribution at 400cm (200mm Tungsten)")
# else:
#     print("Failed to retrieve PDG ID data.")


# In[ ]:


# def plot_1D_histogram_discrete(data, xlabel, title):
#     """
#     Plot a 1D histogram where x-axis has discrete unique values.

#     Args:
#         data (array-like): The data to plot.
#         xlabel (str): Label for the x-axis.
#         title (str): Title of the histogram.
#     """
#     unique_values, counts = np.unique(data, return_counts=True)

#     plt.figure(figsize=(8, 6))
#     plt.bar(unique_values, counts, width=1.0, edgecolor='black', alpha=0.7)

#     plt.xlabel(xlabel)
#     plt.ylabel("Count")
#     plt.title(title)
#     plt.xticks(unique_values)  # Ensure only discrete values are labeled
#     plt.grid(True, axis='y')
#     plt.show()

# # Apply boolean masks to filter PDG ID values
# filtered_PDGid = PDGid[mask_protons_neutrons]
# filtered_PDGid_200 = PDGid_200[mask_protons_neutrons_200]
# # Now call the function
# plot_1D_histogram_discrete(filtered_PDGid, "PDG ID", "PDG ID Distribution at 400cm (#theta = 0, #alpha = 0, #beta = 0)")
# plot_1D_histogram_discrete(filtered_PDGid_200, "PDG ID", "PDG ID Distribution at 400cm (200mm Tungsten)")


# In[ ]:


# def plot_1D_histogram(
#     data,
#     xlabel,
#     title,
#     bins=50,
#     start=None,
#     stop=None,
#     axis_name="variable",
#     category_name=None,
#     category_label=None,
#     category_value=None,
#     is_integer=False
# ):
#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(8, 6))  # <-- add this

#     if is_integer:
#         unique_vals = np.unique(data)
#         main_axis = hist.axis.IntCategory(
#             unique_vals,
#             name=axis_name,
#             label=xlabel
#         )
#     else:
#         if start is None:
#             start = float(np.min(data))
#         if stop is None:
#             stop = float(np.max(data))
#         main_axis = hist.axis.Regular(
#             bins,
#             start,
#             stop,
#             name=axis_name,
#             label=xlabel
#         )

#     if category_name and category_label and category_value:
#         cat_axis = hist.axis.StrCategory(
#             categories=[category_value],
#             name=category_name,
#             label=category_label
#         )
#         h = Hist(main_axis, cat_axis)
#         h.fill(**{axis_name: data, category_name: category_value})
#     else:
#         h = Hist(main_axis)
#         h.fill(**{axis_name: data})

#     h.plot()
#     if category_name:
#         plt.legend()
#     plt.title(title)  # <- optional, improves context
#     plt.show()


# In[ ]:


# def plot_2D_histogram(
#     data_x,
#     data_y,
#     xlabel,
#     ylabel,
#     title,
#     bins=50,
#     x_start=None,
#     x_stop=None,
#     y_start=None,
#     y_stop=None,
#     x_name="x",
#     y_name="y"
# ):
#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(8, 6))  # <-- add this

#     if x_start is None:
#         x_start = float(np.min(data_x))
#     if x_stop is None:
#         x_stop = float(np.max(data_x))
#     if y_start is None:
#         y_start = float(np.min(data_y))
#     if y_stop is None:
#         y_stop = float(np.max(data_y))

#     x_axis = hist.axis.Regular(bins, x_start, x_stop, name=x_name, label=xlabel)
#     y_axis = hist.axis.Regular(bins, y_start, y_stop, name=y_name, label=ylabel)

#     h2d = Hist(x_axis, y_axis)
#     h2d.fill(**{x_name: data_x, y_name: data_y})

#     h2d.plot2d()
#     plt.title(title)
#     plt.show()


# In[ ]:


# # Check if PDG ID data was successfully extracted
# if PDGid is not None and PDGid_200 is not None:
#     # Convert to numpy array (for compatibility)
#     PDGid = ak.to_numpy(PDGid).astype(int)
#     PDGid_200 = ak.to_numpy(PDGid_200).astype(int)

#     # Apply PDG ID filtering (e.g., only select muons PDG ID = ±13)
#     mask_muons = filter_by_pdgid(PDGid, 13, -13)
#     mask_muons_200 = filter_by_pdgid(PDGid_200, 13, -13)

#     # Extract positions x, y for muons
#     x = ak.to_numpy(get_virtual_detector_data(dataset, detector_key, "x")[mask_muons])
#     y = ak.to_numpy(get_virtual_detector_data(dataset, detector_key, "y")[mask_muons])

#     x_200 = ak.to_numpy(get_virtual_detector_data(dataset_200, detector_key, "x")[mask_muons_200])
#     y_200 = ak.to_numpy(get_virtual_detector_data(dataset_200, detector_key, "y")[mask_muons_200])

#     # Plot 1D histogram of PDG IDs (Full dataset) with integer category axis and legend
#     plot_1D_histogram(
#         data=PDGid,
#         xlabel="PDG ID",
#         title="PDG ID Distribution at 400 cm (#theta = 0, #alpha = 0, #beta = 0)",
#         axis_name="pdg_id",
#         is_integer=True,
#         category_name="target",
#         category_label="Target",
#         category_value="1000mm"
#     )

#     plot_1D_histogram(
#         data=PDGid_200,
#         xlabel="PDG ID",
#         title="PDG ID Distribution at 400 cm (200mm Tungsten)",
#         axis_name="pdg_id",
#         is_integer=True,
#         category_name="target",
#         category_label="Target",
#         category_value="200mm"
#     )

#     # Plot 2D histogram (x vs y) for filtered muons
#     plot_2D_histogram(
#         data_x=x,
#         data_y=y,
#         xlabel="x (mm)",
#         ylabel="y (mm)",
#         title="Muon x-y Distribution at 400 cm (#theta = 0, #alpha = 0, #beta = 0)",
#         x_name="x",
#         y_name="y"
#     )

#     plot_2D_histogram(
#         data_x=x_200,
#         data_y=y_200,
#         xlabel="x (mm)",
#         ylabel="y (mm)",
#         title="Muon x-y Distribution at 400 cm (200mm Tungsten)",
#         x_name="x",
#         y_name="y"
#     )
# else:
#     print("Failed to retrieve PDG ID data.")


# In[ ]:





# In[ ]:




