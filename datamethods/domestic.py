"""
File containing some functions to aid with clustering and imputation.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import k_modes
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score
)
import warnings
warnings.filterwarnings("ignore")

# Constants
epc_fields = [
    "epc_rating_current",
    "epc_rating_potential",
    "d_epc_mainht_fuel",
    "d_epc_mainht_plant",
    "d_epc_mainht_room",
    "d_epc_secondht_fuel",
    "d_epc_dhw_fuel",
    "d_epc_dhw_plant",
    "d_epc_envelope_flr_type",
    "d_epc_envelope_flr_insulation",
    "d_epc_envelope_rf_type",
    "d_epc_envelope_rf_insulation",
    "d_epc_envelope_wall_type",
    "d_epc_envelope_wall_insulation",
    "d_epc_glazing_type"
]

def get_epc_val(rows: pd.DataFrame, field_name: str) -> str:
    """
    Function to extract values of epc columns.
    """
    for i in range(len(rows)):
        strval = rows.iloc[i][field_name]
        if pd.notna(strval):
            return strval
    return pd.NA


def clustering_summary(df, cols_of_interest):
    # Extract the cluster labels
    cluster_labels = df['Cluster'].unique()

    # Initialize an empty dataframe for the summary
    summary_df = pd.DataFrame(index=cols_of_interest)

    # Iterate through each cluster label
    for label in cluster_labels:
        # Filter rows based on the cluster label
        cluster_rows = df[df['Cluster'] == label]

        # Calculate mode for each column
        mode_values = {}
        for column in summary_df.index:
            if pd.api.types.is_numeric_dtype(df[column]):
                # For numeric fields, use the mode from the mode() function
                try:
                    mode_values[column] = cluster_rows[column].mode().iloc[0]
                except IndexError:
                    mode_values[column] = cluster_rows[column].mode()
            else:
                # For categorical fields, use the most frequent value using value_counts
                try:
                    mode_values[column] = cluster_rows[column].value_counts().idxmax()
                except ValueError:
                    mode_values[column] = pd.NA

        # Add mode values as a column in the summary dataframe
        summary_df[label] = pd.Series(mode_values)

    return summary_df




def compute_cluster_curve_numeric(
        df: pd.DataFrame, 
        fields: list,
        cluster_nums: list,
        verbose: bool = True
        ) -> tuple[list, list, list, list]:
    """
    Returns a vector of cluster numbers and a vector of inertias,
    so that the inertia vs cluster num plot can be made. It also
    returns a list of lists: this is the list of cluster labels, 
    for each cluster num. It also returns a list of dictionaries:
    one dictionary for each cluster number. Each dictionary is nested.
    The top level keys are the cluster label; their associated values
    are then another dictionary whose keys are all the column names
    and whose values are the typical or mode values of each column.
    """
    
    # Subset the DataFrame with the selected columns
    subset_df = df[fields]

    # Lists to hold the scores and other data
    inertias = []
    cluster_label_list = []
    typical_value_list = []

    # Iterate over the cluster values
    for i in cluster_nums:
        if verbose:
            print(f"Clustering with {i} clusters.")

        # Create k-prototypes model
        kproto = KMeans(i)

        # Fit the model
        clusters = kproto.fit_predict(subset_df.values)

        # Add cluster labels to the data frame
        df["Cluster"] = clusters

        # Store the cluster labels
        df = df.sort_index()
        cluster_label_list.append(df["Cluster"].values)

        # Extract typical values from each cluster
        # and store in a dictionary
        typical_values = {}
        for cluster_label in df["Cluster"].unique():

            # Create a dict for this cluster
            typical_values[cluster_label] = {}

            # Find the typical value for each field
            cluster_data = df[df["Cluster"] == cluster_label]
            for field in fields:
                typical_values[cluster_label][field] = cluster_data[field].mode().iloc[0]

        # Record scores
        inertias.append(kproto.inertia_)

        # Record typical value dict
        typical_value_list.append(typical_values)

    return cluster_nums, inertias, cluster_label_list, typical_value_list


def compute_cluster_curve(
        df: pd.DataFrame, 
        fields: list,
        cluster_nums: list,
        verbose: bool = True
        ) -> tuple[list, list, list, list]:
    """
    Returns a vector of cluster numbers and a vector of inertias,
    so that the inertia vs cluster num plot can be made. It also
    returns a list of lists: this is the list of cluster labels, 
    for each cluster num. It also returns a list of dictionaries:
    one dictionary for each cluster number. Each dictionary is nested.
    The top level keys are the cluster label; their associated values
    are then another dictionary whose keys are all the column names
    and whose values are the typical or mode values of each column.
    """
    
    # Subset the DataFrame with the selected columns
    subset_df = df[fields]

    # Separate numerical and categorical columns
    num_cols = subset_df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = subset_df.select_dtypes(include=['object']).columns

    # Standardize numerical columns
    scaler = StandardScaler()
    subset_df[num_cols] = scaler.fit_transform(subset_df[num_cols])

    # Convert categorical columns to string type
    subset_df[cat_cols] = subset_df[cat_cols].astype(str)

    # Lists to hold the scores and other data
    inertias = []
    cluster_label_list = []
    typical_value_list = []

    # Iterate over the cluster values
    for i in cluster_nums:
        if verbose:
            print(f"Clustering with {i} clusters.")

        # Create k-prototypes model
        kproto = KPrototypes(n_clusters=i, init='Cao', verbose=0)

        # Fit the model
        clusters = kproto.fit_predict(subset_df.values, categorical=list(range(len(num_cols), len(subset_df.columns))))

        # Add cluster labels to the data frame
        df["Cluster"] = clusters

        # Store the cluster labels
        df = df.sort_index()
        cluster_label_list.append(df["Cluster"].values)

        # Extract typical values from each cluster
        # and store in a dictionary
        typical_values = {}
        for cluster_label in df["Cluster"].unique():

            # Create a dict for this cluster
            typical_values[cluster_label] = {}

            # Find the typical value for each field
            cluster_data = df[df["Cluster"] == cluster_label]
            for field in fields:
                typical_values[cluster_label][field] = cluster_data[field].mode().iloc[0]

        # Record scores
        inertias.append(kproto.cost_)

        # Record typical value dict
        typical_value_list.append(typical_values)

    return cluster_nums, inertias, cluster_label_list, typical_value_list



def compute_cluster_curve_with_impute(
        df: pd.DataFrame,
        partial_df: pd.DataFrame,
        fields: list,
        cluster_nums: list,
        verbose: bool = True
        ) -> tuple[list, list, list, list]:
    
    # Subset the DataFrame with the selected columns
    subset_df = df[fields]

    # Separate numerical and categorical columns
    num_cols = subset_df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = subset_df.select_dtypes(include=['object']).columns

    # Standardize numerical columns
    scaler = StandardScaler()
    subset_df[num_cols] = scaler.fit_transform(subset_df[num_cols])

    # Convert categorical columns to string type
    subset_df[cat_cols] = subset_df[cat_cols].astype(str)

    # Lists to hold the scores and other data
    inertias = []
    cluster_label_list = []
    typical_value_list = []

    # Iterate over the cluster values
    for i in cluster_nums:
        if verbose:
            print(f"Clustering with {i} clusters.")

        # Create k-prototypes model
        kproto = KPrototypes(n_clusters=i, init='Cao', verbose=0)

        # Fit the model
        clusters = kproto.fit_predict(subset_df.values, categorical=list(range(len(num_cols), len(subset_df.columns))))

        # Add cluster labels to the data frame
        df["Cluster"] = clusters

        # Extract typical values from each cluster
        # and store in a dictionary
        typical_values = {}
        for cluster_label in df["Cluster"].unique():

            # Create a dict for this cluster
            typical_values[cluster_label] = {}

            # Find the typical value for each field
            cluster_data = df[df["Cluster"] == cluster_label]
            for field in fields:
                typical_values[cluster_label][field] = cluster_data[field].mode().iloc[0]

        # Now lets do imputation
        closest_clusters = []
        imputation_fields = ["height", "age", "nofloors"]
        age_mapping = {"PRE-1914": 1, "1918-1939": 2, "1945-1980": 3, "POST-1980": 4}
        # For each row in the partial_df (the df to be imputed)
        for _, row in partial_df.iterrows():

            # Now go through each of the clusters
            # and see which is the closest
            closest_cluster = None
            shortest_distance = np.inf
            dists = []
            for cluster_label in df["Cluster"].unique():

                # Get the typical value from this cluster
                # we find the typical value of each field in imputation_fields,
                # by looking them up in typical_values[cluster_label].
                for imp_field in imputation_fields:

                    # Get the value of the row we are trying to assign
                    rows_value = row[imp_field]
                    if imp_field == "age":
                        rows_value = age_mapping[rows_value]

                    # Now get this cluster's typical value for imp field
                    typ = typical_values[cluster_label][imp_field]
                    if imp_field == "age":
                        typ = age_mapping[typ]

                    # Now get this dimension's distance
                    dist = np.abs(typ - rows_value)
                    dists.append(dist**2)

                # Now take the euclidean distance over all dimensions
                total_dist = np.sqrt(sum(dists))
                if total_dist < shortest_distance:
                    shortest_distance = total_dist
                    closest_cluster = cluster_label

            # Store closest cluster
            closest_clusters.append(closest_cluster)

        # Add imputed cluster labels back in
        partial_df["Cluster"] = closest_clusters

        # Join with the rest of the df
        temp_df = pd.concat([df, partial_df])

        # Store the cluster labels
        temp_df = temp_df.sort_index()
        cluster_label_list.append(temp_df["Cluster"].values)

        # Record scores
        inertias.append(kproto.cost_)

        # Record typical value dict
        typical_value_list.append(typical_values)

    return cluster_nums, inertias, cluster_label_list, typical_value_list


def cluster_and_impute(
        df: pd.DataFrame,
        partial_df: pd.DataFrame,
        fields: list[str],
        cluster_num: int
    ):

    # Subset the DataFrame with the selected columns
    subset_df = df[fields]

    # Separate numerical and categorical columns
    num_cols = subset_df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = subset_df.select_dtypes(include=['object']).columns

    # Standardize numerical columns
    scaler = StandardScaler()
    subset_df[num_cols] = scaler.fit_transform(subset_df[num_cols])

    # Convert categorical columns to string type
    subset_df[cat_cols] = subset_df[cat_cols].astype(str)

    # Create k-prototypes model
    kproto = KPrototypes(n_clusters=cluster_num, init='Cao', verbose=0)

    # Fit the model
    clusters = kproto.fit_predict(subset_df.values, categorical=list(range(len(num_cols), len(subset_df.columns))))

    # Add cluster labels to the data frame
    df["Cluster"] = clusters

    # Extract typical values from each cluster
    # and store in a dictionary
    typical_values = {}
    for cluster_label in df["Cluster"].unique():

        # Create a dict for this cluster
        typical_values[cluster_label] = {}

        # Find the typical value for each field
        cluster_data = df[df["Cluster"] == cluster_label]
        for field in fields:
            typical_values[cluster_label][field] = cluster_data[field].mode().iloc[0]

    # Now lets do imputation
    closest_clusters = []
    imputation_fields = ["height", "age", "nofloors"]
    age_mapping = {"PRE-1914": 1, "1918-1939": 2, "1945-1980": 3, "POST-1980": 4}
    # For each row in the partial_df (the df to be imputed)
    for _, row in partial_df.iterrows():

        # Now go through each of the clusters
        # and see which is the closest
        closest_cluster = None
        shortest_distance = np.inf
        dists = []
        for cluster_label in df["Cluster"].unique():

            # Get the typical value from this cluster
            # we find the typical value of each field in imputation_fields,
            # by looking them up in typical_values[cluster_label].
            for imp_field in imputation_fields:

                # Get the value of the row we are trying to assign
                rows_value = row[imp_field]
                if imp_field == "age":
                    rows_value = age_mapping[rows_value]

                # Now get this cluster's typical value for imp field
                typ = typical_values[cluster_label][imp_field]
                if imp_field == "age":
                    typ = age_mapping[typ]

                # Now get this dimension's distance
                dist = np.abs(typ - rows_value)
                dists.append(dist**2)

            # Now take the euclidean distance over all dimensions
            total_dist = np.sqrt(sum(dists))
            if total_dist < shortest_distance:
                shortest_distance = total_dist
                closest_cluster = cluster_label

        # Store closest cluster
        closest_clusters.append(closest_cluster)

    # Add imputed cluster labels back in
    partial_df["Cluster"] = closest_clusters

    # Join with the rest of the df
    temp_df = pd.concat([df, partial_df])

    # Store the cluster labels
    temp_df = temp_df.sort_index()

    return temp_df, typical_values


def finite_difference_derivative(arr, step_size=1):
    """
    Compute the derivative of a 1D numpy array using finite differences.
    """
    # Using central difference for interior points
    arr = np.array(arr)
    derivative = (arr[2:] - arr[:-2])/(2*step_size)

    # Using forward/backward difference for boundary points
    derivative = np.concatenate(([arr[1] - arr[0]], derivative, [arr[-1] - arr[-2]])) / step_size

    return derivative


def compare_cluster_curves(
        curve: list[list],
        truth_curve: list[list]
        ) -> tuple[list]:
    """
    Function to compare a clustering curve to the ground
    truth clustering curve. It take as input two lists of lists:
    each list contains a list of clustering labels for each clustering
    number. This function returns a list of clustering scores.
    """
    ari_vec, nmis_vec = [], []
    for i in range(len(curve)):

        # These clustering objects are both a list 
        # of cluster labels
        clustering = curve[i]
        true_clustering = truth_curve[i]

        # Compare the two clusterings
        ari = adjusted_rand_score(true_clustering, clustering)
        nmis = normalized_mutual_info_score(true_clustering, clustering)

        # Record these
        ari_vec.append(ari)
        nmis_vec.append(nmis)

    return ari_vec, nmis_vec

