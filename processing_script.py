"""
This script takes the data, having already been imputed, and performs 
clustering to assign constructions.
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from datamethods.domestic import *
from shapely.geometry import Polygon
from shapely.wkt import loads


# Do clustering on 5 nodes with flag imputation
df = pd.read_csv("data/domtest.csv", index_col=0)

# Identify rows with at least one pd.NA in the epc_fields columns
na_rows = df[epc_fields].apply(lambda row: any(pd.isna(value) for value in row), axis=1)

# Split the DataFrame
df_without_epc = df[na_rows]
df_with_epc = df[~na_rows]
df_with_epc = df_with_epc.sort_index()

# The columns we'd like to use for clustering
cols_to_ignore = [
    "d_epc_mainht_fuel",
    "d_epc_mainht_plant",
    "d_epc_mainht_room",
    "d_epc_envelope_flr_insulation",
    "d_epc_envelope_rf_insulation"
]
useful_epc_vals = [item for item in epc_fields if item not in cols_to_ignore]
cluster_col_list = [
    "nofloors", "height", "age", "FLOOR_1: use", *useful_epc_vals
    ]

# Handle missing epc fields with a flag
df_without_epc[useful_epc_vals] = "missing"

# Zip this back together and try clustering
full_dom_df = pd.concat([df_with_epc, df_without_epc])
full_dom_df = full_dom_df.sort_index()
_, _, cluster_label_list, _ = compute_cluster_curve(
        full_dom_df,
        cluster_col_list,
        [5]
    )

# Add cluster labels to the data frame
full_dom_df["Cluster"] = cluster_label_list[0]

# Also pull in the the non dom data and cluster it 
# using kprototypes with n = 3.
nondom_df = pd.read_csv("data/nondomtest.csv", index_col=0)
nondom_df['area'] = nondom_df['polygon'].apply(lambda polygon: loads(polygon).area)
_, _, cluster_label_list, _ = compute_cluster_curve(
        nondom_df,
        ["area", "FLOOR_1: use"],
        [2]
    )
nondom_df["Cluster"] = cluster_label_list[0]
nondom_df = nondom_df.drop(columns=["area"])


# Before stitching dom and non-dom back together, lets prefix their
# cluster labels with dom and nondom
def nondom_prefix(value):
    return f'nondom{int(value)}'
nondom_df['Cluster'] = nondom_df['Cluster'].apply(nondom_prefix)
def dom_prefix(value):
    return f'dom{int(value)}'
full_dom_df['Cluster'] = full_dom_df['Cluster'].apply(dom_prefix)

# Zip the tow clustered data frames together and output
# together with a cluster report data frame
clustered_df = pd.concat([full_dom_df, nondom_df], axis=0).sort_index()
clustered_df.to_csv("data/clustered.csv")

cols_of_interest = ["FLOOR_1: use", "age", *epc_fields, "height", "nofloors", "wwr"]
cluster_guide = clustering_summary(clustered_df, cols_of_interest)
cluster_guide.to_csv("data/cluster_guide.csv")





