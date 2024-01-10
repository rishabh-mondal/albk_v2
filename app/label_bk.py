import os
import numpy as np
from glob import glob
import streamlit as st
import xarray as xr

import matplotlib.pyplot as plt


# Functions
def get_state():
    return st.session_state


def get_var(key):
    return st.session_state[key]


def set_state(key, value):
    st.session_state[key] = value


def set_state_if_not_none(key, value):
    if key not in get_state():
        set_state(key, value)


def get_filename(path):
    return os.path.splitext(os.path.basename(path))[0]


# Page config
st.set_page_config(layout="wide")

# Paths
load_dir = st.text_input("Load Directory")
save_dir = st.text_input("Save Directory")
annotator_name = st.text_input("Annotator name")
assert os.path.exists(load_dir), "Load path does not exist"
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

# Load all and labeled files
if "all_files" not in get_state():
    all_files = set(glob(os.path.join(load_dir, "*.zarr")))
    all_files = set(map(get_filename, all_files))
    set_state("all_files", all_files)
else:
    all_files = get_var("all_files")

if "labeled_files" not in get_state():
    labeled_files = set(glob(os.path.join(save_dir, "*.nc")))
    labeled_files = set(map(get_filename, labeled_files))
    set_state("labeled_files", labeled_files)
else:
    labeled_files = get_var("labeled_files")

# Set current file or load it
if "current_file" not in get_state():
    remaining_files = sorted(all_files - labeled_files)
    # st.write("First few files", remaining_files[:5])
    set_state("current_file", remaining_files.pop())

current_file = get_var("current_file")

# Layout
st.write(f"Found total {len(all_files)} files")
st.write(f"Found {len(labeled_files)} labeled files")
st.write(f"## Labeling {get_var('current_file')}")

col1, col2 = st.columns([0.85, 0.15])

# Load data and create fig
if "fig" not in get_state():
    full_path = os.path.join(load_dir, current_file + ".zarr")
    ds = xr.open_zarr(full_path, consolidated=False)
    data = ds["data"].transpose("lat_lag", "lon_lag", "channel", "row", "col")
    blocks = []
    for i in range(5):
        blocks.append([])
        for j in range(5):
            blocks[i].append(data[i, j, :, :, :])

    full_img = np.block(blocks)
    full_img = np.swapaxes(full_img, 0, 2)
    full_img = np.swapaxes(full_img, 0, 1)

    fig, ax = plt.subplots()
    ax.imshow(full_img)
    ax.vlines(range(224, 224 * 5, 224), 0, 224 * 5, color="r", lw=0.5, ls="--", alpha=0.5)
    ax.hlines(range(224, 224 * 5, 224), 0, 224 * 5, color="r", lw=0.5, ls="--", alpha=0.5)
    ax.set_axis_off()
    col1.pyplot(fig)
    set_state("fig", fig)
else:
    fig = get_var("fig")
    col1.pyplot(fig)


def on_click_label(i, j):
    if get_var("label_box")[i, j] == "O":
        get_var("label_box")[i, j] = "Z"
    elif get_var("label_box")[i, j] == "Z":
        get_var("label_box")[i, j] = "F"
    elif get_var("label_box")[i, j] == "F":
        get_var("label_box")[i, j] = "O"


col2.markdown(
    f""""### Click on the boxes to label the images as O, Z, F
    
* Z: ZigZag
* F: FCK
* O: Other 
              """
)
if "label_box" not in get_state():
    set_state("label_box", np.empty((5, 5), dtype=str))
    get_var("label_box")[:, :] = "O"


# Show label box with buttons
for i in range(5):
    cont = col2.container()
    cols = cont.columns(5)
    for j, col in enumerate(cols):
        col.button(get_var("label_box")[i, j], key=f"label_{i}_{j}", on_click=on_click_label, args=(i, j))


def on_click_label():
    save_ds = xr.Dataset(
        {
            "label": (("lat_lag", "lon_lag"), get_var("label_box")),
        },
        coords={"lat_lag": range(-2, 3), "lon_lag": range(-2, 3)},
        attrs={"annotator": annotator_name},
    )
    full_save_path = os.path.join(save_dir, current_file + ".nc")
    save_ds.to_netcdf(full_save_path)

    # add to labeled files
    get_var("labeled_files").add(current_file)

    # Set next file
    remaining_files = get_var("all_files") - get_var("labeled_files")
    st.session_state["current_file"] = remaining_files.pop()

    # Reset label box
    get_state().pop("label_box")

    # Reset fig
    get_state().pop("fig")


save_and_next = col2.button("Save & Next", key="Save_And_Next", on_click=on_click_label)