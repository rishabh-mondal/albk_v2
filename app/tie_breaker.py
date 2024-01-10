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
load_dir = st.text_input("Load Directory (image files)")
save_dir = st.text_input("Save Directory (for moderator labels)")
label_dir1 = st.text_input("Label Directory 1")
annotator1 = st.text_input("Annotator 1 Name")
label_dir2 = st.text_input("Label Directory 2")
annotator2 = st.text_input("Annotator 2 Name")
moderator_name = st.text_input("Moderator name")

assert os.path.exists(load_dir), "Load path does not exist"
assert os.path.exists(label_dir1), "Label path 1 does not exist"
assert os.path.exists(label_dir2), "Label path 2 does not exist"

if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

# Load all and labeled files
if "all_files" not in get_state():
    all_files = set(sorted(glob(os.path.join(load_dir, "*.zarr"))))
    all_files = set(map(get_filename, all_files))
    set_state("all_files", all_files)
else:
    all_files = get_var("all_files")

if "labeled_files1" not in get_state():
    labeled_files1 = set(glob(os.path.join(label_dir1, "*.nc")))
    labeled_files1 = set(map(get_filename, labeled_files1))
    set_state("labeled_files1", labeled_files1)
else:
    labeled_files1 = get_var("labeled_files1")

if "labeled_files2" not in get_state():
    labeled_files2 = set(glob(os.path.join(label_dir2, "*.nc")))
    labeled_files2 = set(map(get_filename, labeled_files2))
    set_state("labeled_files2", labeled_files2)
else:
    labeled_files2 = get_var("labeled_files2")

common_files = set(labeled_files1).intersection(set(labeled_files2))

if "labeled_files" not in get_state():
    labeled_files = set(glob(os.path.join(save_dir, "*.nc")))
    labeled_files = set(map(get_filename, labeled_files))
    set_state("labeled_files", labeled_files)
else:
    labeled_files = get_var("labeled_files")

# assert len(labeled_files1) == len(labeled_files2), "Number of files in label directories do not match"
# assert sorted(labeled_files1) == sorted(
#     labeled_files2
# ), f"Files in label directories do not match. Unmatched files = {set(labeled_files1) ^ set(labeled_files2)}"

# Set current file or load it
if "current_file" not in get_state():
    # Find files with different labels
    diff_files = []
    for file in common_files:
        ds1 = xr.open_dataset(os.path.join(label_dir1, file + ".nc"))
        ds2 = xr.open_dataset(os.path.join(label_dir2, file + ".nc"))
        if not np.all(ds1["label"] == ds2["label"]):
            diff_files.append(file)

    remaining_files = set(diff_files) - set(labeled_files)

    set_state("diff_files", diff_files)

    if len(remaining_files) == 0:
        st.write("No files to label")
        raise st.stop()
    set_state("current_file", remaining_files.pop())
else:
    diff_files = get_var("diff_files")
    remaining_files = set(diff_files) - set(labeled_files)
    current_file = get_var("current_file")

# Layout
st.write(f"Total files = {len(all_files)}")
st.write(f"Total labeled files by annotator 1 = {len(labeled_files1)}")
st.write(f"Total labeled files by annotator 2 = {len(labeled_files2)}")
st.write(f"Total common files = {len(common_files)}")
st.write(f"Total files with different labels = {len(diff_files)}")
st.write(f"Total labeled files by moderator = {len(labeled_files)}")
st.write(f"Total remaining files with different labels = {len(remaining_files)}")
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


if "label_box1" not in get_state():
    labels1 = xr.open_dataset(os.path.join(label_dir1, current_file + ".nc"))["label"].values
    set_state("label_box1", labels1)

if "label_box2" not in get_state():
    labels2 = xr.open_dataset(os.path.join(label_dir2, current_file + ".nc"))["label"].values
    set_state("label_box2", labels2)


# Show label box 1 with buttons
col2.markdown(f"### {annotator1}")
for i in range(5):
    cont = col2.container()
    cols = cont.columns(5)
    for j, col in enumerate(cols):
        col.button(get_var("label_box1")[i, j], key=f"label1_{i}_{j}", disabled=True)

# Show label box 2 with buttons
col2.markdown(f"### {annotator2}")
for i in range(5):
    cont = col2.container()
    cols = cont.columns(5)
    for j, col in enumerate(cols):
        col.button(get_var("label_box2")[i, j], key=f"label2_{i}_{j}", disabled=True)

# Show label box with buttons
col2.markdown(f"Click on the boxes to label the images as O, Z, F")

if "label_box" not in get_state():
    set_state("label_box", labels1.copy())

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
        attrs={"annotator1": annotator1, "annotator2": annotator2, "moderator": moderator_name},
    )
    full_save_path = os.path.join(save_dir, current_file + ".nc")
    save_ds.to_netcdf(full_save_path)

    # add to labeled files
    get_var("labeled_files").add(current_file)

    # Set next file
    remaining_files = set(get_var("diff_files")) - set(get_var("labeled_files"))
    if len(remaining_files) == 0:
        st.write("No files to label")
        raise st.stop()
    st.session_state["current_file"] = remaining_files.pop()

    # Reset label boxes
    get_state().pop("label_box")
    get_state().pop("label_box1")
    get_state().pop("label_box2")

    # Reset fig
    get_state().pop("fig")


save_and_next = col2.button("Save & Next", key="Save_And_Next", on_click=on_click_label)