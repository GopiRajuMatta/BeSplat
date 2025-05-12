import os
import h5py
import torch
import hdf5plugin
import numpy as np
from tqdm import tqdm
from pathlib import Path
from utils import img_utils

def load_timestamps(basedir, args):

    # BeNeRF_Synthetic 
    if args.dataset in ["BeNeRF_Blender", "BeNeRF_Unreal"]:
        time_ts_path = os.path.join(basedir, "poses_ts.txt")
        times_ts = np.loadtxt(time_ts_path)
        times_start = times_ts[:-1]
        times_end = times_ts[1:]
    # TUM-VIE
    elif args.dataset == "TUM_VIE":
        timestamps_path = os.path.join(basedir, "image_timestamps.txt")
        exposures_path = os.path.join(basedir, "image_exposures.txt")
        timestamps = np.loadtxt(timestamps_path)
        exposures = np.loadtxt(exposures_path)
        times_start = timestamps[:] - 0.5 * exposures[:]
        times_end = timestamps[:] + 0.5 * exposures[:]
    # E2NeRF_Real
    elif args.dataset == "E2NeRF_Real":
        time_start_path = os.path.join(basedir, "exposure_start_ts.txt")
        time_end_path = os.path.join(basedir, "exposure_end_ts.txt")
        times_start = np.loadtxt(time_start_path)
        times_end = np.loadtxt(time_end_path)
    # E2NeRF_Synthetic
    elif args.dataset == "E2NeRF_Synthetic":
        eventdir = os.path.join(basedir, "events")
        eventdir_idx = eventdir + "/r_{}".format(args.index * 2)
        events_txt = np.loadtxt(os.path.join(eventdir_idx, "v2e-dvs-events.txt"))
        st, _, _ ,_ = events_txt[0]
        ed, _, _ ,_ = events_txt[len(events_txt) - 1]
        times_start = int(st * 1e19)
        times_end = int(ed * 1e19)
    else:
        print("[ERROR] Cannot load timestamps for images")
        assert False

    if args.dataset == "E2NeRF_Synthetic":
        # record exposure time for rgb camera
        img_ts_start = times_start
        img_ts_end = times_end
        # usually,select more events will be better
        evt_ts_start = times_start - args.event_shift_start * 1e3 
        evt_ts_end = times_end + args.event_shift_end * 1e3
    else:
        # record exposure time for rgb camera    
        img_ts_start = times_start[args.index]
        img_ts_end = times_end[args.index]
        # usually,select more events will be better
        evt_ts_start = times_start[args.index] - args.event_shift_start * 1e3 
        evt_ts_end = times_end[args.index] + args.event_shift_end * 1e3

    return img_ts_start, img_ts_end, evt_ts_start, evt_ts_end

def load_data(
    datadir, args):
    datadir = os.path.expanduser(datadir)
    datasource = args.dataset
    # load start and end timestamps of exposure time
    print("[INFO] Loading timestamps...")
    img_ts_start, img_ts_end, evt_ts_start, evt_ts_end = load_timestamps(datadir, args)
    print("[INFO] Load timestamps successfully!!")
    print(f"Image Timestamp Start: {img_ts_start}")
    print(f"Image Timestamp End: {img_ts_end}")
    print(f"Event Timestamp Start: {evt_ts_start}")
    print(f"Event Timestamp End: {evt_ts_end}")
    # load events
    print("[INFO] Loading events...")
    eventdir = os.path.join(datadir, "events")
    # BeNeRF Synthetic
    if datasource in ["BeNeRF_Blender", "BeNeRF_Unreal"]:
        events = np.load(os.path.join(eventdir, "events.npy"))
        events = np.array(
            [ event for event in events if evt_ts_start <= event[2] <= evt_ts_end]
        )
    # E2NeRF Real
    elif datasource == "E2NeRF_Real":
        events_tensor = torch.load(os.path.join(eventdir, "events.pt"))
        events_numpy = events_tensor.numpy()
        #print(events_numpy)
        events =  np.array(
            [event for event in tqdm(events_numpy) if evt_ts_start <= event[2] <= evt_ts_end]
        )
        print("events=",events)
    # E2NeRF_Synthetic
    elif datasource == "E2NeRF_Synthetic":
        eventdir_idx = eventdir + "/r_{}".format(args.index * 2)
        events_txt = np.loadtxt(os.path.join(eventdir_idx, "v2e-dvs-events.txt"))
        events_list = []
        for row in tqdm(events_txt):
            t, x, y, p = row
            t = t * 1e19
            p = 2 * p - 1
            events_list.append(np.array([x, y, t, p], dtype = np.int64))
        events = np.array(events_list)
    # TUM-VIE
    elif datasource == "TUM_VIE":
        # import h5 file
        h5file = h5py.File(os.path.join(eventdir, "events.h5"))
        # h5group contains h5dataset: [x y t p]
        h5group = h5file["events"]

        # select events corresponding to idx
        h5dataset_ts = h5group["t"]

        # iteratively import timestamps of event data in chunks
        selected_indices = np.array([])
        chunk_size = 500000
        for chunk_idx in tqdm(range(0, len(h5dataset_ts), chunk_size)):
            chunk_indices = np.where(
                (h5dataset_ts[chunk_idx : chunk_idx + chunk_size] >= evt_ts_start) 
                & (h5dataset_ts[chunk_idx : chunk_idx + chunk_size] <= evt_ts_end)
            )
            chunk_indices = chunk_indices[0]
            chunk_indices[:] = chunk_indices[:] + chunk_idx
            selected_indices = np.concatenate((selected_indices, chunk_indices)).astype(np.uint64)
        selected_indices_start = np.array(selected_indices[0], dtype = np.uint64)
        selected_indices_end = np.array(selected_indices[len(selected_indices) - 1] + 1, dtype = np.uint64)
        # creat events array
        events = np.zeros(len(selected_indices))
        h5group_order = ["x", "y", "t", "p"]
        for i in tqdm(range(len(h5group_order))):
            h5dataset_name = h5group_order[i]
            h5dataset = h5group[h5dataset_name][
                selected_indices_start : selected_indices_end
            ]
            events = np.vstack((events, h5dataset))
        events = np.delete(events, 0, axis = 0)
        events = np.transpose(events)

    # sorted according to time
    events = events[events[:, 2].argsort()]
    # create dictionary
    events = {
        "x": events[:, 0].astype(int),
        "y": events[:, 1].astype(int),
        # norm ts(0~1)
        "ts": (events[:, 2] - evt_ts_start) / (evt_ts_end - evt_ts_start),
        "pol": events[:, 3],
    }
    print("[INFO] Load events successfully!!")
    for key, value in events.items():
        print(f"{key}: {value[:10]}")

    return events



