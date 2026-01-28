import os
import re
import json
import copy
from pathlib import Path
from typing import Dict, List, Any

import mne
import h5py
import numpy as np

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
# from fastapi.staticfiles import StaticFiles

import asyncio
import aiofiles

app = FastAPI()

# allow cross domain requests
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# Subjects' PVF data loaded to memory
MAX_NUM_SUBJECTS_ALLOWED_LOAD = 2           # this number is restricted by max RAM size allowed by server process.
subjects_loaded_pvf_data      = dict()
subject_list_index_dict       = []          


# PVF streamlines
STREAMLINES_DOWNSAMPLE_FACTOR   : int = 4   # keep this as 4 - important - 20260117

# PVF data directories
PVF_SUBJECTS_DIR = f"{Path(__file__).parent}/pvf_data/pvf_subjects"
FS_SUBJECTS_DIR  = f"{Path(__file__).parent}/pvf_data/fs_subjects"

# ------ list subject, sessions, pvf files and source estimate files ---------------------------
@app.get("/api/list-subjects")
async def list_subjects():
    """
    List all subject IDs/names under the PVF subjects directory.
    """
    subject_list = []
    if os.path.exists(PVF_SUBJECTS_DIR) == False:
        return subject_list
    else:
        subject_list = [subject_name for subject_name in os.listdir(PVF_SUBJECTS_DIR) if os.path.isdir(os.path.join(PVF_SUBJECTS_DIR, subject_name))]
        return subject_list

@app.get("/api/list-sessions")
async def list_subjects(subject: str = Query(None)):
    """
    List all session folders under PVF subject directory.
    """
    session_list = []
    subject_dir = f"{PVF_SUBJECTS_DIR}/{subject}"
    if os.path.exists(subject_dir) == False:
        return session_list
    else:
        session_list = [session_id for session_id in os.listdir(subject_dir) if os.path.isdir(os.path.join(subject_dir, session_id)) and session_id.startswith("ses-")]
        return session_list

@app.get("/api/list-subjects-files")
async def list_subjects_pvf_files(subject: str = Query(None), session: str = Query(None)):
    """
    list all _metadata.json files under a given subject and session folder.
    """
    session_pvf_dir = f"{PVF_SUBJECTS_DIR}/{subject}/{session}/PVF"
    fname_list      = []
    if os.path.exists(session_pvf_dir) == False:
        return fname_list
    
    fname_list = [fname for fname in os.listdir(session_pvf_dir) if os.path.isfile(os.path.join(session_pvf_dir, fname)) and fname.endswith("_metadata.json")]
    
    return fname_list

@app.get("/api/list-subject-source-estimate")
async def list_subject_source_estimate_files(subject: str = Query(None), session: str = Query(None)):
    """
    list all _metadata.json files for a given subject.
    """
    session_dir = f"{PVF_SUBJECTS_DIR}/{subject}/{session}"
    fname_list  = []
    if os.path.exists(session_dir) == False:
        return fname_list
    
    fname_list = [fname for fname in os.listdir(session_dir) if os.path.isfile(os.path.join(session_dir, fname)) and fname.endswith("-stc.h5")]
    
    print(fname_list)
    return fname_list


# ------ load data ----------------------------
@app.get("/api/load-subjects-files")
async def load_subjects_files(
                            subject         : str             = Query(None),
                            session         : str             = Query(None),
                            file            : str             = Query(None),):
    load_subject_json_files(subject_name=subject, session=session, file_name=file)
    data = resp_pvf_json(subject_name=subject, session=session, file_name=file, timepoint=0)
    task = asyncio.create_task(load_streamlines_all_time_windows(subject_name=subject, session=session, file_name=file))
    return data

@app.get("/api/load-modes")
async def load_subject_file_modes(                            
                            subject         : str             = Query(None),
                            session         : str             = Query(None),
                            file            : str             = Query(None),
                            mode            : int             = Query(None),):
    data = process_modes(subject_name=subject, session=session, file_name=file, mode_id=int(mode))
    return data

@app.get("/api/load-source-estimate")
async def load_subject_source_estimate_file(subject         : str             = Query(None),
                                            session         : str             = Query(None),
                                            file            : str             = Query(None),):
    global subjects_loaded_pvf_data
    load_subject_source_estimate(subject_name=subject, session=session, file_name=file, timepoint=0)
    sensor_signals = subjects_loaded_pvf_data[subject][session]["source_estimate"][file]['sensor_signals'][:-2, :].T # in the form of time * channels
    return {'sensor_signals': sensor_signals.tolist()}

@app.get("/api/update-source-estimate")
async def update_subject_source_estimate(subject         : str             = Query(None),
                                              session         : str             = Query(None),
                                              file            : str             = Query(None),
                                              timepoint       : str             = Query(None)):
    data = load_subject_source_estimate(subject_name=subject, session=session, file_name=file, timepoint=timepoint)    
    return data

@app.get("/api/update-PVF-streamlines")
async def update_pvf_streamlines(subject: str = Query(None), 
                                 session   : str = Query(None),
                                 file     : str  = Query(None),
                                 timepoint: str  = Query(None)):
    """更新特定时间点的流线数据"""
    data = await update_pvf_streamlines_data(subject_name=subject, session=session, file_name=file, timepoint=timepoint)
    return data

@app.get("/api/get-brain-surfaces")
async def get_brain_surfaces(subject: str = Query(None)):
    """Obtain Brain surfaces as vertices and faces"""
    if not subject:
        return {"error": "Subject ID is required"}
    
    lh_surf_path = f"{FS_SUBJECTS_DIR}/{subject}/surf/lh.pial"
    rh_surf_path = f"{FS_SUBJECTS_DIR}/{subject}/surf/rh.pial"
    lh_vertices, lh_faces  = mne.read_surface(lh_surf_path)
    rh_vertices, rh_faces  = mne.read_surface(rh_surf_path)

    print(f"Brain surfaces for subject: {subject} loaded.")
    
    return {"lh_surface": {'vertices': lh_vertices.tolist(), 'faces': lh_faces.tolist()},
            "rh_surface": {'vertices': rh_vertices.tolist(), 'faces': rh_faces.tolist()},
            "subject_id" : subject,     }


# ------ functions to prepare data for api responses --------------
def process_pvf_time_window(subject_name: str, session: str, file_name: str, pvf_time_window_id: int) -> Dict[str, Any]:
    '''
        process PVF vector positions and directions at a specific time window.
    '''
    global subjects_loaded_pvf_data

    file_pvf_data = subjects_loaded_pvf_data[subject_name][session]['meta_files'][file_name]
    print(f"Processing Vx, Vy, Vz at time point: {pvf_time_window_id} of {file_pvf_data['PVF_num_time_points']}")

    file_pvf_data   = subjects_loaded_pvf_data[subject_name][session]['meta_files'][file_name]
    mask_volume     = subjects_loaded_pvf_data[subject_name][session]['volume_mask']
    vol_src         = subjects_loaded_pvf_data[subject_name][session]['vol_src'] 
    volume_vert_ind = subjects_loaded_pvf_data[subject_name][session]['volume_vertex_index']
    vert_no         = volume_vert_ind[mask_volume]
    vx              = np.squeeze(file_pvf_data["Vx"][pvf_time_window_id, :, :, :])     # np.squeeze(pvf_vx[:, :, :, pvf_time_window_id])
    vy              = np.squeeze(file_pvf_data["Vy"][pvf_time_window_id, :, :, :])     # np.squeeze(pvf_vy[:, :, :, pvf_time_window_id])
    vz              = np.squeeze(file_pvf_data["Vz"][pvf_time_window_id, :, :, :])     # np.squeeze(pvf_vz[:, :, :, pvf_time_window_id])
    

    # obtain positions in mm from volume source space
    positions  = np.zeros((np.sum(mask_volume), 3))
    directions = np.zeros((np.sum(mask_volume), 3))

    # id_serial = 0
    # for idz in range(vx.shape[0]):
    #     for idy in range(vx.shape[1]):
    #         for idx in range(vx.shape[2]):
    #             if mask_volume[idz, idy, idx]:
    #                 vert_index               = volume_vert_ind[idz, idy, idx]
    #                 positions[id_serial, :]  = vol_src[0]['rr'][vert_index] * 1000
    #                 directions[id_serial,:]  = [vx[idz, idy, idx], vy[idz, idy, idx], vz[idz, idy, idx]]
    #                 id_serial               += 1

    # positions_2 = np.zeros((np.sum(mask_volume), 3))
    if vol_src is not None:
        vert_no   = volume_vert_ind[mask_volume]
        positions = vol_src[0]['rr'][vert_no] * 1000
        # positions_2 = vol_src[0]['rr'][vert_no] * 1000
    else:
        positions = np.zeros((np.sum(mask_volume), 3))
        # positions_2 = vol_src[0]['rr'][vert_no] * 1000

    # u, v, w = vx[mask_volume] * -1, vy[mask_volume] * -1, vz[mask_volume] * -1
    u, v, w = vx[mask_volume], vy[mask_volume], vz[mask_volume]
    directions = np.vstack([u.flatten().T, v.flatten().T, w.flatten().T,]).T # should not swap x and y
    # directions = np.vstack([v.flatten().T, u.flatten().T, w.flatten().T,]).T # swap x and y because numPy uses row-major order
    # directions_2 = np.vstack([v.flatten().T, u.flatten().T, w.flatten().T,]).T # swap x and y because numPy uses row-major order

    # normalise directions
    directions_norm_max = np.max(np.linalg.norm(directions, ord=2, axis=1))
    directions = directions / directions_norm_max

    return {"positions": positions, "directions": directions}

def process_streamlines_time_window(subject_name: str, session: str, file_name: str, pvf_time_window_id: int) -> List[Any]:
    '''
        process streamlines at a specific time window.
    '''
    global subjects_loaded_pvf_data, STREAMLINES_DOWNSAMPLE_FACTOR
    
    pvf_streamline_all_time_windows = subjects_loaded_pvf_data[subject_name][session]['meta_files'][file_name]['streamlines_time_windows']

    new_streamlines = []
    if len(pvf_streamline_all_time_windows) != 0:
        streamlines = pvf_streamline_all_time_windows[str(pvf_time_window_id)]
        new_streamlines = downsample_streamlines(streamlines, factor=STREAMLINES_DOWNSAMPLE_FACTOR)

    print(f"Processed {len(new_streamlines)} streamlines at time point: {pvf_time_window_id}")
    return new_streamlines

def process_modes(subject_name: str, session: str, file_name: str, mode_id: int) -> Dict[str, Any]:
    '''
        process vector positions and directions for a specific mode.
    '''
    global subjects_loaded_pvf_data
    print(f"Processing modes for {subject_name}'s {file_name} on mode {mode_id}")

    file_pvf_data   = subjects_loaded_pvf_data[subject_name][session]['meta_files'][file_name]
    mask_volume     = subjects_loaded_pvf_data[subject_name][session]['volume_mask']
    vol_src         = subjects_loaded_pvf_data[subject_name][session]['vol_src']
    volume_vert_ind = subjects_loaded_pvf_data[subject_name][session]['volume_vertex_index']
    temporal_modes  = np.asarray(file_pvf_data["mode_data"]["temporal_modes"]).T
    mode_vel_mat    = np.asarray(file_pvf_data["mode_data"]["mode_vel_mat"])
    vx              = np.squeeze(mode_vel_mat[mode_id, 0, :, :, :])   # np.squeeze(pvf_vx[:, :, :, pvf_time_window_id])
    vy              = np.squeeze(mode_vel_mat[mode_id, 1, :, :, :])   # np.squeeze(pvf_vy[:, :, :, pvf_time_window_id])
    vz              = np.squeeze(mode_vel_mat[mode_id, 2, :, :, :])   # np.squeeze(pvf_vz[:, :, :, pvf_time_window_id])
    vert_no         = volume_vert_ind[mask_volume]

    # obtain positions in mm from volume source space
    positions  = np.zeros((np.sum(mask_volume), 3))
    directions = np.zeros((np.sum(mask_volume), 3))

    if vol_src is not None:
        volume_vert_ind = np.asarray(volume_vert_ind)
        vert_no         = volume_vert_ind[mask_volume]
        positions = vol_src[0]['rr'][vert_no] * 1000
        
    else:
        positions = np.zeros((np.sum(mask_volume), 3))
            
    u, v, w = vx[mask_volume], vy[mask_volume], vz[mask_volume]
    directions = np.vstack([u.flatten().T, v.flatten().T, w.flatten().T,]).T # should not swap x and y

    # normalise directions
    directions_norm_max = np.max(np.linalg.norm(directions, ord=2, axis=1))
    directions = directions / directions_norm_max
    
    return {"positions": positions.tolist(), "directions": directions.tolist(), "temporal_modes": temporal_modes.tolist()}

def load_subject_json_files(subject_name: str, session: str, file_name: str) -> Dict[str, Any]:
    global subject_list_index_dict, subjects_loaded_pvf_data

    subject_list_index = len(subject_list_index_dict)
    
    if subject_name not in subjects_loaded_pvf_data and subject_list_index < MAX_NUM_SUBJECTS_ALLOWED_LOAD:
        subject_pvf_data                                = read_subject_pvf_json(subject_name=subject_name, session=session, file_name=file_name)
        subjects_loaded_pvf_data[subject_name]          = dict()
        subjects_loaded_pvf_data[subject_name][session] = subject_pvf_data
        subject_list_index_dict.append(subject_name)
    elif subject_name not in subjects_loaded_pvf_data and subject_list_index == MAX_NUM_SUBJECTS_ALLOWED_LOAD:
        remove_subject_name = subject_list_index_dict[0]
        subjects_loaded_pvf_data.pop(remove_subject_name)
        subject_list_index_dict.pop(0)
        subjects_loaded_pvf_data[subject_name]          = dict()
        subject_pvf_data                                = read_subject_pvf_json(subject_name=subject_name, session=session, file_name=file_name) 
        subjects_loaded_pvf_data[subject_name][session] = subject_pvf_data
        subject_list_index_dict.append(subject_name)
    elif subject_name in subjects_loaded_pvf_data:
        if session not in subjects_loaded_pvf_data[subject_name]:
            subject_pvf_data = read_subject_pvf_json(subject_name=subject_name, session=session, file_name=file_name)
            subjects_loaded_pvf_data[subject_name][session] = subject_pvf_data
        elif file_name not in subjects_loaded_pvf_data[subject_name][session]['meta_files']:
            subjects_loaded_pvf_data[subject_name][session]['meta_files'][file_name] = read_file_pvf_data(subject_name=subject_name, session=session, file_name=file_name)

def read_file_pvf_data(subject_name: str, session: str, file_name: str) -> Dict[str, Any]:

    metadata_path         = f"{PVF_SUBJECTS_DIR}/{subject_name}/{session}/PVF/{file_name}"
    vx_path               = metadata_path.replace("_metadata.json", "_Vx.json")
    vy_path               = metadata_path.replace("_metadata.json", "_Vy.json")
    vz_path               = metadata_path.replace("_metadata.json", "_Vz.json")
    condA_path            = metadata_path.replace("_metadata.json", "_condA.json")
    pattern_path          = metadata_path.replace("_metadata.json", "_pattern_detection.json")
    mode_path             = metadata_path.replace("_metadata.json", "_SVD_spacetime_modes.json")
    pvf_streamline_folder = str(metadata_path.replace("_metadata.json", "_streamlines"))

    file_pvf_data = dict()
    # file_pvf_data['metadata_fname'] = file_name
    file_pvf_data['metadata_path']  = metadata_path

    # ses_id = extract_ses_info(file_name, get_all=False)
    # if ses_id is not None and ses_id != "":
    #     whole_brain_source_space_fname = f"{FS_SUBJECTS_DIR}/{subject_name}/bem/whole_brain_{ses_id}_vol_src.fif"
    # else:
    #     whole_brain_source_space_fname = f"{FS_SUBJECTS_DIR}/{subject_name}/bem/whole_brain_vol_src.fif"
    # file_pvf_data['vol_src']        = mne.read_source_spaces(whole_brain_source_space_fname)

    try:
        with open(metadata_path, "r", encoding="utf8") as f:
            pvf_metadata                         = json.load(f)
            file_pvf_data["subject_ID"]          = subject_name
            file_pvf_data["times"]               = pvf_metadata['times']    
            file_pvf_data["PVF_num_time_points"] = len(pvf_metadata['times'])
            pvf_num_time_points                  = len(pvf_metadata['times'])

            if "sensor_signals" in pvf_metadata.keys():
                file_pvf_data["sensor_signals"] = np.asarray(pvf_metadata['sensor_signals'])
                print(f"Successfully loaded sensor signals from metadata: {metadata_path}")

            print(f"Successfully loaded: {metadata_path}")
            print(f"Number of Timepoints: {pvf_num_time_points}")
    except Exception as e:
        print(f"Failed with errors: {e}")
    
    # read Vx
    try:
        with open(vx_path, "r", encoding="utf8") as f:
            data_vx                        = json.load(f)
            file_pvf_data["Vx"]            = np.asarray(data_vx["Vx"])
            pvf_dimension                  = file_pvf_data["Vx"].shape[-1]
            file_pvf_data["PVF_dimension"] = pvf_dimension
            print(f"PVF dimensions: {pvf_dimension}")
            print(f"Successfully loaded Vx: {vx_path}")
    except Exception as e:
        print(f"Failed with errors: {e}")
    
    # read Vy
    try:
        with open(vy_path, "r", encoding="utf8") as f:
            data_vy             = json.load(f)
            file_pvf_data["Vy"] = np.asarray(data_vy["Vy"])
            print(f"Successfully loaded Vy: {vy_path}")
    except Exception as e:
        print(f"Failed with errors: {e}")
    
    # read Vz
    try:
        with open(vz_path, "r", encoding="utf8") as f:
            data_vz = json.load(f)
            file_pvf_data["Vz"]  = np.asarray(data_vz["Vz"])
            print(f"Successfully loaded Vz: {vz_path}")
    except Exception as e:
        print(f"Failed with errors: {e}")
    
    # read condition number 读取条件数数据
    try:
        with open(condA_path, "r", encoding="utf8") as f:
            file_pvf_data["condA"] = json.load(f)
            print(f"Successfully loaded condA: {condA_path}")
    except Exception as e:
        print(f"Failed with errors: {e}")
    
    # read pattern data
    try:
        with open(pattern_path, "r", encoding="utf8") as f:
            file_pvf_data["pattern_data"] = json.load(f)
            print(f"Successfully loaded patterns: {pattern_path}")
    except Exception as e:
        print(f"Failed with errors: {e}")
    
    # read mode decomposition
    try:
        with open(mode_path, "r", encoding="utf8") as f:
            file_pvf_data["mode_data"] = json.load(f)

            mode_singular_values = file_pvf_data['mode_data']['singular_values']
            squared_Sigma        = np.square(mode_singular_values)
            sum_of_squared_Sigma = np.sum(squared_Sigma)
            percentages_squared  = (squared_Sigma / sum_of_squared_Sigma) * 100
            file_pvf_data['mode_info'] = {}
            for mode in range(len(mode_singular_values)):
                file_pvf_data['mode_info'][mode] = {'singular_value': float(mode_singular_values[mode]),
                                                    'percentage': float(percentages_squared[mode])}
            print(f"Successfully loaded patterns: {mode_path}")
    except Exception as e:
        print(f"Failed with errors: {e}")
    
    # read streamline data 读取流线数据
    try:
        first_tw_path = os.path.join(pvf_streamline_folder, "pvf_streamlines_time_window_0_4.json")
        with open(first_tw_path, "r", encoding="utf8") as f:
            pvf_streamlines_time_windows              = json.load(f)
            file_pvf_data["streamlines_time_windows"] = copy.copy(pvf_streamlines_time_windows)
            print(f"Successfully loaded Streamlines: {pvf_streamline_folder}")
    except Exception as e:
        print(f"Failed with errors: {e}")
    
    return file_pvf_data

def read_subject_pvf_json(subject_name: str, session: str, file_name: str) -> Dict[str, Any]:
    global subjects_loaded_pvf_data
        
    subject_pvf_data                    = dict()
    subject_pvf_data['subject_id']      = subject_name
    subject_pvf_data['meta_files']      = dict()
    subject_pvf_data['source_estimate'] = dict()

    ses_id = extract_ses_info(file_name, get_all=False)
    if ses_id is not None and ses_id != "":
        whole_brain_source_space_fname = f"{FS_SUBJECTS_DIR}/{subject_name}/bem/whole_brain_{ses_id}_vol_src.fif"
    else:
        whole_brain_source_space_fname = f"{FS_SUBJECTS_DIR}/{subject_name}/bem/whole_brain_vol_src.fif"
    subject_pvf_data['vol_src']         = mne.read_source_spaces(whole_brain_source_space_fname)
    print(f"Successfully loaded volume source space: {whole_brain_source_space_fname}")
    
    metadata_path         = f"{PVF_SUBJECTS_DIR}/{subject_name}/{session}/PVF/{file_name}"
    try:
        with open(metadata_path, "r", encoding="utf8") as f:
            pvf_metadata                            = json.load(f)
            subject_pvf_data['volume_mask']         = np.asarray(pvf_metadata['volume_mask'])
            subject_pvf_data['volume_vertex_index'] = np.asarray(pvf_metadata['volume_vertex_index'])
            print(f"Successfully loaded: {metadata_path}")
    except Exception as e:
        print(f"Failed with errors: {e}")

    subject_pvf_data['meta_files'][file_name] = read_file_pvf_data(subject_name=subject_name, session=session, file_name=file_name)
    return subject_pvf_data

async def update_pvf_streamlines_data(subject_name: str, session: str, file_name: str, timepoint: str) -> Dict[str, Any]:
    """更新特定时间点的流线数据及奇点数据"""
    global subjects_loaded_pvf_data

    if subject_name in subjects_loaded_pvf_data:
        if session in subjects_loaded_pvf_data[subject_name]:
            if file_name in subjects_loaded_pvf_data[subject_name][session]['meta_files']:    
                try:
                    return resp_pvf_json(subject_name=subject_name, session=session, file_name=file_name, timepoint=timepoint)
                except ValueError:
                    return {"message": "Invalid timepoint format"}
            if file_name not in subjects_loaded_pvf_data[subject_name][session]['meta_files']:
                subjects_loaded_pvf_data[subject_name][session]['meta_files'][file_name] = read_file_pvf_data(subject_name=subject_name, session=session, file_name=file_name)
                try:
                    return resp_pvf_json(subject_name=subject_name, session=session, file_name=file_name, timepoint=timepoint)
                except ValueError:
                    return {"message": "Invalid timepoint format"}
        else:
            subject_pvf_data = read_subject_pvf_json(subject_name=subject_name, session=session, file_name=file_name)
            subjects_loaded_pvf_data[subject_name][session] = subject_pvf_data
    else:
        load_subject_json_files(subject_name=subject_name, session=session, file_name=file_name)
        try:
            return resp_pvf_json(subject_name=subject_name, session=session, file_name=file_name, timepoint=timepoint)
        except ValueError:
            return {"message": "Invalid timepoint format"}

# async functions to read all streamlines data
async def load_streamlines_all_time_windows(subject_name: str, session: str, file_name: str):
    ''' load all streamlines data from json files in background'''
    global subjects_loaded_pvf_data

    if len(subjects_loaded_pvf_data[subject_name][session]['meta_files'][file_name]['streamlines_time_windows']) < subjects_loaded_pvf_data[subject_name][session]['meta_files'][file_name]["PVF_num_time_points"]:
        metadata_path         = f"{PVF_SUBJECTS_DIR}/{subject_name}/{session}/PVF/{file_name}"
        pvf_streamline_folder = str(metadata_path.replace("_metadata.json", "_streamlines"))
        print(f"In background loading streamlines of all time windows from folder: {pvf_streamline_folder}")
        
        streamline_files = [
            f for f in os.listdir(pvf_streamline_folder)
            if os.path.isfile(os.path.join(pvf_streamline_folder, f)) and f.endswith(".json")
        ]
        
        for file in streamline_files:
            print(f"Loading streamlines from: {file}")
            file_path = os.path.join(pvf_streamline_folder, file)
            async with aiofiles.open(file_path, "r", encoding="utf8") as f:
                content = await f.read()  # 异步读取全部内容
                streamlines_time_windows = json.loads(content)  # 解析 JSON
                # streamlines_time_windows = json.load(f)
                for timepoint, streamlines in streamlines_time_windows.items():
                    subjects_loaded_pvf_data[subject_name][session]['meta_files'][file_name]['streamlines_time_windows'][timepoint] = streamlines
        
        print(f"Completed loading all streamlines from: {len(streamline_files)} json files.")

# downsample streamlines for faster rendering
def downsample_streamlines(streamlines: List[Any], factor: int = 2) -> List[Any]:
    """Downsample streamlines by a given factor"""
    downsampled_streamlines = []
    for sl in streamlines:
        downsampled_sl = sl[::factor]
        downsampled_streamlines.append(downsampled_sl)
    return downsampled_streamlines

def extract_ses_info(input_str, get_all=False):
    """
    From a string to extract ses-XXX format substring using regular expressions.

    Args:
        input_str (str): orignal string to extract from
        get_all (bool): getting all matched patterns or not. True to return all. False to only return the first one.
        
    Returns:
        list/str/None: matched strings, first matched string or all matched string, or None if no match found.
    """
    # Regular expression：
    # ses- : matching prefix "ses-"
    # \d+ ：matching digits (one or more)
    pattern = r'ses-\d+'
    
    # 查找所有匹配项
    matches = re.findall(pattern, input_str)
    
    if not matches:
        return None
    
    # 根据需求返回单个或所有匹配项
    if get_all:
        return matches
    else:
        return matches[0]

# ------------------------------------------------------
# load and prepare source estimate data
def load_subject_source_estimate(subject_name: str, session: str, file_name: str, timepoint: str):
    global subjects_loaded_pvf_data

    # ses_id = extract_ses_info(file_name, get_all=False)
    # if ses_id is not None and ses_id != "":
    #     whole_brain_source_space_fname = f"{FS_SUBJECTS_DIR}/{subject_name}/bem/whole_brain_{ses_id}_vol_src.fif"
    # else:
    #     whole_brain_source_space_fname = f"{FS_SUBJECTS_DIR}/{subject_name}/bem/whole_brain_vol_src.fif"

    if file_name not in subjects_loaded_pvf_data[subject_name][session]['source_estimate']:
        src_est_fname            = f"{PVF_SUBJECTS_DIR}/{subject_name}/{session}/{file_name}"
        print(f"Loading source estimate: {src_est_fname}")
        src_est = mne.read_source_estimate(fname=src_est_fname)# mne.read_source_estimate(fname=src_est_fname, subject=subject_name)
        subjects_loaded_pvf_data[subject_name][session]["source_estimate"][file_name]                   = dict()
        subjects_loaded_pvf_data[subject_name][session]["source_estimate"][file_name]['source_signals'] = src_est.data
        subjects_loaded_pvf_data[subject_name][session]["source_estimate"][file_name]['source_vert_no'] = src_est.vertices[0]
        # subjects_loaded_pvf_data[subject_name]["source_estimate"][file_name]['sensor_signals'] = sensor_signals




    if 'sensor_signals' not in subjects_loaded_pvf_data[subject_name][session]["source_estimate"][file_name].keys():
        sensor_signals_file_name = file_name.replace("-stc.h5", ".mat")
        sensor_signals_fname     = f"{PVF_SUBJECTS_DIR}/{subject_name}/{session}/{sensor_signals_file_name}"
        print(f"Loading sensor signals from {sensor_signals_fname}")
        sensor_signal_file       = h5py.File(sensor_signals_fname)
        if 'sensor_signals' in sensor_signal_file.keys():
            sensor_signals = np.asarray(sensor_signal_file['sensor_signals'])
        else:
            sensor_signals = np.asarray(sensor_signal_file['source_data'])

        subjects_loaded_pvf_data[subject_name][session]["source_estimate"][file_name]['sensor_signals'] = sensor_signals

    source_data_time = subjects_loaded_pvf_data[subject_name][session]["source_estimate"][file_name]['source_signals'][:, int(timepoint)]
    source_vert_no   = subjects_loaded_pvf_data[subject_name][session]["source_estimate"][file_name]['source_vert_no']
    source_positions = subjects_loaded_pvf_data[subject_name][session]['vol_src'][0]['rr'][source_vert_no] * 1000
    if np.max(source_data_time) < 1e-10:
        source_data_time = source_data_time * 1e16
    else:
        source_data_time = source_data_time * 1000
    return {'positions': source_positions.tolist(), 'values': source_data_time.tolist()}


# ------ response of pvf (json) data to api requests --------------
def resp_pvf_json(subject_name: str, session: str, file_name: str, timepoint: int) -> Dict[str, Any]:
    '''
    Generate PVF JSON response for a given timepoint.
    '''
    
    global subjects_loaded_pvf_data

    file_pvf_data = subjects_loaded_pvf_data[subject_name][session]['meta_files'][file_name]
    timepoint_int = int(timepoint)
    pvf_data      = process_pvf_time_window(subject_name=subject_name, session=session, file_name=file_name, pvf_time_window_id=timepoint_int)
    streamlines   = process_streamlines_time_window(subject_name=subject_name, session=session, file_name=file_name, pvf_time_window_id=timepoint_int)

    return {
        "subject_ID"         : file_pvf_data["subject_ID"],
        "metadata_fname"     : file_name,
        "pvf_positions"      : pvf_data["positions"].tolist(),
        "pvf_directions"     : pvf_data["directions"].tolist(),
        "times"              : file_pvf_data["times"][:-2],  # here, because time stencil, last two time points are not computed
        "condA"              : sum(file_pvf_data["condA"][str(timepoint)]) / len(file_pvf_data["condA"][str(timepoint)]),
        "patterns"           : file_pvf_data["pattern_data"].get(str(timepoint), []),
        "mode_info"          : file_pvf_data['mode_info'],
        "streamlines"        : streamlines,
        "PVF_dimension"      : file_pvf_data["PVF_dimension"],
        "PVF_num_time_points": file_pvf_data["PVF_num_time_points"]-2, # here, because time stencil, last two time points are not computed
        }


# start up server
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="server ip address and port number")
    
    # add IP address（-i/--ip，required=False）
    parser.add_argument(
        "-i", "--ip",
        type     = str,
        default  = "127.0.0.1",
        help     = "server IP address, default: 127.0.0.1",
        required = False,
    )
    
    # add port number parameter（-p/--port，type=int ）
    parser.add_argument(
        "-p", "--port",
        type     = int,
        default  = 32123,
        help     = "Server port number (1024-65535), default:32123",
        required = False,
    )
    
    # parse input parameters
    args = parser.parse_args()
    
    # validate port number
    if not (1024 <= args.port <= 65535):
        raise ValueError("port number must be between 1024 and 65535")
    
    host_ip   = args.ip
    host_port = args.port
    uvicorn.run("3dpvf_server:app", host=host_ip, port=host_port, reload=True)