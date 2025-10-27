import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import copy

import numpy as np

from fastapi import FastAPI, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()
# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# PVF metadata
subject_id          : str            = ""
pvf_metadata_fname  : str            = ""
pvf_metadata        : Dict[str, Any] = {}
pvf_num_time_points: int             = 0
pvf_times           : List[float]    = []
pvf_dimension       : int            = 50

# PVF data
pvf_mask_volume = []
pvf_vx          = []
pvf_vy          = []
pvf_vz          = []

dim_shift                : List[int]           = [25, 25, 25]
temporary_map_dim_shifts: Dict[str, List[int]] = {
    # "sub-003": [25, 25, 20],
    # "sub-005": [25, 25, 17],
}

# PVF condition numbers
pvf_condA_fname: str            = ""
pvf_condA_data : Dict[str, Any] = {}

# PVF condition numbers
pvf_pattern_fname: str            = ""
pvf_pattern_data : Dict[str, Any] = {}

# PVF streamlines
pvf_streamline_folder          : str            = ""
pvf_streamlines_time_windows   : Dict[str, Any] = {}
pvf_streamline_all_time_windows: Dict[str, Any] = {}
pvf_streamlines_tmin           : int            = 0
pvf_streamlines_tmax           : int            = 4

# PVF data directories
PVF_SUBJECTS_DIR = f"{Path(__file__).parent}/pvf_data/pvf_subjects"
FS_SUBJECTS_DIR  = f"{Path(__file__).parent}/pvf_data/fs_subjects"


@app.get("/api/list-subjects")
async def list_subjects():
    """
    List all subject IDs under the PVF subjects directory.
    """
    subject_id_list = []
    if os.path.exists(PVF_SUBJECTS_DIR) == False:
        return subject_id_list
    else:
        subject_id_list = [subject_id for subject_id in os.listdir(PVF_SUBJECTS_DIR) if os.path.isdir(os.path.join(PVF_SUBJECTS_DIR, subject_id))]
        return subject_id_list


@app.get("/api/list-subjects-files")
async def list_subjects_pvf_files(subject: str = Query(None)):
    """
    list all _metadata.json files for a given subject.
    """
    subject_id  = subject
    subject_dir = f"{PVF_SUBJECTS_DIR}/{subject_id}"
    fname_list  = []
    if os.path.exists(subject_dir) == False:
        return fname_list
    
    fname_list = [fname for fname in os.listdir(subject_dir) if os.path.isfile(os.path.join(subject_dir, fname)) and fname.endswith("_metadata.json")]
    
    return fname_list


@app.get("/api/load-subjects-files")
async def load_subjects_files(
                            subject         : str             = Query(None),
                            file            : str             = Query(None),
                            background_tasks: BackgroundTasks = None):
    
    global subject_id, pvf_metadata_fname
    
    print(f"Current subject: {subject_id}, file: {pvf_metadata_fname}")
    
    if subject_id == subject and pvf_metadata_fname == file:
        data = await resp_pvf_json(0)
        return data
    elif subject and file:
        print(f"Loading subject: {subject}, file: {file}")
        
        # using background task to load all streamlines
        background_tasks.add_task(load_streamlines_all_time_windows)
        
        data = await read_pvf_json(subject, file)
        
        # await load_streamlines_all_time_windows()
        return data
    else:
        return {"message": "No data found."}


@app.get("/api/update-PVF-streamlines")
async def update_pvf_streamlines(subject: str = Query(None), file: str = Query(None), timepoint: str = Query(None)):
    """更新特定时间点的流线数据"""
    global subject_id, pvf_metadata_fname
    
    if subject_id == subject and pvf_metadata_fname == file and timepoint:
        print(f"Updating PVF streamlines at timepoint: {timepoint}")
        data = await update_pvf_streamlines_data(timepoint)
        return data
    else:
        print("Subject or meta file mismatch when updating streamlines.")
        return {"message": "Subject or file mismatch"}


# functions to process PVF data
def process_pvf_time_window(pvf_time_window_id: int) -> Dict[str, Any]:
    """处理特定时间窗口的PVF数据"""
    global pvf_vx, pvf_vy, pvf_vz, pvf_num_time_points, dim_shift, pvf_mask_volume
    print(f"Processing Vx, Vy, Vz at time point: {pvf_time_window_id} of {pvf_num_time_points}")

    mask_volume = pvf_mask_volume
    vx          = np.squeeze(pvf_vx[:, :, :, pvf_time_window_id])
    vy          = np.squeeze(pvf_vy[:, :, :, pvf_time_window_id])
    vz          = np.squeeze(pvf_vz[:, :, :, pvf_time_window_id])

    y, x, z = np.meshgrid(np.arange(0, vx.shape[0], 1),
                        np.arange(0, vx.shape[0], 1),
                        np.arange(0, vx.shape[0], 1))
    positions = np.vstack([x[mask_volume].flatten(),
                        y[mask_volume].flatten(),
                        z[mask_volume].flatten()]).T
    
    # VF arrows
    positions = (positions + np.repeat([[-dim_shift[0], -dim_shift[1], -dim_shift[2]]],
                                   positions.shape[0], axis=0)) * 5

    u, v, w = vx[mask_volume], vy[mask_volume], vz[mask_volume]
    directions = np.vstack([u.flatten().T, v.flatten().T, w.flatten().T,]).T

    
    return {"positions": positions, "directions": directions}


def process_streamlines_time_window(pvf_time_window_id: int) -> List[Any]:
    """处理特定时间窗口的流线数据"""
    global pvf_streamline_all_time_windows, dim_shift
    streamlines = pvf_streamline_all_time_windows[str(pvf_time_window_id)]
    new_streamlines = []
    for s, streamline in enumerate(streamlines):
        streamline = np.asarray(streamline)
        n_pos = streamline.shape[0]

        new_streamline = ((streamline + np.repeat([[-dim_shift[0], -dim_shift[1], -dim_shift[2]]], n_pos, axis=0)) * 5 )
        # vf space ijk                     #src-vf dim shift
        new_streamlines.append(new_streamline.tolist())

    print(f"Processed {len(new_streamlines)} streamlines at time point: {pvf_time_window_id}")
    return new_streamlines


# async functions to read all streamlines data
def load_streamlines_all_time_windows():
    """loading all streamlines from folder in background"""
    global pvf_streamline_all_time_windows, pvf_streamline_folder, pvf_streamlines_time_windows
    
    print(f"Background loading streamlines of all time windows from folder: {pvf_streamline_folder}")
    # pvf_streamline_all_time_windows = copy.copy(pvf_streamlines_time_windows)
    streamline_files = [
        f for f in os.listdir(pvf_streamline_folder)
        if os.path.isfile(os.path.join(pvf_streamline_folder, f)) and f.endswith(".json")
    ]
    
    for file in streamline_files:
        print(f"Loading streamlines from: {file}")
        file_path = os.path.join(pvf_streamline_folder, file)
        with open(file_path, "r", encoding="utf8") as f:
            streamlines_time_windows = json.load(f)
            for timepoint, streamlines in streamlines_time_windows.items():
                pvf_streamline_all_time_windows[timepoint] = streamlines
    
    print(f"Completed loading all streamlines from: {len(streamline_files)} json files.")


async def read_pvf_json(subject_name: str, file_name: str) -> Dict[str, Any]:
    """Read PVF JSON files"""
    global subject_id, pvf_metadata_fname, pvf_metadata, pvf_vx, pvf_vy, pvf_vz
    global pvf_condA_fname, pvf_condA_data, pvf_pattern_fname, pvf_pattern_data
    global pvf_streamline_folder, pvf_streamlines_time_windows, pvf_num_time_points
    global pvf_dimension, dim_shift, pvf_mask_volume, pvf_streamline_all_time_windows
    global pvf_times
    
    subject_id         = subject_name
    metadata_path      = f"{PVF_SUBJECTS_DIR}/{subject_name}/{file_name}"
    pvf_metadata_fname = file_name
    
    vx_path               = metadata_path.replace("_metadata.json", "_Vx.json")
    vy_path               = metadata_path.replace("_metadata.json", "_Vy.json")
    vz_path               = metadata_path.replace("_metadata.json", "_Vz.json")
    condA_path            = metadata_path.replace("_metadata.json", "_condA.json")
    pattern_path          = metadata_path.replace("_metadata.json", "_pattern_detection.json")
    pvf_streamline_folder = str(metadata_path.replace("_metadata.json", "_streamlines"))
    
    resp_value: Dict[str, Any] = {}
    
    # 读取元数据
    try:
        with open(metadata_path, "r", encoding="utf8") as f:
            pvf_metadata                      = json.load(f)
            pvf_mask_volume                   = np.asarray(pvf_metadata['volume_mask'])
            resp_value["subject_ID"]          = subject_name
            resp_value["times"]               = pvf_metadata['times']
            resp_value["PVF_num_time_points"] = len(pvf_metadata['times'])
            pvf_num_time_points               = len(pvf_metadata['times'])
            pvf_times                         = pvf_metadata['times']

            if subject_id in temporary_map_dim_shifts:
                dim_shift = temporary_map_dim_shifts[subject_id]
            else:
                dim_shift = pvf_metadata['dim_shift']
            
            print(f"Successfully loaded: {metadata_path}")
            print(f"Number of Timepoints: {pvf_num_time_points}")
            print(f"Dim shift: {dim_shift}")
    except Exception as e:
        print(f"处理失败: {e}")
    
    # 读取Vx数据
    try:
        with open(vx_path, "r", encoding="utf8") as f:
            data_vx             = json.load(f)
            pvf_vx              = np.asarray(data_vx["Vx"])
            pvf_num_time_points = pvf_vx.shape[3]
            pvf_dimension       = pvf_vx.shape[0]
            resp_value["PVF_dimension"] = pvf_dimension
            # resp_value["PVF_num_time_points"] = pvf_num_time_points
            print(f"PVF dimensions: {pvf_dimension}")
            print(f"Successfully loaded Vx: {vx_path}")
    except Exception as e:
        print(f"处理失败: {e}")
    
    # 读取Vy数据
    try:
        with open(vy_path, "r", encoding="utf8") as f:
            data_vy = json.load(f)
            pvf_vy  = np.asarray(data_vy["Vy"])
            print(f"Successfully loading Vy: {vy_path}")
    except Exception as e:
        print(f"处理失败: {e}")
    
    # 读取Vz数据
    try:
        with open(vz_path, "r", encoding="utf8") as f:
            data_vz = json.load(f)
            pvf_vz  = np.asarray(data_vz["Vz"])
            print(f"Successfully loading Vz: {vz_path}")
    except Exception as e:
        print(f"处理失败: {e}")
    
    # 读取条件数数据
    try:
        with open(condA_path, "r", encoding="utf8") as f:
            pvf_condA_data = json.load(f)
            print(f"Successfully loading condA: {condA_path}")
    except Exception as e:
        print(f"处理失败: {e}")
    
    # 读取模式检测数据
    try:
        with open(pattern_path, "r", encoding="utf8") as f:
            pvf_pattern_data = json.load(f)
            print(f"Successfully loading Patterns: {pattern_path}")
    except Exception as e:
        print(f"处理失败: {e}")
    
    # 读取流线数据
    try:
        first_tw_path = os.path.join(pvf_streamline_folder, "pvf_streamlines_time_window_0_4.json")
        with open(first_tw_path, "r", encoding="utf8") as f:
            pvf_streamlines_time_windows = json.load(f)
            pvf_streamline_all_time_windows = copy.copy(pvf_streamlines_time_windows)
            print(f"Successfully loading Streamlines: {pvf_streamline_folder}")
    except Exception as e:
        print(f"处理失败: {e}")
    
    # 处理默认时间点数据
    default_first_timepoint      = 0
    pvf_data                     = process_pvf_time_window(default_first_timepoint)
    streamlines                  = process_streamlines_time_window(default_first_timepoint)
    resp_value["pvf_positions"]  = pvf_data["positions"].tolist()
    resp_value["pvf_directions"] = pvf_data["directions"].tolist()
    resp_value["condA"]          = sum(pvf_condA_data[str(default_first_timepoint)]) / len(pvf_condA_data[str(default_first_timepoint)])
    resp_value["patterns"]       = pvf_pattern_data.get(str(default_first_timepoint), {})
    resp_value["streamlines"]    = streamlines
    
    return resp_value


async def resp_pvf_json(timepoint: int) -> Dict[str, Any]:
    """生成PVF JSON响应"""
    global subject_id, pvf_dimension, pvf_num_time_points
    global pvf_condA_data, pvf_pattern_data, pvf_times
    pvf_data    = process_pvf_time_window(timepoint)
    streamlines = process_streamlines_time_window(timepoint)
    return {
        "subject_ID"         : subject_id,
        "pvf_positions"      : pvf_data["positions"].tolist(),
        "pvf_directions"     : pvf_data["directions"].tolist(),
        "times"              : pvf_times,
        "condA"              : sum(pvf_condA_data[str(timepoint)]) / len(pvf_condA_data[str(timepoint)]),
        "patterns"           : pvf_pattern_data[str(timepoint)],
        "streamlines"        : streamlines,
        "PVF_dimension"      : pvf_dimension,
        "PVF_num_time_points": pvf_num_time_points,
    }


async def update_pvf_streamlines_data(timepoint: str) -> Dict[str, Any]:
    """更新特定时间点的流线数据"""
    global pvf_condA_data, pvf_pattern_data, pvf_times
    try:
        timepoint_int = int(timepoint)
        pvf_data    = process_pvf_time_window(timepoint_int)
        streamlines = process_streamlines_time_window(timepoint_int)
        return {
            "pvf_positions" : pvf_data["positions"].tolist(),
            "pvf_directions": pvf_data["directions"].tolist(),
            "times"         : pvf_times,
            "condA"         : sum(pvf_condA_data[timepoint]) / len(pvf_condA_data[timepoint]),
            "patterns"      : pvf_pattern_data[timepoint],
            "streamlines"   : streamlines,
        }
    except ValueError:
        return {"message": "Invalid timepoint format"}
    

if __name__ == "__main__":
    uvicorn.run("3dpvf_server:app", host="127.0.0.1", port=3000, reload=True)