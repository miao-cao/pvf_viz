# convert_freesurfer_to_obj.py
import numpy as np
import mne
import sys
import os

def convert_freesurfer_surface_to_obj(freesurfer_file, output_obj_file):
    """
    将FreeSurfer表面文件转换为OBJ格式
    
    参数:
        freesurfer_file: FreeSurfer表面文件路径 (如 lh.pial)
        output_obj_file: 输出的OBJ文件路径
    """
    try:
        # 使用MNE读取FreeSurfer表面文件
        vertices, faces = mne.read_surface(freesurfer_file)
        
        # 写入OBJ文件
        with open(output_obj_file, 'w') as f:
            # OBJ文件头
            f.write("# Converted from FreeSurfer surface file\n")
            f.write(f"# Vertices: {len(vertices)}, Faces: {len(faces)}\n")
            
            # 写入顶点数据
            for vertex in vertices:
                f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
            
            # 写入面数据 (注意OBJ格式的面索引从1开始)
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
                
        print(f"成功转换: {freesurfer_file} -> {output_obj_file}")
        print(f"顶点数: {len(vertices)}, 面数: {len(faces)}")
        
    except Exception as e:
        print(f"转换失败: {str(e)}")
        raise

def convert_subject_data(subject_id, data_dir="data"):
    """
    转换一个受试者的所有FreeSurfer表面文件
    
    参数:
        subject_id: 受试者ID (如 'p1')
        data_dir: 数据目录
    """
    subject_dir = f"{data_dir}/{subject_id}"
    
    # 转换左半球
    lh_pial_fs  = f"{subject_dir}/surf/lh.pial"      # os.path.join(subject_dir, "/surf/lh.pial")
    lh_pial_obj = f"{subject_dir}/surf/lh.pial.obj"  # os.path.join(subject_dir, "/surf/lh.pial.obj")
    
    if os.path.exists(lh_pial_fs):
        convert_freesurfer_surface_to_obj(lh_pial_fs, lh_pial_obj)
    else:
        print(f"警告: 找不到文件 {lh_pial_fs}")
    
    # 转换右半球
    rh_pial_fs = f"{subject_dir}/surf/rh.pial"          # os.path.join(subject_dir, "/surf/rh.pial")
    rh_pial_obj = f"{subject_dir}/surf/rh.pial.obj"     # os.path.join(subject_dir, "/surf/rh.pial.obj")
    
    if os.path.exists(rh_pial_fs):
        convert_freesurfer_surface_to_obj(rh_pial_fs, rh_pial_obj)
    else:
        print(f"警告: 找不到文件 {rh_pial_fs}")

def convert_pvf_data_2json(pvf_data_fname):
    """
    占位函数: 将PVF数据转换为JSON格式
    """
    import json
    import h5py
    import hdf5storage as hdf

    res_fname = pvf_data_fname    
    pvf_data = hdf.loadmat(f"{pvf_data_fname}.mat")

    # with h5py.File(f"{pvf_data_fname}.mat", "r") as f:
    #     pvf_data = f
        
    params              = pvf_data['params']
    gen_time            = pvf_data['gen_time']
    times               = pvf_data['times']
    Vx                  = pvf_data['Vx']
    Vy                  = pvf_data['Vy']
    Vz                  = pvf_data['Vz']
    dim_shift           = [20, 20, 20]      # pvf_data['dim_shift']
    volume_mask         = pvf_data['volume_mask']
    volume_vertex_index = pvf_data['volume_vertex_index']
    vol_vert_FS_RAS_ind = pvf_data['vol_vert_FS_RAS_ind']


    n_times = times.shape[0]
    cond_num = {}
    for t in range(n_times):
        cond_num[f"{t}"] = [7] * 10
    # if params['compute_condion_number'] == True: # save condition numbers if computed
    #     cond_num = {}

    with open(f"{res_fname}_condA.json", 'w') as file:
        json.dump(cond_num, file)

    # dict_keys(['Vx', 'Vy', 'Vz', 'dim_shift', 'gen_time', 'orig_src_volumes', 'params', 'simulation_params', 'times', 'vol_vert_FS_RAS_ind', 'volume_mask', 'volume_mask_interp', 'volume_vertex_index', 'wave', 'wave_analytical'])

    with open(f"{res_fname}_metadata.json", 'w') as file: # save pvf's metadata as .json format
        vf_data_dict                        = {}
        vf_data_dict['params']              = params
        vf_data_dict['gen_time']            = gen_time
        vf_data_dict['times']               = times.tolist()
        vf_data_dict['vol_vert_FS_RAS_ind'] = vol_vert_FS_RAS_ind.tolist()
        vf_data_dict['dim_shift']           = dim_shift
        vf_data_dict['volume_mask']         = volume_mask.tolist()
        vf_data_dict['volume_vertex_index'] = volume_vertex_index.tolist()
        json.dump(vf_data_dict, file)
    
    with open(f"{res_fname}_Vx.json", 'w') as file: # save pvf's Vx as .json format
        vf_data_dict                        = {}
        vf_data_dict['params']              = params
        vf_data_dict['gen_time']            = gen_time
        vf_data_dict['Vx']                  = Vx.tolist()
        json.dump(vf_data_dict, file)
    with open(f"{res_fname}_Vy.json", 'w') as file: # save pvf's Vy as .json format
        vf_data_dict                        = {}
        vf_data_dict['params']              = params
        vf_data_dict['gen_time']            = gen_time
        vf_data_dict['Vy']                  = Vy.tolist()
        json.dump(vf_data_dict, file)

    with open(f"{res_fname}_Vz.json", 'w') as file: # save pvf's Vz as .json format
        vf_data_dict                        = {}
        vf_data_dict['params']              = params
        vf_data_dict['gen_time']            = gen_time
        vf_data_dict['Vz']                  = Vz.tolist()
        json.dump(vf_data_dict, file)


if __name__ == "__main__":
    
    data_dir = "/Users/miaoc/Desktop/Vectorfields/fs_subjects"   
    subject_ids = ['sub19'] # ["sub-003", "sub-005"]                            # ["p1", "p2", "p3"]

    for subject_id in subject_ids:
        print(f"\n转换受试者 {subject_id} 的数据...")
        convert_subject_data(subject_id, data_dir=data_dir)

    # pvf_data_fname = "/Users/miaoc/Documents/Neuroscience/workspace/pvf_viz/pvf_data/pvf_subjects/sub19/pvf_propspiral_locInterphigh_signalamp1_spiralfreq1_spiralwavelength3_addnoiseTrue_noiseamp0.4_20260105154036"
    # convert_pvf_data_2json(pvf_data_fname)