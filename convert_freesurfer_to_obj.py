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

if __name__ == "__main__":
    
    data_dir = "/Users/miaoc/Desktop/Vectorfields/fs_subjects"   
    subject_ids = ["sub-003", "sub-005"]                            # ["p1", "p2", "p3"]

    for subject_id in subject_ids:
        print(f"\n转换受试者 {subject_id} 的数据...")
        convert_subject_data(subject_id, data_dir=data_dir)