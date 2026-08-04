# PVF visualisation
Using three.js to visualise velocity fields, singular points, streamlines and spatiotemporal modes.

Freesurfer processed brain surfaces are used for visualisation.

## Data locations.
Under pvf_data, use soft links (Linux/Unix) to point to PVF data folder and Freesurfer subjects folder to access PVF data and Freesurfer surfaces.

PVF data
> pvf_data/pvf_subjects

> ln -s /PATH_TO_pvf_subjects pvf_data/pvf_subjects

Freesurfer subjects
> pvf_data/fs_subjects

> ln -s /PATH_TO_fs_subjects pvf_data/fs_subjects


# To do list - PVF publications
 - Publication-level figure exports (publication-level fine tuning of colarmaps, vectors and view angles).
 - left-right

# To do list - FDRI project stream 4, phase 2
 - Source estimate viewer alone.
 - MRI viewer (traditional 3 views, grey value intensity) integration.
 - Diffusion MRI/tractography viewer integration.
