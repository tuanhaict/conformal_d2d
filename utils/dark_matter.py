import h5py

def load_dark_matter_data(dim, file_path="dark_matter.h5"):
    dists_xyz_in, dists_xyz_out = [], []

    with h5py.File(file_path, "r") as f:
        for i in range(len(f["input"])):
            Xin  = f[f"input/{i}"][:]
            Xout = f[f"output/{i}"][:]
            dists_xyz_in.append(Xin[:, :dim])
            dists_xyz_out.append(Xout[:, :dim])
    return dists_xyz_in, dists_xyz_out