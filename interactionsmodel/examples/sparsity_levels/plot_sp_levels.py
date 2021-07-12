import plotly.graph_objects as go
import json
import os
import numpy as np
from interactionsmodel import path_data, path_tex

if __name__ == "__main__":
    path_save = os.path.join(path_tex, "blocks_interactions", "prebuilt_images")

    with open(os.path.join(path_data, "sp_levels.json")) as json_file:
        data = json.load(json_file)

    l1_max, l2_max = data["lambda_max"]
    n_lambda = len(np.unique(data["CD"]["lambda_1"]))
    print(l1_max, l2_max)
    l1 = np.unique(data["CD"]["lambda_1"])
    l2 = np.unique(data["CD"]["lambda_2"])

    z_data = np.array(data["CD"]["sp_theta"]).reshape(n_lambda, n_lambda).T
    z_data = np.flip(z_data)
    fig = go.Figure(data=[go.Surface(z=z_data[:8, :8], x=l1[:8], y=l2[:8])])
    fig.update_traces(
        contours_z=dict(
            show=True, usecolormap=True, highlightcolor="limegreen", project_z=True
        )
    )
    fig.update_layout(
        title="",
        autosize=False,
        scene_camera_eye=dict(x=1.87, y=0.88, z=0.3),
        width=800,
        height=600,
        margin=dict(r=40, l=10, b=10, t=5),
        font=dict(size=18),
        scene=dict(
            xaxis_title="l1 penalty",
            xaxis_tickfont=dict(size=12),
            yaxis_title="l2 penalty",
            yaxis_tickfont=dict(size=12),
            zaxis_title="nnz(Θ) / q",
            zaxis_tickfont=dict(size=12),
        ),
    )
    fig.update_layout(xaxis_type="log", yaxis_type="log")
    fig.write_image(os.path.join(path_save, "theta_surface_sparsity.pdf"))
    fig.show()
