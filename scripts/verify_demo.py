"""Audit exported synthetic evidence with Python's standard library."""
import csv
import json
import math
from pathlib import Path
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def records(path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None, f"{path}: missing header")
        require(len(reader.fieldnames) == len(set(reader.fieldnames)), f"{path}: duplicate columns")
        rows = list(reader)
    require(all(None not in row and None not in row.values() for row in rows),
            f"{path}: CSV header/row width mismatch")
    return rows


def close(actual, reported, tolerance, label):
    require(math.isfinite(actual) and math.isfinite(reported), f"{label}: nonfinite")
    require(abs(actual-reported) <= tolerance,
            f"{label}: independently calculated {actual}, reported {reported}")


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def mul(matrix, vector):
    return [dot(row, vector) for row in matrix]


def angle(a, b):
    cosine = dot(a, b)/(math.sqrt(dot(a, a))*math.sqrt(dot(b, b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def audit(directory):
    report = json.loads((directory/"report.json").read_text(encoding="utf-8"))
    require(report["seed"] == 2026 and report["points"] == 240, "Unexpected fixture")
    require(report["calibration"] == [525, 510, 320, 240], "Unexpected pinhole K")
    require(report["known_rotation_y_deg"] == 5, "Unexpected known rotation")
    require(report["known_translation_scene_units"] == [-0.6, 0.02, 0.04], "Unexpected known translation")
    require(report["opencv_threads"] == 1 and report["opencl"] is False, "Uncontrolled runtime")
    baseline = math.sqrt(0.6**2+0.02**2+0.04**2)
    close(baseline, report["known_baseline_scene_units"], 1e-12, "Baseline")
    scene = records(directory/"scene.csv")
    points = records(directory/"estimated_points.csv")
    metrics = records(directory/"metrics.csv")
    tracks = records(directory/"tracking.csv")
    require(len(scene) == 240 and len(metrics) == 2, "Unexpected scene/metric rows")
    require(sum(int(row["known_mismatch"]) for row in scene) == 35, "Known mismatch count")
    require([int(row["index"]) for row in scene] == list(range(240)), "Scene index order")
    noisy = next(row for row in metrics if row["scenario"] == "noise_and_mismatches")
    clean = next(row for row in metrics if row["scenario"] == "clean")
    for row in metrics:
        require(int(row["model_inliers"]) >= int(row["accepted_points"]), "Consensus/acceptance order")
        require(int(row["accepted_points"]) == int(row["true_points"])+int(row["retained_mismatches"]),
                "Acceptance accounting")
        require(int(row["retained_mismatches"]) == 0, "Known mismatch retained")
    require(int(clean["accepted_points"]) == 240 and int(noisy["accepted_points"]) == len(points),
            "Point count/report disagreement")
    indices = [int(row["input_index"]) for row in points]
    require(len(indices) == len(set(indices)), "Duplicate accepted index")
    require(all(0 <= index < 240 for index in indices), "Invalid accepted index")
    R = [report["estimated_rotation_row_major"][3*i:3*i+3] for i in range(3)]
    t = report["estimated_translation_baseline_units"]
    require(all(math.isfinite(value) for row in R for value in row), "Nonfinite pose")
    for i in range(3):
        for j in range(3):
            close(dot(R[i], R[j]), float(i == j), 1e-9, "Rotation orthogonality")
    close(math.sqrt(dot(t, t)), 1, 1e-9, "Translation norm")
    radians = math.radians(5)
    truth_R = [[math.cos(radians), 0, math.sin(radians)], [0, 1, 0],
               [-math.sin(radians), 0, math.cos(radians)]]
    rotation_error = math.degrees(math.acos(max(-1.0, min(1.0,
        (sum(dot(truth_R[i], R[i]) for i in range(3))-1)/2))))
    close(rotation_error, float(noisy["rotation_error_deg"]), 1e-8, "Rotation error")
    close(angle([-0.6, 0.02, 0.04], t), float(noisy["translation_error_deg"]), 1e-8,
          "Translation direction error")
    fx, fy, cx, cy = report["calibration"]
    square_errors, residuals, ray_angles = [], [], []
    center = [-dot([R[j][i] for j in range(3)], t) for i in range(3)]
    xyz = []
    for row, index in zip(points, indices):
        reference = scene[index]
        X = [float(row[key]) for key in ["x_baseline", "y_baseline", "z_baseline"]]
        require(all(math.isfinite(value) for value in X), "Nonfinite cloud")
        Y = [value+shift for value, shift in zip(mul(R, X), t)]
        require(0 < X[2] <= 100 and 0 < Y[2] <= 100, "Both-camera depth gate")
        require(reference["known_mismatch"] == "0", "Known mismatch in cloud")
        truth = [float(reference[key]) for key in ["x_scene", "y_scene", "z_scene"]]
        square_errors.append(sum((baseline*a-b)**2 for a, b in zip(X, truth)))
        projected1 = [fx*X[0]/X[2]+cx, fy*X[1]/X[2]+cy]
        projected2 = [fx*Y[0]/Y[2]+cx, fy*Y[1]/Y[2]+cy]
        residuals.append(max(math.hypot(projected1[0]-float(reference["u1_px"]),
                                        projected1[1]-float(reference["v1_px"])),
                             math.hypot(projected2[0]-float(reference["u2_px"]),
                                        projected2[1]-float(reference["v2_px"]))))
        ray_angles.append(angle(X, [a-b for a, b in zip(X, center)]))
        xyz.append(X)
    rmse = math.sqrt(sum(square_errors)/len(square_errors))
    close(rmse, float(noisy["structure_rmse_scene"]), 1e-10, "3D RMSE")
    close(max(residuals), float(noisy["max_reprojection_px"]), 1e-4, "Maximum reprojection")
    close(min(ray_angles), float(noisy["min_angle_deg"]), 1e-8, "Minimum ray angle")
    require(rmse < 0.35 and max(residuals) < 2.001 and min(ray_angles) > 1, "Accuracy bounds")
    ply = (directory/"cloud.ply").read_text().splitlines()
    require("SYNTHETIC" in ply[2], "Missing synthetic PLY provenance")
    body = ply[ply.index("end_header")+1:]
    require(f"element vertex {len(points)}" in ply and len(body) == len(points), "PLY count")
    require([[float(value) for value in line.split()] for line in body] == xyz, "PLY/CSV disagreement")
    require(len(tracks) == 500 and len({row["id"] for row in tracks}) == len(tracks),
            "Frontend observation count/identity")
    errors = sorted(math.hypot(float(row["u_current"])-float(row["u_previous"])-4,
                              float(row["v_current"])-float(row["v_previous"])-3)
                    for row in tracks if row["has_previous"] == "1")
    close(errors[len(errors)//2], report["tracker_median_warp_error_px"], 1e-6,
          "Frontend warp upper median")
    require(len(errors) == report["tracker_temporal_correspondences"], "Frontend temporal count")
    print(f"Audited {len(points)} synthetic points; RMSE={rmse:.6g}; "
          f"{len(errors)} tracks; CSV schemas, pose, PLY and quality metrics agree.")


if __name__ == "__main__":
    audit(Path(sys.argv[1] if len(sys.argv) == 2 else "examples/offline"))
