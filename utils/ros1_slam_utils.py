import csv
import numpy as np

def parse_ros1_bag_csv(path):
    out = []

    with open(path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            out.append([
                float(row["field.header.stamp"]),
                float(row["field.pose.position.x"]),
                float(row["field.pose.position.y"]),
                float(row["field.pose.position.z"]),
                float(row["field.pose.orientation.x"]),
                float(row["field.pose.orientation.y"]),
                float(row["field.pose.orientation.z"]),
                float(row["field.pose.orientation.w"]),
            ])

    return np.array(out)