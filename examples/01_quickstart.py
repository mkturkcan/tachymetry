"""Run the full pipeline on a public sample clip and print a summary."""

import tachymetry as tm
from tachymetry.data import sample_video

if __name__ == "__main__":
    run = tm.track(sample_video("intersection-day"), out="runs/quickstart", render=True)
    print(run.camera.describe())
    print(run.summary())
    df = run.tracks.to_dataframe()
    cars = df[(df["class"] == "car") & (df["state"] == "moving")]
    print(cars.groupby("track_id")["speed_kmh"].median().describe())
