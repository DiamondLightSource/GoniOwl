import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

dodgy_data = []
dodgy_data.append("2024-08-01-09-02-27")

dir = "/dls_sw/i23/logs/GoniOwl"
start_date_cutoff = "2023-01-01"  # Only process files from this date onwards (YYYY-MM-DD format)

csvs = glob.glob(os.path.join(dir, "*.csv"))
csvs = [f for f in csvs if os.path.basename(f)[:10] >= start_date_cutoff]

df = pd.concat((pd.read_csv(f, header=None) for f in csvs), ignore_index=True)
df.columns = ["DateTime", "Histogram", "GoniOwl", "Human", "Image1", "Image2"]

df["Histogram"] = df["Histogram"].replace({"histogram_pin_detected": "on", "histogram_pin_not_detected": "off", "histogram_not_matched": "fail"})
df["GoniOwl"] = df["GoniOwl"].replace({"goniowl_pin_off": "off", "goniowl_pin_on": "on", "ERROR_READING_PV": "error", "goniowl_dark": "dark", "goniowl_light": "light", "PIN_ON": "on", "PIN_OFF": "off"})
df["Human"] = df["Human"].replace({"human_no": "off", "human_yes": "on"})

df = df[~df["DateTime"].isin(dodgy_data)]
df = df[~df["GoniOwl"].isin(["error", "light", "dark"])]
os.makedirs("outputs", exist_ok=True)
df.to_csv("outputs/GoniOwlMetrics_output.csv", index=False)

total_rows = len(df)
same_histogram_goniowl_human = ((df["Histogram"] == df["GoniOwl"]) & (df["GoniOwl"] == df["Human"])).mean() * 100
same_goniowl_human = (df["GoniOwl"] == df["Human"]).mean() * 100
same_histogram_human = (df["Histogram"] == df["Human"]).mean() * 100

print(f"Percentage of time Histogram, GoniOwl, and Human are the same: {same_histogram_goniowl_human:.2f}%")
print(f"Percentage of time GoniOwl and Human are the same: {same_goniowl_human:.2f}%")
print(f"Percentage of time Histogram and Human are the same: {same_histogram_human:.2f}%")

df[~((df["Histogram"] == df["GoniOwl"]) & (df["GoniOwl"] == df["Human"]))].to_csv("outputs/histogramvsgoniowlvshuman.csv", index=False)
df[df["GoniOwl"] != df["Human"]].to_csv("outputs/goniowlvshuman.csv", index=False)
df[df["Histogram"] != df["Human"]].to_csv("outputs/histogramvshuman.csv", index=False)


histogram_human_same_goniowl_not = df[(df["Histogram"] == df["Human"]) & (df["GoniOwl"] != df["Human"])]
histogram_human_same_goniowl_not.to_csv("outputs/histogram_human_same_goniowl_not.csv", index=False)

goniowl_human_same_histogram_not = df[(df["GoniOwl"] == df["Human"]) & (df["Histogram"] != df["Human"])]
goniowl_human_same_histogram_not.to_csv("outputs/goniowl_human_same_histogram_not.csv", index=False)

histogram_human_same_goniowl_not["DateTime"] = pd.to_datetime(histogram_human_same_goniowl_not["DateTime"].str[:10])
goniowl_human_same_histogram_not["DateTime"] = pd.to_datetime(goniowl_human_same_histogram_not["DateTime"].str[:10])

occurrences_histogram_human_same_goniowl_not = histogram_human_same_goniowl_not["DateTime"].value_counts().sort_index().cumsum()
occurrences_goniowl_human_same_histogram_not = goniowl_human_same_histogram_not["DateTime"].value_counts().sort_index().cumsum()

plt.plot(occurrences_histogram_human_same_goniowl_not, marker='o', label='Histogram and Human same, GoniOwl not')
plt.plot(occurrences_goniowl_human_same_histogram_not, marker='x', label='GoniOwl and Human same, Histogram not')

plt.xlabel("Date")
plt.ylabel("Cumulative Occurrences")
plt.title("Cumulative Occurrences Over Time")
plt.legend()
plt.grid(True)
plt.show()

disagree_df = df[df["GoniOwl"] != df["Human"]]

# Create disagreements folder if it doesn't exist
disagreements_dir = "outputs/disagreements"
if not os.path.exists(disagreements_dir):
    os.makedirs(disagreements_dir)

# Copy disagreement images to the folder
import shutil
for index, row in disagree_df.iterrows():
    img_path = row["Image1"]
    goniowl_value = row["GoniOwl"]
    human_value = row["Human"]
    
    print(f"GoniOwl = {goniowl_value}, Human = {human_value}, image = {img_path}")
    
    # Copy image to disagreements folder
    if os.path.exists(img_path):
        filename = os.path.basename(img_path)
        dest_path = os.path.join(disagreements_dir, filename)
        shutil.copy(img_path, dest_path)
    
    # img = mpimg.imread(img_path)
    # plt.imshow(img)
    # plt.title(f"GoniOwl = {goniowl_value}, Human = {human_value}")
    # plt.show()