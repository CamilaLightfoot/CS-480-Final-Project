import pandas as pd
from pathlib import Path

# CSVs live in: data/LFW/
DATA_DIR = Path("data/LFW")

# Images live in: data/LFW/lfw-deepfunneled/
LFW_IMAGES = DATA_DIR / "lfw-deepfunneled"

def img_path_from_name_and_index(name, index):
    """
    Build a path like:
      data/LFW/lfw-deepfunneled/Aaron_Peirsol/Aaron_Peirsol_0001.jpg
    from:
      name="Aaron_Peirsol", index=1
    """
    idx = int(index)
    filename = f"{name}_{idx:04d}.jpg"
    return str(LFW_IMAGES / name / filename)

def convert_match_file(csv_name: str) -> pd.DataFrame:
    """
    matchpairsDev*.csv format:
      name,imagenum1,imagenum2
    """
    csv_path = DATA_DIR / csv_name
    print(f"[INFO] Reading match file: {csv_path}")
    df = pd.read_csv(csv_path)

    out = pd.DataFrame()
    out["img1"] = [
        img_path_from_name_and_index(row["name"], row["imagenum1"])
        for _, row in df.iterrows()
    ]
    out["img2"] = [
        img_path_from_name_and_index(row["name"], row["imagenum2"])
        for _, row in df.iterrows()
    ]
    out["label"] = 1
    return out

def convert_mismatch_file(csv_name: str) -> pd.DataFrame:
    """
    mismatchpairsDev*.csv formats:
      A) name1,name2,imagenum1,imagenum2
      B) name,imagenum1,name.1,imagenum2   (your file)
    """
    csv_path = DATA_DIR / csv_name
    print(f"[INFO] Reading mismatch file: {csv_path}")
    df = pd.read_csv(csv_path)

    cols = set(df.columns)

    # Case A: standard naming
    if {"name1", "name2", "imagenum1", "imagenum2"}.issubset(cols):
        name1_col = "name1"
        name2_col = "name2"
        img1_col = "imagenum1"
        img2_col = "imagenum2"

    # Case B: your file: name, imagenum1, name.1, imagenum2
    elif {"name", "name.1", "imagenum1", "imagenum2"}.issubset(cols):
        name1_col = "name"
        name2_col = "name.1"
        img1_col = "imagenum1"
        img2_col = "imagenum2"

    else:
        raise ValueError(
            f"Unexpected columns in {csv_path}: {df.columns}. "
            "Expected either "
            "{name1,name2,imagenum1,imagenum2} or "
            "{name,name.1,imagenum1,imagenum2}."
        )

    out = pd.DataFrame()
    out["img1"] = [
        img_path_from_name_and_index(row[name1_col], row[img1_col])
        for _, row in df.iterrows()
    ]
    out["img2"] = [
        img_path_from_name_and_index(row[name2_col], row[img2_col])
        for _, row in df.iterrows()
    ]
    out["label"] = 0
    return out

def main():
    # Quick sanity checks
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR does not exist: {DATA_DIR}")
    if not LFW_IMAGES.exists():
        raise FileNotFoundError(f"LFW_IMAGES does not exist: {LFW_IMAGES}")

    match_train    = convert_match_file("matchpairsDevTrain.csv")
    match_test     = convert_match_file("matchpairsDevTest.csv")
    mismatch_train = convert_mismatch_file("mismatchpairsDevTrain.csv")
    mismatch_test  = convert_mismatch_file("mismatchpairsDevTest.csv")

    all_pairs = pd.concat(
        [match_train, match_test, mismatch_train, mismatch_test],
        ignore_index=True,
    )

    out_path = Path("data/lfw_pairs.csv")
    all_pairs.to_csv(out_path, index=False)

    print(f"[INFO] Saved {out_path} with {len(all_pairs)} pairs.")

if __name__ == "__main__":
    main()