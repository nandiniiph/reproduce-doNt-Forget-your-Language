import os
import re
import pandas as pd

BASE_PATH = "outputs/amazon"


# =========================
# Extract untuk NFL & Baseline
# =========================
def extract_metrics(log_path):
    if not os.path.exists(log_path):
        return None, None, None

    with open(log_path, "r") as f:
        text = f.read()

    # ambil semua eval_accuracy
    accs = re.findall(r"'eval_accuracy': ([0-9.]+)", text)

    if len(accs) >= 3:
        biased = float(accs[-3])
        robust = float(accs[-2])
        delta = round(robust - biased, 3)
        return biased, robust, delta

    return None, None, None


# =========================
# Extract DFR
# =========================
def extract_dfr(log_path):
    if not os.path.exists(log_path):
        return {}

    with open(log_path, "r") as f:
        text = f.read()

    # split berdasarkan kemunculan run
    blocks = text.split("Namespace(")

    results = {}

    for block in blocks:
        if "data_percentage=0.05" in block:
            b = re.findall(r"biased acc=([0-9.]+)", block)
            u = re.findall(r"unbiased acc=([0-9.]+)", block)
            if b and u:
                results["DFR (5%)"] = (float(b[-1]), float(u[-1]))

        if "data_percentage=1.0" in block:
            b = re.findall(r"biased acc=([0-9.]+)", block)
            u = re.findall(r"unbiased acc=([0-9.]+)", block)
            if b and u:
                results["DFR (100%)"] = (float(b[-1]), float(u[-1]))

    return results


# =========================
# MAIN
# =========================
def main():
    rows = []

    models = [
        ("RoBERTa", f"{BASE_PATH}/none/log.txt"),
        ("NFL-F", f"{BASE_PATH}/nfl_f/log.txt"),
        ("NFL-CO", f"{BASE_PATH}/nfl_co/log.txt"),
        ("NFL-CP", f"{BASE_PATH}/nfl_cp/log.txt"),
        ("NFL-PT", f"{BASE_PATH}/nfl_pt/log.txt"),
    ]

    # ambil baseline + NFL
    for name, path in models:
        b, r, d = extract_metrics(path)
        rows.append({
            "Method": name,
            "Biased Acc": b,
            "Robust Acc": r,
            "Δ (Gap)": d
        })

    # ambil DFR
    dfr_path = f"{BASE_PATH}/dfr/log.txt"
    dfr = extract_dfr(dfr_path)

    if "DFR (5%)" in dfr:
        b, r = dfr["DFR (5%)"]
        rows.append({
            "Method": "DFR (5%)",
            "Biased Acc": b,
            "Robust Acc": r,
            "Δ (Gap)": round(r - b, 3)
        })

    if "DFR (100%)" in dfr:
        b, r = dfr["DFR (100%)"]
        rows.append({
            "Method": "DFR (100%)",
            "Biased Acc": b,
            "Robust Acc": r,
            "Δ (Gap)": round(r - b, 3)
        })

    # ideal model
    b, r, d = extract_metrics(f"{BASE_PATH}/ideal/log.txt")
    rows.append({
        "Method": "Ideal Model",
        "Biased Acc": b,
        "Robust Acc": r,
        "Δ (Gap)": d
    })

    # =========================
    # DataFrame
    # =========================
    df = pd.DataFrame(rows)

    # formatting
    df = df.round(4)

    print("\n===== AMAZON TABLE =====\n")
    print(df.to_string(index=False))

    # save
    df.to_csv("amazon_table.csv", index=False)

    print("\nSaved to amazon_table.csv")


if __name__ == "__main__":
    main()