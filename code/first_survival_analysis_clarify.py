import lifelines
import pandas as pd

gt_df = pd.read_excel("F:/CLAUDIO/BREAST_CANCER_DATASETS/CLARIFY JANUARY 2024/unified_clinical_info_CBDC_jan2024.xlsx")

gt_eofus = gt_df["EOFUS"].unique()
gt_meta = gt_df["Meta_life"].unique()

print("GHola")