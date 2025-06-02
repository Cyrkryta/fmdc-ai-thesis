set -euo pipefail

if [ $# -ne 3 ]; then
  echo "Usage: $0 <MAP_DIR> <METRIC> <WARP_BASE>" >&2
  exit 1
fi

MAP_DIR="$1"    # Inputs\ nMAP_DIR="$1"    # directory containing *_<METRIC>.nii.gz
METRIC="$2"     # delta_r or absdiff
WARP_BASE="$3"  # root of processed datasets (dataset/subject/...)
TEMPLATE="${FSLDIR}/data/standard/MNI152_T1_2mm_brain.nii.gz"

echo "[1/3] Warping *_${METRIC}.nii.gz from ${MAP_DIR} into MNI space..."

# Step 1: Warp each subject map with premat from func2struct
for mapfile in "${MAP_DIR}"/*_${METRIC}.nii.gz; do
  combined=$(basename "${mapfile}" "_${METRIC}.nii.gz")
  dataset=${combined%%_*}
  subject=${combined#*_}
  premat="${WARP_BASE}/${dataset}/${subject}/func2struct.mat"
  warpfile="${WARP_BASE}/${dataset}/${subject}/T1w_to_MNI_warp.nii.gz"
  outfile="${MAP_DIR}/${combined}_${METRIC}_MNI.nii.gz"

  # Check existence of transforms
  if [ ! -f "${premat}" ]; then
    echo "Error: Premat (func2struct) not found: ${premat}" >&2
    exit 1
  fi
  if [ ! -f "${warpfile}" ]; then
    echo "Error: Warpfile not found: ${warpfile}" >&2
    exit 1
  fi

  # Apply both affine and nonlinear warps
  applywarp \
    --ref="${TEMPLATE}" \
    --in="${mapfile}" \
    --warp="${warpfile}" \
    --premat="${premat}" \
    --out="${outfile}" \
    --interp=trilinear

  # Replace any NaNs with zeros
  fslmaths "${outfile}" -nan "${outfile}"

  echo "  Warped and cleaned: ${mapfile} → ${outfile}"
done

# Step 2: Merge warped maps into 4D and compute group mean
merged="${MAP_DIR}/all_${METRIC}_MNI.nii.gz"
groupmean="${MAP_DIR}/group_${METRIC}_MNI.nii.gz"

echo "[2/3] Merging into ${merged} and computing mean → ${groupmean}..."
fslmerge -t "${merged}" "${MAP_DIR}"/*_${METRIC}_MNI.nii.gz
fslmaths "${merged}" -Tmean "${groupmean}"
 echo "  Generated group mean map: ${groupmean}"


echo "Done. All steps completed."