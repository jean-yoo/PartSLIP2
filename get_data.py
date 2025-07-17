from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="HCMUE-Research/SAM-vit-h",
    repo_type="model",  # or "dataset" if it's a dataset repo, but this is a model repo
    allow_patterns="sam_vit_h_4b8939.pth",   # Only fetch the .pth file
    local_dir="models",                   # Optional: where to save locally
)