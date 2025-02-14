"""
RAVE Latent Space Analysis to PCA Coordinates
Consolidated version that supports exporting either a 2D or 3D PCA mapping (or any n-dimensional mapping)
from the RAVE latent space.

Author: Moisés Horta Valenzuela
Last Modified: 11/02/2025
"""

import argparse
import torch
import librosa as li
import numpy as np
from tqdm import tqdm
import os
import json

# For PCA and T-SNE:
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# For UMAP (make sure to install: pip install umap-learn)
from umap import UMAP

def main():
    # Set up command-line arguments
    parser = argparse.ArgumentParser(
        description="RAVE Latent Space Analysis to Dimensionality Reduced Coordinates and JSON Output (supports PCA, T-SNE, and UMAP)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pre-trained RAVE model (TorchScript file)"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Directory containing the .wav audio dataset"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Filename for output JSON data. If not provided, a default name is used."
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=2,
        help="Number of components to compute (default: 2 for 2D output, or 3 for 3D output)"
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=48000,
        help="Sampling rate for loading audio files (default: 48000)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="pca",
        help="Dimensionality reduction method: 'pca', 'tsne', or 'umap' (default: pca)"
    )
    args = parser.parse_args()

    # Set device: GPU if available, otherwise CPU.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the pre-trained RAVE model.
    print(f"Loading RAVE model from {args.model_path}...")
    rave = torch.jit.load(args.model_path).to(device)
    rave.eval()

    # List all .wav files in the specified audio directory.
    audio_files = [
        os.path.join(args.audio_dir, f)
        for f in os.listdir(args.audio_dir)
        if f.lower().endswith('.wav')
    ]
    if not audio_files:
        print(f"No audio files found in directory: {args.audio_dir}")
        return

    # Initialize a list to store latent vectors from all files.
    latent_vectors = []

    print("Encoding audio files to latent space...")
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        # Load audio file using librosa.
        x, _ = li.load(audio_file, sr=args.sr, mono=True)
        # Reshape to (1, 1, samples) and move to the device.
        x = torch.from_numpy(x).reshape(1, 1, -1).float().to(device)

        # Encode the audio into latent representation using the RAVE model.
        with torch.no_grad():
            z = rave.encode(x)  # Expected shape: (1, n_dimensions, encoded_sample_length)
            z = z.cpu().numpy().squeeze()  # Remove batch dimension: (n_dimensions, encoded_sample_length)
            latent_vectors.append(z)

    # Concatenate latent vectors along the time axis and transpose:
    # Final shape: (total_samples, n_dimensions)
    latent_vectors = np.concatenate(latent_vectors, axis=-1).T

    # Dimensionality reduction.
    method = args.method.lower()
    if method == "tsne":
        print("Performing T-SNE on latent vectors...")
        tsne = TSNE(n_components=args.n_components, random_state=42)
        latent_reduced = tsne.fit_transform(latent_vectors)
        # Scale T-SNE coordinates so that the maximum absolute value is 5.
        max_val = np.abs(latent_reduced).max()
        if max_val != 0:
            latent_reduced = latent_reduced * (5.0 / max_val)
        print("Scaled T-SNE coordinates to a maximum of 5.")
    elif method == "umap":
        print("Performing UMAP on latent vectors...")
        umap_model = UMAP(n_components=args.n_components, random_state=42)
        latent_reduced = umap_model.fit_transform(latent_vectors)
        # Scale UMAP coordinates so that the maximum absolute value is 5.
        max_val = np.abs(latent_reduced).max()
        if max_val != 0:
            latent_reduced = latent_reduced * (5.0 / max_val)
        print("Scaled UMAP coordinates to a maximum of 5.")
    else:
        print("Performing PCA on latent vectors...")
        pca = PCA(n_components=args.n_components)
        latent_reduced = pca.fit_transform(latent_vectors)

    # Prepare the JSON data.
    output_data = {"mapping": []}
    for red_coord, latent_coord in zip(latent_reduced, latent_vectors):
        if args.n_components == 2:
            coord_dict = {
                "Reduced coordinate": {
                    "x": float(red_coord[0]),
                    "y": float(red_coord[1])
                },
                "Latent coordinate": latent_coord.tolist()
            }
        elif args.n_components == 3:
            coord_dict = {
                "Reduced coordinate": {
                    "x": float(red_coord[0]),
                    "y": float(red_coord[1]),
                    "z": float(red_coord[2])
                },
                "Latent coordinate": latent_coord.tolist()
            }
        else:
            coord_dict = {
                "Reduced coordinate": {f"component_{i}": float(val) for i, val in enumerate(red_coord)},
                "Latent coordinate": latent_coord.tolist()
            }
        output_data["mapping"].append(coord_dict)

    # Determine the output JSON filename.
    if args.output_json is None:
        base_name = os.path.basename(args.model_path).split('.')[0]
        output_json = f"{args.n_components}D_{method}_latent_mapping_{base_name}.json"
    else:
        output_json = args.output_json

    # Save the output data to a JSON file.
    with open(output_json, "w") as json_file:
        json.dump(output_data, json_file, indent=4)

    print(f"Dimensionality reduced latent mapping data saved to {output_json}")

if __name__ == "__main__":
    main()
