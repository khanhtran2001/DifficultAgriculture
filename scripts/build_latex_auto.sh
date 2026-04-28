#!/usr/bin/env bash
set -euo pipefail

workspace_root="/home/khanh/Projects/DifficultyAgri"
texinputs="${workspace_root}/docs/Research_Paper:"

target_input="${1:-}"

resolve_target() {
    local input_path="$1"

    local gce_dir="${workspace_root}/docs/GCCE_2026"

    if [[ -n "$input_path" && -f "$input_path" && "$input_path" == *.tex ]]; then
        printf '%s\n' "$input_path"
        return 0
    fi

    if [[ -n "$input_path" && -d "$input_path" ]]; then
        find "$input_path" -maxdepth 1 -type f -name '*.tex' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
        return 0
    fi

    if [[ -n "$input_path" && -f "$input_path" ]]; then
        local dir_path
        dir_path="$(dirname "$input_path")"
        local stem
        stem="$(basename "$input_path")"
        stem="${stem%.*}"

        if [[ -f "$dir_path/$stem.tex" ]]; then
            printf '%s\n' "$dir_path/$stem.tex"
            return 0
        fi

        find "$dir_path" -maxdepth 1 -type f -name '*.tex' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
        return 0
    fi

    if [[ -d "$gce_dir" ]]; then
        local gce_target
        gce_target="$(find "$gce_dir" -maxdepth 1 -type f -name 'MinImageScorer_GCCE2026_v*.tex' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
        if [[ -n "$gce_target" ]]; then
            printf '%s\n' "$gce_target"
            return 0
        fi
    fi

    find "$workspace_root" -type f -name '*.tex' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
}

target_file="$(resolve_target "$target_input")"

if [[ -z "$target_file" || ! -f "$target_file" ]]; then
    echo "No LaTeX source file found to build." >&2
    exit 1
fi

target_dir="$(dirname "$target_file")"
target_name="$(basename "$target_file")"

cd "$target_dir"
TEXINPUTS="$texinputs" pdflatex -interaction=nonstopmode -halt-on-error "$target_name"
TEXINPUTS="$texinputs" pdflatex -interaction=nonstopmode -halt-on-error "$target_name"