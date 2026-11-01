"""Tests for tools/upload_hf.py, with the Hugging Face API replaced by an in-memory fake."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.upload_hf import main, upload_checkpoints  # noqa: E402


class _FakeHfApi:
    def __init__(self, existing=()):
        self.files = set(existing)
        self.uploads = []

    def file_exists(self, repo_id, filename, repo_type=None):
        return filename in self.files

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):
        self.uploads.append((path_or_fileobj, path_in_repo, repo_id))
        self.files.add(path_in_repo)


def _export(run_dir, *steps):
    export_dir = run_dir / "ckpts" / "exported"
    export_dir.mkdir(parents=True)
    for step in steps:
        (export_dir / f"model_{step}.safetensors").write_bytes(b"weights")
    return str(export_dir)


def test_uploads_new_steps_and_skips_existing(tmp_path):
    export_dir = _export(tmp_path, 1000, 2000)
    api = _FakeHfApi(existing={"run-a/model_1000.safetensors"})
    urls = upload_checkpoints(api, "u/repo", "run-a", export_dir, [1000, 2000])

    assert [path for _, path, _ in api.uploads] == ["run-a/model_2000.safetensors"]
    assert set(urls) == {1000, 2000}
    assert urls[2000] == "https://huggingface.co/u/repo/blob/main/run-a/model_2000.safetensors"


def test_missing_export_uploads_nothing(tmp_path):
    export_dir = _export(tmp_path, 1000)
    api = _FakeHfApi()
    with pytest.raises(FileNotFoundError, match="model_2000"):
        upload_checkpoints(api, "u/repo", "run-a", export_dir, [1000, 2000])
    assert api.uploads == []


def test_main_names_the_folder_after_the_run(tmp_path, monkeypatch):
    import huggingface_hub

    run_dir = tmp_path / "b2-jointdisp"
    _export(run_dir, 1000)
    api = _FakeHfApi()
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    main(["--output-dir", f"{run_dir}/", "--steps", "1000", "--repo", "u/repo"])
    assert api.uploads[0][1:] == ("b2-jointdisp/model_1000.safetensors", "u/repo")
