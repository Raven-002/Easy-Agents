import re
from dataclasses import dataclass

from git import Diff, DiffIndex, Repo

HUNK_RE = re.compile(r"^@@.*?@@", re.MULTILINE)


@dataclass
class DiffHunk:
    hunk: str


@dataclass
class DiffFile:
    path_a: str
    path_b: str
    change_type: str | None
    permissions_a: str | None
    permissions_b: str | None
    is_binary: bool
    hunks: list[DiffHunk]


def git_diff(target_branch: str) -> list[DiffFile]:
    """
    Return a structured diff between the current tree and the target branch.
    Includes patch text for all files, including empty diffs.
    """
    repo = Repo()
    diff_index: DiffIndex[Diff] = repo.index.diff(target_branch, create_patch=True)
    diff_files: list[DiffFile] = []

    for diff in diff_index:
        patch_text = diff.diff.decode("utf-8", errors="ignore") if diff.diff else ""  # type: ignore
        assert isinstance(patch_text, str)
        hunks: list[DiffHunk] = []
        for m in HUNK_RE.finditer(patch_text):
            # Find the start of the hunk header
            start = m.start()
            # Find the next hunk or end of patch
            next_hunk = HUNK_RE.search(patch_text, pos=m.end())
            end = next_hunk.start() if next_hunk else len(patch_text)
            hunk_text = patch_text[start:end].strip()
            # Only include if hunk has more than the header line
            if len(hunk_text.splitlines()) > 1:
                hunks.append(DiffHunk(hunk=hunk_text))

        permissions_a = oct(diff.a_blob.mode)[-3:] if diff.a_blob else None
        permissions_b = oct(diff.b_blob.mode)[-3:] if diff.b_blob else None

        diff_file = DiffFile(
            path_a=diff.a_path or "",
            path_b=diff.b_path or "",
            change_type=diff.change_type,
            permissions_a=permissions_a,
            permissions_b=permissions_b,
            is_binary=(diff.b_blob.mime_type not in ["text/plain"] if diff.b_blob else False),
            hunks=hunks,
        )

        diff_files.append(diff_file)

    return diff_files


def summerize_diff_files(diff: list[DiffFile]) -> str:
    diff_str: str = ""
    for file in diff:
        if file.change_type == "D":
            diff_str += f"- {file.path_b} Deleted\n"
        elif file.change_type == "A":
            diff_str += f"- {file.path_a} Added\n"
        else:
            if file.permissions_a != file.permissions_b:
                diff_str += f"- {file.path_b}:{file.permissions_b} --> {file.path_a}:{file.permissions_a}\n"
            elif file.path_a != file.path_b:
                diff_str += f"- {file.path_b} --> {file.path_a}\n"
            else:
                diff_str += f"- {file.path_a} Modified\n"
    return diff_str
