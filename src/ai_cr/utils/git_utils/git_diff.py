from git import Diff, DiffIndex, Repo


def git_diff(target_branch: str) -> DiffIndex[Diff]:
    """Get diff between current branch and target branch."""
    repo = Repo()
    return repo.git.diff(target_branch)
