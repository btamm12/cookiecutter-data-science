from {{cookiecutter.pkg_name}} import constants


def full_path(path: str, make_dir: bool = True):
    """Returns the full path from a path that is relative to the repository."""
    full_path = constants.DIR_REPO.joinpath(path)
    if make_dir:
        full_path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    return str(full_path)
