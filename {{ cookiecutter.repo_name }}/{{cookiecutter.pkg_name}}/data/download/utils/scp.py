from paramiko import AutoAddPolicy, SSHClient
from scp import SCPClient
import sys


# Define progress callback that prints the current percentage completed for the file
def _progress(filename, size, sent):
    sys.stdout.write("%s: %.2f%%   \r" % (filename, float(sent) / float(size) * 100))


def _connect_ssh(
    target_ssh_url: str,
    jump_ssh_url: str = None,
    username: str = None,
    password: str = None,
    verbose: bool = True,
):
    if verbose:
        print(f"Connecting to SSH server: {target_ssh_url} ...")

    target_ssh = SSHClient()
    target_ssh.set_missing_host_key_policy(AutoAddPolicy())

    if jump_ssh_url is None:
        jump_ssh = None
        jump_channel = None
    else:
        # Source: https://gist.github.com/tintoy/443c42ea3865680cd624039c4bb46219
        jump_ssh = SSHClient()
        jump_ssh.set_missing_host_key_policy(AutoAddPolicy())
        jump_ssh.connect(jump_ssh_url, username=username, password=password)

        jump_transport = jump_ssh.get_transport()
        src_addr = (jump_ssh_url, 22)
        dest_addr = (target_ssh_url, 22)
        jump_channel = jump_transport.open_channel("direct-tcpip", dest_addr, src_addr)

    target_ssh.connect(
        target_ssh_url, username=username, password=password, sock=jump_channel,
    )

    # ------------------------------------------------------------------------

    if verbose:
        print("Connection successful.")

    return target_ssh, jump_ssh


def _close_ssh(target_ssh: SSHClient, jump_ssh: SSHClient = None):
    target_ssh.close()
    if jump_ssh is not None:
        jump_ssh.close()


def _scp(
    target_ssh: SSHClient,
    remote_path: str,
    local_path: str,
    recursive: bool = False,
    verbose: bool = True,
):

    if verbose:
        print("Downloading data from SSH server...")

    # Download int16 audio data.
    scp = SCPClient(target_ssh.get_transport(), progress=_progress)
    scp.get(
        remote_path=remote_path, local_path=local_path, recursive=recursive,
    )
    scp.close()


def scp_file(
    target_ssh_url: str,
    remote_path: str,
    local_path: str,
    jump_ssh_url: str = None,
    username: str = None,
    password: str = None,
    verbose: bool = True,
):

    # Create SSH connection.
    target_ssh, jump_ssh = _connect_ssh(
        target_ssh_url, jump_ssh_url, username, password, verbose
    )

    # Perform SCP.
    _scp(target_ssh, remote_path, local_path, recursive=False, verbose=verbose)

    # Close SSH connection.
    _close_ssh(target_ssh, jump_ssh)


def scp_dir(
    target_ssh_url: str,
    remote_path: str,
    local_path: str,
    jump_ssh_url: str = None,
    username: str = None,
    password: str = None,
    verbose: bool = True,
):

    # Create SSH connection.
    target_ssh, jump_ssh = _connect_ssh(
        target_ssh_url, jump_ssh_url, username, password, verbose
    )

    # Perform SCP.
    _scp(target_ssh, remote_path, local_path, recursive=True, verbose=verbose)

    # Close SSH connection.
    _close_ssh(target_ssh, jump_ssh)

