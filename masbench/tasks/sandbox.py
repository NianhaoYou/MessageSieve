from __future__ import annotations

import ctypes
from dataclasses import dataclass
import itertools
import os
from pathlib import Path
import re
import resource
import secrets
import shutil
import signal
import subprocess
import tempfile
import threading
from typing import Any


_SANDBOX_LAYOUT_VERSION = "masbench-python312-numpy-v2"
_SANDBOX_ROOT = Path(tempfile.gettempdir()) / f"masbench-{_SANDBOX_LAYOUT_VERSION}"
_SANDBOX_PYTHON = Path("/usr/bin/python3")
_LEGACY_SUCCESS_EXIT_CODE = 73
_SANDBOX_SUCCESS_EXIT_CODES = tuple(
    code for code in range(64, 126) if code != _LEGACY_SUCCESS_EXIT_CODE
)
_SANDBOX_UIDS = itertools.count(60_000)
_SANDBOX_BUILD_LOCK = threading.Lock()
_SANDBOX_ENV = {
    "HOME": "/work",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/bin",
    "PYTHONHASHSEED": "0",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


@dataclass(frozen=True, slots=True)
class SandboxResult:
    passed: bool
    stdout: str
    stderr: str
    timed_out: bool = False


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    resolved = source.resolve()
    try:
        os.link(resolved, destination)
    except OSError:
        shutil.copy2(resolved, destination)


def _copy_tree_readonly(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        copy_function=lambda src, dst: _link_or_copy(Path(src), Path(dst)),
        dirs_exist_ok=True,
        symlinks=False,
    )


def _runtime_path(expression: str) -> Path:
    completed = subprocess.run(
        [str(_SANDBOX_PYTHON), "-I", "-c", expression],
        check=True,
        capture_output=True,
        text=True,
        env=_SANDBOX_ENV,
    )
    return Path(completed.stdout.strip()).resolve()


def _shared_libraries(binary: Path) -> set[Path]:
    completed = subprocess.run(
        ["ldd", str(binary)],
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "LANG": "C"},
    )
    paths: set[Path] = set()
    for line in completed.stdout.splitlines():
        match = re.search(r"=>\s+(/\S+)\s+\(", line)
        if not match:
            match = re.match(r"\s*(/\S+)\s+\(", line)
        if match:
            paths.add(Path(match.group(1)))
    return paths


def _remove_sandbox_escape_modules(root: Path, stdlib_relative: Path) -> None:
    stdlib = root / stdlib_relative
    for relative in (
        "ctypes",
        "socket.py",
        "ssl.py",
    ):
        target = stdlib / relative
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    dynload = stdlib / "lib-dynload"
    for pattern in ("_ctypes*.so", "_socket*.so", "_ssl*.so"):
        for target in dynload.glob(pattern):
            target.unlink()


def _build_sandbox_root() -> Path:
    if _SANDBOX_ROOT.is_dir():
        return _SANDBOX_ROOT
    if os.geteuid() != 0:
        raise RuntimeError("Code execution requires root for the chroot sandbox")
    if not _SANDBOX_PYTHON.is_file():
        raise RuntimeError("the isolated system Python runtime is unavailable")
    with _SANDBOX_BUILD_LOCK:
        if _SANDBOX_ROOT.is_dir():
            return _SANDBOX_ROOT
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{_SANDBOX_LAYOUT_VERSION}-",
                dir=tempfile.gettempdir(),
            )
        )
        try:
            python_source = _SANDBOX_PYTHON.resolve()
            python_target = temporary / python_source.relative_to("/")
            _link_or_copy(python_source, python_target)
            alias = temporary / _SANDBOX_PYTHON.relative_to("/")
            alias.parent.mkdir(parents=True, exist_ok=True)
            alias.symlink_to(python_target.name)

            stdlib_source = _runtime_path(
                "import sysconfig; print(sysconfig.get_paths()['stdlib'])"
            )
            numpy_source = _runtime_path("import numpy; print(numpy.__path__[0])")
            numpy_libs_source = numpy_source.parent / "numpy.libs"
            _copy_tree_readonly(
                stdlib_source,
                temporary / stdlib_source.relative_to("/"),
            )
            _copy_tree_readonly(
                numpy_source,
                temporary / numpy_source.relative_to("/"),
            )
            if numpy_libs_source.is_dir():
                _copy_tree_readonly(
                    numpy_libs_source,
                    temporary / numpy_libs_source.relative_to("/"),
                )

            dynamic_objects = [python_source]
            dynamic_objects.extend(stdlib_source.rglob("*.so"))
            dynamic_objects.extend(numpy_source.rglob("*.so"))
            if numpy_libs_source.is_dir():
                dynamic_objects.extend(numpy_libs_source.rglob("*.so*"))
            libraries: set[Path] = set()
            for binary in dynamic_objects:
                libraries.update(_shared_libraries(binary))
            for library in libraries:
                _link_or_copy(library, temporary / library.relative_to("/"))

            _remove_sandbox_escape_modules(
                temporary,
                stdlib_source.relative_to("/"),
            )
            (temporary / "work").mkdir(mode=0o755)
            os.chmod(temporary, 0o755)
            os.replace(temporary, _SANDBOX_ROOT)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    return _SANDBOX_ROOT


def _isolate_network(libc: Any) -> None:
    if not hasattr(libc, "unshare"):
        raise RuntimeError("network namespace isolation is unavailable")
    if libc.unshare(0x40000000) != 0:  # CLONE_NEWNET
        raise OSError(
            ctypes.get_errno(), "unable to isolate the sandbox network namespace"
        )


def _sandbox_preexec(
    root: Path,
    work_relative: Path,
    timeout_sec: int,
    sandbox_uid: int,
) -> None:
    cpu_limit = max(1, int(timeout_sec))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + 1))
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    libc = ctypes.CDLL(None, use_errno=True)
    _isolate_network(libc)
    os.chroot(root)
    os.chdir(Path("/") / work_relative)
    os.setgroups([])
    os.setgid(sandbox_uid)
    os.setuid(sandbox_uid)
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), "unable to set no_new_privs")


def run_python_program_in_sandbox(
    script: str,
    timeout_sec: int,
    *,
    stdin_text: str = "",
    require_completion: bool = False,
) -> SandboxResult:
    root = _build_sandbox_root()
    sandbox_uid = next(_SANDBOX_UIDS)
    host_work = Path(tempfile.mkdtemp(prefix="run-", dir=root / "work"))
    os.chown(host_work, sandbox_uid, sandbox_uid)
    work_relative = host_work.relative_to(root)
    script_path = host_work / "candidate_and_tests.py"
    stdout_path = host_work / "stdout.txt"
    stderr_path = host_work / "stderr.txt"
    success_exit_code = (
        secrets.choice(_SANDBOX_SUCCESS_EXIT_CODES) if require_completion else 0
    )
    guarded_script = script.rstrip()
    if require_completion:
        guarded_script += f"\n\nraise SystemExit({success_exit_code})\n"
    script_path.write_text(guarded_script, encoding="utf-8")
    os.chmod(script_path, 0o444)
    completed: subprocess.Popen[str] | None = None
    timed_out = False
    return_code = -1
    try:
        with (
            stdout_path.open("w", encoding="utf-8") as stdout_handle,
            stderr_path.open("w", encoding="utf-8") as stderr_handle,
        ):
            completed = subprocess.Popen(
                [str(_SANDBOX_PYTHON), "-I", "candidate_and_tests.py"],
                stdin=subprocess.PIPE,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                env=_SANDBOX_ENV,
                start_new_session=True,
                preexec_fn=lambda: _sandbox_preexec(
                    root,
                    work_relative,
                    timeout_sec,
                    sandbox_uid,
                ),
            )
            try:
                completed.communicate(input=stdin_text, timeout=timeout_sec)
                return_code = int(completed.returncode or 0)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(completed.pid, signal.SIGKILL)
                completed.communicate()
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")[:4096]
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        passed = not timed_out and return_code == success_exit_code
        return SandboxResult(
            passed=passed,
            stdout=stdout,
            stderr=("TIMEOUT" if timed_out else stderr),
            timed_out=timed_out,
        )
    finally:
        if completed is not None and completed.poll() is None:
            os.killpg(completed.pid, signal.SIGKILL)
            completed.wait()
        os.chmod(script_path, 0o644)
        shutil.rmtree(host_work, ignore_errors=True)
