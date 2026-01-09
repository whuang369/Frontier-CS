import os
import tarfile
import tempfile
import subprocess
import glob
import re


class Solution:
    def solve(self, src_path: str) -> bytes:
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                with tarfile.open(src_path, 'r:*') as tf:
                    tf.extractall(tmpdir)
            except Exception:
                return b'A' * 21

            root = tmpdir
            entries = [e for e in os.listdir(tmpdir) if not e.startswith('.')]
            if len(entries) == 1:
                single = os.path.join(tmpdir, entries[0])
                if os.path.isdir(single):
                    root = single

            harness_files = []
            for dirpath, dirnames, filenames in os.walk(root):
                for fn in filenames:
                    if not fn.endswith(('.c', '.cc', '.cpp', '.cxx', '.C')):
                        continue
                    path = os.path.join(dirpath, fn)
                    try:
                        with open(path, 'r', errors='ignore') as f:
                            content = f.read()
                    except Exception:
                        continue
                    if 'LLVMFuzzerTestOneInput' in content:
                        harness_files.append(path)

            if not harness_files:
                return b'A' * 21

            harness = harness_files[0]

            c_files = []
            cpp_files = []
            main_re = re.compile(r'\bmain\s*\(')

            for dirpath, dirnames, filenames in os.walk(root):
                for fn in filenames:
                    if not fn.endswith(('.c', '.cc', '.cpp', '.cxx', '.C')):
                        continue
                    path = os.path.join(dirpath, fn)

                    if path == harness:
                        if fn.endswith('.c'):
                            c_files.append(path)
                        else:
                            cpp_files.append(path)
                        continue

                    try:
                        with open(path, 'r', errors='ignore') as f:
                            text = f.read(4096)
                    except Exception:
                        continue

                    if 'LLVMFuzzerTestOneInput' in text:
                        continue
                    if main_re.search(text):
                        continue

                    if fn.endswith('.c'):
                        c_files.append(path)
                    else:
                        cpp_files.append(path)

            poc = self._try_libfuzzer(root, harness, c_files, cpp_files)
            if poc is not None:
                return poc

            poc = self._try_manual_asan(root, harness, c_files)
            if poc is not None:
                return poc

            return b'A' * 21

    def _have_compiler(self, name: str) -> bool:
        try:
            r = subprocess.run(
                [name, '--version'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
            )
            return r.returncode == 0
        except Exception:
            return False

    def _try_libfuzzer(self, root: str, harness: str, c_files, cpp_files):
        if not self._have_compiler('clang++'):
            return None

        build_dir = os.path.join(root, '_poc_build_fuzzer')
        os.makedirs(build_dir, exist_ok=True)
        bin_path = os.path.join(build_dir, 'fuzz_bin')

        srcs = list(dict.fromkeys([harness] + c_files + cpp_files))
        if not srcs:
            return None

        cmd = [
            'clang++',
            '-g',
            '-O1',
            '-fno-omit-frame-pointer',
            '-fsanitize=fuzzer,address',
        ] + srcs + ['-o', bin_path]

        try:
            proc = subprocess.run(
                cmd,
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
            )
        except Exception:
            return None

        if proc.returncode != 0 or not os.path.exists(bin_path):
            return None

        env = os.environ.copy()
        env.setdefault('ASAN_OPTIONS', 'detect_leaks=0')
        artifact_prefix = os.path.join(build_dir, 'crash-')

        run_cmd = [
            bin_path,
            '-max_total_time=8',
            f'-artifact_prefix={artifact_prefix}',
        ]

        try:
            subprocess.run(
                run_cmd,
                cwd=build_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                timeout=20,
            )
        except Exception:
            pass

        crash_files = sorted(glob.glob(artifact_prefix + '*'))
        for cf in crash_files:
            try:
                with open(cf, 'rb') as f:
                    data = f.read()
                if data:
                    return data
            except Exception:
                continue
        return None

    def _try_manual_asan(self, root: str, harness: str, c_files):
        if not c_files or harness not in c_files:
            return None

        compiler = None
        for c in ('clang', 'gcc'):
            if self._have_compiler(c):
                compiler = c
                break
        if compiler is None:
            return None

        build_dir = os.path.join(root, '_poc_build_asan')
        os.makedirs(build_dir, exist_ok=True)
        driver_c = os.path.join(build_dir, 'driver.c')
        bin_path = os.path.join(build_dir, 'asan_bin')

        driver_src = r'''
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>

int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size);

int main(void) {
    size_t cap = 4096;
    size_t len = 0;
    uint8_t *buf = (uint8_t*)malloc(cap);
    if (!buf) return 0;
    int c;
    while ((c = getchar()) != EOF) {
        if (len == cap) {
            size_t new_cap = cap * 2;
            uint8_t *tmp = (uint8_t*)realloc(buf, new_cap);
            if (!tmp) {
                free(buf);
                return 0;
            }
            buf = tmp;
            cap = new_cap;
        }
        buf[len++] = (uint8_t)c;
    }
    LLVMFuzzerTestOneInput(buf, len);
    free(buf);
    return 0;
}
'''
        try:
            with open(driver_c, 'w') as f:
                f.write(driver_src)
        except Exception:
            return None

        sources = list(dict.fromkeys([driver_c] + c_files))
        cmd = [
            compiler,
            '-std=c11',
            '-g',
            '-O1',
            '-fno-omit-frame-pointer',
            '-fsanitize=address',
        ] + sources + ['-o', bin_path]

        try:
            proc = subprocess.run(
                cmd,
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
            )
        except Exception:
            return None

        if proc.returncode != 0 or not os.path.exists(bin_path):
            return None

        env = os.environ.copy()
        env.setdefault('ASAN_OPTIONS', 'detect_leaks=0')

        candidates = []
        # lengths around expected PoC size plus some others
        lengths = list(range(1, 65))
        if 21 not in lengths:
            lengths.append(21)
        for L in lengths:
            candidates.append(b'\x00' * L)
            candidates.append(b'\xff' * L)
            candidates.append(b'A' * L)
            pattern = (b'ABCD' * ((L + 3) // 4))[:L]
            candidates.append(pattern)
        # a smaller set of random-like patterns based on simple deterministic sequences
        for L in range(4, 65):
            seq = bytes((i % 256 for i in range(L)))
            candidates.append(seq)

        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)

        for data in unique_candidates:
            try:
                proc = subprocess.run(
                    [bin_path],
                    input=data,
                    cwd=root,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=0.5,
                )
            except subprocess.TimeoutExpired:
                continue
            except Exception:
                continue
            if proc.returncode != 0:
                return data

        return None