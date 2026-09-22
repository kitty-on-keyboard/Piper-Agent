1. **Identify Performance Improvement**: Looking at `src/tools/registry.cpp`, the `read_slice` function formats integer line numbers using `std::to_string(line)` inside a tight `while` loop (around line 1120). As documented in `.jules/bolt.md` under `2026-05-26 - Zero-Allocation Line Numbering for Tool File Observations`, using `std::to_string(line)` triggers frequent heap allocations per read. We can eliminate these allocations by using `std::to_chars` with a stack buffer, mirroring the optimization previously applied to `number_lines()`.

2. **Implement Zero-Allocation Numbering**:
   - In `src/tools/registry.cpp`, modify the loop in `read_slice`.
   - Add a stack buffer: `char num_buf[32];` outside the loop.
   - Replace `outp += std::to_string(line);` with:
     ```cpp
     const auto [ptr, ec] = std::to_chars(num_buf, num_buf + sizeof(num_buf), line);
     outp.append(num_buf, static_cast<std::size_t>(ptr - num_buf));
     ```

3. **Verify the change**: Rebuild the C++ component to verify the change compiles and passes tests:
   ```bash
   cd build && cmake .. && make -j$(nproc) test_registry && ctest -R test_registry
   ```
   (I will discover the appropriate test target name or use a general compilation check).

4. **Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.**

5. **Create a PR (or commit)**: Create a commit with title `⚡ Bolt: Replace std::to_string with std::to_chars in read_slice for zero-allocation line numbering` and a description outlining the performance benefit.
