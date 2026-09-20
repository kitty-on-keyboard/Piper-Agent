## 2024-05-18 - Socket Path Truncation via strncpy
**Vulnerability:** Socket Path Truncation via strncpy in connect/bind calls.
**Learning:** Copying a socket path using `strncpy` without checking the length against the buffer size can lead to silent path truncation. This causes the program to bind or connect to a shorter, unintended path, potentially allowing attackers to hijack or spoof the socket connection.
**Prevention:** Always verify that the string length is strictly less than the destination buffer size before copying into fixed-size structures like `sockaddr_un::sun_path`.
