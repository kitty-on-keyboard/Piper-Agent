# Security

Please report vulnerabilities privately. Do not open a public issue for a crash that
escapes the workspace sandbox, a prompt injection that runs a command the operator did
not approve, or a path that reads files outside the workspace root.

Use GitHub's **Report a vulnerability** on this repository
(Security → Advisories → New draft security advisory).

This project is a local coding agent: it can run shell commands and edit files in the
workspace you open, subject to approval cards. Treat an untrusted workspace the way you
would treat untrusted source.
