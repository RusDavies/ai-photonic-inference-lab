# Security Policy

Photonic Inference Lab is a research simulation codebase. It is not intended
to run as a network service or control physical hardware directly.

## Reporting A Vulnerability

Use the repository's private vulnerability reporting feature or private
security advisory workflow once the public repository exists.

Please include:

- affected commit or version;
- affected file and function, if known;
- a minimal reproduction or proof sketch;
- impact and assumptions;
- whether the issue can affect generated artifacts, dataset handling, or local
  execution safety.

Do not open a public issue for a vulnerability until it has been triaged.

## Supported Versions

Until the first public release is tagged, only the default branch is supported.
After tagged releases exist, this policy should be updated with the supported
release window.

## Scope

In scope:

- unsafe archive extraction or dataset handling;
- arbitrary file write/read behavior outside documented output directories;
- command-line behavior that can unexpectedly execute external programs;
- dependency or packaging issues that materially affect local execution safety.

Out of scope:

- claims that require access to private infrastructure;
- denial-of-service claims based only on intentionally long-running research
  experiments;
- hardware-safety issues, because this repository does not control hardware.
