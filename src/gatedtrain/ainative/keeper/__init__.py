"""The keeper: a daemon that alone holds the sealed items and signs the trust records, reached over a Unix socket
(Linux only; tests/README.md section 13, AN-12 to AN-19). It runs under the owner's own OS user (same-user mode, the
default: the procedure is followed, nothing is sealed from the account) or under its own (separate-user mode, the
sealed-data guarantee); the mode is observed, never chosen (AN-27)."""
