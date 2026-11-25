"""Sealed-set protocol limits of AI-native projects (design 05 AN-6, AN-12, AN-25): trust invariants.

They are fixed in code and are not settings: a limit an agent or a user could raise would change what the keeper
accepts and what a report promises. `tests/test_single_source.py` pins each value, and the derived ones are computed
here from the one number they depend on.
"""

from __future__ import annotations

MIB = 1024 * 1024

# The largest data file an intake accepts (AN-6): the keeper holds the whole file in memory to split and seal it.
MAX_DATA_BYTES = 64 * MIB
assert MAX_DATA_BYTES % MIB == 0                        # the text below states whole MiB
MAX_DATA_TEXT = f"{MAX_DATA_BYTES // MIB} MiB"          # the limit as the refusals state it

# The longest request line the keeper reads (AN-12). A data file travels base64-encoded (4 bytes per 3), and the rest of
# the line is the request's envelope: MAX_DATA_BYTES * 3 / 2 leaves 1/8 more than the encoding
# needs (64 MiB becomes 85 1/3 MiB in base64; the cap is 96 MiB).
MAX_REQUEST_BYTES = MAX_DATA_BYTES * 3 // 2
MAX_RESPONSE_BYTES = MAX_REQUEST_BYTES

# The longest predictions file one container run may write (AN-25, AN-9).
MAX_PREDICTIONS_BYTES = 256 * MIB
