"""Walking-skeleton prototype of the Composable Post-Training Framework.

prototype, unpromotable. The Tier A trainer and evaluator are stand-ins
marked skeleton-stub (MVP-SPEC section 3).
"""

__version__ = "0.4.0"

PACKAGE = "gatedtrain"
# The analysis code version (05 section 3.7, design revision 24): every registration records
# ANALYSIS_CODE_VERSION and is refused (REG-FIELD) when it differs from the running code. It names the gate
# statistics (gates.py) and the registration semantics (registration.py), not the package release. Bump
# ANALYSIS_VERSION only when that code's behaviour changes, which invalidates every existing registration; a package
# release that leaves it unchanged leaves existing registrations valid. It is not tied to __version__.
ANALYSIS_VERSION = "0.3.0"
ANALYSIS_CODE_VERSION = f"{PACKAGE} {ANALYSIS_VERSION}"

# Tags carried by every output (MVP-SPEC section 2.11).
TAGS = ("prototype", "unpromotable")
REPORT_TAGS = ("prototype", "unpromotable", "registration: unsigned")
STUB_MARK = "skeleton-stub"
NOT_ASSESSED = "not-assessed (skeleton)"
