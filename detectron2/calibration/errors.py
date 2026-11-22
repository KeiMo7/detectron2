# Copyright (c) Facebook, Inc. and its affiliates.
"""Exception types for the score-calibration package."""


class CalibrationError(Exception):
    """Base class for all calibration-related errors."""


class DegenerateFitError(CalibrationError):
    """Raised when a calibration fit has no usable signal (e.g. a single label value)."""


class CalibrationRangeError(CalibrationError):
    """Raised when calibrated outputs fall outside [0, 1].

    Carries the *indices* of the offending classes. Class names are
    deliberately not included so the error can be logged safely.
    """

    def __init__(self, class_indices):
        self.class_indices = sorted(int(c) for c in class_indices)
        super().__init__(
            "calibrated scores out of range [0, 1] for class indices: "
            + ", ".join(str(c) for c in self.class_indices)
        )


class CalibrationFormatError(CalibrationError):
    """Raised when a serialized calibration artifact is malformed."""


class CalibrationVersionError(CalibrationError):
    """Raised when a serialized calibration artifact has an unsupported version."""


class CalibrationMismatchError(CalibrationError):
    """Raised when the class set stored in a calibration artifact does not
    match the class set expected at load time.

    Only counts and positions are exposed in the message; the stored and
    expected class lists are available as attributes for programmatic use.
    """

    def __init__(self, stored_classes, expected_classes):
        self.stored_classes = list(stored_classes)
        self.expected_classes = list(expected_classes)
        first_diff = next(
            (
                i
                for i, (a, b) in enumerate(zip(self.stored_classes, self.expected_classes))
                if a != b
            ),
            min(len(self.stored_classes), len(self.expected_classes)),
        )
        self.first_diff = first_diff
        super().__init__(
            "calibration class set mismatch: artifact holds "
            f"{len(self.stored_classes)} classes, expected {len(self.expected_classes)} "
            f"classes; first difference at position {first_diff}. "
            "Refusing to align classes by position."
        )
