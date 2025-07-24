import glob
import os

from omegaconf import OmegaConf

license_url_files = glob.glob(
    os.path.expandvars("$REPO_ROOT/src/organize/license_configs/urls/*.yaml")
)

LICENSE_URL = []
for file in license_url_files:
    LICENSE_URL.extend(OmegaConf.load(file))

LICENSE_DETAILS = OmegaConf.load(
    os.path.expandvars("$REPO_ROOT/src/organize/license_configs/licenses.yaml")
)

URL_TO_LICENSE_FAMILY = {item.url: item.license_family for item in LICENSE_URL}

URL_TO_LICENSE = {item.url: item.license for item in LICENSE_URL}

from enum import StrEnum


class LicenseUsage(StrEnum):
    COMMERCIAL_DERIVATIVE_OK = "com"
    NONCOMMERCIAL_DERIVATIVE_OK = "nc"
    NO_DERIVATIVE = "nd"


def url_to_license(url: str) -> str:
    """Converts a license URL to its short name (e.g., 'cc-by-sa-4.0')."""
    return URL_TO_LICENSE[url]


def url_to_license_family(url: str) -> str:
    """Converts a license URL to its family (e.g., 'CC-BY-SA')."""
    url = url.replace("https://", "http://")
    return URL_TO_LICENSE_FAMILY[url]


def check_license_derivative_ok(license_family):
    """Checks if a license family permits derivative works."""
    return LICENSE_DETAILS[license_family].derivative


def check_license_commercial_ok(license_family):
    """Checks if a license family permits commercial use."""
    return LICENSE_DETAILS[license_family].commercial


def license_family_to_usage(license_family: str) -> LicenseUsage:
    """Determines the usage category (com, nc, nd) from a license family."""
    if check_license_derivative_ok(license_family):
        if check_license_commercial_ok(license_family):
            return LicenseUsage.COMMERCIAL_DERIVATIVE_OK
        else:
            return LicenseUsage.NONCOMMERCIAL_DERIVATIVE_OK
    else:
        return LicenseUsage.NO_DERIVATIVE
