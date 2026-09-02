from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import wrds


CORE_REQUIREMENTS = {
    "crsp": {
        "description": "CRSP stock returns and security metadata",
        "tables": ("msf", "msenames", "mseall"),
    },
    "comp": {
        "description": "Compustat North America fundamentals",
        "tables": ("funda", "company"),
    },
    "crsp_a_ccm": {
        "description": "CRSP/Compustat Merged linking tables",
        "tables": ("ccmxpf_linktable",),
    },
}


@dataclass(frozen=True)
class LibraryCheck:
    library: str
    description: str
    available: bool
    present_tables: tuple[str, ...]
    missing_tables: tuple[str, ...]


def _normalize(items: Iterable[str]) -> set[str]:
    return {item.lower() for item in items}


def check_wrds_access() -> list[LibraryCheck]:
    """Return availability checks for the WRDS libraries FactorForge needs."""
    db = wrds.Connection()

    try:
        available_libraries = _normalize(db.list_libraries())
        results: list[LibraryCheck] = []

        for library, requirement in CORE_REQUIREMENTS.items():
            library_available = library in available_libraries
            tables = tuple(requirement["tables"])

            if library_available:
                available_tables = _normalize(db.list_tables(library=library))
                present_tables = tuple(table for table in tables if table in available_tables)
                missing_tables = tuple(table for table in tables if table not in available_tables)
            else:
                present_tables = ()
                missing_tables = tables

            results.append(
                LibraryCheck(
                    library=library,
                    description=str(requirement["description"]),
                    available=library_available,
                    present_tables=present_tables,
                    missing_tables=missing_tables,
                )
            )

        return results
    finally:
        db.close()


def print_access_report(results: list[LibraryCheck]) -> None:
    """Print a compact WRDS access report."""
    print("FactorForge WRDS Access Check")
    print("=" * 31)

    for result in results:
        status = "OK" if result.available and not result.missing_tables else "MISSING"
        print(f"\n[{status}] {result.library}: {result.description}")
        print(f"  Library available: {result.available}")
        print(f"  Present tables: {', '.join(result.present_tables) or 'none'}")
        print(f"  Missing tables: {', '.join(result.missing_tables) or 'none'}")

    missing = [
        result.library
        for result in results
        if not result.available or result.missing_tables
    ]

    if missing:
        print("\nNext step: confirm your WRDS account includes these libraries:")
        print(", ".join(missing))
    else:
        print("\nReady: your WRDS account appears to cover the core FactorForge data.")


def main() -> None:
    print_access_report(check_wrds_access())


if __name__ == "__main__":
    main()
