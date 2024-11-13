"""Convert CalDAV calendars to ICS files.

Fetches all entries from a CalDAV server
 and saves each calendar as a separate ICS file.
For faster access, data is cached locally,
with all password details stripped before saving.
Re-syncing with the server requires a password.

To integrate with Google Calendar,
either upload the ICS files or host them publicly
(consider basic auth or similar for access control;
for strong security, use a different method).

Example config (`config.toml`):
```toml
url = "https://example.com/remote.php/dav/calendars/user/"
user = "user"
save-dir = "calendars"
```

Usage:
    >>> python caldav2ics.py config.toml
    Password: <password>
    >>> cat .password | python caldav2ics.py config.toml -s

Creates:
```
calendars/
    calendar1.ics
    calendar2.ics
    cache.pkl
```
"""

from __future__ import annotations

import argparse
import getpass
import logging
import pickle
import string
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import msgspec
import requests.auth
from caldav.davclient import DAVClient
from caldav.objects import SynchronizableCalendarObjectCollection
from icalendar import Calendar, Event
from typing_extensions import Self, TypeGuard

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Sequence


logger = logging.getLogger(__name__)


def _typeguard_cache(
    cache: Any,
) -> TypeGuard[dict[str, SynchronizableCalendarObjectCollection]]:
    if not isinstance(cache, dict):
        raise TypeError("Cache is not a dict")
    if not all(
        isinstance(key, str)
        and isinstance(value, SynchronizableCalendarObjectCollection)
        for key, value in cache.items()
    ):
        raise TypeError(
            "Cache is not a dict[str, SynchronizableCalendarObjectCollection]"
        )
    return True


Config = TypedDict("Config", {"url": str, "user": str, "save-dir": str})


class CalendarCache(dict[str, SynchronizableCalendarObjectCollection]):
    """Cache for calendars."""

    def __init__(self, cals: Iterable[Calendar]) -> None:
        """Initialize Cache with an Iterable of syncable calendars."""
        for cal in cals:
            start = time.perf_counter()
            elem = cal.objects(load_objects=True)
            remove_passwd(elem)
            self[cal.name] = elem
            logger.info(
                "Loaded %s objects from %s in %ss",
                len(self[cal.name]),
                cal.name,
                round(time.perf_counter() - start, 2),
            )

    @classmethod
    def _load(cls, cache: Any) -> Self:
        """Load cache from `path`."""
        if _typeguard_cache(cache):
            obj = cls.__new__(cls)
            for key, value in cache.items():
                obj[key] = value

            return obj
        raise ValueError("Invalid cache")

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load cache from `path`."""
        cache = pickle.loads(Path(path).read_bytes())
        return cls._load(cache)

    def save(self, path: str | Path) -> None:
        """Save cache to `path`."""
        parent = dict(self.items())
        Path(path).write_bytes(pickle.dumps(parent))

    def sync(self, passwd: str) -> None:
        """Sync cache with server."""
        n_updated = 0
        start = time.monotonic()
        for obj in self.values():
            with temp_passwd(obj, passwd):
                updated = obj.sync()
            obj.objects = list(obj.objects)
            n_updated += len(updated[0])
        logger.info(
            "Updated %s objects in %ss", n_updated, round(time.monotonic() - start, 2)
        )


def remove_passwd(elem: SynchronizableCalendarObjectCollection) -> None:
    """Remove password from `elem`."""
    client: DAVClient = elem.calendar.client
    client.auth = None
    client.password = None


@contextmanager
def temp_passwd(
    elem: SynchronizableCalendarObjectCollection, passwd: str
) -> Generator[None]:
    """Temporarily set password to `passwd`."""
    client: DAVClient = elem.calendar.client
    if client.username is None:
        raise ValueError("No username set")
    try:
        client.password = passwd
        client.auth = requests.auth.HTTPBasicAuth(client.username, client.password)
        yield
    finally:
        remove_passwd(elem)


def isalnum(c: str) -> bool:
    """Is `c` an alphanumeric character?"""
    return c in set(string.ascii_letters + string.digits)


def get_options(name: str, owner: str = "Owner") -> tuple[str, dict[str, str]]:
    """Get options for ics file."""
    alnum = "".join([c for c in name if isalnum(c)])
    return alnum, {
        "version": "2.0",
        "prodid": f"-//{owner}//{alnum}//EN",
        "calscale": "GREGORIAN",
        "method": "PUBLISH",
        "refresh-interval;value=duration": "PT15M",
        "x-published-ttl": "PT15M",
        "x-wr-calname": name,
        "x-wr-caldesc": name,
    }


def write_calendar(
    name: str, cal: SynchronizableCalendarObjectCollection, owner: str, save_dir: Path
) -> None:
    """Create and write ics file."""
    calendar = Calendar()
    timezones_cache = []

    alnum, options = get_options(name, owner)
    for key, value in options.items():
        calendar.add(key, value)

    events = []
    for obj in cal:
        cal_ = Calendar.from_ical(obj._data)  # noqa: SLF001
        for timezone in cal_.walk("VTIMEZONE"):
            if timezone["tzid"] not in timezones_cache:
                timezones_cache.append(timezone["tzid"])
                calendar.add_component(timezone)
        for event in cal_.walk("VEVENT"):
            event_copy = Event(event)
            events.append(event_copy)

    for event in events:
        # event_copy.add('categories', category) # noqa: ERA001
        calendar.add_component(event)

    cal_path = save_dir / (alnum + ".ics")
    cal_path.write_bytes(calendar.to_ical())


def create_cache(url: str, user: str, passwd: str) -> CalendarCache:
    """Create a new cache."""
    client = DAVClient(url=url, username=user, password=passwd)
    principal = client.principal()  # type: ignore[no-untyped-call]
    calendars = principal.calendars()
    return CalendarCache(calendars)


def update_cache(config: Config, passwd: str) -> None:
    """Updates existing cache, else creates a new one."""
    url, user, save_dir = config["url"], config["user"], config["save-dir"]
    save_dir_path = Path(save_dir)
    cache_path = save_dir_path / "cache.pkl"
    logger.info("Cache path: %s", cache_path)

    if not cache_path.exists():
        logger.info("Creating cache...")
        cache = create_cache(url, user, passwd)

    else:
        logger.info("Loading cache...")
        cache = CalendarCache.load(cache_path)
        cache.sync(passwd)

    cache.save(cache_path)

    start = time.monotonic()
    for name, cal in cache.items():
        write_calendar(name, cal, user, save_dir_path)
    logger.info("Wrote calendars in %s seconds", round(time.monotonic() - start, 2))


def parse_args(argv: Sequence[str] | None = None) -> tuple[dict[str, Config], str]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to config file, containing blocks with url, user, and save-dir.",
    )
    parser.add_argument(
        "-s",
        "--stdin",
        action="store_true",
        help="Read password from stdin. If not provided, will prompt for password.",
        default=False,
    )
    args = parser.parse_args(argv)
    config_path: Path = args.config

    if args.stdin:
        # remove the newline character
        # don't strip as spaces could be part of the password
        passwd = sys.stdin.read().removesuffix("\n")
    else:
        passwd = getpass.getpass("Password: ")
        if not passwd:
            raise ValueError("Password not provided")

    try:
        content = config_path.read_bytes()
        unvalidated = msgspec.toml.decode(content)
        configs = msgspec.convert(unvalidated, dict[str, Config])
    except (FileNotFoundError, msgspec.ValidationError) as e:
        raise ValueError(f"Invalid config file: {e}") from e

    for value in configs.values():
        save_dir_path = Path(value["save-dir"])
        if not save_dir_path.is_absolute():
            raise ValueError("Save directory must be an absolute path")
        if save_dir_path.is_file():
            raise ValueError("Save directory must be a directory or non existent")
        save_dir_path.mkdir(parents=True, exist_ok=True)
    return configs, passwd


def main(argv: Sequence[str] | None = None) -> None:
    """Main entrypoint."""
    logging.basicConfig(level=logging.INFO)

    config, passwd = parse_args(argv)

    for server, conf in config.items():
        start = time.monotonic()
        update_cache(conf, passwd)
        logger.info(
            "Finished %s in %s seconds", server, round(time.monotonic() - start, 2)
        )


if __name__ == "__main__":
    main()


__all__ = ["main"]
