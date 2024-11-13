# CalDAV2ICS

A simple script to download all calendars from a CalDAV server and save them as .ics files.

# Installation

```bash
pip install .
```

# Usage
- Create a config .toml file
```toml
[server_a]
url = "https://caldav.server_a.com/calendar/remote.php/dav/calendars/user/"
username = "user"
save-dir = "calendars/server_a"

[calendar_b]
...
```
- Run the script
```bash
caldav2ics -c config.toml
```
or without to be prompted for the password
```bash
cat .password | caldav2ics -c config.toml
```

# TODO
- tests
