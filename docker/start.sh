#! /bin/sh

/usr/local/bin/adduser2

exec tini -g -- "$@"
