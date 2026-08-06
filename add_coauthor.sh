#!/bin/bash

START_COMMIT=$1
END_COMMIT=$2
COAUTHOR_NAME=$3
COAUTHOR_EMAIL=$4

if [ -z "$START_COMMIT" ] || [ -z "$END_COMMIT" ]; then
  echo "Usage:"
  echo "./add_coauthor.sh <start_commit> <end_commit> \"Name\" \"email\""
  exit 1
fi

git filter-repo --commit-callback "
if commit.original_id.decode() >= b'$START_COMMIT' and commit.original_id.decode() <= b'$END_COMMIT':
    commit.message += b'\n\nCo-authored-by: $COAUTHOR_NAME <$COAUTHOR_EMAIL>\n'
"
