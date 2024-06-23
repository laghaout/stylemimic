#!/bin/sh

PACKAGE=stylemimic
CURR_DIR="."

rm *~
rm log.txt
rm -rf "$CURR_DIR"/*~
rm -rf "$CURR_DIR"/lesson*/
rm -rf "$CURR_DIR"/.pytest_cache/
rm -rf "$CURR_DIR"/__pycache__/
rm -rf "$CURR_DIR"/.ipynb_checkpoints/
rm -rf "$CURR_DIR"/*/*~
rm -rf "$CURR_DIR"/*/.pytest_cache/
rm -rf "$CURR_DIR"/*/__pycache__/
rm -rf "$CURR_DIR"/build/ "$CURR_DIR"/dist/ "$CURR_DIR"/$PACKAGE.egg-info/
