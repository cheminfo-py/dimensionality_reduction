# -*- coding: utf-8 -*-
"""Test if the API responds in the Docker image"""
import sys

import requests


r = requests.post("http://localhost:8091/version/")
keys = r.json().keys()

if "version" in keys:
    print("PXRD prediction API works")
    sys.exit(0)
else:
    sys.exit(1)
