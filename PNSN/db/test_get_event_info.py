#!/usr/bin/env python3

# ./test_get_event_info.py 61886506

import sys
from get_event_info import get_event_info

evid = sys.argv[1]

orid, ordate, lat, lon, dep, strmag, mindist, maxdist, netstas, distkm, analyst_class = get_event_info(evid)

print(orid, ordate, lat, lon, dep, strmag, mindist, maxdist, netstas, distkm, analyst_class)

