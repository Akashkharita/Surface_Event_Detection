#!/usr/bin/env python3

# Given an evid and a psycopg2 connection to the AQMS db, return 
#   event info and a list of the 10 net.sta with the earliest picks.
#   Uses P picks for Binder and Jiggle events, and the picks that 
#   go into subnet triggers.
#  Alex Hutko, UW 2024

import os
import psycopg2
from datetime import datetime, timedelta

#----- Convert unix/epoc time to truetime which includes leap seconds.
#      Input is a timestamp, e.g. 1568130188.11
#      Output is a datetime object, e.g. datetime.datetime(2019, 9, 10, 15, 42, 41, 117380)
def unix_to_true_time(unixtime):
    leap_seconds = {2272060800: 10 ,2287785600: 11 ,2303683200: 12 ,2335219200: 13 ,2366755200: 14 ,2398291200: 15 ,2429913600: 16 ,2461449600: 17 ,2492985600: 18 ,2524521600: 19 ,2571782400: 20 ,2603318400: 21 ,2634854400: 22 ,2698012800: 23 ,2776982400: 24 ,2840140800: 25 ,2871676800: 26 ,2918937600: 27 ,2950473600: 28 ,2982009600: 29 ,3029443200: 30 ,3076704000: 31 ,3124137600: 32, 3345062400: 33 ,3439756800: 34 ,3550089600: 35 ,3644697600: 36 ,3692217600: 37}
    time1900 = unixtime + 2208988800
    seconds_to_sub = 0
    for utime in leap_seconds:
        if ( time1900 >= utime ):
            seconds_to_sub = leap_seconds[utime] - 10
    t = datetime.utcfromtimestamp(unixtime) - timedelta(seconds=seconds_to_sub)
    return t

def get_event_info(evid):

    analyst_class = 'N/A'

    #----- Connect to database

    dbname = os.environ['AQMS_DB']
    dbuser = os.environ['AQMS_USER']
    hostname = os.environ['AQMS_HOST2']
    dbpass = os.environ['AQMS_PASSWD']

    conn = psycopg2.connect(dbname=dbname, user=dbuser, host=hostname, password=dbpass)
    cursor = conn.cursor()

    evinfo = {}
    # select a.evid, t.net, t.sta, t.location, t.seedchan, to_timestamp(t.datetime) at time zone 'utc' from trig_channel t inner join nettrig n on t.ntid = n.ntid inner join assocnte a on n.ntid = a.ntid where a.evid = 62027876 and t.datetime > 0 order by t.datetime;
    # Get event type: event or subnet trigger

    cursor.execute('select etype, prefmag from event where evid = (%s)',(evid,))
    for record in cursor:
        etype = record[0]
        prefmag = record[1]

    #----- Binder or Jiggle (HYPO) events of any eventtype
    netstas, distkm = [], []
    if etype != 'st':
        if prefmag is not None:
            cursor.execute('select o.evid,  o.orid, o.datetime, o.lat, o.lon, o.depth, o.distance, o.wrms, o.algorithm, o.rflag, n.magnitude, n.magtype, n.uncertainty, n.nsta, n.magalgo from origin o inner join event e on o.evid = e.evid inner join netmag n on n.magid = e.prefmag where e.evid = (%s) order by o.datetime desc', (  evid, ) )
            for record in cursor:
                evid = record[0]
                orid = record[1]
                odate = unix_to_true_time(record[2])
                lat = record[3]
                lon = record[4]
                dep = record[5]
                mindist = record[6]
                orms = record[7]
                mag = record[10]
                unc = record[11]
                nsta = record[12]
                magalgo = record[13]
                algorithm = record[8]
                if 'HYP' in algorithm:
                    analyst_class = etype
            #----- Get preferred magnitude
            try:
                cursor.execute('select n.magnitude, n.magtype from netmag n inner join event e on n.magid = e.prefmag where e.evid = (%s) and e.selectflag = (%s)',(evid,1))
                for record in cursor:
                    prefmag = record[0]
                    magtype = record[1]
            except:
                prefmag = mag
                magtype = '_'
            strmag = 'M{:1s}{:3.1f}'.format(magtype,prefmag)
        else:
            cursor.execute('select o.evid,  o.orid, o.datetime, o.lat, o.lon, o.depth, o.distance, o.wrms, o.algorithm, o.rflag from origin o inner join event e on o.evid = e.evid where e.evid = (%s) order by o.datetime desc', (  evid, ) )
            for record in cursor:
                evid = record[0]
                orid = record[1]
                odate = unix_to_true_time(record[2])
                lat = record[3]
                lon = record[4]
                dep = record[5]
                mindist = record[6]
                orms = record[7]
                mag, unc, nsta, magalgo, magtype, strmag = 0., 0., 0, 'None', '_', 'Md0.0'
        evinfo[evid] = [ orid, odate, lat, lon, dep, mindist, orms, mag, unc, nsta, magalgo ]

        #----- Go through each evid, find the associated arrivals, get 10 closest P picks
        for evid in evinfo:
            netstas, distkm, ncount, mindist, maxdist = [], [], 0, -99, -99
            orid = evinfo[evid][0]
            ordate = evinfo[evid][1]
            lat = evinfo[evid][2]
            lon = evinfo[evid][3]
            dep = evinfo[evid][4]
            mag = evinfo[evid][7]
            cursor.execute('select a.net, a.sta, a.location, a.seedchan, a.datetime, a.iphase, a.qual, a.quality, s.importance, s.wgt, s.timeres, s.delta, s.seaz from arrival a inner join assocaro s on a.arid = s.arid where a.iphase = (%s) and s.orid = (%s) order by s.delta',('P',orid,))
            for record in cursor:
                adate = unix_to_true_time(record[4])
                if ncount == 0:
                    mindist = record[11]
                if ncount <= 10 and record[0] != 'US':
                    ncount += 1
                    netstas.append( record[0] + '.' + record[1] )
                    distkm.append(round(record[11]))
                    maxdist = record[11]
    
    #----- Subnet trigger
    elif etype == 'st':
        cursor.execute('select t.net, t.sta, t.datetime from trig_channel t inner join nettrig n on t.ntid = n.ntid inner join assocnte a on n.ntid = a.ntid where a.evid = (%s) and t.datetime > 0 order by t.datetime',(evid,))
        n = 0
        for record in cursor:
            if n == 0:
                ordate = unix_to_true_time(record[2])
            if n < 10:
                netsta = record[0] + '.' + record[1]
                netstas.append(netsta)
                distkm.append(0)
                n += 1
        orid, lat, lon, dep, strmag, mindist, maxdist = 0, 0, 0, 0, 'Md0.0', 0, 0

    return (orid, ordate, lat, lon, dep, strmag, mindist, maxdist, netstas, distkm, analyst_class)

