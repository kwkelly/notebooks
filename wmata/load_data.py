import bz2
import pandas as pd
import json
import http.client
import urllib.request
import urllib.parse
import urllib.error
import itertools
import api_key
import datetime
import os

#eastern = pytz.timezone('US/Eastern')
headers = {
    # Request headers
    'api_key': api_key.wmata_key,
}

# =================================================

def query_api(endpoint, params, headers):
    try:
        conn = http.client.HTTPSConnection('api.wmata.com')
        conn.request("GET", endpoint % params, "{body}", headers)
        response = conn.getresponse()
        data = response.read().decode('utf-8')
        parsed = json.loads(data)
        conn.close()
        return parsed
    except Exception as e:
        print("[Errno {0}] {1}".format(e.errno, e.strerror))


def get_station_list(line_code):
    # Positions
    params = urllib.parse.urlencode({
        'LineCode': line_code,
    })
    return query_api("/Rail.svc/json/jStations?%s",
                     params, headers)


def get_circuits():
    # Cricuits
    params = urllib.parse.urlencode({
    })
    return query_api("/TrainPositions/TrackCircuits?contentType=json&%s",
                     params, headers)


def get_routes():
    # Standard Routes
    params = urllib.parse.urlencode({
    })
    return query_api("/TrainPositions/StandardRoutes?contentType=json&%s",
                     params, headers)



def date_parse(unix_time):
    # returns the datetime in local time from the unix timestamp
    t = datetime.datetime.fromtimestamp(float(unix_time))
    return t


class train_info_maps():

    def __init__(self):
        self.routes = get_routes()
        self.circuits = get_circuits()
        # get all the lines
        self.line_codes = ['RD', 'YL', 'BL', 'OR', 'GR', 'SV']
        self.track_nums = [1, 2]
        self.directions = [1, 2]
        self.stations_dict = self.create_stations_dict()
        self.circ_seq_dict = self.get_circuit_to_sequence_dict()
        self.circ_track_dict = self.get_circuit_to_track_dict()
        self.circ_station_dict = self.get_circuit_station_dict()

    def create_stations_dict(self):
        """
        creates a dict that maps station codes to names
        """
        stations_dict = {}
        for line_code in self.line_codes:
            stations = get_station_list(line_code)
            for station in stations['Stations']:
                stations_dict[station['Code']] = station['Name']
        return stations_dict

    def get_circuit_to_sequence_dict(self):
        """
        returns a dict that maps line_codes and track nums to sequence numbers
        """
        circ_seq_dict = {}
        for line_code in self.line_codes:
            for track_num in self.track_nums:
                circuits = [route for route in self.routes['StandardRoutes']
                            if route['LineCode'] == line_code
                            and route['TrackNum'] == track_num]
                circuits = circuits[0]['TrackCircuits']  # should only be one
                circ_seq_dict['{}{}'.format(line_code, track_num)] =\
                    {circuit['CircuitId']: circuit['SeqNum']
                     for circuit in circuits}
        return circ_seq_dict

    def get_circuit_to_track_dict(self):
        """
        creates a dict that maps a circuit id to a particular track (1 or 2)
        """
        circuits = self.circuits['TrackCircuits']
        return {circuit['CircuitId']: circuit['Track'] for circuit in circuits}

    def get_circuit_station_dict(self):
        """
        returns a dict that maps line_codes and track nums to sequence numbers
        """
        circ_station_dict = {}
        for line_code in self.line_codes:
            for track_num in self.track_nums:
                circuits = [route for route in self.routes['StandardRoutes']
                            if route['LineCode'] == line_code
                            and route['TrackNum'] == track_num]
                circuits = circuits[0]['TrackCircuits']  # should only be one
                circ_station_dict['{}{}'.format(line_code, track_num)] =\
                    {circuit['CircuitId']: circuit['StationCode']
                     for circuit in circuits if circuit['StationCode'] is not None}
        return circ_station_dict

class TripInfo():

    def __init__(self, date):
        self.start_time_dict = {}
        self.trip_dict = {}
        self.counter = itertools.count()
        self.date = date

    def id_gen(self):
        return "{date}_{count}".format(date=self.date.strftime("%Y-%m-%d"), count=next(self.counter))


    #id_dest_map = {}
    def start_time(self, row):
        """
        If the trian id and destination exists, then it is part of the same trip.
        If it is a new destination or the train is not in there, it is a new trip.
        """
        train_id = row['TrainId']
        cur_dest = row['DestinationStationCode']
        cur_time = row['Time']
        if train_id not in self.start_time_dict:
            self.start_time_dict[train_id] = (cur_dest, cur_time)
            return cur_time
        else:
            old_dest, old_time = self.start_time_dict[train_id]
            if cur_dest != old_dest: # if the train has a new dest, we overwrite the map
                self.start_time_dict[train_id] = (cur_dest, cur_time)
                return cur_time
            else:
                return old_time

    def unique_trips(self, row):
        train_id = row['TrainId']
        dest = row['DestinationStationCode']
        start = row['StartTime']
        trip = (train_id, dest, start)
        if trip not in self.trip_dict:
            trip_id = self.id_gen()
            self.trip_dict[trip] = trip_id
            return trip_id
        else:
            return self.trip_dict[trip]


def read_data(file_list):
    """
    Given a list of WMATA files from pushshift.io, read them in
    one at a time and then concatenate. Return a dataframe
    """
    df_list = []
    for file_name in file_list:
        with open(file_name, 'rb') as f:
            data_c = f.read()
            data_u = bz2.decompress(data_c)
            l = [json.loads(line.decode('utf-8')) for line in data_u.splitlines()]
            df_list.append(pd.DataFrame(l))
    df = pd.concat(df_list, axis=0, ignore_index=True)
    df = df[df['ServiceType'] != 'NoPassengers']
    df = df[df['ServiceType'] != 'Unknown']
    return df

def add_track(row, circ_track_dict):
    c_id = row['CircuitId']
    return circ_track_dict.get(c_id)

def add_seq_num(row, circ_seq_dict, circ_track_dict):
    c_id = row['CircuitId']
    line_code = row['LineCode']
    track = circ_track_dict.get(c_id)
    line = circ_seq_dict.get("{}{}".format(line_code, track), {})
    return line.get(c_id)

def add_station_code(row, circ_station_dict, circ_track_dict):
    c_id = row['CircuitId']
    line_code = row['LineCode']
    track = circ_track_dict.get(c_id)
    line = circ_station_dict.get("{}{}".format(line_code, track), {})
    return line.get(c_id)


def add_all_extras(df, date):
    m = train_info_maps()
    #f = partial(add_extra_data, train_info_maps=m)
    #df[['Track', 'SeqNum', 'StationCode', 'Time']] = df.apply(f, axis=1)
    circ_track_dict = m.circ_track_dict
    circ_seq_dict = m.circ_seq_dict
    circ_track_dict = m.circ_track_dict
    circ_station_dict = m.circ_station_dict
    df['Track'] = df.apply(lambda row: add_track(row, circ_track_dict), axis=1)
    df['SeqNum'] = df.apply(lambda row: add_seq_num(row, circ_seq_dict, circ_track_dict), axis=1)
    df['StationCode'] = df.apply(lambda row: add_station_code(row, circ_station_dict, circ_track_dict), axis=1)
    df['Time'] = df.apply(lambda row: date_parse(row['retrieved_on']), axis=1)
    df = df.dropna(subset=['Track', 'SeqNum', 'StationCode'])
    df.Track = df.Track.astype(int)
    df.SeqNum = df.SeqNum.astype(int)

    # now the tricky part of getting trips
    t = TripInfo(date)
    df['StartTime'] = df.apply(lambda row: t.start_time(row), axis=1)
    df['TripId'] = df.apply(lambda row: t.unique_trips(row), axis=1)
    return df


def process_day(date):
    # first we need to determine which files correspond to this date
    # The date will be eastern time, the filenames from pushshift are
    # UTC.
    # Note that each date will be 4am eastern until 4am eastern the next day
    # (or 3 to 3, it depends on if we're in DST or not)
    # trains should not be running before or after that time really.
    # this corresponds to getting 5hrs ahead
    day_1 = date.strftime("%Y-%m-%d")
    day_2 = (date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    download_dir = "data/"
    out_file = download_dir + "WMATA_train_trips_{}.csv".format(day_1)
    if os.path.isfile(out_file):
        print("{} already exists. Skipping".format(out_file))
        return

    times_day_1 = list(range(8,24))
    times_day_2 = list(range(0,9))
    fnames = ["WMATA_trains_{day}_{time:02d}.bz2".format(day=day_1, time=time) for time in times_day_1] \
        + ["WMATA_trains_{day}_{time:02d}.bz2".format(day=day_2, time=time) for time in times_day_2]
    paths = [download_dir + fname for fname in fnames]
    # check that each one is there
    files = os.listdir(download_dir)
    for fname in fnames:
        if fname not in files:
            print("Day {} not complete. Skipping".format(day_1))
            return
    # read and process
    print("Processing {}".format(day_1))
    try:
        df = read_data(paths)
    except KeyError:
        print("All files for {} may have been empty. Skipping. Check".format(day_1))
        return
    df = add_all_extras(df, date)
    df = df.drop_duplicates(['StationCode','LineCode','DestinationStationCode', 'TripId'])
    df = df.reset_index(drop=True)
    df.to_csv(out_file, index=False)
    return




if __name__ == "__main__":
    file_list = ['data/WMATA_trains_2016-08-25_{:02d}.bz2'.format(i) for i in range(1)]
    df = read_data(file_list)
    df = add_all_extras(df)

    df = df.drop_duplicates(['StationCode','LineCode','DestinationStationCode', 'TripId'])

