import pandas as pd
from datetime import timedelta
import xml.etree.ElementTree as et
from scipy.sparse.linalg import eigs
import numpy as np


def xml_to_pandas(xml_file):
    """Convert the station status xml file to a pandas dataframe"""
    tree = et.parse(xml_file)
    root = tree.getroot()

    l = []
    for station in root:
        d = {}
        for attrib in station:
            d[str(attrib.tag)] = str(attrib.text)
        l.append(d)

    df = pd.DataFrame.from_dict(l)
    return df

# read the data
data = pd.read_csv('data/2016-Q1-Trips-History-Data.csv',
                   parse_dates=['Start date', 'End date'],
                   infer_datetime_format=True)
data['Duration'] = data['Duration (ms)'].apply(lambda x:
                                               timedelta(milliseconds=int(x)))

data['End station number'] = data['End station number'].astype(int)
data['Start station number'] = data['Start station number'].astype(int)

data = data.dropna()
# after dropping some columns we can convert from float to int
bike_stations = xml_to_pandas('data/bike_stations.xml')
bike_stations['terminalName'] = bike_stations['terminalName'].astype(int)
bike_stations['lat'] = bike_stations['lat'].astype(float)
bike_stations['long'] = bike_stations['long'].astype(float)
station_locations = bike_stations[['terminalName']]
station_locations['location'] = list(zip(bike_stations['lat'],
                                         bike_stations['long']))

# merge location and usage info
data = data.merge(station_locations, how='left',
                  left_on='Start station number', right_on='terminalName')
data.columns = [w if w != 'location' else
                'start location' for w in data.columns]
data.drop('terminalName', axis=1, inplace=True)

data = data.merge(station_locations,
                  left_on='End station number', right_on='terminalName')
data.columns = [w if w != 'location' else
                'end location' for w in data.columns]
data.drop('terminalName', axis=1, inplace=True)
data = data[data['end location'].map(lambda x: isinstance(x, tuple))]
data = data[data['start location'].map(lambda x: isinstance(x, tuple))]

grouped = data.groupby(['Start station number', 'End station number']) \
    .Duration.count().unstack(level=1, fill_value=0)
m, n = grouped.shape

def k_max_indices(v, k=1):
    """
    Returns the indices of the k largest elements in a numpy array
    """
    # sloooow don't really use
    ind = np.argsort(np.absolute(v),axis=0)[-k:].flatten().tolist()[::-1]
    return ind


def eigenvector_centrality(A, num=1):
    """
    Given a square matrix returns the index of the most central
    node (column)
    """
    w, v = eigs(A, k=1, which='LM')
    ind = k_max_indices(v, num)
    return ind


def page_rank(A, p=0.15, num=1):
    """
    Compute the page rank of a given graph A and return the num largest
    values
    """
    m, n = A.shape
    assert(m == n)
    A = A/A.sum(axis=0)[None, :]
    B = np.full((m, n), 1/m, dtype=A.dtype)
    C = p*B + (1-p)*A
    w, v = eigs(C, k=1, which='LM')
    ind = k_max_indices(v, num)
    return ind


def hits(A, num=1):
    # https://cs7083.wordpress.com/2013/01/31/demystifying-the-pagerank-and-hits-algorithms/
    m, n = A.shape
    Hu = np.dot(A.T, A)
    Au = np.dot(A, A.T)
    w, a = eigs(Au, k=1, which='LM')
    w, h = eigs(Hu, k=1, which='LM')
    i, j = k_max_indices(a, num), k_max_indices(h, num)
    return i, j


def degree_centrality(A, num=1):
    m, n = A.shape
    max_in = A.sum(axis=1)
    max_out = A.sum(axis=0)
    max_in.shape = (m, 1)
    max_out.shape = (m, 1)
    in_degree = k_max_indices(max_in, num)
    out_degree = k_max_indices(max_out, num)
    return in_degree, out_degree


#grouped.iloc[0, 1] = 100000.0
A = grouped.as_matrix().astype(float)
# we transpose the matrix so that the link directions are swapped.
# This allows me to use my preferred column oriented (right) eigenvectors.
A = A.T

# Degree Centrality
num = 6
mi, mo = degree_centrality(A, num=num)
print("\nIn degree centrality: ")
for i in range(num):
    terminal_name = grouped.index[mi[i]]
    print(bike_stations[bike_stations['terminalName'] == terminal_name].name.values)
    print('Out degree: {}, In degree: {}'.format(A[:,mi[i]].sum(), A[mi[i],:].sum()))

print("\nOut degree centrality: ")
for i in range(num):
    terminal_name = grouped.index[mo[i]]
    print(bike_stations[bike_stations['terminalName'] == terminal_name].name.values)
    print('Out degree: {}, In degree: {}'.format(A[:,mo[i]].sum(), A[mo[i],:].sum()))

# Eigevector centrality
ma = eigenvector_centrality(A, num=num)
print("\nEigenvector centrality: ")
for i in range(num):
    terminal_name = grouped.index[ma[i]]
    print(bike_stations[bike_stations['terminalName'] == terminal_name].name.values)
    print('Out degree: {}, In degree: {}'.format(A[:,ma[i]].sum(), A[ma[i],:].sum()))

# PageRank
ma = page_rank(A, num=num)
print("\nPage rank: ")
for i in range(num):
    terminal_name = grouped.index[ma[i]]
    print(bike_stations[bike_stations['terminalName'] == terminal_name].name.values)
    print('Out degree: {}, In degree: {}'.format(A[:,ma[i]].sum(), A[ma[i],:].sum()))

# Hits
ma, mh = hits(A, num=num)
print("\nAuthority: ")
for i in range(num):
    terminal_name = grouped.index[ma[i]]
    print(bike_stations[bike_stations['terminalName'] == terminal_name].name.values)
    print('Out degree: {}, In degree: {}'.format(A[:,ma[i]].sum(), A[ma[i],:].sum()))

print("\nHub: ")
for i in range(num):
    terminal_name = grouped.index[mh[i]]
    print(bike_stations[bike_stations['terminalName'] == terminal_name].name.values)
    print('Out degree: {}, In degree: {}'.format(A[:,mh[i]].sum(), A[mh[i],:].sum()))
