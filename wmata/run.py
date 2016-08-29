import load_data
import datetime
import pandas as pd

start = datetime.datetime(year=2016, month=7, day=21)
end = datetime.datetime(year=2016, month=8, day=25)
# process all the days!
d = datetime.datetime(year=2016, month=8, day=24)
dates = pd.date_range(start, end)
for date in dates:
    load_data.process_day(date)
