import rosbag
import rospy
import pandas
import os
import numpy as np
from itertools import chain

from optparse import OptionParser
import sys, os

def extract_chunks(file_in, m_per_chunk=1000):
    bagfile = rosbag.Bag(file_in)
    messages = bagfile.get_message_count()
    #m_per_chunk = int(round(float(messages) / float(chunks)))
    chunks = int(round(float( messages) / float(m_per_chunk) )) + 1
    chunk = 0
    m = 0

    dirname = os.path.basename(file_in).split('.bag')[0] + "_chunked"
    dirname = os.path.join(os.path.dirname(file_in), dirname)
    os.mkdir(dirname)

    basename = os.path.basename(file_in).split('.bag')[0] + "_chunk_"
    basename = os.path.join(dirname, basename)

    chunkname = basename + "%0" + str(int(np.log10(chunks))+2) + "d.bag"
    outbag = rosbag.Bag(chunkname % chunk, 'w')

    time_to_chunk = []
    for topic, msg, t in bagfile.read_messages():
        m += 1
        if m % m_per_chunk == 0:
            if len(time_to_chunk) > 0:
                time_to_chunk[-1]['time_end'] = t.secs + t.nsecs*1e-9
                print('ending chunk: ', t.secs + t.nsecs*1e-9)
            
            outbag.close()
            chunk += 1
            chunkname = basename + "%0" + str(int(np.log10(chunks))+2) + "d.bag"
            outbag = rosbag.Bag(chunkname % chunk, 'w')

            new_time_to_chunk = {'chunkname': chunkname % chunk, 'time_start': t.secs + t.nsecs*1e-9}
            time_to_chunk.append(new_time_to_chunk)
            print('new chunk: ', t.secs + t.nsecs*1e-9)

        outbag.write(topic, msg, t)

    if len(time_to_chunk) > 0:
        time_to_chunk[-1]['time_end'] = t.secs + t.nsecs*1e-9
        print('ending chunk: ', t.secs + t.nsecs*1e-9)

    outbag.close()

    pd = pandas.DataFrame(time_to_chunk)
    pd.to_hdf( os.path.join(dirname, "time_to_chunk_dict.hdf"), "time_to_chunk")

    return pandas.DataFrame(time_to_chunk)


def load_messages_for_chunked_bag(time_to_chunk, t0, t1):

    print(t0)
    print(time_to_chunk.time_start.min())
    print(time_to_chunk.time_start.max())
    first_chunk_idx = time_to_chunk[time_to_chunk.time_start>=t0].iloc[0:1].index[0] -1
    last_chunk_idx = time_to_chunk[time_to_chunk.time_start<=t1].iloc[-1:].index[0] + 1

    chunknames = time_to_chunk.query("index >= " + str(first_chunk_idx) + " and index <= " + str(last_chunk_idx)).chunkname.values

    master_msgs = None

    for chunkname in chunknames:
        if master_msgs is None:
            bag = rosbag.Bag(chunkname)
            master_msgs = bag.read_messages(start_time=rospy.Time(t0), end_time=rospy.Time(t1))
        else:
            bag = rosbag.Bag(chunkname)
            msgs = bag.read_messages(start_time=rospy.Time(t0), end_time=rospy.Time(t1))
            master_msgs = chain(master_msgs, msgs)

    return master_msgs

if __name__ == '__main__':
        
    ## Read data #############################################################
    parser = OptionParser()
    parser.add_option('--dvbag', type=str, default='none', help="name and path of the delta video bag file to chunk")
    (options, args) = parser.parse_args()
    

    extract_chunks(options.dvbag)