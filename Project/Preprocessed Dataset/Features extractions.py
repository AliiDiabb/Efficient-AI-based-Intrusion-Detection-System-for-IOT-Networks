import pandas as pd


#This is the features extractions from the Datasest EdgeIIoT
Deploable_Features = [
    'arp.dst.proto_ipv4', 'arp.opcode', 'arp.hw.size', 'arp.src.proto_ipv4',
    'icmp.checksum', 'icmp.seq_le', 'icmp.unused',
    'http.content_length', 'http.request.method', 'http.referer', 'http.request.version', 'http.response',
    'tcp.ack', 'tcp.ack_raw', 'tcp.checksum', 'tcp.connection.fin', 'tcp.connection.rst',
    'tcp.connection.syn', 'tcp.connection.synack', 'tcp.dstport', 'tcp.flags', 'tcp.flags.ack',
    'tcp.len', 'tcp.seq', 'tcp.srcport',
    'udp.port', 'udp.stream', 'udp.time_delta',
    'dns.qry.name.len', 'dns.qry.type', 'dns.retransmission', 'dns.retransmit_request', 'dns.retransmit_request_in',
    'mqtt.conack.flags', 'mqtt.conflag.cleansess', 'mqtt.conflags', 'mqtt.hdrflags',
    'mqtt.len', 'mqtt.msgtype', 'mqtt.proto_len', 'mqtt.protoname', 'mqtt.topic', 'mqtt.topic_len', 'mqtt.ver',
    'mbtcp.len', 'mbtcp.trans_id', 'mbtcp.unit_id'
]




# Load the ML CSV file
path = "Preprocessed DataSet/Preprocessed-ML-EdgeIIoT-dataset.csv"
path1="Preprocessed DataSet/Preprocessed-DNN-EdgeIIoT-dataset.csv"

df = pd.read_csv(path1)

# Select only the deployable features and target columns
selected_columns = Deploable_Features + ['Attack_label', 'Attack_type']
df_filtered = df[selected_columns]

# Save the filtered dataset
df_filtered.to_csv("ML_Depoable_Features.csv", index=False)

print(f"Filtered dataset saved with {len(df_filtered.columns)} columns and {len(df_filtered)} rows")




