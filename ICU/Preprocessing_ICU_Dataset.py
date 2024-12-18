# %% [markdown]
# ### Import Library

# %%
import pandas as pd

# %% [markdown]
# ### Load ICU Dataset

# %%
file_path = r"D:\Hungary\Semester 2\Open-Source Technologies for Data Science\Practice\Project\Datasets\ICU_Dataset.csv"
ICU_data = pd.read_csv(file_path)
ICU_data.head()

# %% [markdown]
# ### Remove unnecessary Columns

# %%
ICU_data = ICU_data.drop(columns=[
    'frame.time_relative', 'frame.len', 'ip.src', 'ip.dst',
    'tcp.srcport', 'tcp.dstport', 'tcp.flags', 'tcp.len', 'tcp.ack',
    'tcp.connection.fin', 'tcp.connection.rst', 'tcp.connection.sack',
    'tcp.connection.syn', 'tcp.flags.fin', 'tcp.flags.syn', 'tcp.flags.urg',
    'tcp.hdr_len', 'tcp.payload', 'tcp.pdu.size', 'tcp.window_size_value',
    'tcp.checksum', 'mqtt.clientid', 'mqtt.clientid_len', 'mqtt.conack.flags',
    'mqtt.conack.val', 'mqtt.conflag.passwd', 'mqtt.conflag.qos',
    'mqtt.conflag.reserved', 'mqtt.conflag.retain', 'mqtt.conflag.willflag',
    'mqtt.conflags', 'mqtt.dupflag', 'mqtt.kalive', 'mqtt.len', 'mqtt.msg',
    'mqtt.topic', 'mqtt.topic_len', 'mqtt.willmsg_len', 'ip.proto', 'ip.ttl'
])



# %%
ICU_data

# %% [markdown]
# ### Save Dataset File

# %%
file_path = r"D:\Hungary\Semester 2\Open-Source Technologies for Data Science\Practice\Project\Datasets\Modified_ICU_Dataset.csv"
ICU_data.to_csv(file_path, index=False)
print(f"File saved successfully at {file_path}")

# %%



