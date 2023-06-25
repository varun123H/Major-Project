from influxdb_client import InfluxDBClient

# Specify your InfluxDB connection details
token = 'psMVu9OR0PbWeZzYAdKDMCYAKUCSd0ohsz1CLpU0kDuzLkcKL3eWRQBOxEMM6ZH0odO8rvvZbOPPZvQjW_A-Gw=='
org = 'major'
bucket = 'mine'

# Create an InfluxDB client
client = InfluxDBClient(url="http://localhost:8086", token=token)

# Construct the Flux query
query = f'from(bucket:"{bucket}") |> range(start: 0)'

# Execute the query
result = client.query_api().query(org=org, query=query)

# Print the query result
for table in result:
    for record in table.records:
        print(record.values)
